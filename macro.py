"""Investigate effect of micro vs. macro."""
import logging
import pathlib
import tempfile
from typing import Collection, Iterable, Optional, Tuple, Type

import click
import numpy
import pandas
import seaborn
from docdata import get_docdata
from matplotlib.ticker import PercentFormatter
from pykeen.datasets import Dataset, dataset_resolver, get_dataset
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

DEFAULT_DIRECTORY = pathlib.Path(tempfile.gettempdir(), "pykeen-macro")


def _triples(dataset_cls: Type[Dataset]) -> int:
    """Extract the number of triples from docdata."""
    return get_docdata(dataset_cls)["statistics"]["triples"]


def _iter_counts(
    max_triples: Optional[int],
) -> Iterable[Tuple[str, str, str, int, pandas.Series]]:
    """Iterate over datasets/splits/targets and the respective counts."""
    for dataset_name, dataset_cls in sorted(
        dataset_resolver.lookup_dict.items(), key=lambda pair: _triples(pair[1])
    ):
        # skip large datasets
        if max_triples and _triples(dataset_cls) > max_triples:
            continue

        # skip unavailable datasets
        try:
            dataset = get_dataset(dataset=dataset_name)
        except Exception as error:
            logging.error(str(error))
            continue

        # investigate triples for all splits
        for split, factory in dataset.factory_dict.items():

            # convert to pandas dataframe
            df = pandas.DataFrame(
                factory.mapped_triples.numpy(), columns=["h", "r", "t"]
            )

            # for each side prediction
            for target in "ht":

                # group by other columns
                keys = [c for c in df.columns if c != target]
                group = df.groupby(by=keys)

                # count number of unique targets per key
                counts = group.agg({target: "nunique"})[target]

                yield dataset_name, split, target, df.shape[0], counts


def _save(df: pandas.DataFrame, output_directory: pathlib.Path, *keys: str):
    """Save dataframe under directory."""
    path = output_directory.joinpath(*keys).with_suffix(suffix=".tsv.gz")
    path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(path, sep="\t", index=False)
    logging.debug(f"Written to {path}")


@click.group()
def main():
    """Run command."""


@main.command()
@click.option("-m", "--max-triples", type=int, default=None)
@click.option("-o", "--output-directory", type=pathlib.Path, default=DEFAULT_DIRECTORY)
def collect(
    max_triples: Optional[int],
    output_directory: pathlib.Path,
):
    """Collect statistics across datasets."""
    # logging setup
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("pykeen.triples").setLevel(level=logging.ERROR)

    # aggregate basic statistics across all datasets
    summary = []

    # iterate over all datasets
    with logging_redirect_tqdm(), tqdm(
        _iter_counts(max_triples=max_triples)
    ) as progress:
        for dataset_name, split, target, num_triples, counts in progress:
            progress.set_postfix(datase=dataset_name, split=split, target=target)

            # append basic statistics to summary data(-frame)
            description = counts.describe()
            summary.append(
                (dataset_name, split, target, num_triples, *description.tolist())
            )

            # store distribution
            df = pandas.DataFrame(data=dict(counts=counts))
            _save(df, output_directory, dataset_name, split, target)

    # store summary
    df = pandas.DataFrame(
        data=summary,
        columns=[
            "dataset",
            "split",
            "side",
            "num_triples",
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
        ],
    )
    _save(df, output_directory, "summary")


@main.command()
@click.option("-i", "--input-root", type=pathlib.Path, default=DEFAULT_DIRECTORY)
@click.option(
    "-d",
    "--dataset",
    multiple=True,
    type=str,
    default=[],#("fb15k237", "kinships", "nations", "wn18rr"),
)
@click.option(
    "-s",
    "--split",
    type=click.Choice(["training", "testing", "validation"]),
    default="testing",
)
def plot(
    input_root: pathlib.Path,
    dataset: Collection[str],
    split: str,
):
    """Create plot."""
    if not dataset:
        dataset = [path.name for path in input_root.iterdir() if path.is_dir()]
    data = []
    for ds in dataset:
        for target in "ht":
            df = pandas.read_csv(
                input_root.joinpath(ds, split, target).with_suffix(".tsv.gz"), sep="\t"
            )
            cs = df["counts"].sort_values(ascending=False)
            cs = cs.cumsum()
            cdf = cs / cs.iloc[-1]
            x = numpy.linspace(0, 1, num=len(cdf) + 1)[1:]
            data.extend((ds, target, xx, yy) for xx, yy in zip(x, cdf))
    df = pandas.DataFrame(data, columns=["dataset", "target", "x", "y"])
    kwargs = dict(style="target") if len(dataset) < 5 else dict(col="target")
    grid: seaborn.FacetGrid = seaborn.relplot(
        data=df,
        x="x",
        y="y",
        hue="dataset",
        kind="line",
        facet_kws=dict(xlim=[0, 1], ylim=[0, 1]),
        **kwargs,
    )
    for ax in grid.axes.flat:
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    grid.set_xlabels(label="Percentage of unique ranking tasks")
    grid.set_ylabels(label="Percentage of evaluation triples")
    grid.savefig(input_root.joinpath("plot.pdf"))


if __name__ == "__main__":
    main()
