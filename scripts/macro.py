"""Investigate effect of micro vs. macro evaluation."""

import logging
from pathlib import Path
from typing import Collection, Iterable, Optional, Tuple, Type

import click
import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn
from docdata import get_docdata
from matplotlib.ticker import PercentFormatter
from pykeen.datasets import Dataset, dataset_resolver, get_dataset
from sklearn.metrics import auc
from tqdm.auto import tqdm
from tqdm.contrib.itertools import product as tqdm_product
from tqdm.contrib.logging import logging_redirect_tqdm

from constants import CHARTS_DIRECTORY, COLLATION_DIRECTORY, DEFAULT_CACHE_DIRECTORY


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
            df = pandas.DataFrame(factory.mapped_triples.numpy(), columns=["h", "r", "t"])

            # for each side prediction
            for target in "ht":
                # group by other columns
                keys = [c for c in df.columns if c != target]
                group = df.groupby(by=keys)

                # count number of unique targets per key
                counts = group.agg({target: "nunique"})[target]

                yield dataset_name, split, target, df.shape[0], counts


def _save(df: pandas.DataFrame, output_directory: Path, *keys: str):
    """Save dataframe under directory."""
    path = output_directory.joinpath(*keys).with_suffix(suffix=".tsv.gz")
    path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(path, sep="\t", index=False)
    logging.debug(f"Written to {path}")


directory_option = click.option("--directory", type=Path, default=DEFAULT_CACHE_DIRECTORY)


@click.group()
def main():
    """Run command."""


@main.command()
@click.option("-m", "--max-triples", type=int, default=None)
@directory_option
def collect(
    max_triples: Optional[int],
    directory: Path,
):
    """Collect statistics across datasets."""
    # logging setup
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("pykeen.triples").setLevel(level=logging.ERROR)

    # make sure the directory exists
    directory.mkdir(exist_ok=True, parents=True)

    # aggregate basic statistics across all datasets
    summary = []

    # iterate over all datasets
    with logging_redirect_tqdm(), tqdm(_iter_counts(max_triples=max_triples)) as progress:
        for dataset_name, split, target, num_triples, counts in progress:
            progress.set_postfix(datase=dataset_name, split=split, target=target)

            # append basic statistics to summary data(-frame)
            description = counts.describe()
            summary.append((dataset_name, split, target, num_triples, *description.tolist()))

            # store distribution
            df = pandas.DataFrame(data=dict(counts=counts))
            _save(df, directory, dataset_name, split, target)

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
    _save(df, directory, "summary")


@main.command()
@directory_option
@click.option(
    "-d",
    "--dataset",
    multiple=True,
    type=str,
    default=[],  # ("fb15k237", "kinships", "nations", "wn18rr"),
)
@click.option(
    "-s",
    "--split",
    type=click.Choice(["training", "testing", "validation"]),
    default="testing",
)
@click.option("-p", "--palette", type=str, default=None)
@click.option("-g", "--grid", is_flag=True)
@click.option("-h", "--height", type=float, default=5, show_default=True)
def plot(
    directory: Path,
    dataset: Collection[str],
    split: str,
    palette: Optional[str],
    grid: bool,
    height: Optional[float],
):
    """Create plot."""
    # logging setup
    logging.basicConfig(level=logging.INFO)

    if not dataset:
        suffix = f"{split}_full"
        dataset = [path.name for path in directory.iterdir() if path.is_dir()]
        logging.info(f"Inferred datasets: {dataset} (by crawling {directory})")
    else:
        suffix = split

    logging.info("Calculating CDFs")
    data = []
    for ds, target in tqdm_product(dataset, "ht"):
        df = pandas.read_csv(directory.joinpath(ds, split, target).with_suffix(".tsv.gz"), sep="\t")
        cs = df["counts"].sort_values(ascending=False)
        cs = cs.cumsum()
        cdf = cs / cs.iloc[-1]
        x = numpy.linspace(0, 1, num=len(cdf) + 1)[1:]
        data.extend((ds, target, xx, yy) for xx, yy in zip(x, cdf))
    df = pandas.DataFrame(data, columns=["dataset", "target", "x", "y"])
    df["target"] = df["target"].apply({"t": "tail", "h": "head"}.__getitem__)
    df.to_csv(COLLATION_DIRECTORY.joinpath(f"macro_{suffix}.tsv.gz"), sep="\t", index=False)

    logging.info(f"Creating plot for {len(dataset)} datasets.")
    kwargs = dict(style="target") if len(dataset) < 5 else dict(col="target")
    facet_grid: seaborn.FacetGrid = seaborn.relplot(
        data=df,
        x="x",
        y="y",
        hue="dataset",
        kind="line",
        facet_kws=dict(xlim=[0, 1], ylim=[0, 1]),
        **kwargs,
        palette=palette,
        height=height,
    )
    for ax in facet_grid.axes.flat:
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        if grid:
            ax.grid()

    # Calculate area under the curve
    auc_rows = [
        (dataset, side, auc(sdf["x"], sdf["y"]))
        for (dataset, side), sdf in df.groupby(["dataset", "target"])
    ]
    auc_df = pandas.DataFrame(auc_rows, columns=["dataset", "target", "auc"])
    auc_df.to_csv(COLLATION_DIRECTORY.joinpath(f"macro_{suffix}_auc.tsv"), sep="\t", index=False)

    logging.info("Calculating area under the curve")
    auc_diffs = []
    for dataset, sdf in auc_df.groupby("dataset"):
        head = sdf[sdf.target == "head"].iloc[0].auc
        tail = sdf[sdf.target == "tail"].iloc[0].auc
        size = _triples(dataset_resolver.lookup(dataset))
        auc_diffs.append((dataset, size, head - tail))
    auc_diffs_df = pandas.DataFrame(auc_diffs, columns=["dataset", "size", "diff"])
    auc_diffs_df.sort_values("diff", inplace=True)
    auc_diffs_df.to_csv(
        COLLATION_DIRECTORY.joinpath(f"macro_{suffix}_auc_diff.tsv"),
        sep="\t",
        index=False,
    )
    fig, ax = plt.subplots()
    seaborn.barplot(data=auc_diffs_df, y="dataset", x="diff", ax=ax)
    ax.set_ylabel("")
    ax.set_xlabel("AUC Difference (head - tail)")
    fig.tight_layout()
    auc_diff_stub = CHARTS_DIRECTORY.joinpath(f"macro_{suffix}_auc_diff")
    fig.savefig(auc_diff_stub.with_suffix(".png"), dpi=300)
    fig.savefig(auc_diff_stub.with_suffix(".pdf"))
    fig.savefig(auc_diff_stub.with_suffix(".svg"))
    plt.close(fig)

    facet_grid.set_xlabels(label="Percentage of unique ranking tasks")
    facet_grid.set_ylabels(label="Percentage of evaluation triples")
    facet_grid.tight_layout()
    output_stub = CHARTS_DIRECTORY.joinpath(f"macro_{suffix}_plot")
    path = output_stub.with_suffix(".pdf")
    facet_grid.savefig(path)
    facet_grid.savefig(output_stub.with_suffix(".svg"))
    facet_grid.savefig(output_stub.with_suffix(".png"), dpi=300)
    logging.info(f"Saved to {path}")


if __name__ == "__main__":
    main()
