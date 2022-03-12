"""Investigate effect of micro vs. macro evaluation."""

import logging
from pathlib import Path
from typing import Collection, Iterable, Optional, Tuple

import click
import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import auc
from tqdm.auto import tqdm
from tqdm.contrib.itertools import product as tqdm_product
from tqdm.contrib.logging import logging_redirect_tqdm

from constants import CHARTS_DIRECTORY, COLLATION_DIRECTORY, DEFAULT_CACHE_DIRECTORY
from pykeen.datasets import Dataset, dataset_resolver
from pykeen.datasets.utils import iter_dataset_instances, max_triples_option


def _iter_counts(
    *,
    min_triples: Optional[int] = None,
    max_triples: Optional[int] = None,
    directory: Path,
    force: bool = False,
) -> Iterable[Tuple[str, str, str, int, pandas.Series]]:
    """Iterate over datasets/splits/targets and the respective counts."""
    for dataset_name, dataset in iter_dataset_instances(
        min_triples=min_triples, max_triples=max_triples
    ):
        # investigate triples for all splits
        split_it = tqdm(dataset.factory_dict.items(), desc="Splits", leave=False)
        for split, factory in split_it:
            split_it.set_postfix(split=split)

            # convert to pandas dataframe
            split_df = pandas.DataFrame(factory.mapped_triples.numpy(), columns=["h", "r", "t"])
            split_triples = split_df.shape[0]

            # for each side prediction
            target_it = tqdm("ht", desc="Target", leave=False)
            for target in target_it:
                target_it.set_postfix(target=target)

                path = directory.joinpath(dataset_name, split, target).with_suffix(suffix=".tsv.gz")
                if path.is_file() and not force:
                    logging.info(f"using cache at {path}")
                    counts_series = pandas.read_csv(path, sep="\t", squeeze=True)
                else:
                    # group by other columns
                    keys = [column for column in split_df.columns if column != target]
                    group = split_df.groupby(by=keys)

                    # count number of unique targets per key
                    counts_series = group.agg({target: "nunique"})[target]

                    # store distribution

                    # df = pandas.DataFrame(data=dict(counts=counts))
                    counts_series.to_csv(path, sep="\t", index=False)
                    logging.debug(f"wrote cache to {path}")

                description = counts_series.describe().tolist()
                yield dataset_name, split, target, split_triples, counts_series, description


directory_option = click.option("--directory", type=Path, default=DEFAULT_CACHE_DIRECTORY)


@click.group()
def main():
    """Run command."""


@main.command()
@max_triples_option
@directory_option
def collect(
    max_triples: Optional[int],
    directory: Path,
):
    """Collect statistics across datasets."""
    # logging setup
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("pykeen.triples").setLevel(level=logging.ERROR)
    directory.mkdir(exist_ok=True, parents=True)

    with logging_redirect_tqdm():
        # aggregate basic statistics across all datasets
        rows = [
            (dataset_name, split, target, num_triples, *description)
            for dataset_name, split, target, num_triples, counts_series, description in _iter_counts(
                max_triples=max_triples, directory=directory
            )
        ]

    # store summary
    summary_df = pandas.DataFrame(
        data=rows,
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
    summary_path = directory.joinpath("summary").with_suffix(suffix=".tsv.gz")
    summary_df.to_csv(summary_path, sep="\t", index=False)
    logging.debug(f"Wrote summary to {summary_path}")


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
        path = directory.joinpath(ds, split, target).with_suffix(".tsv.gz")
        counts_series: pandas.Series = pandas.read_csv(path, sep="\t", squeeze=True)
        cs = counts_series.sort_values(ascending=False).cumsum()
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
        size = Dataset.triples_sort_key(dataset_resolver.lookup(dataset))
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
    facet_grid.set_ylabels(label=f"Percentage of {split} triples")
    facet_grid.tight_layout()
    output_stub = CHARTS_DIRECTORY.joinpath(f"macro_{suffix}_plot")
    path = output_stub.with_suffix(".pdf")
    facet_grid.savefig(path)
    facet_grid.savefig(output_stub.with_suffix(".svg"))
    facet_grid.savefig(output_stub.with_suffix(".png"), dpi=300)
    logging.info(f"Saved to {path}")


if __name__ == "__main__":
    main()
