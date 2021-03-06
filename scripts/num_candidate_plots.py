"""Generate plot of number of candidates vs expectation and variance."""

import click
import numpy as np
import pandas as pd
import seaborn as sns
from constants import CHARTS_DIRECTORY, COLLATION_DIRECTORY
from more_click import force_option
from pykeen.metrics import RankBasedMetric
from pykeen.metrics.ranking import (
    ArithmeticMeanRank,
    GeometricMeanRank,
    HarmonicMeanRank,
    HitsAtK,
    InverseArithmeticMeanRank,
    InverseGeometricMeanRank,
    InverseHarmonicMeanRank,
)
from tqdm import tqdm

CANDIDATE_PATH = COLLATION_DIRECTORY.joinpath("candidates.tsv")
CHART_STUB = CHARTS_DIRECTORY.joinpath("candidate_plot")


@click.command()
@force_option
def main(
    force: bool,
    num_samples: int = 1_000,
    min_log10: int = 0,
    max_log10: int = 6,
    points: int = 40,
    legend_bottom: bool = False
):
    sizes = np.logspace(min_log10, max_log10, num=points).astype(int)

    normal_metrics = [
        HarmonicMeanRank(),
        ArithmeticMeanRank(),
        GeometricMeanRank(),
    ]
    inverse_metrics = [
        InverseHarmonicMeanRank(),
        InverseArithmeticMeanRank(),
        InverseGeometricMeanRank(),
        *(HitsAtK(10**x) for x in range(max_log10 // 2)),
    ]
    metrics = [
        *((metric, False) for metric in normal_metrics),
        *((metric, True) for metric in inverse_metrics),
    ]

    if CANDIDATE_PATH.is_file() and not force:
        df = pd.read_csv(CANDIDATE_PATH, sep="\t")
    else:
        df = _get_df(
            metrics,
            sizes=sizes,
            num_samples=num_samples,
        )
        df.to_csv(CANDIDATE_PATH, sep="\t", index=False)

    data = df.melt(
        id_vars=["metric", "candidate", "inverted"],
        value_vars=["expectation", "variance"],
    )
    data["metric"] = data["metric"].map(lambda s: s.replace("_", " ").title())
    grid: sns.FacetGrid = sns.relplot(
        data=data,
        x="candidate",
        y="value",
        col="variable",
        row="inverted",
        hue="metric",
        kind="line",
        facet_kws={"sharey": False, "sharex": True},
        height=2.3,
        aspect=1.6,
    )
    grid.set(xscale="log", yscale="log", ylabel="Value", xlabel="Number of Candidates")
    if legend_bottom:
        sns.move_legend(
            grid,
            "lower center",
            bbox_to_anchor=(0.40, -0.175),
            ncol=3,
            title="Metric",
            frameon=True,
        )
    grid.tight_layout()
    grid.savefig(CHART_STUB.with_suffix(".png"), dpi=300)
    grid.savefig(CHART_STUB.with_suffix(".svg"))
    grid.savefig(CHART_STUB.with_suffix(".pdf"))


def _get_df(
    metrics: list[RankBasedMetric],
    sizes: np.ndarray,
    num_samples: int,
) -> pd.DataFrame:
    rows = []
    outer_it = tqdm(sizes)
    for size in outer_it:
        size_item = size.item()
        outer_it.set_postfix(size=size_item)
        num_candidates = np.array([size_item for _ in range(size_item)])
        inner_it = tqdm(metrics, leave=False)
        for metric, inverted in inner_it:
            inner_it.set_postfix(metric=metric)
            rows.append(
                (
                    metric.key.removeprefix("inverse_"),
                    inverted,
                    size_item,
                    metric.expected_value(num_candidates=num_candidates, num_samples=num_samples),
                    metric.variance(num_candidates=num_candidates, num_samples=num_samples),
                )
            )
    return pd.DataFrame(
        rows,
        columns=["metric", "inverted", "candidate", "expectation", "variance"],
    )


if __name__ == "__main__":
    main()
