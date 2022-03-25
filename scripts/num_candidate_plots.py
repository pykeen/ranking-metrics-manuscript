"""Generate plot of number of candidates vs expectation and variance."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.contrib.itertools import product

from constants import CHARTS_DIRECTORY
from pykeen.metrics import RankBasedMetric
from pykeen.metrics.ranking import (
    ArithmeticMeanRank, GeometricMeanRank, HarmonicMeanRank, HitsAtK, InverseArithmeticMeanRank,
    InverseGeometricMeanRank, InverseHarmonicMeanRank,
)

sns.set_style("white")


def main():
    num_samples = 500  # any bigger is unnecessary
    num_candidates = np.logspace(np.log10(2), 9, num=60).reshape(-1, 1).astype(int)
    inverse_metrics = [
        InverseHarmonicMeanRank(),
        InverseArithmeticMeanRank(),
        InverseGeometricMeanRank(),
        HitsAtK(10),
    ]
    metrics = [
        HarmonicMeanRank(),
        ArithmeticMeanRank(),
        GeometricMeanRank(),
    ]
    inverse_df = _get_dfs(
        inverse_metrics, num_candidates=num_candidates, num_samples=num_samples,
    )
    normal_df = _get_dfs(
        metrics, num_candidates=num_candidates, num_samples=num_samples,
    )

    fig, ((lax, rax), (lax_inv, rax_inv)) = plt.subplots(2, 2, figsize=(10, 7))

    sns.lineplot(data=normal_df, x="candidate", y="expectation", hue="metric", ax=lax)
    lax.set_xlabel("")
    lax.set_ylabel("Expectation")
    lax.set_xscale("log")
    # lax.set_yscale("log")

    sns.lineplot(data=normal_df, x="candidate", y="variance", hue="metric", ax=rax)
    rax.set_xlabel("")
    rax.set_ylabel("Variance")
    rax.set_xscale("log")

    sns.lineplot(data=inverse_df, x="candidate", y="expectation", hue="metric", ax=lax_inv)
    lax_inv.set_xlabel("Number of Candidates")
    lax_inv.set_ylabel("Expectation")
    lax_inv.set_xscale("log")

    sns.lineplot(data=inverse_df, x="candidate", y="variance", hue="metric", ax=rax_inv)
    rax_inv.set_xlabel("Number of Candidates")
    rax_inv.set_ylabel("Variance")
    rax_inv.set_xscale("log")

    fig.tight_layout()
    fig.savefig(CHARTS_DIRECTORY.joinpath("candidate_plot.svg"))
    fig.savefig(CHARTS_DIRECTORY.joinpath("candidate_plot.png"), dpi=300)
    fig.savefig(CHARTS_DIRECTORY.joinpath("candidate_plot.pdf"))


def _get_dfs(
    metrics: list[RankBasedMetric],
    num_candidates: np.ndarray,
    num_samples: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            (
                metric.__class__.__name__,
                candidate[0].item(),
                metric.expected_value(candidate, num_samples),
                metric.variance(candidate, num_samples),
            )
            for metric, candidate in product(metrics, num_candidates, unit_scale=True, desc="calculating properties")
        ],
        columns=["metric", "candidate", "expectation", "variance"]
    )


if __name__ == '__main__':
    main()
