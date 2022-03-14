"""Generate plot of number of candidates vs expectation and variance."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from constants import CHARTS_DIRECTORY
from pykeen.metrics.ranking import (
    ArithmeticMeanRank, GeometricMeanRank, HarmonicMeanRank, HitsAtK, InverseArithmeticMeanRank,
    InverseGeometricMeanRank, InverseHarmonicMeanRank,
)

sns.set_style("white")


def main():
    num_samples = 500  # any bigger is unnecessary
    num_candidates = np.logspace(np.log10(2), 7, num=60).reshape(-1, 1)
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
    inverse_expectations, inverse_variances = _get_dfs(
        inverse_metrics, num_candidates=num_candidates, num_samples=num_samples,
    )
    expectations, variances = _get_dfs(
        metrics, num_candidates=num_candidates, num_samples=num_samples,
    )

    fig, ((lax, rax), (lax_inv, rax_inv)) = plt.subplots(2, 2, figsize=(10, 7))

    sns.lineplot(data=expectations, x="x", y="y", hue="metric", ax=lax)
    lax.set_xlabel("")
    lax.set_ylabel("Expectation")
    lax.set_xscale("log")

    sns.lineplot(data=variances, x="x", y="y", hue="metric", ax=rax)
    rax.set_xlabel("")
    rax.set_ylabel("Variance")
    rax.set_xscale("log")

    sns.lineplot(data=inverse_expectations, x="x", y="y", hue="metric", ax=lax_inv)
    lax_inv.set_xlabel("Number of Candidates")
    lax_inv.set_ylabel("Expectation")
    lax_inv.set_xscale("log")

    sns.lineplot(data=inverse_variances, x="x", y="y", hue="metric", ax=rax_inv)
    rax_inv.set_xlabel("Number of Candidates")
    rax_inv.set_ylabel("Variance")
    rax_inv.set_xscale("log")

    fig.tight_layout()
    fig.savefig(CHARTS_DIRECTORY.joinpath("candidate_plot.svg"))
    fig.savefig(CHARTS_DIRECTORY.joinpath("candidate_plot.png"), dpi=300)
    fig.savefig(CHARTS_DIRECTORY.joinpath("candidate_plot.pdf"))


def _get_dfs(metrics, num_candidates, num_samples):
    expectations = pd.DataFrame(
        [
            (metric.__class__.__name__, x[0].item(), metric.expected_value(x, num_samples))
            for metric in metrics
            for x in num_candidates
        ],
        columns=["metric", "x", "y"]
    )
    variances = pd.DataFrame(
        [
            (metric.__class__.__name__, x[0].item(), metric.variance(x, num_samples))
            for metric in metrics
            for x in num_candidates
        ],
        columns=["metric", "x", "y"]
    )
    return expectations, variances


if __name__ == '__main__':
    main()
