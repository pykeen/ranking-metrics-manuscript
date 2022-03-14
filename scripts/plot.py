"""Generate plot."""

import matplotlib.pyplot as plt
import pandas as pd
import scipy.constants
import seaborn as sns

from constants import CHARTS_DIRECTORY, MELTED_PATH

MODEL_TITLES = {
    "complex": "ComplEx",
    "tucker": "TuckER",
    "rotate": "RotatE",
    "transe": "TransE",
}
DATASET_TITLES = {
    "fb15k237": "FB15k-237",
    "wn18rr": "WN18-RR",
    "nations": "Nations",
    "kinships": "Kinships",
}
ORDER = [
    "Original",
    "Adjusted Index",
    "z-Adjusted Metric",
]
METRICS = {
    "mean_reciprocal_rank": {
        "base_title": "Mean Reciprocal Rank",
        "base_ylim": [0, 1],
        "metrics": [
            "inverse_harmonic_mean_rank",
            "adjusted_inverse_harmonic_mean_rank",
            "z_inverse_harmonic_mean_rank",
        ],
        "short": ["MRR", "AMRR", "ZMRR"],
    },
    # "arithmetic_mean_rank": {
    #     "base_title": "Mean Rank",
    #     "base_yscale": "log",
    #     "metrics": [
    #         "arithmetic_mean_rank",
    #         "adjusted_arithmetic_mean_rank_index",
    #         "z_arithmetic_mean_rank",
    #     ],
    # },
    "hits_at_10": {
        "base_title": "Hits at 10",
        "base_ylim": [0, 1],
        "metrics": [
            "hits_at_10",
            "adjusted_hits_at_10",
            "z_hits_at_10",
        ],
        "short": ["$H_{10}$", "$AH_{10}$", "$ZH_{10}$"],
    },
}


def main():
    melted_df = pd.read_csv(MELTED_PATH, sep="\t")
    melted_df.loc[:, "model"] = melted_df["model"].map(MODEL_TITLES)
    melted_df.loc[:, "dataset"] = melted_df["dataset"].map(DATASET_TITLES)
    _plot_summary(melted_df)

    for base_metric_key, metadata in METRICS.items():
        metrics = metadata["metrics"]
        df = melted_df[melted_df["variable"].isin(metrics)].copy()
        metric_order = [
            f"{order} ({short})"
            for order, short in zip(ORDER, metadata["short"])
        ]
        df.loc[:, "variable"] = df["variable"].map(dict(zip(metrics, metric_order)))
        grid: sns.FacetGrid = sns.catplot(
            data=df,
            x="model",
            y="value",
            col="variable",
            col_order=metric_order,
            # hue="model",
            hue="dataset",
            hue_order=["FB15k-237", "WN18-RR", "Nations", "Kinships"],
            sharey=False,
            # facet_kws=dict(sharey=False),
            kind="bar",
            height=2.2,
            aspect=scipy.constants.golden,
        )
        # grid.set_xticklabels(rotation=30, ha="right")
        for key, ax in grid.axes_dict.items():
            if key.startswith("z-Adjusted Metric"):
                ax.set_yscale("log")
            elif key.startswith("Adjusted Index"):
                ax.set_ylim([0, 1])
            else:  # base metric
                if "base_ylim" in metadata:
                    ax.set_ylim(metadata["base_ylim"])
                if "base_yscale" in metadata:
                    ax.set_yscale(metadata["base_yscale"])
        grid.set_ylabels(label=metadata["base_title"])
        grid.set_xlabels(label="")
        grid.set_titles(col_template="{col_name}", size=13)
        grid.tight_layout()

        chart_path_stub = CHARTS_DIRECTORY.joinpath(f"{base_metric_key}_plot")
        grid.savefig(chart_path_stub.with_suffix(".pdf"))
        grid.savefig(chart_path_stub.with_suffix(".svg"))
        grid.savefig(chart_path_stub.with_suffix(".png"), dpi=300)


def _plot_summary(melted_df: pd.DataFrame):
    # Plot summary chart
    sliced_melted_df = melted_df[
        ~melted_df.variable.isin(
            {
                "variance",
                "count",
                "median_rank",
                "inverse_median_rank",
                "standard_deviation",
                "median_absolute_deviation",
                "geometric_mean_rank",
                "harmonic_mean_rank",
                "inverse_geometric_mean_rank",
                "inverse_arithmetic_mean_rank",
                "hits_at_1",
                "hits_at_3",
                "adjusted_hits_at_1",
                "adjusted_hits_at_3",
                "z_hits_at_1",
                "z_hits_at_3",
            }
        )
    ]
    sns.relplot(
        data=sliced_melted_df,
        x="dataset_triples",
        y="value",
        hue="model",
        col="variable",
        col_wrap=4,
        kind="line",
        facet_kws={"sharey": False, "sharex": True},
    ).set(xscale="log")
    summary_path_stub = CHARTS_DIRECTORY.joinpath("summary")
    plt.savefig(summary_path_stub.with_suffix(".png"), dpi=300)
    plt.savefig(summary_path_stub.with_suffix(".svg"))


if __name__ == "__main__":
    main()
