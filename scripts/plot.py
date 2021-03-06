"""Generate plot."""

import click
import matplotlib.pyplot as plt
import pandas as pd
import scipy.constants
import seaborn as sns
from constants import CHARTS_DIRECTORY, MELTED_PATH
from pykeen.datasets import dataset_resolver

SIGIL = r"\mathcal{T}_{train}"


def _lookup_key(d):
    return dataset_resolver.docdata(d, "statistics", "training")


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
DATASET_TITLES = {
    key: f"{value} ($|{SIGIL}|={_lookup_key(key):,}$)" for key, value in DATASET_TITLES.items()
}
# show datasets in increasing order of entity size
DATASET_ORDER = [
    v for _, v in sorted(DATASET_TITLES.items(), key=lambda pair: _lookup_key(pair[0]))
]
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
    "arithmetic_mean_rank": {
        "base_title": "Mean Rank",
        "base_yscale": "log",
        "metrics": [
            "arithmetic_mean_rank",
            "adjusted_arithmetic_mean_rank_index",
            "z_arithmetic_mean_rank",
        ],
        "short": ["MR", "AMRI", "ZMR"],
        "has_negative_z": True,
    },
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
    "geometric_mean_rank": {
        "base_title": "Geometric Mean Rank",
        "base_yscale": "log",
        "metrics": [
            "geometric_mean_rank",
            "adjusted_geometric_mean_rank_index",
            "z_geometric_mean_rank",
        ],
        "short": ["GMR", "AGMRI", "ZGMR"],
        "has_negative_z": True,
    },
}


@click.command()
@click.option(
    "--wrap", is_flag=True, help="Use 2x2 grid instead of 1x3 grid for embedding in slides."
)
def main(wrap: bool = False):
    melted_df = pd.read_csv(MELTED_PATH, sep="\t")
    melted_df.loc[:, "model"] = melted_df["model"].map(MODEL_TITLES)
    melted_df.loc[:, "dataset"] = melted_df["dataset"].map(DATASET_TITLES)
    _plot_summary(melted_df)

    for base_metric_key, metadata in METRICS.items():
        metrics = metadata["metrics"]
        df = melted_df[melted_df["variable"].isin(metrics)].copy()
        metric_order = [f"{order} ({short})" for order, short in zip(ORDER, metadata["short"])]
        df.loc[:, "variable"] = df["variable"].map(dict(zip(metrics, metric_order)))
        grid: sns.FacetGrid = sns.catplot(
            data=df,
            x="model",
            y="value",
            col="variable",
            col_order=metric_order,
            col_wrap=2 if wrap else None,
            # hue="model",
            hue="dataset",
            hue_order=DATASET_ORDER,
            sharey=False,
            # facet_kws=dict(sharey=False),
            kind="bar",
            height=2.4,
            aspect=scipy.constants.golden,
        )
        # TODO calculate this
        has_negative_z = metadata.get("has_negative_z", False)
        # grid.set_xticklabels(rotation=30, ha="right")
        for key, ax in grid.axes_dict.items():
            if key.startswith("z-Adjusted Metric"):
                if has_negative_z:
                    ax.set_yscale("symlog")
                else:
                    ax.set_yscale("log")
            elif key.startswith("Adjusted Index"):
                ax.set_ylim([-0.1, 1])
            else:  # base metric
                if "base_ylim" in metadata:
                    ax.set_ylim(metadata["base_ylim"])
                if "base_yscale" in metadata:
                    ax.set_yscale(metadata["base_yscale"])
        grid.set_ylabels(label=metadata["base_title"])
        grid.set_xlabels(label="")
        grid.set_titles(col_template="{col_name}", size=13)
        if wrap:
            sns.move_legend(grid, "upper left", bbox_to_anchor=(0.425, 0.40), title="Dataset")
        grid.tight_layout()
        chart_path_stub = CHARTS_DIRECTORY.joinpath(f"{base_metric_key}_plot")
        grid.savefig(chart_path_stub.with_suffix(".pdf"))
        grid.savefig(chart_path_stub.with_suffix(".svg"))
        grid.savefig(chart_path_stub.with_suffix(".png"), dpi=300)
        plt.close(grid.fig)


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
                "harmonic_mean_rank",
                "inverse_geometric_mean_rank",
                "inverse_arithmetic_mean_rank",
                "hits_at_1",
                "hits_at_3",
                "hits_at_5",
                "adjusted_hits_at_1",
                "adjusted_hits_at_3",
                "adjusted_hits_at_5",
                "z_hits_at_1",
                "z_hits_at_3",
                "z_hits_at_5",
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
        height=2.2,
        aspect=scipy.constants.golden,
    ).set(xscale="log")
    summary_path_stub = CHARTS_DIRECTORY.joinpath("summary")
    plt.savefig(summary_path_stub.with_suffix(".png"), dpi=300)
    plt.savefig(summary_path_stub.with_suffix(".svg"))


if __name__ == "__main__":
    main()
