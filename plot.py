"""Generate plot."""

import pandas
import scipy.constants
import seaborn

from collate import HERE, MELTED_PATH

CHARTS_DIRECTORY = HERE.joinpath("charts")
CHARTS_DIRECTORY.mkdir(exist_ok=True)


def main():
    melted_df = pandas.read_csv(MELTED_PATH, sep="\t")
    base_metric_name = "Mean Reciprocal Rank"
    base_metric_key = base_metric_name.lower().replace(" ", "_")
    metrics = [
        "inverse_harmonic_mean_rank",
        "adjusted_inverse_harmonic_mean_rank",
        "z_inverse_harmonic_mean_rank",
    ]

    # filter
    df = melted_df[melted_df["variable"].isin(metrics) & (melted_df["model"] != "rotate")].copy()

    rename = {
        "inverse_harmonic_mean_rank": "Original",
        "z_inverse_harmonic_mean_rank": "z-Adjusted Metric",
        "adjusted_inverse_harmonic_mean_rank": "Adjusted Index",
    }
    df.loc[:, "variable"] = df["variable"].apply(rename.__getitem__)
    metrics = [rename[m] for m in metrics]

    grid: seaborn.FacetGrid = seaborn.catplot(
        data=df,
        # x="dataset",
        # order=["fb15k237", "wn18rr", "nations", "kinships"],
        x="model",
        y="value",
        col="variable",
        col_order=metrics,
        # hue="model",
        hue="dataset",
        hue_order=["fb15k237", "wn18rr", "nations", "kinships"],
        sharey=False,
        # facet_kws=dict(sharey=False),
        kind="bar",
        height=3,
        aspect=scipy.constants.golden ** (-1),
    )
    grid.set_xticklabels(rotation=45, ha="right")
    for key, ax in grid.axes_dict.items():
        if key == "z-Adjusted Metric":
            ax.set_yscale("log")
        else:
            ax.set_ylim([0, 1])
    grid.set_ylabels(label=base_metric_name)
    grid.set_xlabels(label="")
    grid.set_titles(col_template="{col_name}")
    grid.tight_layout()

    chart_path_stub = CHARTS_DIRECTORY.joinpath(f"{base_metric_key}_plot")
    grid.savefig(chart_path_stub.with_suffix(".pdf"))
    grid.savefig(chart_path_stub.with_suffix(".svg"))
    grid.savefig(chart_path_stub.with_suffix(".png"), dpi=300)


if __name__ == '__main__':
    main()
