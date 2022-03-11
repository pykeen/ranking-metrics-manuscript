"""Generate plot."""
import pandas
import scipy.constants
import seaborn

from collate import MELTED_PATH

df2 = pandas.read_csv(MELTED_PATH, sep="\t")
metrics = [
    "inverse_harmonic_mean_rank",
    "adjusted_inverse_harmonic_mean_rank",
    "z_inverse_harmonic_mean_rank",
]

# filter
df = df2[df2["variable"].isin(metrics) & (df2["model"] != "rotate")].copy()

rename = {
    "inverse_harmonic_mean_rank": "original",
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
grid.set_xticklabels(rotation=90)
for key, ax in grid.axes_dict.items():
    if key == "z-Adjusted Metric":
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, 1)
grid.set_ylabels(label="")
grid.set_xlabels(label="")
grid.set_titles(col_template="{col_name}")
grid.tight_layout()
grid.savefig("/tmp/plot.pdf")
