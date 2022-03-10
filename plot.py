import json
from pathlib import Path

import pandas as pd
import seaborn as sns
from docdata import get_docdata
from pykeen.datasets import dataset_resolver
import click

HERE = Path(__file__).parent.resolve()
RESULTS = HERE.joinpath("glb_results")
COLLATED_PATH = HERE.joinpath("collated.tsv")
MELTED_PATH = HERE.joinpath("melted.tsv")
CHART_PATH_SVG = HERE.joinpath("chart.svg")
CHART_PATH_PNG = HERE.joinpath("chart.png")


def main(force: bool = False):
    if MELTED_PATH.exists() and not force:
        melted_df = pd.read_csv(MELTED_PATH, sep="\t")
    else:
        rows = []
        for dataset_directory in RESULTS.iterdir():
            if not dataset_directory.is_dir():
                continue

            dataset_cls = dataset_resolver.lookup(dataset_directory.name)
            dataset_statistics = get_docdata(dataset_cls)["statistics"]
            dataset_triples = dataset_statistics["triples"]

            for model_directory in dataset_directory.iterdir():
                if not model_directory.is_dir():
                    continue
                configuration_path = model_directory.joinpath("configuration_copied.json")
                if not configuration_path.is_file():
                    print(f"Missing file: {configuration_path}")
                    continue

                configuration = json.loads(configuration_path.read_text())
                results = json.loads(
                    model_directory.joinpath("replicates", "replicate-00000", "results.json").read_text())
                row = dict(
                    dataset=configuration["pipeline"]["dataset"],
                    dataset_triples=dataset_triples,
                    model=configuration["pipeline"]["model"],
                    **results["metrics"]["both"]["realistic"],
                    **results["times"],
                )
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(COLLATED_PATH, sep="\t", index=False)

        id_vars = ['dataset', 'dataset_triples', 'model', 'evaluation', 'training']
        melted_df = pd.melt(df, id_vars=id_vars, value_vars=[v for v in df.columns if v not in id_vars])
        melted_df.to_csv(MELTED_PATH, sep="\t", index=False)

    print(melted_df.head())

    g = sns.catplot(df=melted_df, x="dataset", y="value", hue="variable")
    g.savefig(CHART_PATH_PNG, dpi=300)


if __name__ == '__main__':
    main()
