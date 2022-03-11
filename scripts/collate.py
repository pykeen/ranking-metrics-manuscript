import json

import click
import pandas as pd
from docdata import get_docdata
from pykeen.datasets import dataset_resolver

from constants import COLLATED_PATH, MELTED_PATH, MODEL_DIRECTORY


@click.command()
def collate():
    """Collate the data."""
    rows = []
    for dataset_directory in MODEL_DIRECTORY.iterdir():
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
                model_directory.joinpath(
                    "replicates", "replicate-00000", "results.json"
                ).read_text()
            )
            row = dict(
                dataset=configuration["pipeline"]["dataset"],
                dataset_triples=dataset_triples,
                model=configuration["pipeline"]["model"],
                **results["metrics"]["both"]["realistic"],
                **results["times"],
            )
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(by=["dataset", "model"])
    # sort columns for reproducible output
    prefix_columns = ["dataset", "dataset_triples", "model", "training", "evaluation"]
    df = df[prefix_columns + sorted(c for c in df.columns if c not in prefix_columns)]
    df.to_csv(COLLATED_PATH, sep="\t", index=False)

    id_vars = ["dataset", "dataset_triples", "model"]
    melted_df = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=[
            v
            for v in df.columns
            if v not in id_vars and v not in ("evaluation", "training")
        ],
    ).sort_values(by=["dataset", "model", "variable"])
    melted_df.to_csv(MELTED_PATH, sep="\t", index=False)


if __name__ == "__main__":
    collate()
