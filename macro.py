"""Investigate effect of micro vs. macro."""
import logging
import pathlib
import tempfile
from typing import Iterable, Optional, Tuple, Type

import click
import numpy
import pandas
from docdata import get_docdata
from pykeen.datasets import Dataset, dataset_resolver, get_dataset
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


def _triples(pair: Tuple[str, Type[Dataset]]) -> int:
    """Extract the number of triples from docdata."""
    return get_docdata(pair[1])["statistics"]["triples"]


def _iter_counts(max_triples: Optional[int]) -> Iterable[Tuple[str, str, str, pandas.Series]]:
    """Iterate over datasets/splits/targets and the respective counts."""
    for dataset_name, dataset_cls in sorted(dataset_resolver.lookup_dict.items(), key=_triples):
        # skip large datasets
        if max_triples and _triples((dataset_name, dataset_cls)) > max_triples:
            continue

        # skip unavailable datasets
        try:
            dataset = get_dataset(dataset=dataset_name)
        except Exception as error:
            logging.error(str(error))
            continue

        # investigate triples for all splits
        for split, factory in dataset.factory_dict.items():

            # convert to pandas dataframe
            df = pandas.DataFrame(dataset.testing.mapped_triples.numpy(), columns=["h", "r", "t"])

            # for each side prediction
            for target in "ht":

                # group by other columns
                keys = [c for c in df.columns if c != target]
                group = df.groupby(by=keys)

                # count number of unique targets per key
                counts = group.agg({target: "nunique"})[target]

                yield dataset_name, split, target, counts


def _save(df: pandas.DataFrame, output_directory: pathlib.Path, *keys: str):
    """Save dataframe under directory."""
    path = output_directory.joinpath(*keys).with_suffix(suffix=".tsv.gz")
    path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(path, sep="\t", index=False)
    logging.debug(f"Written to {path}")


@click.command()
@click.option("-m", "--max-triples", type=int, default=None)
@click.option("-o", "--output-directory", type=pathlib.Path, default=pathlib.Path(tempfile.gettempdir(), "pykeen-macro"))
def main(
    max_triples: Optional[int],
    output_directory: pathlib.Path,
):
    """Collect statistics across datasets."""
    # logging setup
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("pykeen.triples").setLevel(level=logging.ERROR)

    # aggregate basic statistics across all datasets
    summary = []

    # iterate over all datasets
    with logging_redirect_tqdm(), tqdm(_iter_counts(max_triples=max_triples)) as progress:
        for dataset_name, split, target, counts in progress:
            progress.set_postfix(datase=dataset_name, split=split, target=target)

            # append basic statistics to summary data(-frame)
            description = counts.describe()
            summary.append((dataset_name, split, target, *description.tolist()))

            # store distribution
            unique_counts, frequency = numpy.unique(counts.values, return_counts=True)
            df = pandas.DataFrame(data=dict(counts=unique_counts, frequency=frequency))
            _save(df, output_directory, dataset_name, split, target)

    # store summary
    df = pandas.DataFrame(data=summary, columns=["dataset", "split", "side", "count", "mean", "std", "min", "25%", "50%", "75%", "max"])
    _save(df, output_directory, "summary")


if __name__ == "__main__":
    main()
