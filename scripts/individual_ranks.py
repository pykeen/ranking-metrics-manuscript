"""Plot distribution of individual ranks, and different mean ranks."""
import itertools
import logging
import pathlib
import tempfile
from typing import Collection, Mapping, Optional, Tuple

import click
import more_click
import numpy
import pandas
import seaborn
import torch
from matplotlib import pyplot as plt
from pykeen.datasets import dataset_resolver, get_dataset
from pykeen.evaluation import RankBasedEvaluator, RankBasedMetricResults
from pykeen.evaluation.rank_based_evaluator import _iter_ranks
from pykeen.models import model_resolver
from pykeen.typing import ExtendedTarget, RankType

ROOT = pathlib.Path(__file__).parents[1]
logger = logging.getLogger(__name__)


class RankBasedEvaluatorAdapter(RankBasedEvaluator):
    """Adapter of rank-based evaluator to give access to raw ranks."""

    raw: Mapping[
        Tuple[ExtendedTarget, RankType], Tuple[numpy.ndarray, numpy.ndarray]
    ] = None

    def finalize(self) -> RankBasedMetricResults:  # noqa: D102
        self.raw = {
            (side, rank_type): (ranks, candidates)
            for side, rank_type, ranks, candidates in _iter_ranks(
                ranks=self.ranks, num_candidates=self.num_candidates
            )
        }
        return super().finalize()


def buffered_raw_evaluation(
    dataset: str,
    model: str,
    model_root: pathlib.Path,
    force: bool = False,
    buffer_root: Optional[pathlib.Path] = None,
) -> Tuple[
    Mapping[Tuple[ExtendedTarget, RankType], Tuple[numpy.ndarray, numpy.ndarray]],
    Mapping[str, float],
]:
    model = model_resolver.normalize(model)
    dataset = dataset_resolver.normalize(dataset)
    buffer_root = buffer_root or pathlib.Path(tempfile.gettempdir(), "ranks")
    path = buffer_root.joinpath(dataset, model, "ranks.pt")
    if path.is_file() and not force:
        logger.info(f"Loading data from {path}")
        return torch.load(path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    # load model
    replicate_root = model_root.joinpath(
        dataset, model, "replicates", "replicate-00000"
    )
    model_path = replicate_root.joinpath("trained_model.pkl")
    logger.info(f"Loaded model from {model_path}")
    model_instance = torch.load(model_path, map_location=device)
    logger.info(f"Loaded model\n{model_instance}")

    # load dataset
    dataset_instance = get_dataset(dataset=dataset)
    logger.info(f"Loaded dataset\n{dataset_instance}")

    evaluator = RankBasedEvaluatorAdapter()
    result = evaluator.evaluate(
        model=model_instance,
        mapped_triples=dataset_instance.testing.mapped_triples,
        additional_filter_triples=[
            dataset_instance.training.mapped_triples,
            dataset_instance.validation.mapped_triples,
        ],
    ).to_flat_dict()
    path.parent.mkdir(exist_ok=True, parents=True)
    torch.save((evaluator.raw, result), path)
    logger.info(f"Saved data to {path}")

    return evaluator.raw, result


def load_data(
    datasets: Collection[str],
    models: Collection[str],
    model_root: pathlib.Path,
    metrics: Collection[str] = (
        "arithmetic_mean_rank",
        "geometric_mean_rank",
        "harmonic_mean_rank",
    ),
    rank_type: RankType = "realistic",
    target: ExtendedTarget = "both",
    force: bool = False,
    buffer_root: Optional[pathlib.Path] = None,
) -> Tuple[pandas.DataFrame, Mapping[Tuple[str, str], Mapping[str, float]]]:
    data = []
    metric_values = {}
    for dataset, model in itertools.product(datasets, models):
        raw, result = buffered_raw_evaluation(
            dataset=dataset,
            model=model,
            model_root=model_root,
            force=force,
            buffer_root=buffer_root,
        )
        ranks, candidates = raw[target, rank_type]
        data.extend((dataset, model, r) for r in ranks)
        metric_values[dataset, model] = {
            metric: result[f"{target}.{rank_type}.{metric}"] for metric in metrics
        }
    df = pandas.DataFrame(data=data, columns=["dataset", "model", "rank"])
    return df, metric_values


DATASETS = ("fb15k237", "kinships", "nations", "wn18rr")
MODELS = ("complex", "rotate", "transe", "tucker")


def add_lines(model, dataset, *, metrics, palette, labels, color, log: bool = False):
    assert model.nunique() == 1
    assert dataset.nunique() == 1
    model = model.unique().item()
    dataset = dataset.unique().item()
    d = metrics.get((dataset, model), {})
    keys = sorted(d.keys())
    for i, key in enumerate(keys, start=1):
        v = d[key]
        if log:
            v = numpy.log(v)
        plt.axvline(v, color=palette[i], label=labels[key])


@click.command()
@click.option("-r", "--model-root", type=pathlib.Path, default=ROOT.joinpath("models"))
@click.option("-b", "--buffer-root", type=pathlib.Path, default=None)
@click.option(
    "-d",
    "--dataset",
    type=click.Choice(DATASETS, case_sensitive=False),
    multiple=True,
    default=DATASETS,
)
@click.option(
    "-m",
    "--model",
    type=click.Choice(MODELS, case_sensitive=False),
    multiple=True,
    default=MODELS,
)
@more_click.log_level_option()
def main(
    model_root: pathlib.Path,
    buffer_root: Optional[pathlib.Path],
    dataset: Collection[str],
    model: Collection[str],
    log_level: str,
):
    """Plot distribution of individual ranks, and different mean ranks."""
    logging.basicConfig(level=log_level)

    df, metrics = load_data(datasets=dataset, models=model, model_root=model_root, buffer_root=buffer_root)
    df["rank [log]"] = df["rank"].apply(numpy.log)

    palette = seaborn.color_palette(n_colors=4)
    grid = seaborn.displot(
        data=df,
        x="rank [log]",
        # x="rank",
        col="model",
        # log_scale=True,
        bins=10,
        # cf. https://github.com/mwaskom/seaborn/issues/2557
        # common_bins=False,
        row="dataset",
        facet_kws=dict(
            # sharex="row",
            sharey="row",
        ),
    )
    grid.map(
        add_lines,
        "model",
        "dataset",
        metrics=metrics,
        palette=palette,
        labels={
            "arithmetic_mean_rank": "AMR",
            "geometric_mean_rank": "GMR",
            "harmonic_mean_rank": "HMR",
        },
        log=True,
    )
    grid.set(xlabel="Rank [log]", ylabel="Frequency", xlim=(0, None))
    # grid.set(xlabel="Rank", ylabel="Frequency", xlim=(1, None))
    # num_entities = {
    #     dsn: get_dataset(dataset=dsn).num_entities
    #     for dsn in dataset
    # }
    # for (dataset, model), ax in grid.axes_dict.items():
    #     ax.set_xlim(1, num_entities[dataset])
    for ax in grid.axes.flat:
        ax.set_yscale("log")
    grid.tight_layout()
    grid.savefig("/tmp/plot.pdf")


if __name__ == "__main__":
    main()
