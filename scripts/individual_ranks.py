"""Plot distribution of individual ranks, and different mean ranks."""
import logging
import pathlib
from typing import Mapping, Tuple

import click
import more_click
import numpy
import seaborn
import torch
from matplotlib import pyplot as plt
from pykeen.datasets import get_dataset
from pykeen.evaluation import RankBasedEvaluator, RankBasedMetricResults
from pykeen.evaluation.rank_based_evaluator import _iter_ranks
from pykeen.typing import ExtendedTarget, RankType

ROOT = pathlib.Path(__file__).parents[1]
logger = logging.getLogger(__name__)


class RankBasedEvaluatorAdapter(RankBasedEvaluator):
    """Adapter of rank-based evaluator to give access to raw ranks."""

    raw: Mapping[Tuple[ExtendedTarget, RankType], Tuple[numpy.ndarray, numpy.ndarray]] = None

    def finalize(self) -> RankBasedMetricResults:  # noqa: D102
        self.raw = {
            (side, rank_type): (ranks, candidates)
            for side, rank_type, ranks, candidates in _iter_ranks(ranks=self.ranks, num_candidates=self.num_candidates)
        }
        return super().finalize()


@click.command()
@click.option("-r", "--model-root", type=pathlib.Path, default=ROOT.joinpath("models"))
@click.option("-d", "--dataset", type=click.Choice(["fb15k237", "kinships", "nations", "wn18rr"]), default="nations")
@click.option("-m", "--model", type=click.Choice(["complex", "rotate", "transe", "tucker"]), default="tucker")
@more_click.log_level_option()
def main(
    model_root: pathlib.Path,
    dataset: str,
    model: str,
    log_level: str,
):
    """Plot distribution of individual ranks, and different mean ranks."""
    logging.basicConfig(level=log_level)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")

    replicate_root = model_root.joinpath(dataset, model, "replicates", "replicate-00000")

    # load model
    model_path = replicate_root.joinpath("trained_model.pkl")
    logger.info(f"Loaded model from {model_path}")
    model = torch.load(model_path, map_location=device)
    logger.info(f"Loaded model\n{model}")

    # load dataset
    dataset = get_dataset(dataset=dataset)
    logger.info(f"Loaded dataset\n{dataset}")

    evaluator = RankBasedEvaluatorAdapter()
    result = evaluator.evaluate(
        model=model, mapped_triples=dataset.testing.mapped_triples, additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ],
    )
    realistic_ranks = evaluator.raw["both", "realistic"]
    palette = seaborn.color_palette(n_colors=4)
    ax = seaborn.distplot(a=realistic_ranks, color=palette[0])
    for i, (label, key) in enumerate((
        ("AMR", "both.realistic.arithmetic_mean_rank"),
        ("GMR", "both.realistic.geometric_mean_rank"),
        ("HMR", "both.realistic.harmonic_mean_rank"),
    ), start=1):
        v = result.get_metric(name=key)
        ax.axvline(x=v, label=label, color=palette[i])
    ax.set_xlim(1, dataset.num_entities)
    ax.set_xlabel("Rank")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
