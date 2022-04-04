"""Fixes a bug in computation of z-adjusted metrics post-hoc."""

import numpy
import pandas
from constants import COLLATED_PATH, MELTED_PATH
from pykeen.datasets import get_dataset
from pykeen.evaluation.evaluator import get_candidate_set_size
from pykeen.metrics import rank_based_metric_resolver

df = pandas.read_csv(COLLATED_PATH, sep="\t")
pairs = {}
for column in df.columns:
    if not column.startswith("z_"):
        continue
    original_column = column[2:]
    pairs[column] = original_column
pairs["z_geometric_mean_rank"] = "geometric_mean_rank"

hits_prefix = "hits_at_"
expectation, variance = {}, {}
for dataset_name in df["dataset"].unique():
    dataset = get_dataset(dataset=dataset_name)
    candidate_df = get_candidate_set_size(
        mapped_triples=dataset.testing.mapped_triples,
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ],
    )
    num_candidates = numpy.concatenate(
        [candidate_df[column] for column in ("head_candidates", "tail_candidates")]
    )
    for z_metric_name, metric_name in pairs.items():
        # normalize metric
        kwargs = {}
        norm_metric_name = metric_name
        if hits_prefix in metric_name:
            kwargs["k"] = int(metric_name[len(hits_prefix) :])
            norm_metric_name = hits_prefix
        metric = rank_based_metric_resolver.make(norm_metric_name, pos_kwargs=kwargs)
        mean = metric.expected_value(num_candidates=num_candidates)
        std = metric.std(num_candidates=num_candidates)
        value = (df.loc[df["dataset"] == dataset_name, metric_name] - mean) / std
        if "geometric_mean_rank" in metric_name:
            value *= -1
        df.loc[df["dataset"] == dataset_name, z_metric_name] = value
    # adjusted geometric mean rank index
    metric_name = "geometric_mean_rank"
    metric = rank_based_metric_resolver.make(metric_name)
    mean = metric.expected_value(num_candidates=num_candidates)
    std = metric.std(num_candidates=num_candidates)
    value = (df.loc[df["dataset"] == dataset_name, metric_name] - mean) / (1 - mean)
    df.loc[df["dataset"] == dataset_name, "adjusted_geometric_mean_rank_index"] = value

df.to_csv("collated/collated.tsv", sep="\t", index=False)

id_vars = ["dataset", "dataset_triples", "model"]
melted_df = pandas.melt(
    df,
    id_vars=id_vars,
    value_vars=[v for v in df.columns if v not in id_vars and v not in ("evaluation", "training")],
).sort_values(by=["dataset", "model", "variable"])
melted_df.to_csv(MELTED_PATH, sep="\t", index=False)
