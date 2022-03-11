# Ranking Metrics Manuscript Supplement

This repository contains analysis and supplementary information
for _A Unified Framework for Rank-based Evaluation
Metrics for Link Prediction_, non-archivally submitted to
[GLB 2022](https://graph-learning-benchmarks.github.io/glb2022).

![Results](charts/mean_reciprocal_rank_plot.svg)

## Citation

A citation for _A Unified Framework for Rank-based Evaluation
Metrics for Link Prediction_ will be added soon.

## Build

After installing `tox` with `pip install tox`, do the following:

1. `tox -e collate` to build the combine results files
2. `tox -e plot` to summarize the results files as plots
3. `tox -e macro` to run the analysis for micro vs. macro evaluation (not
   actually presented in the paper)
