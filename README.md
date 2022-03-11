# glb2022_metrics

Results for the Ranking Metrics submission @ GLB 2022

![Results](charts/mean_reciprocal_rank_plot.svg)

## Build

After installing `tox` with `pip install tox`, do the following:

1. `tox -e collate` to build the combine results files
2. `tox -e plot` to summarize the results files as plots
3. `tox -e macro` to run the analysis for micro vs. macro evaluation (not
   actually presented in the paper)
