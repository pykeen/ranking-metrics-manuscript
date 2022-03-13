# Ranking Metrics Manuscript Supplement

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6347429.svg)](https://doi.org/10.5281/zenodo.6347429)

This repository contains analysis and supplementary information for _A Unified
Framework for Rank-based Evaluation Metrics for Link Prediction_, non-archivally
submitted to [GLB 2022](https://graph-learning-benchmarks.github.io/glb2022).

📣 **Main Results** 📣 There's a dataset size-correlation for common rank-based
evaluation metrics like mean rank (MR), mean reciprocal rank (MRR), and hits at
k (H@K) that makes them impossible to compare across datasets. We used the
expectation and variance of each metric to define adjusted metrics that don't
have a dataset size-correlation and are therefore comparable across datasets.

![Results](charts/mean_reciprocal_rank_plot.svg)

🖼️ **Figure Summary** 🖼️ While the MRR on, e.g., Nations and WN18-RR appears
similar for ComplEx, the
_adjusted index_ reveals that when adjusting for chance, the performance on (the
larger) WN18-RR is more remarkable. The _z-adjusted_ metric allows an easier
direct comparison against the baseline that suggests the results on smaller
datasets are less considerable, despite achieving better unnormalized
performance. ¬

## Citation

```bibtex
@book{hoyt2022,
   author = {Hoyt, Charles Tapley and Berrendorf, Max and Gaklin, Mikhail and Tresp, Volker and Gyori, Benjamin M},
   title = {{A Unified Framework for Rank-based Evaluation Metrics for Link Prediction}},
   year = {2022}
}
```

## Build

After installing `tox` with `pip install tox`, do the following:

1. `tox -e collate` to build the combine results files
2. `tox -e plot` to summarize the results files as plots
3. `tox -e macro` to run the analysis for micro vs. macro evaluation (not
   actually presented in the paper)
