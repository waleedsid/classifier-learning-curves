# Dataset attribution

Alpaydin, E. and Kaynak, C. (1998). Optical Recognition of Handwritten Digits. UCI Machine Learning Repository. [DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49).

The UCI dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The experiment uses the 1,797 observation subset bundled with scikit-learn through [load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html), rather than concatenating the original UCI training and test files.

No source archive is redistributed here. Transformations consist of fold specific scaling for logistic regression and KNN, repeated training subsets, scoring and aggregation into learning curves. The tree uses unscaled pixels. Attribution applies to derivative numerical results and digit visualizations. No endorsement is implied.
