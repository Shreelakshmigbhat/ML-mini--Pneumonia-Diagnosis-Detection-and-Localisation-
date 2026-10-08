# Stage 10 — XGBoost Extension

> **XGBoost is an extension experiment and is not part of the original reference paper.**

## Why XGBoost

The reference project already evaluates Logistic Regression, a linear SVM, a
Random Forest, and a custom 10-layer CNN. XGBoost adds a gradient-boosted tree
classifier as a different classical approach. It is an experiment, not an
assumption that boosting will outperform the reference models or the CNN.

## Feature representation

The extension reuses the existing fixed split, the shared DICOM image loader,
the 128 × 128 grayscale resize, and the train-fitted normalization statistics.
Each normalized image is flattened to 16,384 float features. Randomized-SVD
PCA then reduces these features to 256 components before XGBoost.

The PCA is fitted only on the training matrix. Validation and test matrices use
only `transform` with that fitted PCA. A memory-mapped temporary feature cache
keeps preprocessing to one pass and limits peak memory; temporary files are
removed when the run completes. Randomized SVD with one power iteration and
four bounded BLAS threads replaces repeated incremental SVD operations.

## Configuration

PCA uses `PCA(n_components=256, svd_solver="randomized",
iterated_power=1, n_oversamples=8, random_state=42)`. This fixed
dimensionality is chosen as a practical reduction from 16,384 pixel features;
the run records its explained variance ratio. XGBoost uses:

| Parameter | Value |
|---|---:|
| Objective | `binary:logistic` |
| Evaluation metric | `logloss` |
| Estimators | 100 |
| Maximum depth | 4 |
| Learning rate | 0.05 |
| Row subsample | 0.8 |
| Column subsample | 0.8 |
| Random seed | 42 |
| Tree method | histogram |
| Device | CPU |

No hyperparameter search or validation-based early stopping is used. The
preflight smoke test uses 500 training and 100 validation images with the same
256-component PCA and 100-tree XGBoost settings; the smoke output is not saved
as a model result.

## Leakage controls and evaluation

The script verifies the saved split checksum, the shared preprocessing
statistics, sample counts, and non-overlapping image IDs before training. PCA
and XGBoost are fit on train only. Validation is evaluation-only; the test set
is reserved for final reporting. Previous Logistic Regression, SVM, and Random
Forest models are loaded from their saved artifacts. The existing CNN
checkpoint is evaluated by inference only; it is not retrained.

The actual PCA variance, sample counts, duration, split checksum, configuration,
and per-split metrics are stored in
[`results/xgboost_results.json`](./results/xgboost_results.json). The
same-test-split comparison, including accuracy, F1, precision, and recall for
all five models, is in
[`results/model_comparison.csv`](./results/model_comparison.csv).

## Results and comparison

Results are generated from the actual run and are intentionally not copied
into this document as manually maintained values. See the JSON and CSV above
for the recorded numerical results. The CSV includes test metrics derived
from saved model predictions, so comparisons use the same fixed test examples.

The dataset contains 14,863 usable labeled images rather than the paper's
reported 17,489-image cohort. Therefore these are local fixed-split results,
not a claim of exact cohort-matched reproduction.

## Limitations

- PCA preserves directions of pixel variance, which need not preserve
  class-discriminative information.
- Flattening discards the explicit spatial structure available to a CNN.
- The fixed 256-component choice is a practical extension setting, not a
  tuned optimum.
- The cohort size differs from the reference paper.
- The Random Forest and CNN comparison values are existing project models,
  not models retrained or retuned for this extension.

## Future improvements

Potential follow-up experiments include comparing a small number of
predeclared PCA dimensions, testing a PCA-free XGBoost baseline on a smaller
image representation, calibrating probabilities, and evaluating repeated
patient-level splits. Such studies should remain separate from this
single-split extension and should not use the test set for model selection.
