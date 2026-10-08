# Reproduction Notes

## Environment and dependencies

The Stage 10 saved result records Python 3.14.4. The repository dependencies are listed in [`requirements.txt`](./requirements.txt); they use minimum version constraints and do not lock every resolved package version.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Dataset preparation

The original RSNA Pneumonia Detection Challenge Stage 2 dataset is not distributed with this repository. Place the local inputs in the following layout:

```text
Data/stage_2_train_images/
Data/stage_2_train_labels.csv
Data/stage_2_detailed_class_info.csv
```

The source code maps `Lung Opacity` to Pneumonia (label 1), `Normal` to Healthy (label 0), excludes other classes, and preserves multiple boxes per image. To create the metadata from scratch:

```powershell
python scripts/create_metadata.py
```

The generated binary metadata has 14,863 rows. The reference description reports 17,489 binary images; the local cohort is not the same.

## Split generation and preprocessing

The fixed split was created using the existing `scripts/create_split.py` entry point:

```powershell
python scripts/create_split.py
```

It creates a seed-42 stratified 70/20/10 image-ID assignment with no overlap between partitions. **Do not rerun this command when reproducing or evaluating against the saved experiment artifacts; use `metadata/split_metadata.csv` as-is.**

Shared preprocessing decodes DICOM images on demand, resizes to 128 × 128 grayscale, scales pixels to [0, 1], then normalizes with mean 0.4898645393 and standard deviation 0.2465253599 fitted on training data. Saved parameters are in `metadata/preprocessing_stats.json`.

## Model training commands

The following entry points exist and train models when run:

```powershell
python scripts/train_logistic_regression.py
python scripts/train_svm.py
python scripts/train_random_forest.py
python scripts/train_cnn.py
```

Existing artifacts are in `results/`; the CNN checkpoint is `models/cnn_best.pth` when available locally. Training is not needed to read the final results. The saved checkpoint metadata reports CNN validation accuracy 0.8977 at epoch 9. A standalone `results/cnn_results.json` and CNN train-accuracy result are not present.

## CAM localization

The repository entry point is:

```powershell
python scripts/run_cam_localization.py
```

This runs localization on the fixed test split and can take time. The existing Stage 9 result is already saved in `results/cam_results.json`; do not rerun CAM just to inspect or validate this documentation.

## XGBoost extension

**XGBoost is an extension and was not part of the original reference methodology.** The available commands are:

```powershell
python scripts/train_xgboost.py --smoke-test
python scripts/train_xgboost.py
```

The smoke test uses 500 training and 100 validation samples. The completed full run uses the fixed split, one randomized 256-component PCA fit on training data, transforms validation/test with that PCA, then fits the fixed 100-tree XGBoost configuration. Its result and timing are in `results/xgboost_results.json`. Do not rerun either command to validate the saved results.

## Notebook instructions

The notebooks live in `notebooks/` and are ordered `01_Dataset_Preprocessing.ipynb` through `05_XGBoost_Extension.ipynb`. Notebook 01 only reads saved metadata; later notebooks include cells that can train models or run CAM. For a documentation-only review, inspect saved `results/*.json` and `results/*.csv` files and do not execute training/full-evaluation cells.

The standalone reference paper citation details beyond the title “Weakly Supervised Pneumonia Localization” are not stored in this repository.
