# Weakly Supervised Pneumonia Localization

## Overview

This repository contains a reproduction-oriented pneumonia classification and weakly supervised localization pipeline, plus a separately identified XGBoost extension. The workflow includes classical image classifiers, a paper-inspired ten-layer CNN, and CAM-based test localization.

The project uses a local RSNA Stage 2 subset. **The RSNA dataset is not included in this repository.** Any local `Data/` directory is ignored by Git; obtain and use the dataset under the RSNA terms and place the required Stage 2 files there.

## Reference Paper

The notebooks identify the reference as **“Weakly Supervised Pneumonia Localization.”** Full citation metadata (authors, venue, DOI) is not present in the project records, so it is not guessed here. The available cohort and some implementation details differ from the paper; see [RESULTS.md](./RESULTS.md).

## Dataset

The Stage 1 scripts expect these local inputs:

```text
Data/
├── stage_2_train_images/
├── stage_2_train_labels.csv
└── stage_2_detailed_class_info.csv
```

The local binary cohort has 14,863 labeled images: 6,012 Pneumonia (`Lung Opacity`) and 8,851 Healthy (`Normal`). The other labeled class is excluded. A fixed seed-42 stratified 70/20/10 split is saved in `metadata/split_metadata.csv`. The paper-reported binary cohort is 17,489 images, so the local results are not cohort matched.

## Project Structure

```text
Data/                 local RSNA data only; ignored by Git
metadata/             saved metadata, preprocessing statistics, fixed split
src/                  data, model, and localization implementations
scripts/              metadata, training, validation, and CAM entry points
notebooks/            01–05 workflow and result notebooks
results/               saved metrics, comparisons, and selected CAM examples
models/                local trained checkpoints (large weights are ignored)
README.md
RESULTS.md
REPRODUCTION.md
requirements.txt
```

## Pipeline

```text
RSNA X-rays
    ↓
DICOM preprocessing (128 × 128 grayscale, train-only normalization)
    ↓
Classical ML on flattened pixels
    ├── Logistic Regression
    ├── SVM
    └── Random Forest
    ↓
10-layer CNN
    ↓
CAM
    ↓
Weakly supervised localization
```

The separate extension path is:

```text
Preprocessed images → flattened pixels → train-only PCA (256) → XGBoost
```

**XGBoost is an extension and was not part of the original reference methodology.**

## Notebooks

- [01_Dataset_Preprocessing.ipynb](./notebooks/01_Dataset_Preprocessing.ipynb) — reads and summarizes existing preprocessing outputs; the actual Stage 1 commands are in `scripts/`.
- [02_Baseline_Models.ipynb](./notebooks/02_Baseline_Models.ipynb) — saved Logistic Regression results and classical baselines.
- [03_CNN_Reproduction.ipynb](./notebooks/03_CNN_Reproduction.ipynb) — CNN architecture, training, and evaluation workflow.
- [04_CAM_Localization.ipynb](./notebooks/04_CAM_Localization.ipynb) — CAM and localization implementation/results.
- [05_XGBoost_Extension.ipynb](./notebooks/05_XGBoost_Extension.ipynb) — PCA/XGBoost extension and saved result review.

Notebook cells that call training or full CAM scripts perform substantial work. Do not run them merely to view the saved results.

## Installation

From the repository root in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The recorded Stage 10 environment used Python 3.14.4. `requirements.txt` specifies minimum package versions rather than a fully locked environment.

## Running the Project

These are the existing script entry points. Data generation and model commands can overwrite saved outputs; do not run them when only reviewing existing results.

```powershell
# Initial dataset metadata and split creation (requires local Data/ files)
python scripts/create_metadata.py
python scripts/create_split.py

# Classification experiments
python scripts/train_logistic_regression.py
python scripts/train_svm.py
python scripts/train_random_forest.py
python scripts/train_cnn.py

# Full CAM test evaluation
python scripts/run_cam_localization.py

# XGBoost smoke test, then one full extension run
python scripts/train_xgboost.py --smoke-test
python scripts/train_xgboost.py
```

The commands above document how to reproduce the experiments; none is required to view the saved output. Reproduction details and split safeguards are in [REPRODUCTION.md](./REPRODUCTION.md).

## Results

See [RESULTS.md](./RESULTS.md), [results/final_model_comparison.csv](./results/final_model_comparison.csv), and [results/final_localization_results.csv](./results/final_localization_results.csv). The original CAM measurements remain in [results/cam_results.json](./results/cam_results.json).

## Reproduction Notes

Logistic Regression, SVM, Random Forest, the CNN, and CAM form the reference-inspired pipeline. XGBoost is a separately reported extension. Dataset mismatch and unspecified paper implementation details are documented in [RESULTS.md](./RESULTS.md).

## Limitations

The local labeled cohort is smaller and differently composed than the paper-reported cohort. CAM localization is weak on the recorded test set (mean IoU 0.0801), and its threshold/clustering/matching choices are implementation details. Some CNN metrics and full bibliographic metadata are unavailable; missing metrics are marked N/A rather than inferred.
