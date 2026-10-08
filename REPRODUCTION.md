# Reproduction Notes

## Environment and dependencies

The saved Stage 10 result records Python 3.14.4. Install the declared dependencies from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` uses minimum version constraints; it is not a fully locked environment. The RSNA dataset and trained CNN checkpoint are not committed.

## Dataset preparation

Place the local RSNA Pneumonia Detection Challenge Stage 2 inputs at:

```text
Data/stage_2_train_images/
Data/stage_2_train_labels.csv
Data/stage_2_detailed_class_info.csv
```

The existing entry point maps `Lung Opacity` to Pneumonia (label 1), `Normal` to Healthy (label 0), excludes other classes, and preserves multiple boxes per image. To recreate metadata from the local source files:

```powershell
python scripts/create_metadata.py
```

The saved binary metadata has 14,863 rows. The reference project description reports 17,489 binary images; the local cohort is not the same.

## Split generation and preprocessing

The fixed split is stored in `metadata/split_metadata.csv`. To generate it from scratch:

```powershell
python scripts/create_split.py
```

This writes a seed-42 stratified 70/20/10 image-ID assignment. **Do not run this during result reproduction or evaluation:** use the committed split file as-is. The command regenerates the split; it does not recover a missing original split from a checkpoint.

Shared preprocessing decodes DICOM images on demand, resizes to 128 × 128 grayscale, scales pixels to [0, 1], then normalizes with mean 0.4898645393 and standard deviation 0.2465253599 fitted on training data. Parameters are saved in `metadata/preprocessing_stats.json`. The existing lightweight check is:

```powershell
python scripts/verify_preprocessing.py --samples 6
```

## Phase 1 model training commands

These existing commands train models when run and are not required to inspect saved results:

```powershell
python scripts/train_logistic_regression.py
python scripts/train_svm.py
python scripts/train_random_forest.py
python scripts/train_cnn.py
```

The CNN checkpoint is `models/cnn_best.pth` when available locally. The saved checkpoint metadata reports validation accuracy 0.8977 at epoch 9. The consolidated comparison is in `results/final_model_comparison.csv`; a standalone `results/cnn_results.json` and CNN train-accuracy result are not present.

## Phase 1 CAM

The existing baseline entry point evaluates the fixed test split:

```powershell
python scripts/run_cam_localization.py
```

This can take time and rewrites result outputs. `results/cam_results.json` already contains the saved result; **do not rerun it merely to inspect or validate documentation.**

## XGBoost extension

XGBoost is an extension, not part of the reference method. Available commands:

```powershell
python scripts/train_xgboost.py --smoke-test
python scripts/train_xgboost.py
```

The smoke test uses 500 training and 100 validation samples. The completed full run uses the fixed split, fits randomized 256-component PCA only on training data, transforms validation/test with that fitted PCA, then fits the fixed 100-tree XGBoost configuration. Results and timing are saved in `results/xgboost_results.json`. The full command performs image loading/preprocessing and model fitting; do not rerun it when the saved result exists.

## Phase 2 error analysis and localization

Stage 12 analysis uses the existing CNN and fixed positive test images. Its script evaluates the threshold grid and writes threshold/error-analysis outputs; this is a full CAM-map/evaluation pass, **not** a lightweight documentation check:

```powershell
python scripts/analyze_cam_errors.py
```

Stage 13 uses a separate Grad-CAM++ map generator with the frozen CNN. The smoke-only command checks a small sample and is not the full result:

```powershell
python scripts/run_improved_cam.py --smoke-only --threshold 0.5 --batch-size 12
```

The full fixed-test evaluation command is:

```powershell
python scripts/run_improved_cam.py --threshold 0.5 --batch-size 16
```

This processes the 601 positive test images and may take time. Its saved output is `results/phase2_improved_cam_results.json`; **do not rerun it when that result exists.**

Stage 14 summarizes the stored baseline and improved results without rerunning localization:

```powershell
python scripts/summarize_phase2_ablation.py
```

Stage 15 final comparison and Stage 16 finalization are documentation/result-consolidation work; no separate training, CAM, split-generation, or tuning command is required. Final saved comparison files are `results/phase2_final_comparison.csv` and `results/phase2_final_results.json`.

## Notebooks and saved results

The nine workflow notebooks are in `notebooks/`:

```text
01_Dataset_Preprocessing.ipynb
02_Baseline_Models.ipynb
03_CNN_Reproduction.ipynb
04_CAM_Localization.ipynb
05_XGBoost_Extension.ipynb
06_CAM_Error_Analysis.ipynb
07_Improved_CAM.ipynb
08_Phase2_Ablation.ipynb
09_Phase2_Final_Comparison.ipynb
```

Some notebook cells train models or evaluate the full test set. For a reproducibility/documentation check, inspect saved artifacts rather than executing expensive cells. The reference paper citation details beyond the title “Weakly Supervised Pneumonia Localization” are not stored in this repository.
