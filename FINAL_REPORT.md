# Weakly Supervised Pneumonia Localization
## Final Project Report

### 1. Project Overview

This project implements a reproduction-oriented image classification and weakly supervised localization study using the locally available RSNA Pneumonia Detection Challenge Stage 2 data.

**Phase 1** covered dataset preparation and a fixed split, classical machine-learning baselines (Logistic Regression, SVM, Random Forest), a paper-inspired 10-layer CNN, CNN CAM localization, and a separate PCA + XGBoost classification extension.

**Phase 2** analyzed the observed CAM failures, evaluated a separate Grad-CAM++ localization method with the frozen CNN, conducted an ablation of the map-generation change, and consolidated the final comparisons. No Phase 2 CNN retraining or split change was performed.

### 2. Reference Paper

The project identifies the reference as **“Weakly Supervised Pneumonia Localization.”** Verified authors, venue, and DOI are not available in the repository and are not inferred. The paper-inspired methodology is distinguished from project extensions below. Differences in local cohort size and implementation details preclude claiming exact reproduction.

### 3. Dataset

The experiments use locally supplied RSNA Stage 2 training images and labels. The usable binary cohort contains **14,863 labeled images**: 6,012 Pneumonia (`Lung Opacity`) and 8,851 Healthy (`Normal`); the other labeled class is excluded. The fixed seed-42 stratified split is 10,405 train, 2,972 validation, and 1,486 test images. CAM evaluation covers the 601 Pneumonia-positive test images.

The project notes describe a paper binary cohort of 17,489 images (8,964 Pneumonia and 8,525 Healthy). The local cohort is therefore not cohort matched. The original dataset is not committed to GitHub; local `Data/` is ignored. Dataset and split details are recorded in `metadata/DATASET_STATS.md` and `metadata/SPLIT_STATS.md`.

### 4. Methodology

The classification and localization path is:

```text
RSNA X-rays and annotations
             ↓
Metadata and fixed train/validation/test assignment
             ↓
DICOM decoding → 128×128 grayscale → train-fitted normalization
             ↓
   +---------+---------+---------+
   |                   |         |
   v                   v         v
Logistic Regression   SVM   Random Forest
             |
             v
      10-layer CNN
             |
             v
  Phase 1 classifier-weighted CAM
             |
             v
 Phase 2 Grad-CAM++ map (frozen CNN)
             |
             v
Threshold/components/boxes → IoU evaluation
```

The separate XGBoost extension path is:

```text
Preprocessed images → flattened pixels → train-only PCA (256) → XGBoost
```

The fixed image-ID split and train-fitted preprocessing statistics are reused across models. Ground-truth localization boxes are evaluation targets only; they are not used by CAM/Grad-CAM++ generation or localization parameter selection.

### 5. Phase 1 Results

The following values are from the saved fixed-test comparison. Accuracy, precision, recall, and F1 are proportions. XGBoost is an **extension**, not a reference-paper model.

| Model | Status | Test Accuracy | Test Precision | Test Recall | Test F1 |
|---|---|---:|---:|---:|---:|
| Logistic Regression | Reference-inspired | 0.832436 | 0.818841 | 0.752080 | 0.784042 |
| SVM | Reference-inspired | 0.835801 | 0.820467 | 0.760399 | 0.789292 |
| Random Forest | Reference-inspired | 0.871467 | 0.875458 | 0.795342 | 0.833479 |
| CNN | Reference-inspired | 0.880215 | 0.920477 | 0.770382 | 0.838768 |
| XGBoost | Extension | 0.833782 | 0.825368 | 0.747088 | 0.784279 |

Train/validation metrics, configuration, and source records are in `results/final_model_comparison.csv`, `results/model_comparison.csv`, and the individual model JSON files. The CNN train-accuracy value is unavailable and is not inferred.

### 6. Phase 1 Localization

The paper-reported reference CAM mean IoU is **0.1508**. The saved Phase 1 baseline result on 601 positive test images is:

- Mean IoU: **0.080087**
- Median IoU: **0.062605**
- IoU > 0: **80.70%**
- IoU ≥ 0.5: **0.33%**

The authoritative local result is `results/cam_results.json`. The Phase 1 result and implementation remain unchanged.

### 7. Phase 2 Motivation

Stage 12 identified **spatial imprecision** as the dominant error. At threshold 0.50, predicted boxes were typically larger than ground-truth boxes: mean predicted area 159,239.87 px² versus mean ground-truth area 77,971.06 px². The mean per-image predicted/ground-truth area ratio was 2.58. Visual examples also showed diffuse/misplaced activation, small predictions, and absent activation. While 80.70% had some overlap, only 0.33% reached IoU ≥ 0.5.

The threshold sensitivity analysis found the best mean IoU in its fixed grid at 0.30 (0.087996), but the high-IoU rate remained 0.33%; threshold adjustment alone did not address the dominant precision issue.

### 8. Phase 2 Improvement

Stage 13 replaced the classifier-weighted CAM map generator with **Grad-CAM++** for the Pneumonia class, using the existing frozen Stage 8 CNN. The 0.50 inclusive threshold, 8-connected component extraction, two-sigma cluster-size filter, native-DICOM box scaling, fixed split, preprocessing, and greedy one-to-one IoU evaluation were retained. Bounding boxes were not used to generate maps, select regions, or tune the method.

The 12-image smoke test checked CAM and mask generation, connected regions, finite values, valid in-bounds boxes, and visualization alignment before the full test evaluation.

### 9. Ablation Study

Stage 14 compared the Phase 1 baseline, the full Grad-CAM++ method, and “Grad-CAM++ removed.” Because map generation was Stage 13’s only changed component, removing Grad-CAM++ restores the Phase 1 algorithm exactly; the removed-component ablation therefore has the same results as Phase 1. The Stage 14 summary reuses the saved Stage 9 and Stage 13 outcomes and does not claim an independent repeat.

The full Phase 2 method is best by mean IoU. The saved ablation supports an association between Grad-CAM++ and changes in mean IoU, median IoU, and IoU > 0. However, IoU ≥ 0.5 is unchanged, so gains are not consistent across all metrics.

### 10. Final Results

The reference paper reports mean IoU only; unavailable reference metrics are shown as N/A. Local results use the same 601 positive test images and the same matching strategy.

| Method | Mean IoU | Median IoU | IoU > 0 | IoU ≥ 0.5 |
|---|---:|---:|---:|---:|
| Reference Paper CAM | 0.1508 | N/A | N/A | N/A |
| Phase 1 Baseline CAM | 0.080087 | 0.062605 | 80.70% | 0.33% |
| Phase 2 Improved CAM | 0.090277 | 0.068268 | 90.18% | 0.33% |

Phase 2 versus Phase 1:

- Mean IoU: **+0.010190** absolute, **+12.72%** relative
- Median IoU: **+0.005664** absolute, **+9.05%** relative
- IoU > 0: **+9.484 percentage points**, **+11.75%** relative
- IoU ≥ 0.5: **0.000 percentage points**, **0.00%** relative

### 11. Discussion

On the saved fixed-test evaluation, Grad-CAM++ increased mean and median IoU and the proportion with any overlap. Stage 14 indicates that the map-generation change is the only difference associated with those aggregates. The high-overlap rate did not improve, and Phase 2 qualitative examples include worsened as well as improved and unchanged cases.

Phase 1 classification results are descriptive results on a local cohort that differs from the paper-reported cohort. Several paper implementation details are incomplete in the project record. XGBoost is a separate extension; its test accuracy was 0.833782 and its training fit used train-only PCA with 256 components. The saved full XGBoost run took 645.75 seconds, mostly in DICOM preprocessing (about 585.8 seconds); PCA fitting took 6.69 seconds and XGBoost fitting 1.95 seconds.

### 12. Limitations

- The available labeled cohort (14,863) differs from the paper-reported binary cohort (17,489).
- Complete citation information and some reference architecture/hyperparameter details are absent from project records.
- The localization comparison uses one fixed split; it is not a repeated-split estimate.
- CAM/Grad-CAM++ localization remains weak in high-overlap terms: only 0.33% of images reach IoU ≥ 0.5 for either local method.
- Stage 13 qualitative results are mixed, so aggregate gains do not mean every image improved.
- The reference comparison is not cohort matched and does not establish exact reproduction.
- XGBoost’s 256-component PCA and fixed tree configuration are practical choices, not tuned optima.
- Results do not establish clinical validity or generalization to other populations.

### 13. Conclusion

Phase 2 demonstrates a measurable aggregate localization improvement over the implemented Phase 1 CAM baseline on the same 601-image test set: mean IoU rose from 0.080087 to 0.090277 (+12.72%). Median IoU and IoU > 0 also rose; IoU ≥ 0.5 remained unchanged at 0.33%. Phase 2 remains below the reference paper’s 0.1508 mean IoU and is not claimed to reproduce it exactly.
