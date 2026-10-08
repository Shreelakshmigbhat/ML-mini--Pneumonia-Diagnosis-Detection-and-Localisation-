# Results

## 1. Project Objective

This project reproduces the classification and weakly supervised localization pipeline described by the reference titled **“Weakly Supervised Pneumonia Localization”**, and adds XGBoost as a separate extension. The paper title is present in the project notebooks; the repository does not record verified authors, venue, or DOI, so those bibliographic details are not inferred here.

## 2. Dataset

The experiments use the locally available RSNA Pneumonia Detection Challenge Stage 2 data. The original dataset is not included in the repository.

| Item | Local experiment |
|---|---:|
| Usable labeled cohort | 14,863 images |
| Pneumonia (`Lung Opacity`) | 6,012 |
| Healthy (`Normal`) | 8,851 |
| Train | 10,405 |
| Validation | 2,972 |
| Test | 1,486 |

The paper description recorded in the project reports 28,989 images total and a 17,489-image binary cohort after excluding 11,500 Diseased/No Pneumonia images. The local Stage 2 binary cohort is 2,626 images smaller (6,012 Pneumonia and 8,851 Healthy), so this is not a cohort-matched replication. Split counts and assignment details are documented in [`metadata/SPLIT_STATS.md`](./metadata/SPLIT_STATS.md).

## 3. Experimental Setup

- Original DICOM X-rays are decoded on demand, resized to 128 × 128 grayscale, scaled to [0, 1], and standardized using global pixel mean and standard deviation fitted on the training split only (mean 0.4898645393; standard deviation 0.2465253599).
- The existing deterministic, image-ID-disjoint 70/20/10 split uses seed 42. The same split is used across experiments; it was not recreated for this consolidation.
- Logistic Regression, linear SVM, and Random Forest use flattened 16,384-pixel vectors.
- The paper-inspired CNN has ten 3 × 3 ReLU convolution layers, four max-pooling stages, global average pooling, and one linear classifier. The local channel schedule is an implementation choice where the paper description is incomplete.
- CAM uses the saved CNN checkpoint and evaluates its localization against test annotations; annotation boxes are evaluation targets, not CAM inputs.
- XGBoost is a separate extension using 256-component randomized-SVD PCA fitted on train only, followed by CPU histogram XGBoost with 100 estimators, depth 4, learning rate 0.05, row/column subsampling 0.8, and seed 42.

## 4. Classification Results

Values below are actual saved results. Accuracy and other test metrics are proportions (for example, 0.88 is 88%). The comparison uses the fixed test split. CNN train accuracy is **N/A** because no CNN results JSON or saved train metric is present; its validation accuracy is available from the saved checkpoint and its test metrics from the saved comparison artifact.

| Model | Category | Representation | Train Accuracy | Validation Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Reference-paper model | Flattened normalized pixels | 0.8587 | 0.8503 | 0.8324 | 0.8188 | 0.7521 | 0.7840 |
| SVM | Reference-paper model | Flattened normalized pixels | 0.8586 | 0.8530 | 0.8358 | 0.8205 | 0.7604 | 0.7893 |
| Random Forest | Reference-paper model | Flattened normalized pixels | 0.9879 | 0.8779 | 0.8715 | 0.8755 | 0.7953 | 0.8335 |
| CNN | Reference-paper model | Learned 10-layer CNN features | N/A | 0.8977 | 0.8802 | 0.9205 | 0.7704 | 0.8388 |
| XGBoost | **Extension; not in the paper** | Train-only PCA (256 components) | 0.8737 | 0.8513 | 0.8338 | 0.8254 | 0.7471 | 0.7843 |

Random Forest and CNN achieved the highest recorded test accuracy in this local comparison. XGBoost is close to Logistic Regression and SVM on test accuracy/F1, and below Random Forest and CNN. These differences are descriptive only: the cohort does not match the paper, and some reference hyperparameters are unspecified.

The row-level values are in [`results/final_model_comparison.csv`](./results/final_model_comparison.csv). The source artifacts are the existing model JSON files and `results/model_comparison.csv`; no models were retrained for this table.

## 5. Localization Results

| Method | Dataset split | Mean IoU | Median IoU | IoU > 0 | IoU ≥ 0.5 | Source |
|---|---|---:|---:|---:|---:|---|
| CNN + CAM | Test, reference paper | 0.1508 | N/A | N/A | N/A | Paper-reported test IoU |
| CNN + CAM | Test, our result | 0.080087 | 0.062605 | 80.70% | 0.33% | Existing `results/cam_results.json`; 601 positive test images |

The local mean IoU is below the paper-reported 0.1508. The full metrics and method details are in [`results/cam_results.json`](./results/cam_results.json); that source file was not modified. Consolidated values are in [`results/final_localization_results.csv`](./results/final_localization_results.csv).

## 6. Reference Paper vs Our Reproduction

| Component | Reference paper / project-recorded target | Our implementation |
|---|---|---|
| Dataset | 28,989 total; 17,489 after excluding the reported Diseased/No Pneumonia class | Local Stage 2 labeled binary cohort: 14,863 (6,012 Pneumonia, 8,851 Healthy); not cohort matched |
| Split | Project notes report 70/20/10 | Fixed seed-42 stratified 70/20/10 image-ID split; disjoint across partitions |
| Preprocessing | Exact details are not fully established in the repository's paper notes | DICOM to 128 × 128 grayscale; scaled then standardized using training-only global statistics |
| LR | Reference test accuracy 0.7302 | Test accuracy 0.8324; NumPy L-BFGS logistic objective due a documented SciPy DLL loading restriction |
| SVM | Reference test accuracy 0.5818; kernel/settings not specified in the project record | Linear squared-hinge SVM, C=1.0; test accuracy 0.8358 |
| RF | Reference test accuracy 0.8300; hyperparameters unspecified | 100 trees, max depth 20, sqrt feature selection; test accuracy 0.8715 |
| CNN | Reference train/test accuracy 0.9307 / 0.9247 | Ten-layer CNN; test accuracy 0.8802. Local channel widths and pooling locations are documented choices because the paper description is incomplete |
| CAM | Reference test mean IoU 0.1508 | Test mean IoU 0.080087; local threshold, connected-component clustering, filtering, and box-matching rules are documented in `cam_results.json` |

The project identifies the source paper by the title “Weakly Supervised Pneumonia Localization”; complete citation metadata was not available in the repository. Reference values above are the targets already recorded in project artifacts, not a claim of exact replication. The CNN architecture, SVM settings, Random Forest settings, local dataset cohort, and localization implementation include differences or choices that should not be attributed to the paper.

## 7. Discussion of Differences

The measured cohort differs from the paper-reported binary cohort by 2,626 images, with fewer Pneumonia cases and more Healthy cases locally. This alone prevents a like-for-like comparison. In addition, the project notes do not fully specify CNN channel widths or pooling locations, SVM kernel/settings, or Random Forest hyperparameters; the implementation records these choices rather than presenting them as paper facts.

Localization depends on the implemented normalized-CAM threshold (0.5), 8-connected DFS regions, an area filter, and greedy one-to-one IoU matching for images with multiple boxes. The repository does not establish that these choices exactly match the paper. No additional causal explanation for the score difference is established by the saved results.

## 8. XGBoost Extension

**XGBoost was introduced as an extension and was not part of the original reference methodology.**

The model receives 256 PCA features from the flattened 16,384-pixel input. PCA uses randomized SVD and is fitted once on training data; validation and test data are transformed with that fitted PCA. XGBoost uses the fixed 100-estimator, depth-4, learning-rate-0.05 CPU histogram configuration recorded in `xgboost_results.json`; no hyperparameter search was run.

On the test split, XGBoost achieved accuracy 0.8338, F1 0.7843, precision 0.8254, and recall 0.7471. This is similar to the LR/SVM baseline scores and below RF/CNN in this comparison. The full Stage 10 run took 645.75 seconds; DICOM loading/preprocessing dominated (about 585.8 seconds), while PCA fitting took 6.69 seconds and XGBoost fitting took 1.95 seconds. The measured trade-off is therefore mostly in serial image preparation, not tree fitting.

## 9. Limitations

- Weakly supervised CAM does not use localization annotations during prediction, and its measured mean IoU is 0.0801.
- Localization depends on the fixed CAM threshold, clustering, filtering, and matching interpretation.
- The locally available labeled cohort differs from the paper-reported cohort.
- The reference description in the repository is incomplete for parts of the CNN architecture and some classical-model settings.
- Results use one fixed split and seed; they are not repeated-split estimates.
- XGBoost dimensionality/configuration is a fixed practical choice, not an optimized setting.
- A standalone CNN results JSON and train-accuracy value are absent; those table fields are marked N/A where unavailable.

## 10. Conclusion

The project completed the reference-inspired classification pipeline and CAM localization on its fixed local cohort, and evaluated XGBoost as a separate extension. CNN and Random Forest had the strongest recorded classification test accuracy (0.8802 and 0.8715 respectively). CAM mean IoU was 0.0801 versus the paper-reported 0.1508, and the cohort and implementation differences preclude an exact-reproduction claim. XGBoost achieved 0.8338 test accuracy with a 1.95-second model fit after train-only PCA; image decoding and preprocessing dominated its total runtime.
