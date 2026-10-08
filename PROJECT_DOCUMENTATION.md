# Weakly Supervised Pneumonia Localization
## Detailed Project Documentation and Manuscript Draft

> **Authoring note:** This draft is based on the implementation and saved results in this repository. It is not a verified transcription of the reference paper. Replace the citation placeholder, verify paper-specific claims against the source publication, and adapt the framing to the intended venue before submission. The reference is identified in the project records by the title *Weakly Supervised Pneumonia Localization*, but authors, venue, year, and DOI are not recorded here.

## Suggested manuscript title

**Weakly Supervised Pneumonia Classification and Localization in Chest Radiographs: A Reproduction Study and Grad-CAM++ Extension**

Alternative, more conservative title:

**A Reproduction-Oriented Study of Pneumonia Classification and Weakly Supervised Localization on RSNA Chest Radiographs**

## Abstract

### Draft

Chest radiographs provide a widely used imaging modality for assessing pneumonia, but image-level classification does not by itself identify where an abnormality is located. This project implements a reproduction-oriented classification and weakly supervised localization pipeline using the locally available labeled RSNA Pneumonia Detection Challenge Stage 2 data. The usable binary cohort contains 14,863 images: 6,012 labeled `Lung Opacity` and 8,851 labeled `Normal`. A deterministic, stratified 70/20/10 split was used for training, validation, and testing. Logistic Regression, linear Support Vector Machine, Random Forest, a ten-convolution-layer CNN, and an additional PCA + XGBoost classifier were evaluated. The XGBoost model is treated as an extension, not as part of the reference methodology.

The CNN achieved test accuracy 0.8802 and F1 0.8388. The Phase 1 classifier-weighted CAM baseline, evaluated on the 601 Pneumonia-positive test images, obtained mean IoU 0.0801, median IoU 0.0626, and any-overlap rate 80.70%; 0.33% of images reached IoU ≥ 0.5. Error analysis indicated frequent spatial imprecision and predicted regions larger than the annotated boxes. Phase 2 replaced only the CAM map-generation method with Grad-CAM++ while retaining the frozen CNN, threshold, region post-processing, split, and IoU matching procedure. Mean IoU increased to 0.0903 (+0.0102, or 12.72% relative), median IoU and any-overlap rate also increased, but the IoU ≥ 0.5 rate remained 0.33%. The reference-paper mean IoU recorded by the project is 0.1508; the local result does not reach that value. The available cohort differs from the paper-reported cohort, so these results should be interpreted as a local reproduction study rather than an exact replication.

### Keywords

Chest radiograph; pneumonia; weakly supervised localization; class activation mapping; Grad-CAM++; convolutional neural network; intersection over union; RSNA.

## 1. Introduction

### 1.1 Background

Pneumonia classification from chest radiographs is commonly framed as an image-level prediction problem. A classifier may estimate whether an image belongs to a Pneumonia or Normal category, but a class label alone does not show which image regions contributed to that decision. Localization is important for interpretability and for assessing whether a model’s salient regions overlap annotated abnormalities.

Weakly supervised localization seeks spatial evidence from image-level supervision without using localization boxes to train the model. Class activation mapping (CAM) provides one such approach when a CNN architecture exposes spatial feature maps and a compatible final linear classifier. The resulting heatmap can be thresholded and converted to regions, which may then be evaluated against box annotations held out from prediction.

### 1.2 Project aims

The project had two phases:

1. **Phase 1 — reproduction-oriented baseline:** construct a fixed image-level classification pipeline with classical classifiers and a paper-inspired CNN, then evaluate classifier-weighted CAM localization against the available RSNA bounding boxes. A separate XGBoost classifier was included as an explicitly labeled extension.
2. **Phase 2 — analysis and localization extension:** quantify baseline CAM failure modes, evaluate Grad-CAM++ using the same frozen CNN and test images, compare results against the baseline, and conduct a minimal component ablation.

The principal localization question was whether replacing the Phase 1 CAM map generator with Grad-CAM++ changed localization quality while keeping the model checkpoint, preprocessing, threshold, box-processing steps, and evaluation protocol fixed.

### 1.3 Contributions and scope

The work comprises:

- A documented binary cohort and a deterministic, stratified, image-ID-disjoint split.
- Shared DICOM preprocessing with normalization statistics fitted on training images.
- Classification comparisons for Logistic Regression, SVM, Random Forest, and a ten-layer CNN.
- A train-only PCA and fixed-configuration XGBoost classification extension.
- A Phase 1 CAM baseline with explicit thresholding, connected-component processing, coordinate scaling, and multi-box IoU matching.
- A threshold sensitivity and failure analysis on the saved Phase 1 CAM maps.
- A Phase 2 Grad-CAM++ localization result and a three-row ablation/comparison using the same saved baseline and Phase 2 evaluation outputs.

This work does not establish clinical validity, generalization to other populations, or an exact reproduction of the reference paper. It does not claim state-of-the-art performance.

## 2. Reference paper and reproduction scope

The project records the reference title as **“Weakly Supervised Pneumonia Localization.”** The complete bibliographic citation is unavailable in the repository and must be added by the author from the source paper:

> **Citation to complete:** [Authors]. “[Verified paper title].” [Journal/conference], [year], [volume/issue/pages], [DOI or URL].

Project records attribute the following comparison targets to the reference:

| Reference measure | Project-recorded value |
|---|---:|
| Binary cohort size after excluding the reported Diseased/No Pneumonia category | 17,489 |
| Pneumonia images | 8,964 |
| Healthy images | 8,525 |
| Reported CAM mean IoU | 0.1508 |
| Other reference localization metrics available in this project | Not available |

The local data and implementation do not match every reference specification. The project source records an SVM and Random Forest comparison, but some corresponding reference-paper settings are not fully documented. CNN channel widths and pooling placement are also implementation choices where the available paper description was incomplete. Comparisons against paper-reported values are therefore descriptive and not cohort matched.

## 3. Materials and methods

### 3.1 Dataset and labels

The experiments use locally supplied RSNA Pneumonia Detection Challenge Stage 2 training images and labels. The data are not committed to the repository. The original labeled training metadata includes three image-level class values:

- `Lung Opacity`, mapped to **Pneumonia (label 1)**.
- `Normal`, mapped to **Healthy (label 0)**.
- `No Lung Opacity / Not Normal`, excluded from the binary cohort.

The local binary metadata contains **14,863 unique images**, consisting of 6,012 Pneumonia images (40.45%) and 8,851 Healthy images (59.55%). There are 11,821 unique images in the excluded class. The supplied label tables contain one or more rows per positive image depending on the number of annotated boxes. The generated metadata has one row per image and preserves multiple boxes as aligned coordinate arrays. All 6,012 Pneumonia images have at least one positive box; Healthy images have no positive box.

The project notes report 17,489 binary images for the reference cohort. The local cohort is 2,626 images smaller, with 2,952 fewer Pneumonia images and 326 more Healthy images than the project-recorded reference counts. This discrepancy is retained and documented rather than altering the dataset to force a match. A further 3,000 image files exist in the local RSNA test folder but are not labeled in the supplied CSV files; they are not part of this supervised binary cohort.

### 3.2 Split design

One fixed split was generated at image-ID level with seed **42** and stratified by binary label:

| Split | Total images | Pneumonia | Healthy |
|---|---:|---:|---:|
| Training | 10,405 | 4,209 | 6,196 |
| Validation | 2,972 | 1,202 | 1,770 |
| Test | 1,486 | 601 | 885 |
| **Total** | **14,863** | **6,012** | **8,851** |

Image identifiers are unique in the source metadata and assigned to one split only; pairwise intersections between splits are empty. Assignment is deterministic and stored in `metadata/split_metadata.csv`. The same split is used across the model comparisons and is the source for the 601 Pneumonia-positive test images used in localization evaluation. No Phase 2 split was created.

### 3.3 Image preprocessing

Images are grayscale DICOM files. The shared preprocessing path:

1. Reads the stored pixel values and DICOM pixel representation.
2. Accounts for MONOCHROME1 inversion when applicable.
3. Scales grayscale values to [0, 1].
4. Resizes the image to **128 × 128** using bilinear interpolation.
5. Standardizes using global resized-pixel statistics computed from the training split only.

The stored training statistics are mean **0.4898645393** and standard deviation **0.2465253599**. These values are saved in `metadata/preprocessing_stats.json` and reused for validation and test preprocessing. DICOM image arrays are decoded on demand in the model pipeline; the whole dataset is not embedded in the repository.

### 3.4 Classification models

#### 3.4.1 Flattened-pixel classical models

Logistic Regression, linear SVM, and Random Forest operate on each normalized 128 × 128 grayscale image reshaped into a vector of **16,384** features.

- **Logistic Regression:** the project used an L2-regularized binary logistic objective. A NumPy L-BFGS implementation was used because the selected environment could not load a SciPy DLL under its Windows application-control setup.
- **SVM:** a linear squared-hinge Support Vector Machine with **C = 1.0** was used. The available project record does not specify the reference paper’s SVM kernel/settings.
- **Random Forest:** the saved comparison describes **100 trees**, maximum depth **20**, and square-root feature selection. These are the local implementation settings; unspecified reference-paper settings are not inferred.

The training/validation/test membership and the shared train-fitted normalization are reused. The saved comparison metrics are reported in Section 4.

#### 3.4.2 Ten-layer CNN

The CNN accepts a one-channel 128 × 128 input and contains ten 3 × 3 convolutional layers with ReLU activations. Its channel schedule is:

`32, 32, 64, 64, 96, 96, 128, 128, 128, 128`.

Each convolution uses stride 1 and padding 1. A 2 × 2 max-pooling layer follows convolutional layers 2, 4, 6, and 8. The final feature maps pass through adaptive global average pooling and a single two-class linear classifier. The implementation has **1,092,610 trainable parameters** according to the saved checkpoint metadata.

The training script specifies a maximum of 20 epochs, batch size 32, Adam optimizer, learning rate 0.0001, cross-entropy objective, and seed 42. The retained checkpoint is from epoch 9, selected using validation accuracy, with saved validation accuracy **0.897712** and **757,154 trainable parameters**. Phase 2 reuses this checkpoint and does not retrain the CNN.

### 3.5 XGBoost extension

XGBoost is a project extension and is not presented as part of the reference method. Each preprocessed image is flattened to 16,384 features. A randomized-SVD PCA with **256 components** is fitted once on the training feature matrix. Validation and test features are transformed with that already-fitted PCA; neither split is used to fit or refit PCA.

The fixed CPU histogram XGBoost configuration is:

| Parameter | Value |
|---|---:|
| `n_estimators` | 100 |
| `max_depth` | 4 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `random_state` | 42 |

No hyperparameter search was conducted. The recorded full-run wall time was about 645.75 s; DICOM image loading/preprocessing accounted for about 585.8 s. PCA fitting took about 6.69 s and XGBoost fitting about 1.95 s. The timed PCA-plus-XGBoost fit subtotal was about 9.59 s. Thus, preprocessing—not PCA or tree fitting—was the principal runtime cost in the recorded run.

### 3.6 Phase 1 CAM localization

#### CNN CAM construction

The final convolutional feature tensor is denoted \(A_k(x,y)\), with channel index \(k\) and spatial coordinates \((x,y)\). For the Pneumonia class \(c\), the final linear classifier provides channel weights \(w_k^c\). The raw class activation map is:

\[
M_c(x,y) = \operatorname{ReLU}\left(\sum_k w_k^c A_k(x,y)\right).
\]

Each image map is normalized to [0, 1] and bilinearly resized to 128 × 128. The classifier checkpoint, split checksum, architecture, and CAM compatibility are validated by the evaluation pipeline. Localization maps are generated from image inputs and classifier weights; box annotations are not passed into CAM generation.

#### Thresholding and region conversion

The official Phase 1 configuration uses an inclusive threshold of **0.50**, i.e. activated pixels satisfy \(M_c(x,y) \geq 0.50\). Activated pixels are grouped by iterative depth-first search with **8-connectivity**. Each component is converted to a rectangular extent. A per-image cluster-size filter retains component pixel counts within the mean ± two population standard deviations of the component sizes. The component extents are then scaled from the 128 × 128 map to the original DICOM image dimensions.

The exact implemented filter is an operationalization of a two-standard-deviation area-filter description; the project does not claim that unspecified paper implementation details are identical.

#### Ground-truth matching and IoU

Positive RSNA box annotations for the 601 Pneumonia-positive test images are used only for evaluation. For an individual predicted/ground-truth box pair, the continuous-coordinate intersection-over-union (IoU) is:

\[
\operatorname{IoU}(P,G) = \frac{|P \cap G|}{|P \cup G|}.
\]

When an image contains multiple predicted and/or ground-truth boxes, all candidate pairwise IoUs are sorted in descending order and matched greedily one-to-one. No prediction or reference box may be matched more than once. The image-level score is the sum of matched IoUs divided by the larger of the predicted-box count and ground-truth-box count; unmatched boxes therefore contribute zero and both missed and extra predictions are penalized. If both counts are zero, the implementation returns 0.

Reported localization metrics are the mean and median of these image-level scores, the percentage of images with score > 0, and the percentage with score ≥ 0.5.

### 3.7 Phase 2 error analysis

Stage 12 evaluated the frozen Phase 1 CNN and generated baseline CAMs once, then reused those maps for a threshold grid from **0.20 through 0.80** in increments of 0.10. This was diagnostic sensitivity analysis; the official Phase 1 score remains the threshold-0.50 result. The threshold sweep did not change the Phase 1 output.

Failure analysis at threshold 0.50 measured predicted box counts and areas, ground-truth box areas, and per-image predicted-to-ground-truth area ratios. Twelve indexed examples were saved to cover representative localization outcomes. This review identified diffuse or misplaced activations, broad predicted boxes, small predictions, multiple components, and missed/irrelevant activation. The diagnosis motivated changing the map generator while retaining the rest of the pipeline.

### 3.8 Phase 2 Grad-CAM++ localization

Stage 13 replaced only the map-generation method with a Grad-CAM++-based class-1 map from the same frozen Stage 8 CNN. The implementation exploits the model’s final linear classifier after global average pooling to compute its feature-map derivative weights in closed form, with numerical epsilon **1e-8**. The positive map is normalized per image and resized to 128 × 128.

The following were retained from Phase 1:

- The CNN checkpoint and preprocessing statistics.
- The fixed split and the same 601 Pneumonia-positive test images.
- The inclusive threshold of 0.50.
- 8-connected component extraction and the two-sigma component-size filter.
- Original-resolution box scaling.
- The one-to-one greedy IoU matching strategy and metrics.

Ground-truth boxes are evaluation-only. Threshold 0.50 was retained rather than selecting a new value from the Stage 12 threshold sweep. A deterministic 12-image smoke test verified finite activation maps, masks, component/box generation, valid image-coordinate bounds, and visualization alignment before the recorded full evaluation.

### 3.9 Ablation design

The Stage 14 minimal ablation contains three rows:

1. **Phase 1 Baseline:** classifier-weighted ReLU CAM and the saved Phase 1 result.
2. **Ablation: Grad-CAM++ removed:** the Phase 1 map generator with Phase 2 post-processing and evaluation settings. Because the map generator was the only Stage 13 change, this is algorithmically the same as Phase 1.
3. **Phase 2 Full Method:** Grad-CAM++ maps with all shared processing and evaluation settings retained.

The ablation summarizes existing Stage 9 and Stage 13 results. It does not independently rerun either 601-image experiment. This should be described as a controlled saved-result comparison, not as an independent replication or repeated-split statistical test.

## 4. Results

### 4.1 Classification performance

The following are saved test-set proportions from the fixed test split. The CNN train accuracy is unavailable in the saved comparison and is not estimated.

| Model | Role | Test accuracy | Test precision | Test recall | Test F1 |
|---|---|---:|---:|---:|---:|
| Logistic Regression | Reference-inspired baseline | 0.832436 | 0.818841 | 0.752080 | 0.784042 |
| Linear SVM | Reference-inspired baseline | 0.835801 | 0.820467 | 0.760399 | 0.789292 |
| Random Forest | Reference-inspired baseline | 0.871467 | 0.875458 | 0.795341 | 0.833479 |
| Ten-layer CNN | Reference-inspired model | 0.880215 | 0.920477 | 0.770383 | 0.838768 |
| PCA + XGBoost | Extension | 0.833782 | 0.825368 | 0.747088 | 0.784279 |

The CNN has the highest test accuracy and F1 among the saved model comparison rows. Random Forest has the next-highest test accuracy and F1. These are descriptive results for this split and cohort; they do not establish generalization or statistical significance.

### 4.2 Phase 1 localization result

| Metric | Phase 1 CAM, threshold 0.50 |
|---|---:|
| Number of Pneumonia-positive test images | 601 |
| Mean image-level IoU | 0.080087 |
| Median image-level IoU | 0.062605 |
| IoU > 0 | 80.70% |
| IoU ≥ 0.5 | 0.33% |

### 4.3 CAM threshold sensitivity

The saved Stage 12 results are:

| Threshold | Mean IoU | Median IoU | IoU > 0 | IoU ≥ 0.5 |
|---:|---:|---:|---:|---:|
| 0.20 | 0.085399 | 0.074030 | 91.01% | 0.17% |
| 0.30 | 0.087996 | 0.072232 | 89.02% | 0.33% |
| 0.40 | 0.085164 | 0.068574 | 85.19% | 0.33% |
| 0.50 | 0.080087 | 0.062605 | 80.70% | 0.33% |
| 0.60 | 0.070602 | 0.048774 | 75.37% | 0.67% |
| 0.70 | 0.055802 | 0.026677 | 66.89% | 0.67% |
| 0.80 | 0.036188 | 0.003740 | 53.58% | 0.17% |

Within the explored grid, threshold 0.30 yielded the highest mean IoU. It did not produce a material change in the high-overlap rate compared with threshold 0.50. Because this grid was evaluated on the test images, it is reported as post-hoc diagnostic sensitivity analysis; it is not used to claim a tuned or independently validated threshold. Phase 2 retained 0.50.

### 4.4 Failure analysis

At the Phase 1 threshold of 0.50:

| Predicted-box outcome | Images | Percentage |
|---|---:|---:|
| No predicted box | 1 | 0.17% |
| Exactly one predicted box | 542 | 90.18% |
| Multiple predicted boxes | 58 | 9.65% |

Across 659 retained predicted boxes, the mean area was **159,239.87 px²** and median area **150,144 px²** at original DICOM dimensions. Across 946 ground-truth boxes, the mean area was **77,971.06 px²** and median area **62,553 px²**. The mean per-image ratio of total predicted area to total ground-truth area was **2.58** (median **1.70**); the ratio of aggregate predicted area to aggregate ground-truth area was **1.42**.

These statistics are consistent with the visual review’s finding that broad/spatially imprecise regions were common. The area ratio is an aggregate description and does not imply that every prediction was oversized. Some examples instead had tiny or no useful activation, and a saved example showed saliency centered in lower central/diaphragmatic anatomy rather than the annotated opacity.

### 4.5 Phase 2 result and ablation

| Method | Mean IoU | Median IoU | IoU > 0 | IoU ≥ 0.5 |
|---|---:|---:|---:|---:|
| Phase 1 Baseline | 0.080087 | 0.062605 | 80.70% | 0.33% |
| Ablation: Grad-CAM++ removed | 0.080087 | 0.062605 | 80.70% | 0.33% |
| Phase 2 Full Method | 0.090277 | 0.068268 | 90.18% | 0.33% |

The full method had the highest saved mean IoU, median IoU, and any-overlap percentage. Removing Grad-CAM++ restores the baseline and baseline metrics. The IoU ≥ 0.5 percentage is identical for baseline and full method.

### 4.6 Phase 2 versus Phase 1 and paper reference

| Measure | Phase 1 CAM | Phase 2 Grad-CAM++ | Absolute change | Relative change |
|---|---:|---:|---:|---:|
| Mean IoU | 0.080087 | 0.090277 | +0.010190 | +12.72% |
| Median IoU | 0.062605 | 0.068268 | +0.005664 | +9.05% |
| IoU > 0 | 80.70% | 90.18% | +9.484 percentage points | +11.75% |
| IoU ≥ 0.5 | 0.33% | 0.33% | 0.000 percentage points | 0.00% |

The Phase 2 mean IoU is **0.060523 below** the project-recorded reference value of **0.1508**. The Phase 1 value was 0.070713 below that reference; Phase 2 narrows the numeric difference by 0.010190 but does not reach or reproduce the paper result.

## 5. Discussion

### 5.1 Classification

The local CNN achieved test accuracy 0.8802 and F1 0.8388 on the fixed split. The Random Forest result was also comparatively strong in this saved table, with test accuracy 0.8715 and F1 0.8335. XGBoost’s test accuracy and F1 were lower than the CNN and Random Forest in this comparison. Because the binary cohort differs from the project-recorded reference cohort and because only one fixed split is evaluated, these observations should be described as within-project model comparisons rather than broad evidence of model superiority.

### 5.2 Localization and interpretation

Grad-CAM++ improved the saved aggregate mean IoU, median IoU, and any-overlap rate over the Phase 1 threshold-0.50 CAM baseline. The unchanged post-processing and evaluation protocol mean that the map-generation algorithm is the only intended localization-method difference. The minimal ablation is consistent with the map generator accounting for the aggregate changes: the no-Grad-CAM++ configuration is the baseline, while the full-method saved results are higher on three measures.

The result is not uniform across metrics. The percentage of images reaching IoU ≥ 0.5 remains 0.33%. The qualitative examples contain successes, partial changes, and regressions. Thus, the evidence supports a modest aggregate improvement on the fixed evaluation, not uniformly more accurate boxes or reliable localization for clinical use.

The threshold sensitivity grid has its highest mean IoU at 0.30, but that is a post-hoc analysis on the same test cases. The Phase 2 comparison deliberately retained the Phase 1 threshold of 0.50 rather than selecting the threshold from that test sweep. Any future threshold selection should be based on a separate validation set and then evaluated once on an untouched test set.

### 5.3 Comparison with the reference

The mean IoU of 0.0903 remains below the project-recorded paper result of 0.1508. Several factors prevent a controlled paper-to-project attribution: the local binary cohort differs in size and class composition, available paper-specific implementation details are incomplete, and the project’s box conversion and matching implementation is explicitly documented but not confirmed to be identical to the paper’s. The observed numerical gap cannot be assigned to any single cause from the available experiments.

### 5.4 Computational considerations

The XGBoost experiment’s recorded end-to-end runtime was dominated by image decoding and preprocessing, rather than estimator fitting. This motivates caching or streamlining feature preparation if the experiment is repeated, while preserving the fixed split and fitting transforms on training data only. The project already limits PCA to one train-only fit and transforms validation/test with the same fitted PCA. Phase 2 inference used the trained CNN checkpoint and did not retrain the classifier.

## 6. Limitations

1. **Cohort mismatch:** the local usable cohort has 14,863 images, compared with the project-recorded reference binary cohort of 17,489.
2. **Incomplete reference citation and protocol:** authors/venue/DOI and some architecture/model/evaluation details are absent from the project records and must be verified from the paper.
3. **Single split:** results are from one deterministic split and are not repeated-run estimates; no confidence intervals or significance tests were produced.
4. **Low high-overlap rate:** only 0.33% of test images reached IoU ≥ 0.5 for either Phase 1 or Phase 2.
5. **Mixed per-image behavior:** aggregate Grad-CAM++ improvements coexist with qualitative regressions.
6. **Post-hoc threshold analysis:** the threshold sweep was diagnostic on the test cases and should not be interpreted as an independently validated operating point.
7. **Evaluation definition:** the reported image-level IoU uses greedy one-to-one matching and normalizes by the larger box count; other studies may use different matching/aggregation rules.
8. **No clinical validation:** the results evaluate agreement with the available annotations only and do not establish diagnostic utility, safety, or deployment readiness.
9. **XGBoost scope:** XGBoost is an extension with a fixed configuration, not a tuned comparison or a reference-paper model.

## 7. Conclusion

This project documents a reproduction-oriented binary classification and weakly supervised localization pipeline for the locally available RSNA Stage 2 cohort. The ten-layer CNN achieved test accuracy 0.8802 and F1 0.8388. Baseline CAM localization on the 601 Pneumonia-positive test images achieved mean IoU 0.080087. Replacing only the map-generation step with Grad-CAM++ raised mean IoU to 0.090277, an absolute increase of 0.010190 or 12.72%, with accompanying increases in median IoU and the any-overlap rate. The IoU ≥ 0.5 rate remained 0.33%, and the local Phase 2 mean remains below the reference value of 0.1508.

The saved results support a measurable aggregate improvement over this project’s Phase 1 baseline under a fixed evaluation protocol. They do not show uniformly improved localization, exact replication of the reference paper, or clinical validity.

## 8. Reproducibility and artifact map

The original dataset is local and is not included. Do not regenerate the split or rerun full experiments merely to read the saved results. The key records are:

| Artifact | Contents |
|---|---|
| `metadata/DATASET_STATS.md` | Local cohort and annotation counts |
| `metadata/SPLIT_STATS.md` | Fixed split counts and assignment rules |
| `metadata/preprocessing_stats.json` | Train-fitted normalization statistics |
| `results/final_model_comparison.csv` | Classification model comparison |
| `results/xgboost_results.json` | XGBoost configuration, scores, and timing |
| `results/cam_results.json` | Authoritative Phase 1 CAM result |
| `results/cam_threshold_analysis.csv` | Stage 12 threshold sensitivity |
| `results/cam_failure_statistics.json` | Stage 12 quantitative failure statistics |
| `results/phase2_improved_cam_results.json` | Stage 13 Grad-CAM++ result |
| `results/phase2_ablation.csv` | Stage 14 ablation comparison |
| `results/phase2_final_comparison.csv` | Stage 15 consolidated localization table |
| `results/phase2_final_results.json` | Stage 15 final values and changes |

Entry points and detailed running cautions are documented in [`REPRODUCTION.md`](./REPRODUCTION.md). The numbered notebooks are under `notebooks/`; several contain training or full-test evaluation cells and should not be executed merely to review saved results.

## 9. Author verification checklist

Before using this as a formal paper:

- Add and verify the complete reference-paper citation, including authors, publication venue, year, and DOI/URL.
- Check every claim about the original paper’s data, models, metrics, and protocol against the source publication.
- Preserve the distinction between paper methods, local reproduction decisions, and project extensions.
- Cite the RSNA dataset and confirm its applicable usage terms.
- Decide whether the manuscript will call the work a reproduction attempt, implementation study, or extension study; do not call it an exact replication without resolving the cohort/protocol differences.
- Report the fixed split, positive-test count, image-level IoU matching formula, and threshold analysis transparently.
- Do not describe the test-set threshold grid as validation or independent tuning.
- Do not infer missing CNN train accuracy, reference-paper metrics, or statistical significance.
- Do not claim clinical validity, state-of-the-art performance, or generalization from these results.
- If new evaluation or threshold selection is performed, use a documented validation/test protocol and preserve these saved results as the historical project record.
