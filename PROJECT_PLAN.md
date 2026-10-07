# Weakly Supervised Pneumonia Localization

## Dataset audit

Audit performed on the locally present RSNA Stage 2 dataset. No dataset files were changed, moved, renamed, downloaded, or preprocessed.

| Item | Observed result |
|---|---:|
| `stage_2_train_images` files / unique filename IDs | 26,684 |
| `stage_2_test_images` files / unique filename IDs | 3,000 |
| Train/test filename-ID overlap | 0 |
| Unique filename IDs across both image folders | 29,684 |
| `stage_2_detailed_class_info.csv` | 30,227 data rows, 2 columns: `patientId`, `class` |
| `stage_2_train_labels.csv` | 30,227 data rows, 6 columns: `patientId`, `x`, `y`, `width`, `height`, `Target` |
| Unique IDs in each CSV | 26,684 |
| Sample train image | `.dcm`, 1024 × 1024 pixels, DICOM with JPEG Baseline transfer syntax (UID `1.2.840.10008.1.2.4.50`) |

### Class counts

Class counts by unique image ID, deduplicating repeated rows for images with multiple boxes:

| `class` value | Unique images |
|---|---:|
| `Lung Opacity` | 6,012 |
| `No Lung Opacity / Not Normal` | 11,821 |
| `Normal` | 8,851 |
| **Total labeled training images** | **26,684** |

The raw class-info row counts are 9,555 `Lung Opacity`, 11,821 `No Lung Opacity / Not Normal`, and 8,851 `Normal`. The difference for `Lung Opacity` comes from a row per bounding box rather than a row per image.

There are **9,555 positive (`Target=1`) bounding-box rows** belonging to **6,012 unique images**. **3,398 images** have more than one box; the maximum observed is four boxes for one image. The label CSV's box coordinates are `x`, `y`, `width`, and `height`; `Target` indicates whether that row is positive. The 20,672 `Target=0` rows have no pneumonia box.

The 3,000 test-folder images have no corresponding labeled IDs in these CSVs; the CSVs describe the 26,684-image training set.

## Reference paper information

The reference-paper description provided for this project reports:

- 28,989 images total: 8,964 Pneumonia, 8,525 Healthy, and 11,500 Diseased/No Pneumonia.
- The Diseased/No Pneumonia group is removed, leaving 17,489 images (8,964 Pneumonia and 8,525 Healthy).
- A 70% / 20% / 10% train / validation / test split.

For planning, `Lung Opacity` is the provisional counterpart of Pneumonia and `Normal` of Healthy. This mapping and the eligibility criteria must be checked against the paper before building the reproduction cohort.

## Reproduction targets

1. Confirm the paper's class mapping, inclusion/exclusion rules, model, preprocessing, training protocol, and evaluation metrics from the reference paper.
2. Reconstruct the reported 17,489-image cohort and document any differences in dataset version or filtering.
3. Reproduce the 70% / 20% / 10% split, keeping related images/patients from leaking across partitions where the paper's protocol permits.
4. Implement and evaluate the paper's method only after the audit and protocol are reconciled; compare results using the paper's reported metrics.

## Extension target

Quantitatively assess localization quality against the available RSNA boxes, reporting box-overlap (IoU) and a point-localization metric alongside classification performance. Compare the localization results with the reproduction baseline.

## Planned stages

1. **Project and dataset audit — complete.** Counts, labels, box multiplicity, image encoding, and repository ignore rules recorded.
2. **Metadata generation — complete.** `scripts/create_metadata.py` generates one row per usable image, excludes all non-Pneumonia/non-Normal classes, and preserves multiple boxes as aligned JSON coordinate arrays. The output remains smaller than the reference cohort; see `metadata/DATASET_STATS.md`.
3. **Fixed data split — complete.** `scripts/create_split.py` writes the single seed-42, label-stratified 70% / 20% / 10% image-ID assignment to `metadata/split_metadata.csv`; see `metadata/SPLIT_STATS.md`.
4. **Reference protocol review.** Extract the paper's precise cohort definition and experimental details; resolve differences before implementation.
5. **Reproduction implementation and evaluation.** Build the baseline and compare results to the paper. No training has been performed in these completed stages.
6. **Localization extension.** Evaluate localization with the selected box/point metrics and compare to the baseline.
7. **Results and documentation.** Record reproducibility details, limitations, and paper-to-project comparisons.

## Dataset discrepancy to resolve

This directory is the RSNA Pneumonia Detection Challenge **Stage 2** layout (`stage_2_*` files and `.dcm` images), but its labeled training cohort does not match the reference-paper counts:

| Category | Reference paper | Local unique labeled IDs | Difference (local − paper) |
|---|---:|---:|---:|
| Pneumonia / provisional `Lung Opacity` | 8,964 | 6,012 | -2,952 |
| Healthy / provisional `Normal` | 8,525 | 8,851 | +326 |
| Diseased/no pneumonia / provisional `No Lung Opacity / Not Normal` | 11,500 | 11,821 | +321 |
| **Total** | **28,989** | **26,684** | **-2,305** |

After provisionally excluding `No Lung Opacity / Not Normal`, the local cohort would contain **14,863** images, **2,626 fewer** than the paper's 17,489. The on-disk total of 29,684 includes 3,000 test images that have no labels in the supplied CSVs, so it is not directly comparable to the paper's labeled class total. The paper's source/version or filtering rules must be reconciled before claiming an exact reproduction.
