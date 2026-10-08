# Phase 2 Plan

## Phase 1 Baseline

- Existing Stage 8 CNN checkpoint and fixed test split; no training or split changes were made for Stage 12.
- Phase 1 CAM at threshold 0.50: mean IoU 0.080087, median IoU 0.062605, IoU > 0 on 80.70% of images, and IoU >= 0.5 on 0.33%.
- The localization evaluation covers 601 Pneumonia-positive test images. CAM inference used no box labels.

## Observed Failure Modes

- At threshold 0.50, 90.18% of images have one predicted box, 9.65% have multiple boxes, and 0.17% have no predicted box.
- Across retained boxes, mean predicted area is 159,239.87 px² (median 150,144 px²), compared with mean ground-truth box area of 77,971.06 px² (median 62,553 px²).
- The mean per-image predicted/ground-truth total area ratio is 2.58; its median is 1.70. The ratio of aggregate predicted area to aggregate ground-truth area is 1.42.
- Visual review shows both diffuse/oversized predicted regions and salient activation away from the annotated pulmonary opacity, including a lower central/diaphragmatic hotspot. Very small or absent activations also occur.
- Most images have some overlap, but almost none achieve IoU >= 0.5. This points to poor spatial precision rather than a complete inability to produce activation.
- See `results/phase2_error_analysis/examples.csv` and its 12 indexed image panels for the representative cases.

## Threshold Analysis

The same frozen checkpoint, existing test split, CAM maps, 8-connected clustering, two-sigma component filter, coordinate scaling, and IoU matching were used at every threshold.

| Threshold | Mean IoU | Median IoU | IoU > 0 | IoU >= 0.5 |
|---:|---:|---:|---:|---:|
| 0.20 | 0.085399 | 0.074030 | 91.01% | 0.17% |
| 0.30 | 0.087996 | 0.072232 | 89.02% | 0.33% |
| 0.40 | 0.085164 | 0.068574 | 85.19% | 0.33% |
| 0.50 | 0.080087 | 0.062605 | 80.70% | 0.33% |
| 0.60 | 0.070602 | 0.048774 | 75.37% | 0.67% |
| 0.70 | 0.055802 | 0.026677 | 66.89% | 0.67% |
| 0.80 | 0.036188 | 0.003740 | 53.58% | 0.17% |

Threshold 0.30 is best by mean IoU within this exploratory grid. It raises mean IoU by 0.007909 over the 0.50 Phase 1 operating point, but does not resolve the low high-IoU rate. This test-set sweep is diagnostic only; it does not replace or edit the official Phase 1 result.

## Proposed Improvement

Replace plain CAM with **Grad-CAM++** for localization, using the same frozen Stage 8 CNN and Pneumonia image-level class output. Keep box annotations out of map generation and training; use them only for evaluation. Retain the existing box conversion and IoU protocol initially so the comparison isolates the localization-map change.

## Why It Should Help

The reviewed examples show diffuse or misplaced class activation and bounding boxes substantially larger than the annotated regions. Grad-CAM++ uses gradient-weighted class-specific activations rather than the baseline CAM's classifier-weighted feature-map sum, providing a distinct weakly supervised localization signal that may emphasize more spatially relevant evidence. It requires no box-label training and adds inference/backpropagation cost without a new training run. Improvement is a hypothesis to test, not a guaranteed outcome.

## Evaluation Strategy

1. Use the existing CNN checkpoint, preprocessing statistics, and fixed train/validation/test assignments; do not retrain the CNN or recreate the split.
2. Implement Grad-CAM++ as a separate localization method, leaving the official Phase 1 CAM implementation and results untouched.
3. Generate class-1 maps from image-level inference only. Ground-truth boxes remain evaluation-only.
4. Select any Grad-CAM++ threshold using validation data, then evaluate the selected setting once on the fixed test set using the same image-level IoU matching protocol.
5. Report mean/median IoU, IoU > 0, IoU >= 0.5, box-count distribution, box areas, and representative visual examples. Compare against both the official threshold-0.50 baseline and the Stage 12 exploratory threshold-0.30 result.
6. Keep inference batched and save only a small, indexed set of examples.

## Success Criteria

- Primary: test mean IoU exceeds the official Phase 1 value of 0.080087, using a threshold selected without test-set tuning.
- Stronger target: exceed the exploratory best fixed-threshold CAM result of 0.087996.
- Avoid trading all overlap away for a few strong matches: report median IoU, IoU > 0, and IoU >= 0.5 alongside the primary metric.
- Demonstrate the intended spatial improvement in examples and area statistics, with no bounding-box supervision and no changes to the split or Phase 1 artifacts.
