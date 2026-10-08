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

The reviewed examples show diffuse or misplaced class activation and bounding boxes substantially larger than the annotated regions. Grad-CAM++ uses gradient-weighted class-specific activations rather than the baseline CAM's classifier-weighted feature-map sum, providing a distinct weakly supervised localization signal that may emphasize more spatially relevant evidence. It requires no box-label training or new training run; its feature-map weighting is calculated during inference. Improvement is a hypothesis to test, not a guaranteed outcome.

## Evaluation Strategy

1. Use the existing CNN checkpoint, preprocessing statistics, and fixed train/validation/test assignments; do not retrain the CNN or recreate the split.
2. Implement Grad-CAM++ as a separate localization method, leaving the official Phase 1 CAM implementation and results untouched.
3. Generate class-1 maps from image-level inference only. Ground-truth boxes remain evaluation-only.
4. For the Stage 13 controlled comparison, retain the Phase 1 threshold of 0.50 without threshold tuning, then evaluate once on the fixed test set using the same image-level IoU matching protocol.
5. Report mean/median IoU, IoU > 0, IoU >= 0.5, box-count distribution, box areas, and representative visual examples. Compare against both the official threshold-0.50 baseline and the Stage 12 exploratory threshold-0.30 result.
6. Keep inference batched and save only a small, indexed set of examples.

## Success Criteria

- Primary: test mean IoU exceeds the official Phase 1 value of 0.080087, using a threshold selected without test-set tuning.
- Stronger target: exceed the exploratory best fixed-threshold CAM result of 0.087996.
- Avoid trading all overlap away for a few strong matches: report median IoU, IoU > 0, and IoU >= 0.5 alongside the primary metric.
- Demonstrate the intended spatial improvement in examples and area statistics, with no bounding-box supervision and no changes to the split or Phase 1 artifacts.

## Implemented Improvement

Phase 2 implements and evaluates an improved CAM localization strategy based on **Grad-CAM++** maps from the frozen Stage 8 CNN. Stage 12 identified spatial imprecision—diffuse or misplaced activation, oversized predicted regions, and very few high-IoU cases—as the dominant failure mode. Grad-CAM++ replaces the classifier-weighted CAM map with a gradient-weighted class activation map; it does not use bounding-box annotations to generate or process predictions.

The existing 0.50 inclusive threshold, 8-connected components, per-image two-sigma component-size filter, native-resolution box scaling, and Phase 1 IoU matching rules were held fixed. The threshold was not tuned on test labels. Other recorded settings were Pneumonia class index 1, input/map size 128 x 128, batch size 16 for full evaluation, and numerical epsilon 1e-8 (configurable with `--epsilon`).

The 12-image deterministic smoke test passed checks for finite CAM values, activation masks, components, in-bounds boxes, and overlay alignment. Full evaluation on the same 601 positive test images completed without retraining or changing the split:

| Metric | Phase 1 CAM | Phase 2 Grad-CAM++ |
|---|---:|---:|
| Mean IoU | 0.080087 | 0.090277 |
| Median IoU | 0.062605 | 0.068268 |
| IoU > 0 | 80.70% | 90.18% |
| IoU >= 0.5 | 0.33% | 0.33% |

Mean IoU increased by 0.010190 (12.72% relative to Phase 1); it also exceeds Stage 12's best fixed-threshold CAM mean IoU of 0.087996. The proportion reaching IoU >= 0.5 did not change, and qualitative examples include worsened cases. The result is an improvement on the reported aggregate test metrics, not evidence that every localization improved or that the method generalizes beyond this evaluation.

Artifacts: `src/localization/improved_cam.py`, `scripts/run_improved_cam.py`, `results/phase2_improved_cam_results.json`, `results/phase2_baseline_vs_improved.csv`, `results/phase2_improved_examples/`, and `notebooks/07_Improved_CAM.ipynb`. The original Phase 1 implementation and `results/cam_results.json` remain unchanged.
