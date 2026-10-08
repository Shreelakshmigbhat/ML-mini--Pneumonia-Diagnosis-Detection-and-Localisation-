# Phase 2 — Improved Weakly Supervised Localization

## 1. Motivation

Stage 12 error analysis found that the Phase 1 CAM often produced spatially imprecise localization: predicted regions were frequently larger than the annotated boxes, salient activation could fall outside annotated pulmonary opacity, and strong overlap was rare. Threshold analysis showed that lowering the threshold to 0.30 raised mean IoU only modestly and did not improve the IoU >= 0.5 rate. Phase 2 therefore tested a distinct localization-map generator while keeping the trained classifier and box-evaluation procedure fixed.

## 2. Phase 1 Baseline

The official Phase 1 baseline is the frozen Stage 8 CNN with classifier-weighted ReLU CAM, followed by 8-connected component extraction, the existing per-image two-sigma cluster-size filter, native-image box scaling, and greedy one-to-one IoU matching.

On the 601 Pneumonia-positive images in the fixed test split:

- Mean IoU: **0.080087**
- Median IoU: **0.062605**
- IoU > 0: **80.70%**
- IoU >= 0.5: **0.33%**

The saved Phase 1 result remains unchanged.

## 3. Error Analysis

At the Phase 1 threshold of 0.50, 90.18% of images had one predicted box, 9.65% had multiple boxes, and 0.17% had no predicted box. The mean predicted box area was 159,239.87 px², versus a mean ground-truth box area of 77,971.06 px². The mean per-image predicted-to-ground-truth area ratio was 2.58 (median 1.70).

Representative Stage 12 panels showed broad or misplaced activations, including an example centered in lower central/diaphragmatic anatomy, alongside very small or absent activations. The dominant problem was spatial imprecision rather than simply absence of activation.

## 4. Proposed Improvement

Stage 13 replaced only the CAM map-generation method with **Grad-CAM++** using the same frozen Stage 8 CNN and Pneumonia class. The inclusive threshold (0.50), 8-connected DFS regions, two-sigma filtering, box scaling, preprocessing, fixed test split, and IoU matching strategy were held constant. Ground-truth boxes were used only for final evaluation.

A deterministic 12-image smoke test verified finite maps, activation masks, components, valid in-bounds boxes, and visualization alignment before the single full evaluation.

## 5. Ablation Study

Stage 14 compared Phase 1, the full Phase 2 method, and “Grad-CAM++ removed.” Because Grad-CAM++ map generation was the only Stage 13 change, removing it restores the Phase 1 algorithm exactly; that ablation therefore has the same metrics as the saved Phase 1 result. Stage 14 reused the verified Stage 9 and 13 outputs and did not repeat either full evaluation.

The full method was best by mean IoU. The ablation associates the change in mean, median, and any-overlap metrics with Grad-CAM++ map generation, but this is not an independent repeat and does not establish universal causal improvement. IoU >= 0.5 did not change.

## 6. Final Results

All local methods use the same 601 fixed-test images and matching strategy. The paper reference is reported as mean IoU only; other reference metrics are unavailable here.

| Method | Mean IoU | Median IoU | IoU > 0 | IoU >= 0.5 |
|---|---:|---:|---:|---:|
| Reference Paper CAM | 0.1508 | N/A | N/A | N/A |
| Phase 1 Baseline CAM | 0.080087 | 0.062605 | 80.70% | 0.33% |
| Phase 2 Improved CAM | 0.090277 | 0.068268 | 90.18% | 0.33% |

Phase 2 minus Phase 1:

| Metric | Absolute change | Relative change |
|---|---:|---:|
| Mean IoU | +0.010190 | +12.72% |
| Median IoU | +0.005664 | +9.05% |
| IoU > 0 | +9.484 percentage points | +11.75% |
| IoU >= 0.5 | 0.000 percentage points | 0.00% |

## 7. Reference Comparison

Phase 2 mean IoU is **0.060523 below** the paper-reported value of 0.1508. Phase 1 was 0.070713 below the reference, so Phase 2 narrows that difference by 0.010190 but does **not** reproduce the paper result. The local cohort and implementation are not claimed to be an exact replication.

## 8. Discussion

Mean IoU increased from 0.080087 to 0.090277, an absolute increase of 0.010190 and relative increase of 12.72%. Median IoU and the proportion with any overlap also increased. The fixed-checkpoint, fixed-post-processing comparison and Stage 14 ablation identify Grad-CAM++ map generation as the only changed component.

The improvement was not consistent across all metrics: the fraction of images with IoU >= 0.5 remained 0.33%. Stage 13 qualitative examples include both improved and worsened cases. The results support a modest aggregate improvement on this fixed test evaluation, not universal per-image improvement or generalization to other cohorts.

## 9. Conclusion

Phase 2 demonstrates a measurable improvement over the implemented Phase 1 CAM baseline on the same 601-image test set, primarily in mean IoU, median IoU, and any-overlap rate. It does not improve the high-IoU rate and remains below the paper reference IoU of 0.1508. No models were retrained, no data split or Phase 1 results were modified, and no new experiments were run for this final comparison.
