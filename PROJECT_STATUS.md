# Project Status

## Phase 1

**COMPLETE**

1. Project and dataset audit
2. Metadata generation
3. Fixed train/validation/test split
4. Logistic Regression baseline
5. SVM baseline
6. Random Forest baseline
7. Custom 10-layer CNN
8. Baseline CAM localization
9. Fixed-split model evaluation and result consolidation
10. XGBoost extension
11. Phase 1 results finalization and documentation consolidation

The split and saved Phase 1 model/localization results remain the project baseline.

`PROJECT_PLAN.md` retains an earlier 1–10 numbering where CAM is stage 8, XGBoost stage 9, and Phase 1 finalization stage 10. The list above follows the later project-stage chronology used for this final status: XGBoost is Stage 10 and Phase 1 finalization is Stage 11.

## Phase 2

**COMPLETE**

12. CAM error analysis and threshold sensitivity
13. Grad-CAM++ improved localization using the frozen CNN
14. Phase 2 ablation study
15. Final Phase 1 / Phase 2 / reference comparison
16. Final documentation, cleanup, reproducibility checks, and submission preparation

## Final Deliverables

- **Source code:** preprocessing/dataset, classical and CNN models, baseline CAM, Grad-CAM++, and scripts for data preparation, training/evaluation, error analysis, improved localization, and ablation summary.
- **Notebooks:** nine numbered notebooks from `01_Dataset_Preprocessing.ipynb` through `09_Phase2_Final_Comparison.ipynb`.
- **Results:** saved classification/model metrics, Phase 1 CAM results, Stage 12 threshold/failure analysis, Stage 13 improved CAM metrics/examples, Stage 14 ablation, and Stage 15 comparison.
- **Documentation:** `README.md`, `RESULTS.md`, `PHASE2_PLAN.md`, `PHASE2_RESULTS.md`, `FINAL_REPORT.md`, `REPRODUCTION.md`, and this status file.

## Known Limitations

- The local binary cohort (14,863 labeled images) differs from the paper-reported cohort (17,489).
- Complete bibliographic metadata and some paper implementation details are not available in the repository.
- Evaluation uses one fixed split; results are not repeated-split estimates.
- Grad-CAM++ raises mean IoU to 0.090277 from 0.080087, but the IoU ≥ 0.5 rate remains 0.33%; per-image examples include regressions.
- The paper-reported mean IoU is 0.1508; the Phase 2 result remains below it.
- XGBoost settings are a fixed mini-project extension configuration rather than a hyperparameter-optimized model.
- Results do not establish clinical validity or out-of-cohort generalization.
