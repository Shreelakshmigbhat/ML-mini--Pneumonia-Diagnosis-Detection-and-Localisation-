# Fixed train/validation/test split statistics

Generated from `metadata/pneumonia_metadata.csv` by `scripts/create_split.py`. Seed: **42**. Splits are stratified by `binary_label` at the image/patient-ID level.

The metadata contains no separate patient-ID field: `image_id` is the RSNA `patientId` carried forward by Stage 1. Each ID occurs exactly once in the source metadata and is assigned to exactly one split. The script rejects duplicate IDs and verifies pairwise split disjointness.

Per-class counts are allocated as closely as possible to 70% / 20% / 10% using largest remainders. Remainder ties use train, validation, test priority. A SHA-256 ordering of seed, class, and image ID makes the assignment deterministic without depending on row order or a third-party random-number implementation.

| Split | Images | % of dataset | Pneumonia | Healthy | Pneumonia within split | Healthy within split |
|---|---:|---:|---:|---:|---:|---:|
| train | 10,405 | 70.01% | 4,209 | 6,196 | 40.45% | 59.55% |
| validation | 2,972 | 20.00% | 1,202 | 1,770 | 40.44% | 59.56% |
| test | 1,486 | 10.00% | 601 | 885 | 40.44% | 59.56% |
| **Total** | **14,863** | **100.00%** | **6,012** | **8,851** | — | — |

## Verification

- Image IDs are unique in the input and assigned once.
- Pairwise intersections of the train, validation, and test ID sets are empty.
- Pneumonia total: 6,012; Healthy total: 8,851.
- The CSV stores one split assignment per image; no model-specific split is generated.
