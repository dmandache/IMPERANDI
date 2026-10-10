# TCGA-LIHC demos

These demos use the public [TCGA-LIHC](https://www.cancerimagingarchive.net/collection/tcga-lihc/) imaging collection to demonstrate and benchmark IMPERANDI.

## Interactive demos

Both notebooks use the same three public patients: `TCGA-BC-A10Y`, `TCGA-DD-A113`, and `TCGA-DD-A4NJ`.

- [**Light demo**](demo_light.ipynb): base IMPERANDI installation; download, ingest, inspect outputs, and metadata-only phase curation.
- [**Full demo**](demo_full.ipynb): continues through NIfTI conversion, liver segmentation, radiomic feature extraction, the interactive viewer and basic TorchIO preprocessing into a PyTorch batch.

The demos are intentionally small and user-facing. They showcase IMPERANDI functionality rather than reproduce the full scientific comparisons.

## Scientific benchmarks

[**Benchmarks**](../benchmarks/) contains the reproducible cohort-building comparisons against published TCGA-LIHC cohorts, including the MRI multiphase and CT portal-venous targets.
