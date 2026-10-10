# CT portal venous cohort

[Paper: Sarfati et al., MICCAI (2023)](https://arxiv.org/abs/2307.04617)
· [Author code](https://github.com/Guerbet-AI/wsp-contrastive)
· [Processed CT release](https://zenodo.org/records/8199165)

## Inclusion and reference

The TCGA-LIHC evaluation cohort (D2_histo) contains **49 patients** with Ishak
scores, one portal venous CT at the earliest available date per patient:
34 labels 0 (Ishak 0–4), 15 labels 1 (Ishak 5–6). Label 0 means lower fibrosis,
not healthy liver. The release also contains later dates and scans without
annotations; all release files are not the 49-patient evaluation cohort.

`reference_cohort.csv` is the author's unchanged
[dataframe_lihc.csv](https://github.com/Guerbet-AI/wsp-contrastive/blob/39adfaf08aabd31e85640b852c0c0a32285cb885/dataframe_lihc.csv),
pinned to commit `39adfaf08aabd31e85640b852c0c0a32285cb885`. Filename fields supply
patient/date/portal phase, not a DICOM series UID. Ishak annotations belong to
the evaluation reference and are not used to choose predicted scans.

## IMPERANDI protocol

`manifest.yaml` retains CT volumes and provides an ordered phase fallback chain:
native metadata rules first, then TotalSegmentator's CT contrast-phase predictor
for still-unresolved eligible volumes. The runner uses metadata only by default
(`TOTALSEG_PHASE=0`), building the cohort from `dicom_index_curated.csv` without
conversion or image-based prediction.

Set `TOTALSEG_PHASE=1` to enable the full chain. Because the predictor consumes
NIfTI, this mode converts the full cleaned inventory before the phase step.
The phase command writes `nifti_index_phased.csv` and its best-per-exam export
`nifti_index_phased_curated.csv`. In either mode, `cohort.py`
retains portal venous candidates and chooses the earliest date per patient,
breaking same-date ties by selection score and stable study/volume IDs.

The benchmark runs on all available CT patients without using the 49 reference
IDs. Extra patients may reflect missing Ishak eligibility rather than wrong
phase selection; inspect `cohort_mismatches.csv` and distinguish those causes.
Missing histology is not resolved by imaging metadata or the phase predictor.

## Preprocessing boundary

The paper uses 512×512 axial CT, HU clipping to [-100, 400] and the central 70%
of liver-containing slices using a privately trained U-Net. The released loader
crops the supplied volume's last axis to its central 70%; it does not load a
mask. Its uint8 conversion/model scaling does not document the upstream HU-to-byte
encoding. IMPERANDI conversion produces native NIfTI as an input to the CT phase
predictor, without claiming equivalent WSP model inputs or adding an undocumented
intensity mapping. Registration and isotropic resampling are not specified for
this cohort.

WSP pretraining uses private CT data; classification retraining is outside this
cohort benchmark.

The copied reference CSV retains the author's
[CC BY-NC-SA 4.0 license](https://github.com/Guerbet-AI/wsp-contrastive/blob/39adfaf08aabd31e85640b852c0c0a32285cb885/LICENSE.txt)
and the release's TCIA data terms; example helper code uses IMPERANDI's license.
