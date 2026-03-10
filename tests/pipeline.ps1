$ErrorActionPreference = "Stop"

$DIR_PATH = ".\tests\data"

imperandi ingest "$DIR_PATH\IRCAD_DICOM" --snapshot_tags --manifest generic
imperandi convert "$DIR_PATH\dicom_index_clean.csv" "$DIR_PATH\IRCAD_nifti"
imperandi segment "$DIR_PATH\nifti_index.csv" --manifest generic
imperandi phase "$DIR_PATH\nifti_index.csv"
imperandi radiomics "$DIR_PATH\nifti_index.csv" --manifest generic

echo "Pipeline test completed successfully."