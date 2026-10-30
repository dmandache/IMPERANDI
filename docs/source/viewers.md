# Quality-control viewers

Use the notebook or browser viewer to inspect CT/MR images and segmentation
masks from a cohort CSV. Install the viewer dependencies in the environment
running the viewer:

```bash
python -m pip install "imperandi[viewer]"
```

From a source checkout, use `python -m pip install -e ".[viewer]"` instead.
The table needs a `nifti_path` column with readable images. Optional
`mask_<output>_path` columns provide overlays; `patient_key`, `date`, `Modality`,
and `phase` support navigation and intensity windowing. Use a processing or
selected-cohort CSV, not the wide selection QC report. Paths in the table must
be accessible from the machine and working directory running the viewer.

## Jupyter notebook

Open a notebook in the environment containing the `viewer` extra, then run:

```python
%matplotlib widget

import pandas as pd
from imperandi.qc.viewer import ScanViewer

df = pd.read_csv("./project/tables/nifti_index_phased.csv")
viewer = ScanViewer(
    df,
    ct_scan_col="nifti_path",
    segmentation_cols=["mask_liver_path", "mask_liver_tumor_path"],
    phase_col="phase",
)
```

Constructing `ScanViewer` displays the controls. Despite the parameter name
`ct_scan_col`, it supports both CT and MR. Omit `segmentation_cols` to discover
mask path columns automatically. CT uses HU windows; MR uses percentile windows
(1st–99th by default). Select a patient, exam, phase, and anatomical plane to
review images and overlays.

For registered results, load `nifti_index_registered.csv` instead. The canonical
image and mask paths select the configured output space. See the
[viewer API](api/qc.rst) for further parameters.

## Browser viewer

From the repository root after installing the editable `viewer` extra:

```bash
panel serve src/imperandi/qc/viewer_web.py --show --args \
  --csv ./project/tables/nifti_index_phased.csv \
  --scan-col nifti_path \
  --phase-col phase \
  --seg-cols mask_liver_path mask_liver_tumor_path
```

`--show` opens the browser. Without it, use the local URL printed by Panel.
The CSV and image paths are read by the server process. Omit `--csv` to select
or upload a table through the interface; uploading a CSV does not upload its
referenced images. Omit `--seg-cols` to discover mask columns automatically.
The interface supports table filtering, image and mask selection, and cohort
navigation.

For a PyPI installation without a source checkout, locate the packaged script:

```bash
python -c "import imperandi; from pathlib import Path; print(Path(imperandi.__file__).parent / 'qc' / 'viewer_web.py')"
```

Use the printed path in place of `src/imperandi/qc/viewer_web.py` in the command
above. Stop the viewer server with Ctrl+C in its terminal.
