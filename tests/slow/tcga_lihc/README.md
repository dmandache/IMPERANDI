# TCGA-LIHC slow dataset

`download.py` uses `idc-index` to download CT/MR series for patients
`TCGA-BC-A10X` and `TCGA-DD-A113` by default into `data/input`.

```bash
python -m pip install -e '.[slow]'
python tests/slow/tcga_lihc/download.py
bash tests/slow/tcga_lihc/pipeline.sh
python -m pytest tests/slow/tcga_lihc -m slow
```

The Bash script accepts optional `input directory`, `work directory`, and
`manifest` positional arguments:

```bash
bash tests/slow/tcga_lihc/pipeline.sh /data/tcga-lihc ./tcga-results generic
```

PowerShell exposes the equivalent named parameters:

```powershell
.\tests\slow\tcga_lihc\pipeline.ps1 -InputDir C:\data\tcga-lihc -WorkDir .\tcga-results -Manifest generic
```

Choose a different subset by repeating `--patient`.

Dataset source: <https://portal.imaging.datacommons.cancer.gov/collections/tcga_lihc/>

Source DICOM files are downloaded locally and are not committed to this
repository. IDC exposes the public files through its cloud download tooling.
