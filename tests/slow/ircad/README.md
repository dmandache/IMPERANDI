# IRCAD slow dataset

`download.py` downloads patients `3Dircadb1.1` and `3Dircadb1.2` by default
from the public 3D-IRCADb-01 collection and extracts their patient CT data
(excluding the labelled/mask/mesh branches) into `data/input`.

```bash
python tests/slow/ircad/download.py
bash tests/slow/ircad/pipeline.sh
python -m pytest tests/slow/ircad -m slow
```

The Bash script accepts optional `input directory`, `work directory`,
`manifest`, and `number of workers` positional arguments. Worker count
defaults to `1`:

```bash
bash tests/slow/ircad/pipeline.sh /data/ircad ./ircad-results generic 4
```

PowerShell exposes the equivalent named parameters:

```powershell
.\tests\slow\ircad\pipeline.ps1 -InputDir C:\data\ircad -WorkDir .\ircad-results -Manifest generic -NumWorkers 4
```

Choose a different subset by repeating `--patient`, for example:

```bash
python tests/slow/ircad/download.py --patient 3Dircadb1.1
```

Dataset source: <https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/>

The dataset is published under CC BY-NC-ND 4.0. Source files are downloaded
locally and are not committed to this repository.
