"""Download selected 3D-IRCADb-01 patients into data/input."""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://cloud.ircad.fr/index.php/s/JN3z7EynBiwYyjy/download"
DEFAULT_PATIENTS = ("3Dircadb1.1", "3Dircadb1.2")
PATIENT_PATTERN = re.compile(r"3Dircadb1\.(?:[1-9]|1[0-9]|20)\Z")


def _safe_extract(archive: Path, destination: Path, patient: str) -> None:
    with zipfile.ZipFile(archive) as source:
        files = [info for info in source.infolist() if not info.is_dir()]
        if not files:
            raise RuntimeError(f"Downloaded archive for {patient} is empty")

        scan_files = [
            info
            for info in files
            if any(
                part.upper().startswith("PATIENT_DICOM")
                for part in Path(info.filename).parts
            )
            or "exam" in {part.lower() for part in Path(info.filename).parts}
        ]
        if scan_files:
            files = scan_files

        roots = {
            Path(info.filename).parts[0] for info in files if Path(info.filename).parts
        }
        extract_root = destination if patient in roots else destination / patient
        extract_root.mkdir(parents=True, exist_ok=True)
        resolved_root = extract_root.resolve()

        for info in files:
            relative = Path(info.filename)
            target = (extract_root / relative).resolve()
            if not target.is_relative_to(resolved_root):
                raise RuntimeError(f"Unsafe archive member: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as reader, target.open("wb") as writer:
                shutil.copyfileobj(reader, writer)


def download_patient(patient: str, destination: Path) -> None:
    if not PATIENT_PATTERN.fullmatch(patient):
        raise ValueError(f"Invalid IRCAD patient name: {patient}")

    patient_dir = destination / patient
    if patient_dir.is_dir() and any(path.is_file() for path in patient_dir.rglob("*")):
        print(f"Skipping {patient}: {patient_dir} already contains data")
        return

    query = urllib.parse.urlencode({"path": f"/{patient}"})
    url = f"{BASE_URL}?{query}"
    print(f"Downloading {patient} from {url}")
    with tempfile.TemporaryDirectory(prefix="imperandi-ircad-") as temp_dir:
        archive = Path(temp_dir) / f"{patient}.zip"
        urllib.request.urlretrieve(url, archive)
        _safe_extract(archive, destination, patient)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patient",
        action="append",
        dest="patients",
        help="Patient folder to download; repeat the option for multiple patients.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "input",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for patient in args.patients or DEFAULT_PATIENTS:
        download_patient(patient, args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
