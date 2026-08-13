"""Download selected TCGA-LIHC patients from NCI Imaging Data Commons."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_PATIENTS = ("TCGA-BC-A10X", "TCGA-DD-A113")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patient",
        action="append",
        dest="patients",
        help="TCGA patient ID; repeat the option for multiple patients.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "input",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        from idc_index import IDCClient
    except ImportError as exc:
        raise SystemExit(
            "TCGA-LIHC download requires idc-index. Install with "
            "`python -m pip install -e '.[slow]'`."
        ) from exc

    patients = args.patients or list(DEFAULT_PATIENTS)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    client = IDCClient()
    index = client.index
    selection = index[
        index["collection_id"].astype(str).str.lower().eq("tcga_lihc")
        & index["PatientID"].isin(patients)
        & index["Modality"].isin(["CT", "MR"])
    ]
    found = set(selection["PatientID"].unique())
    missing = sorted(set(patients) - found)
    if missing:
        raise SystemExit(f"No CT/MR series found for patient(s): {', '.join(missing)}")

    print(
        f"Downloading {len(selection)} series for {len(found)} patient(s) "
        f"into {output_dir}"
    )
    client.download_from_selection(
        downloadDir=str(output_dir),
        seriesInstanceUID=selection["SeriesInstanceUID"].tolist(),
        dirTemplate="%PatientID/%StudyInstanceUID/%Modality/%SeriesInstanceUID",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
