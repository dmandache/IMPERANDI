"""Download TCGA-LIHC radiology DICOMs from IDC for both cohort benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--patient", action="append", help="PatientID; repeat for a subset"
    )
    args = parser.parse_args()
    try:
        from idc_index import IDCClient
    except ImportError as exc:
        raise SystemExit(
            "Install the downloader with: python -m pip install idc-index"
        ) from exc

    client = IDCClient()
    selection = client.index.loc[
        client.index["collection_id"].astype(str).str.lower().eq("tcga_lihc")
        & client.index["Modality"].isin(["CT", "MR"])
    ]
    if args.patient:
        selection = selection.loc[selection["PatientID"].isin(args.patient)]
        missing = set(args.patient) - set(selection["PatientID"])
        if missing:
            parser.error(
                f"No CT/MR series for patient(s): {', '.join(sorted(missing))}"
            )
    if selection.empty:
        parser.error("No TCGA-LIHC CT/MR series in the installed IDC index")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading {len(selection)} CT/MR series for "
        f"{selection['PatientID'].nunique()} patient(s) into {output}"
    )
    client.download_from_selection(
        downloadDir=str(output),
        seriesInstanceUID=selection["SeriesInstanceUID"].drop_duplicates().tolist(),
        dirTemplate="%PatientID/%StudyInstanceUID/%Modality/%SeriesInstanceUID",
    )
    manifest = output / "idc_selection.csv"
    selection[
        ["PatientID", "Modality", "StudyInstanceUID", "SeriesInstanceUID", "series_size_MB"]
    ].to_csv(manifest, index=False)
    print(f"Saved download selection to {manifest}")


if __name__ == "__main__":
    main()
