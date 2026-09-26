"""Benchmark SimpleITK and FireANTs on one reproducibly sampled group.

The two runs consume the same source CSV and manifest.  Only
``mask_registration_backend`` and FireANTs' strict no-fallback guard differ.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml


def _stage_details(row):
    value = row.get("registration_stage_details")
    return {} if pd.isna(value) else json.loads(value)


def _mask_dice(first, second):
    import SimpleITK as sitk

    a = sitk.ReadImage(str(first))
    b = sitk.ReadImage(str(second))
    geometry = (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-5, rtol=0)
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-5, rtol=0)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-5, rtol=0)
    )
    if not geometry:
        raise ValueError("Backend outputs do not share the native image grid")
    av = sitk.GetArrayViewFromImage(a) > 0
    bv = sitk.GetArrayViewFromImage(b) > 0
    denominator = int(av.sum() + bv.sum())
    return (
        1.0 if denominator == 0 else float(2 * np.count_nonzero(av & bv) / denominator)
    )


def _run(source, manifest, root, backend):
    from imperandi.process.registration import register

    directory = root / backend
    directory.mkdir(parents=True, exist_ok=True)
    settings = deepcopy(manifest)
    settings.setdefault("registration", {})["mask_registration_backend"] = backend
    settings["registration"]["fireants_fallback_to_simpleitk"] = False
    manifest_path = directory / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(settings, sort_keys=False))
    output = directory / "registered.csv"
    args = register.build_parser().parse_args(
        [
            "--csv_path",
            str(source),
            "--csv_path_out",
            str(output),
            "--qc_csv_path",
            str(directory / "qc.csv"),
            "--error_csv_path",
            str(directory / "errors.csv"),
            "--output_dir",
            str(directory / "artifacts"),
            "--manifest",
            str(manifest_path),
            "--mask_registration_backend",
            backend,
            "--num_workers",
            "1",
            "--force",
        ]
    )
    started = time.perf_counter()
    register.main(args)
    elapsed = time.perf_counter() - started
    return pd.read_csv(output), elapsed


def _summarize(table, elapsed):
    stage_elapsed = {}
    optimization_elapsed = {}
    peak_gpu = 0
    fireants_stage_count = 0
    fallbacks = []
    rows = []
    for _, row in table.iterrows():
        details = _stage_details(row)
        for stage, detail in details.items():
            if detail.get("backend") == "fireants":
                fireants_stage_count += 1
            stage_elapsed[stage] = stage_elapsed.get(stage, 0.0) + float(
                detail.get("elapsed_seconds") or 0.0
            )
            optimization_elapsed[stage] = optimization_elapsed.get(stage, 0.0) + float(
                detail.get("optimization_elapsed_seconds") or 0.0
            )
            peak_gpu = max(
                peak_gpu,
                int(detail.get("gpu_peak_memory_allocated_bytes") or 0),
            )
            if detail.get("fallback_reason"):
                fallbacks.append(
                    {
                        "scan": row.get("registration_scan_id"),
                        "stage": stage,
                        "reason": detail["fallback_reason"],
                    }
                )
        rows.append(
            {
                "scan": row.get("registration_scan_id"),
                "selected_stage": row.get("registration_selected_stage"),
                "selected_dice": row.get("registration_dice_selected"),
                "consensus_contributors": row.get(
                    "registration_consensus_contributors"
                ),
            }
        )
    return {
        "total_seconds": elapsed,
        "stage_seconds": stage_elapsed,
        "optimization_seconds": optimization_elapsed,
        "peak_gpu_memory_bytes": peak_gpu or None,
        "fireants_stage_count": fireants_stage_count,
        "fallbacks": fallbacks,
        "scans": rows,
    }


def _agreement(simpleitk, fireants, column, fallback_column):
    first = simpleitk.set_index("registration_scan_id")
    second = fireants.set_index("registration_scan_id")
    result = {}
    for scan in first.index.intersection(second.index):
        left, right = first.at[scan, column], second.at[scan, column]
        if pd.isna(left):
            left = first.at[scan, fallback_column]
        if pd.isna(right):
            right = second.at[scan, fallback_column]
        if (
            pd.isna(left)
            or pd.isna(right)
            or not Path(left).is_file()
            or not Path(right).is_file()
        ):
            result[str(scan)] = None
        else:
            result[str(scan)] = _mask_dice(left, right)
    return result


def _select_group(table, config, seed):
    """Select one multi-volume manifest-defined group reproducibly."""
    from imperandi.process.registration.cohort import prepare_cohort

    planned = prepare_cohort(table, config)
    sizes = planned.groupby("registration_group_id", sort=True).size()
    eligible = sorted(str(group_id) for group_id in sizes[sizes >= 2].index)
    if not eligible:
        raise ValueError(
            "Benchmark input has no multi-volume registration groups under the "
            "manifest grouping settings"
        )
    rng = np.random.default_rng(seed)
    group_id = eligible[int(rng.integers(len(eligible)))]
    selected_indices = planned.index[
        planned.registration_group_id.astype(str) == group_id
    ].tolist()
    selected = table.iloc[selected_indices].copy().reset_index(drop=True)
    group_label = planned.loc[selected_indices[0], "registration_group_label"]
    return selected, {
        "seed": seed,
        "eligible_group_count": len(eligible),
        "selected_group_id": group_id,
        "selected_group_label": group_label,
        "selected_group_size": len(selected),
    }


def _markdown(report):
    selection = report["selection"]
    lines = [
        "# FireANTs registration benchmark",
        "",
        "Both backends used the same source CSV and manifest; only the two mask-linear stages changed backend.",
        f"Seed: `{selection['seed']}`. Selected group: `{selection['selected_group_label']}` "
        f"({selection['selected_group_size']} volumes from "
        f"{selection['eligible_group_count']} eligible groups).",
        "",
        "| Metric | SimpleITK | FireANTs |",
        "|---|---:|---:|",
    ]
    for key, label in (
        ("total_seconds", "Total time (s)"),
        ("peak_gpu_memory_bytes", "Peak GPU memory (bytes)"),
    ):
        lines.append(
            f"| {label} | {report['simpleitk'].get(key)} | {report['fireants'].get(key)} |"
        )
    lines.extend(["", "## Per-stage elapsed seconds", ""])
    stages = sorted(
        set(report["simpleitk"]["stage_seconds"])
        | set(report["fireants"]["stage_seconds"])
    )
    lines.extend(["| Stage | SimpleITK | FireANTs |", "|---|---:|---:|"])
    for stage in stages:
        lines.append(
            f"| {stage} | {report['simpleitk']['stage_seconds'].get(stage, 0):.6f} | "
            f"{report['fireants']['stage_seconds'].get(stage, 0):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Selection and consensus",
            "",
            "| Backend | Scan | Selected stage | Selected Dice | Consensus contributors |",
            "|---|---|---|---:|---|",
        ]
    )
    for backend in ("simpleitk", "fireants"):
        for scan in report[backend]["scans"]:
            lines.append(
                f"| {backend} | {scan['scan']} | {scan['selected_stage']} | "
                f"{scan['selected_dice']} | {scan['consensus_contributors']} |"
            )
    lines.extend(
        [
            "",
            "## Agreement",
            "",
            "```json",
            json.dumps(report["agreement"], indent=2),
            "```",
            "",
        ]
    )
    if report["demonstrated_speedup"]:
        lines.append(
            f"This run demonstrated a {report['speedup_ratio']:.3f}x total-time speedup."
        )
    else:
        lines.append("This run did not demonstrate a total-time speedup.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--manifest", default="generic")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    source = args.csv_path.expanduser().resolve()
    table = pd.read_csv(source)
    from imperandi.utils.manifest import load_manifest
    from imperandi.process.registration.config import RegistrationConfig

    manifest = load_manifest(args.manifest, base_path=Path.cwd())
    config = RegistrationConfig.from_mapping(manifest.get("registration", {}))
    selected, selection = _select_group(table, config, args.seed)
    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected_source = root / "benchmark_input.csv"
    selected.to_csv(selected_source, index=False)
    # Probe in a fresh process so CUDA/package initialization is still included
    # in the timed FireANTs run and cannot make the comparison look faster.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from imperandi.process.registration.fireants_backend import _imports; _imports()",
        ],
        check=True,
    )

    simpleitk, simpleitk_elapsed = _run(selected_source, manifest, root, "simpleitk")
    fireants, fireants_elapsed = _run(selected_source, manifest, root, "fireants")
    report = {
        "source_volume_count": len(table),
        "selection": selection,
        "volume_count": len(selected),
        "simpleitk": _summarize(simpleitk, simpleitk_elapsed),
        "fireants": _summarize(fireants, fireants_elapsed),
        "agreement": {
            "native_organ_dice": _agreement(
                simpleitk, fireants, "reg_organ_native_path", "mask_liver"
            ),
            "tumor_mask_dice": _agreement(
                simpleitk,
                fireants,
                "reg_tumor_native_path",
                "mask_liver_tumor",
            ),
        },
    }
    no_fallback = not report["fireants"]["fallbacks"]
    report["demonstrated_speedup"] = bool(
        no_fallback
        and report["fireants"]["fireants_stage_count"] > 0
        and fireants_elapsed < simpleitk_elapsed
    )
    report["speedup_ratio"] = (
        simpleitk_elapsed / fireants_elapsed if report["demonstrated_speedup"] else None
    )
    (root / "benchmark.json").write_text(json.dumps(report, indent=2, default=str))
    (root / "benchmark.md").write_text(_markdown(report))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
