from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from imperandi.process.registration.consensus import TumorComponent


@dataclass(frozen=True)
class LongitudinalAuditConfig:
    max_centroid_shift_mm: float = 25.0
    max_total_volume_change_ratio: float = 0.6


@dataclass(frozen=True)
class AuditFinding:
    patient_key: str
    visit_prev: str
    visit_curr: str
    flag: str
    severity: str
    value: float | None
    detail: str


def _component_centroids(components: Iterable[TumorComponent]) -> np.ndarray:
    arr = np.asarray(
        [
            [float(c.centroid_x_mm), float(c.centroid_y_mm), float(c.centroid_z_mm)]
            for c in components
        ],
        dtype=float,
    )
    if arr.size == 0:
        return np.zeros((0, 3), dtype=float)
    return arr


def _match_centroid_distances(prev: np.ndarray, curr: np.ndarray) -> list[float]:
    if prev.size == 0 or curr.size == 0:
        return []
    remaining = set(range(curr.shape[0]))
    distances: list[float] = []
    for i in range(prev.shape[0]):
        if not remaining:
            break
        p = prev[i]
        best_j = None
        best_dist = float("inf")
        for j in remaining:
            dist = float(np.linalg.norm(p - curr[j]))
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is not None:
            distances.append(best_dist)
            remaining.remove(best_j)
    return distances


def build_longitudinal_audit(
    *,
    patient_key: str,
    sorted_visits: list[str],
    components_by_visit: dict[str, list[TumorComponent]],
    config: LongitudinalAuditConfig,
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    if len(sorted_visits) <= 1:
        return findings

    for prev_visit, curr_visit in zip(sorted_visits[:-1], sorted_visits[1:]):
        prev_components = components_by_visit.get(prev_visit, [])
        curr_components = components_by_visit.get(curr_visit, [])
        prev_count = len(prev_components)
        curr_count = len(curr_components)
        if prev_count != curr_count:
            findings.append(
                AuditFinding(
                    patient_key=patient_key,
                    visit_prev=str(prev_visit),
                    visit_curr=str(curr_visit),
                    flag="tumor_count_mismatch",
                    severity="warning",
                    value=float(curr_count - prev_count),
                    detail=f"count {prev_count} -> {curr_count}",
                )
            )

        if prev_count > 0 and curr_count == 0:
            findings.append(
                AuditFinding(
                    patient_key=patient_key,
                    visit_prev=str(prev_visit),
                    visit_curr=str(curr_visit),
                    flag="missing_lesion",
                    severity="warning",
                    value=float(prev_count),
                    detail="all previously visible lesions are missing",
                )
            )

        prev_centroids = _component_centroids(prev_components)
        curr_centroids = _component_centroids(curr_components)
        centroid_distances = _match_centroid_distances(prev_centroids, curr_centroids)
        if centroid_distances:
            max_shift = float(max(centroid_distances))
            if max_shift > float(config.max_centroid_shift_mm):
                findings.append(
                    AuditFinding(
                        patient_key=patient_key,
                        visit_prev=str(prev_visit),
                        visit_curr=str(curr_visit),
                        flag="suspicious_position_shift",
                        severity="warning",
                        value=max_shift,
                        detail=(
                            "maximum nearest-centroid shift exceeded threshold "
                            f"({config.max_centroid_shift_mm:.1f} mm)"
                        ),
                    )
                )

        prev_total_volume = float(sum(c.volume_ml for c in prev_components))
        curr_total_volume = float(sum(c.volume_ml for c in curr_components))
        denom = max(1e-6, prev_total_volume)
        change_ratio = abs(curr_total_volume - prev_total_volume) / denom
        if change_ratio > float(config.max_total_volume_change_ratio):
            findings.append(
                AuditFinding(
                    patient_key=patient_key,
                    visit_prev=str(prev_visit),
                    visit_curr=str(curr_visit),
                    flag="unstable_segmentation_pattern",
                    severity="warning",
                    value=float(change_ratio),
                    detail=(
                        "total tumor volume changed by "
                        f"{100.0 * change_ratio:.1f}%"
                    ),
                )
            )
        if prev_count > 0 and curr_count > 0 and prev_count != curr_count:
            findings.append(
                AuditFinding(
                    patient_key=patient_key,
                    visit_prev=str(prev_visit),
                    visit_curr=str(curr_visit),
                    flag="possible_merge_or_split",
                    severity="info",
                    value=float(curr_count - prev_count),
                    detail="connected-component count changed across visits",
                )
            )
    return findings
