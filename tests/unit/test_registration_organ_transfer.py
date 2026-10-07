"""Distance-field organ transfer, consensus, seam, and final-QC contracts."""

import numpy as np
import pytest

from imperandi.process.registration import alignment
from imperandi.process.registration.cohort import (
    _fill_unobserved,
    final_organ_qc,
    fuse_organs,
    fuse_tumors,
)

sitk = pytest.importorskip("SimpleITK")


@pytest.fixture(autouse=True)
def single_thread():
    previous = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    yield
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(previous)


def anisotropic_organ(*, shift=(0, 0, 0), island=False):
    shape = (32, 52, 48)
    z, y, x = np.indices(shape)
    center = np.array([16, 26, 24]) + np.asarray(shift)
    values = (
        ((z - center[0]) / 10) ** 2
        + ((y - center[1]) / 15) ** 2
        + ((x - center[2]) / 14) ** 2
        <= 1
    )
    if island:
        values[2:4, 3:5, 3:5] = True
    image = sitk.GetImageFromArray(values.astype(np.uint8))
    image.SetSpacing((1.1, 1.6, 4.0))
    image.SetOrigin((13.0, -9.0, 5.0))
    return image


def full_distance(mask):
    cropped = alignment.organ_distance_field(mask, padding_mm=8.0)
    return alignment.expand_organ_distance(cropped, mask)


def component_count(mask):
    finder = sitk.ConnectedComponentImageFilter()
    finder.SetFullyConnected(False)
    connected = finder.Execute(mask)
    return int(sitk.GetArrayViewFromImage(connected).max())


def test_anisotropic_organ_transfer_thresholds_only_on_destination_grid():
    source = anisotropic_organ()
    field = alignment.organ_distance_field(source, padding_mm=8.0)
    transform = sitk.TranslationTransform(3, (0.45, -0.7, 1.2))

    transferred_field = alignment.resample_organ_distance(field, source, transform)
    transferred = alignment.threshold_organ_distance(transferred_field)

    source_volume = int(sitk.GetArrayViewFromImage(source).sum())
    output_volume = int(sitk.GetArrayViewFromImage(transferred).sum())
    assert transferred.GetSpacing() == source.GetSpacing()
    assert transferred.GetPixelID() == sitk.sitkUInt8
    assert component_count(transferred) == 1
    assert abs(output_volume - source_volume) / source_volume < 0.03


def test_distance_majority_rejects_one_fragmented_outlier():
    first = anisotropic_organ(shift=(0, 0, -1))
    second = anisotropic_organ(shift=(0, 0, 1))
    outlier = anisotropic_organ(shift=(0, 0, 7), island=True)
    fields = [full_distance(mask) for mask in (first, second, outlier)]
    coverages = [alignment.coverage_image(mask) for mask in (first, second, outlier)]

    fused = fuse_organs(fields, coverages, organ_consensus="majority")
    quality = final_organ_qc(fused.mask, first)

    assert quality.component_count == 1
    assert quality.fragment_count == 0
    assert not sitk.GetArrayViewFromImage(fused.mask)[2:4, 3:5, 3:5].any()
    assert abs(quality.volume_change_fraction) < 0.12


def test_coverage_boundary_blends_fields_and_keeps_unobserved_source():
    source = anisotropic_organ()
    consensus = anisotropic_organ(shift=(0, 0, 3))
    source_field = full_distance(source)
    consensus_field = full_distance(consensus)
    coverage_values = np.zeros(tuple(reversed(source.GetSize())), dtype=np.uint8)
    coverage_values[..., :26] = 1
    coverage = sitk.GetImageFromArray(coverage_values)
    coverage.CopyInformation(source)

    transfer = _fill_unobserved(
        consensus_field,
        coverage,
        source_field,
        blend_width_mm=4.0,
    )
    output = sitk.GetArrayViewFromImage(transfer.mask)
    original = sitk.GetArrayViewFromImage(source)

    assert np.array_equal(output[..., 26:], original[..., 26:])
    assert component_count(transfer.mask) == 1
    assert transfer.residual_seam_voxels == 0


def test_final_qc_reports_fragments_without_removing_them():
    source = anisotropic_organ()
    fragmented = anisotropic_organ(island=True)

    quality = final_organ_qc(fragmented, source, residual_seam_voxels=3)

    assert quality.component_count == 2
    assert quality.fragment_count == 1
    assert quality.fragment_sizes_mm3[0] == pytest.approx(8 * np.prod(source.GetSpacing()))
    assert quality.has_fragments is True
    assert quality.has_residual_seam is True
    assert sitk.GetArrayViewFromImage(fragmented)[2:4, 3:5, 3:5].all()


def test_tumor_union_still_preserves_separate_lesions():
    reference = anisotropic_organ()
    first = sitk.Image(reference.GetSize(), sitk.sitkUInt8)
    first.CopyInformation(reference)
    first[10:12, 12:14, 8:10] = 1
    second = sitk.Image(reference.GetSize(), sitk.sitkUInt8)
    second.CopyInformation(reference)
    second[30:32, 35:37, 20:22] = 1
    coverage = alignment.coverage_image(reference)

    fused = fuse_tumors(
        [first, second], [coverage, coverage], tumor_consensus="union"
    )

    assert int(sitk.GetArrayViewFromImage(fused.mask).sum()) == 16
    assert component_count(fused.mask) == 2
