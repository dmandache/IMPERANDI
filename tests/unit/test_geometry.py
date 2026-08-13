import sys
from pathlib import Path

# Ensure src/ is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest

from imperandi.utils import geometry


def test_parse_iop_accepts_array_like():
    data = [1, 2, 3, 4, 5, 6]
    out = geometry.parse_iop(data)
    assert isinstance(out, np.ndarray)
    assert out.dtype == float
    assert out.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    out2 = geometry.parse_iop(np.array(data))
    assert out2.tolist() == out.tolist()


def test_parse_iop_accepts_string_literal_or_fallback_regex():
    out = geometry.parse_iop("[1, 2, 3, 4, 5, 6]")
    assert out.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    out2 = geometry.parse_iop("IOP: 1 2 3 4 5 6")
    assert out2.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    out3 = geometry.parse_iop("(1e0, 2e0, 3e0, 4e0, 5e0, 6e0)")
    assert out3.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    out4 = geometry.parse_iop("vals: -1.5 0.0 2.5 3 4 5")
    assert out4.tolist() == [-1.5, 0.0, 2.5, 3.0, 4.0, 5.0]


def test_parse_iop_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        geometry.parse_iop(None)

    with pytest.raises(ValueError):
        geometry.parse_iop("not numbers")

    with pytest.raises(ValueError):
        geometry.parse_iop("1 2 3 4 5")


def test_standardize_iop_success_and_rounding():
    iop = [1, 0, 0, 0, 1, 0]
    out = geometry.standardize_iop(iop)
    assert isinstance(out, tuple)
    assert out == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    # values below zero_tol should be snapped to 0 and rounded
    iop2 = [1, 1e-8, 0, 0, 1, 0]
    out2 = geometry.standardize_iop(iop2, decimals=3, zero_tol=1e-6)
    assert out2 == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    # non-unit vectors should be normalized
    iop3 = [2, 0, 0, 0, 0, 3]
    out3 = geometry.standardize_iop(iop3, decimals=3)
    assert out3 == (1.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def test_standardize_iop_handles_invalid_inputs():
    assert geometry.standardize_iop(None) is None
    assert geometry.standardize_iop("bad") is None
    assert geometry.standardize_iop([1, 2, 3]) is None
    assert geometry.standardize_iop([0, 0, 0, 1, 0, 0]) is None


def test_as_float_array():
    assert geometry.as_float_array(None) is None

    out = geometry.as_float_array([1, 2, 3])
    assert isinstance(out, np.ndarray)
    assert out.dtype == float

    out2 = geometry.as_float_array("[4, 5]")
    assert out2.tolist() == [4.0, 5.0]

    out3 = geometry.as_float_array((6, 7, 8))
    assert out3.tolist() == [6.0, 7.0, 8.0]

    with pytest.raises((ValueError, SyntaxError)):
        geometry.as_float_array("not a list")


def test_slice_normal_from_iop():
    iop = [1, 0, 0, 0, 1, 0]
    out = geometry.slice_normal_from_iop(iop)
    assert np.allclose(out, np.array([0.0, 0.0, 1.0]))

    assert geometry.slice_normal_from_iop(None) is None
    assert geometry.slice_normal_from_iop([1, 2, 3]) is None

    # colinear row/col -> zero normal
    iop_colinear = [1, 0, 0, 2, 0, 0]
    assert geometry.slice_normal_from_iop(iop_colinear) is None

    # accepts string literal
    out2 = geometry.slice_normal_from_iop("[1, 0, 0, 0, 1, 0]")
    assert np.allclose(out2, np.array([0.0, 0.0, 1.0]))


def test_classify_plane_from_iop_standard_planes():
    plane, angle, axis = geometry.classify_plane_from_iop([1, 0, 0, 0, 1, 0])
    assert plane == "AX"
    assert axis == "Z"
    assert angle == 0.0

    plane, angle, axis = geometry.classify_plane_from_iop([1, 0, 0, 0, 0, 1])
    assert plane == "COR"
    assert axis == "Y"
    assert angle == 0.0

    plane, angle, axis = geometry.classify_plane_from_iop([0, 1, 0, 0, 0, 1])
    assert plane == "SAG"
    assert axis == "X"
    assert angle == 0.0


def test_classify_plane_from_iop_oblique_and_invalid():
    plane, angle, axis = geometry.classify_plane_from_iop(
        [1, 0, 0, 0, 1, 1], angle_thresh_deg=10.0
    )
    assert plane == "OBL"
    assert axis == "Y"
    assert angle > 10.0

    # negative normal should still be axial (abs used)
    plane2, angle2, axis2 = geometry.classify_plane_from_iop([1, 0, 0, 0, -1, 0])
    assert plane2 == "AX"
    assert axis2 == "Z"
    assert angle2 == 0.0

    # angle equal to threshold => not oblique
    plane3, angle3, axis3 = geometry.classify_plane_from_iop(
        [1, 0, 0, 0, 0.984807753, 0.173648178], angle_thresh_deg=10.0
    )
    assert plane3 == "AX"
    assert axis3 == "Z"
    assert angle3 == pytest.approx(10.0, abs=1e-6)

    # angle slightly smaller than threshold
    plane4, angle4, axis4 = geometry.classify_plane_from_iop(
        [1, 0, 0, 0, 0.139173, 0.990268], angle_thresh_deg=10.0
    )
    assert plane4 == "COR"
    assert axis4 == "Y"
    assert angle4 == pytest.approx(8.0, abs=1e-3)

    plane, angle, axis = geometry.classify_plane_from_iop(None)
    assert plane is None
    assert axis is None
    assert np.isnan(angle)
