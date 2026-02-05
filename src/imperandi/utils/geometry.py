import re
from ast import literal_eval

import numpy as np


def parse_iop(iop):
    if iop is None:
        raise ValueError("IOP is None")

    if isinstance(iop, (list, tuple, np.ndarray)):
        return np.asarray(iop, dtype=float)

    if isinstance(iop, str):
        s = iop.strip()
        try:
            val = literal_eval(s)
            return np.asarray(val, dtype=float)
        except (ValueError, SyntaxError):
            pass

        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if len(nums) == 6:
            return np.asarray(nums, dtype=float)

    raise ValueError(f"Cannot parse IOP: {iop}")


def standardize_iop(iop, decimals=3, zero_tol=1e-6):
    try:
        iop = parse_iop(iop)
    except (ValueError, TypeError):
        return None

    if iop.shape != (6,):
        raise ValueError(f"IOP must have length 6, got shape {iop.shape}")

    row = iop[:3]
    col = iop[3:]

    row_norm = np.linalg.norm(row)
    col_norm = np.linalg.norm(col)

    if row_norm == 0 or col_norm == 0:
        raise ValueError(f"Invalid IOP (zero norm): {iop}")

    row = row / row_norm
    col = col / col_norm

    iop_std = np.concatenate([row, col])
    iop_std[np.abs(iop_std) < zero_tol] = 0.0
    iop_std = np.round(iop_std, decimals=decimals)

    return tuple(iop_std)


def as_float_array(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x, dtype=float)
    return np.asarray(literal_eval(x), dtype=float)

def as_tuple(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return tuple(x) 
    return tuple(literal_eval(x))


def slice_normal_from_iop(iop):
    iop = as_float_array(iop)
    if iop is None or iop.size != 6:
        return None
    row = iop[:3]
    col = iop[3:]
    n = np.cross(row, col)
    norm = np.linalg.norm(n)
    if norm == 0:
        return None
    return n / norm


def classify_plane_from_iop(iop, angle_thresh_deg=10.0):
    n = slice_normal_from_iop(iop)
    if n is None:
        return None, np.nan, None

    dots = np.abs(n)
    k = int(np.argmax(dots))
    max_dot = float(dots[k])
    angle = float(np.degrees(np.arccos(np.clip(max_dot, -1.0, 1.0))))

    axis = ["X", "Y", "Z"][k]
    if angle > angle_thresh_deg and not np.isclose(angle, angle_thresh_deg, atol=1e-6):
        return "OBL", angle, axis

    plane = {"X": "SAG", "Y": "COR", "Z": "AX"}[axis]
    return plane, angle, axis
