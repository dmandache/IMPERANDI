"""Sphinx configuration for the IMPERANDI documentation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

project = "IMPERANDI"
author = "Diana Mandache"
copyright = "2026, Diana Mandache"

try:
    from imperandi import __version__
except ImportError:
    __version__ = "0.0.0"

version = release = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autoclass_content = "both"

# Keep API docs buildable with the base installation. These packages are only
# needed by segmentation, radiomics, or the notebook viewers.
autodoc_mock_imports = [
    "IPython",
    "SimpleITK",
    "TotalSegmentator",
    "ipywidgets",
    "matplotlib",
    "param",
    "panel",
    "pyradiomics",
    "radiomics",
    "skimage",
    "torch",
    "xgboost",
]

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_title = "IMPERANDI documentation"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}
