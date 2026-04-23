#!/usr/bin/env python3
"""Strip execution outputs from Jupyter notebooks."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def strip_notebook(path: Path) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8-sig"))
    changed = False

    cells = notebook.get("cells", [])
    if isinstance(cells, list):
        for cell in cells:
            if not isinstance(cell, dict) or cell.get("cell_type") != "code":
                continue

            outputs = cell.get("outputs")
            if outputs:
                cell["outputs"] = []
                changed = True

            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                changed = True

    metadata = notebook.get("metadata")
    if isinstance(metadata, dict) and "widgets" in metadata:
        metadata.pop("widgets", None)
        changed = True

    if changed:
        rendered = json.dumps(notebook, ensure_ascii=False, indent=1)
        path.write_text(rendered + "\n", encoding="utf-8")

    return changed


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "Usage: strip_notebook_outputs.py <notebook.ipynb> [...]", file=sys.stderr
        )
        return 1

    any_changed = False
    for raw_path in argv[1:]:
        path = Path(raw_path)
        try:
            if strip_notebook(path):
                any_changed = True
        except Exception as exc:  # pragma: no cover - defensive for hook execution
            print(f"{path}: {exc}", file=sys.stderr)
            return 1

    return 2 if any_changed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
