"""Executable module entry point for launching the Imperandi CLI.

The definitions in this module are part of the Imperandi codebase and are
intended to be reused by higher-level workflows and CLI entry points.
"""

from imperandi.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
