"""Public package interface for imperandi."""

from ._version import __version__


def main(*args, **kwargs):
    """Lazy proxy to the CLI entrypoint."""
    from imperandi.cli import main as _main

    return _main(*args, **kwargs)


__all__ = ["__version__", "main"]
