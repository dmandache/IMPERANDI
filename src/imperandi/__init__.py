"""Public package interface for imperandi."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("imperandi")
except PackageNotFoundError:
    __version__ = "0.0.0"


def main(*args, **kwargs):
    """Lazy proxy to the CLI entrypoint."""
    from imperandi.cli import main as _main

    return _main(*args, **kwargs)


__all__ = ["__version__", "main"]
