# Project dataset configuration

This directory contains project-local OPERANDI configuration. It is available
when running from this repository but is not installed as part of the
`imperandi` Python package.

- `manifests/operandi.yaml` defines the OPERANDI pipeline.
- `hooks/operandi.py` implements its identifier and derived-column hooks.

Generic configuration shipped to all users lives under
`src/imperandi/builtin_datasets_config/`.
