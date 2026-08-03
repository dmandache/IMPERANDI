#!/usr/bin/env bash
set -euo pipefail

# Supply a v2 project file; the default is suitable for a local IRCAD fixture.
CONFIG_PATH="${1:-./tests/data/imperandi.yaml}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing project configuration: $CONFIG_PATH" >&2
  echo "Create one with: imperandi init $CONFIG_PATH" >&2
  exit 2
fi

imperandi validate "$CONFIG_PATH"
imperandi plan "$CONFIG_PATH"
imperandi run "$CONFIG_PATH"

echo "Pipeline executed successfully."
