#!/usr/bin/env bash
set -euo pipefail

# End-to-end IRCAD usage example for pipeline v2. Pass another project file to
# run the same validate/plan/run workflow against a different real dataset.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/ircad.yaml}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing project configuration: $CONFIG_PATH" >&2
  exit 2
fi

imperandi validate "$CONFIG_PATH"
imperandi plan "$CONFIG_PATH"
imperandi run "$CONFIG_PATH"

echo "IRCAD v2 pipeline completed successfully."
