#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if ! command -v python3 >/dev/null 2>&1; then
  printf '%s\n' 'Python 3.10+ is required. Run this on a DGX Spark with DGX OS.' >&2
  exit 1
fi
exec python3 "$HERE/welcome.py" "$@"
