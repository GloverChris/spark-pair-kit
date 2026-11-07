#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
action="${1:-plan}"
if (($#)); then shift; fi
case "$action" in
  plan|apply|doctor|restore) exec "$HERE/spark-pair.sh" "cluster-$action" "$@" ;;
  *) printf '%s\n' 'Usage: cluster-pair.sh {plan|apply|doctor|restore} [options]' >&2; exit 2 ;;
esac
