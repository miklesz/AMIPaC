#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CONF_FILE="$SCRIPT_DIR/mediamtx.yml"

if command -v mediamtx >/dev/null 2>&1; then
  exec mediamtx "$CONF_FILE"
fi

echo "mediamtx is not installed." >&2
echo "Install it first:" >&2
echo "  macOS: brew install mediamtx" >&2
echo "  Ubuntu: download from https://github.com/bluenviron/mediamtx/releases" >&2
exit 1

