#!/usr/bin/env sh
set -eu

URL=${1:-http://127.0.0.1:8080/stream.ts}

if [ -x /Applications/VLC.app/Contents/MacOS/VLC ]; then
  exec /Applications/VLC.app/Contents/MacOS/VLC --play-and-exit "$URL"
fi

if command -v vlc >/dev/null 2>&1; then
  exec vlc --play-and-exit "$URL"
fi

exec ffplay "$URL"

