#!/usr/bin/env sh
set -eu

PORT=${1:-1234}

if [ -x /Applications/VLC.app/Contents/MacOS/VLC ]; then
  exec /Applications/VLC.app/Contents/MacOS/VLC --play-and-exit "udp://@:$PORT"
fi

if command -v vlc >/dev/null 2>&1; then
  exec vlc --play-and-exit "udp://@:$PORT"
fi

exec ffplay "udp://@:$PORT"
