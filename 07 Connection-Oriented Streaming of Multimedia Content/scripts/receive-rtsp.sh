#!/usr/bin/env sh
set -eu

URL=${1:-rtsp://127.0.0.1:8554/live}

if [ -x /Applications/VLC.app/Contents/MacOS/VLC ]; then
  exec /Applications/VLC.app/Contents/MacOS/VLC --play-and-exit --rtsp-tcp "$URL"
fi

if command -v vlc >/dev/null 2>&1; then
  exec vlc --play-and-exit --rtsp-tcp "$URL"
fi

exec ffplay -rtsp_transport tcp "$URL"

