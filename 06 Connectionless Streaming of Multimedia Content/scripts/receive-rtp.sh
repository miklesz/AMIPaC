#!/usr/bin/env sh
set -eu

SDP_PATH=${1:-/tmp/amipac-06.sdp}

if [ -x /Applications/VLC.app/Contents/MacOS/VLC ]; then
  exec /Applications/VLC.app/Contents/MacOS/VLC --play-and-exit "$SDP_PATH"
fi

if command -v vlc >/dev/null 2>&1; then
  exec vlc --play-and-exit "$SDP_PATH"
fi

exec ffplay -protocol_whitelist file,udp,rtp "$SDP_PATH"
