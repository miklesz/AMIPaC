#!/usr/bin/env sh
set -eu

if [ "${1-}" = "" ]; then
  echo "Usage: $0 INPUT [DESTINATION_IP] [PORT] [SDP_PATH]" >&2
  exit 1
fi

INPUT=$1
DEST_IP=${2:-127.0.0.1}
PORT=${3:-5004}
SDP_PATH=${4:-/tmp/amipac-06.sdp}

exec ffmpeg -re -stream_loop -1 -i "$INPUT" \
  -an -c:v copy \
  -f rtp -sdp_file "$SDP_PATH" "rtp://$DEST_IP:$PORT"
