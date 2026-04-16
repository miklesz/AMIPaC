#!/usr/bin/env sh
set -eu

BASE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
INPUT=${1:-"$BASE_DIR/media/BigBuckBunny_320x180.mp4"}
PORT=${PORT:-12530}
SDP_PATH=${SDP_PATH:-/tmp/amipac-06-smoke.sdp}

if [ ! -f "$INPUT" ]; then
  echo "Missing input file: $INPUT" >&2
  exit 1
fi

rm -f "$SDP_PATH"

ffmpeg -hide_banner -loglevel error \
  -re -stream_loop -1 -t 4 -i "$INPUT" \
  -an -c:v copy \
  -f rtp -sdp_file "$SDP_PATH" "rtp://127.0.0.1:$PORT" >/dev/null 2>&1 &
SEND_PID=$!

while [ ! -s "$SDP_PATH" ]; do
  sleep 1
done

"$BASE_DIR/scripts/receive-rtp.sh" "$SDP_PATH" >/dev/null 2>&1 &
RECV_PID=$!

wait "$SEND_PID"
kill "$RECV_PID" 2>/dev/null || true
wait "$RECV_PID" 2>/dev/null || true

echo "RTP VLC smoke test finished."
