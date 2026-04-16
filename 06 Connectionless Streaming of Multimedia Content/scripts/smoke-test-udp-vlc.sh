#!/usr/bin/env sh
set -eu

BASE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
INPUT=${1:-"$BASE_DIR/media/BigBuckBunny_320x180.mp4"}
PORT=${PORT:-12520}

if [ ! -f "$INPUT" ]; then
  echo "Missing input file: $INPUT" >&2
  exit 1
fi

"$BASE_DIR/scripts/receive-udp.sh" "$PORT" >/dev/null 2>&1 &
RECV_PID=$!
sleep 2

ffmpeg -hide_banner -loglevel error \
  -re -stream_loop -1 -t 4 -i "$INPUT" \
  -c copy -f mpegts "udp://127.0.0.1:$PORT?pkt_size=1316"

kill "$RECV_PID" 2>/dev/null || true
wait "$RECV_PID" 2>/dev/null || true

echo "UDP VLC smoke test finished."
