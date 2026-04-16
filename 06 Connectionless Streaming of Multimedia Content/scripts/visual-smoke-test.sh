#!/usr/bin/env sh
set -eu

BASE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
INPUT=${1:-"$BASE_DIR/media/BigBuckBunny_320x180.mp4"}
PORT=${PORT:-1234}
DURATION=${DURATION:-5}

if [ ! -f "$INPUT" ]; then
  echo "Missing input file: $INPUT" >&2
  exit 1
fi

cleanup() {
  if [ "${SEND_PID-}" != "" ]; then
    kill "$SEND_PID" 2>/dev/null || true
  fi
  if [ "${RECV_PID-}" != "" ]; then
    kill "$RECV_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

ffplay -window_title "AMIPaC visual smoke test" \
  -x 640 -y 360 \
  "udp://127.0.0.1:$PORT?fifo_size=1000000&overrun_nonfatal=1" >/dev/null 2>&1 &
RECV_PID=$!

sleep 1

ffmpeg -hide_banner -loglevel error \
  -re -stream_loop -1 -t "$DURATION" -i "$INPUT" \
  -an \
  -c:v libx264 -preset ultrafast -tune zerolatency \
  -g 24 -x264-params repeat-headers=1 \
  -pix_fmt yuv420p \
  -f mpegts "udp://127.0.0.1:$PORT?pkt_size=1316" >/dev/null 2>&1 &
SEND_PID=$!

wait "$SEND_PID"
SEND_PID=
sleep 1
kill "$RECV_PID" 2>/dev/null || true
wait "$RECV_PID" 2>/dev/null || true
RECV_PID=

echo "Visual smoke test finished."
