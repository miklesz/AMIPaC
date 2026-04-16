#!/usr/bin/env sh
set -eu

MODE=${1:-udp}
BASE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
INPUT="$BASE_DIR/media/BigBuckBunny_320x180.mp4"

if [ ! -f "$INPUT" ]; then
  echo "Missing sample file: $INPUT" >&2
  exit 1
fi

cleanup() {
  if [ "${SEND_PID-}" != "" ]; then
    kill "$SEND_PID" 2>/dev/null || true
  fi
  if [ "${RECV_PID-}" != "" ]; then
    kill "$RECV_PID" 2>/dev/null || true
  fi
  rm -f "${RECV_LOG-}" "${SDP_PATH-}"
}

trap cleanup EXIT INT TERM

RECV_LOG=$(mktemp)

case "$MODE" in
  udp)
    PORT=${PORT:-1234}
    ffmpeg -hide_banner -nostats -loglevel info \
      -i "udp://127.0.0.1:$PORT?fifo_size=1000000&overrun_nonfatal=1" \
      -f null - >"$RECV_LOG" 2>&1 &
    RECV_PID=$!
    sleep 1
    ffmpeg -hide_banner -nostats -loglevel error \
      -re -stream_loop -1 -t 3 -i "$INPUT" \
      -c copy -f mpegts "udp://127.0.0.1:$PORT?pkt_size=1316" &
    SEND_PID=$!
    wait "$SEND_PID"
    SEND_PID=
    sleep 1
    kill "$RECV_PID" 2>/dev/null || true
    wait "$RECV_PID" 2>/dev/null || true
    RECV_PID=
    grep -q "Input #0, mpegts" "$RECV_LOG"
    echo "UDP self-test passed."
    ;;
  rtp)
    PORT=${PORT:-5004}
    SDP_PATH=$(mktemp /tmp/amipac-06-rtp-XXXXXX.sdp)
    ffmpeg -hide_banner -nostats -loglevel error \
      -re -stream_loop -1 -t 3 -i "$INPUT" \
      -an -c:v copy \
      -f rtp -sdp_file "$SDP_PATH" "rtp://127.0.0.1:$PORT" &
    SEND_PID=$!
    while [ ! -s "$SDP_PATH" ]; do
      sleep 1
    done
    ffmpeg -hide_banner -nostats -loglevel info \
      -protocol_whitelist file,udp,rtp \
      -i "$SDP_PATH" -f null - >"$RECV_LOG" 2>&1 &
    RECV_PID=$!
    wait "$SEND_PID"
    SEND_PID=
    sleep 1
    kill "$RECV_PID" 2>/dev/null || true
    wait "$RECV_PID" 2>/dev/null || true
    RECV_PID=
    grep -q "Input #0, sdp" "$RECV_LOG"
    echo "RTP self-test passed."
    ;;
  *)
    echo "Usage: $0 [udp|rtp]" >&2
    exit 1
    ;;
esac
