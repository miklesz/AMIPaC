#!/usr/bin/env sh
set -eu

if [ "${1-}" = "" ]; then
  echo "Usage: $0 INPUT [DESTINATION_IP] [PORT]" >&2
  exit 1
fi

INPUT=$1
DEST_IP=${2:-127.0.0.1}
PORT=${3:-1234}

exec ffmpeg -re -stream_loop -1 -i "$INPUT" \
  -c copy -f mpegts "udp://$DEST_IP:$PORT?pkt_size=1316"
