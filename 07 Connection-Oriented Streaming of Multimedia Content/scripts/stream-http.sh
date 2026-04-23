#!/usr/bin/env sh
set -eu

if [ "${1-}" = "" ]; then
  echo "Usage: $0 INPUT [BIND_IP] [PORT]" >&2
  exit 1
fi

INPUT=$1
BIND_IP=${2:-0.0.0.0}
PORT=${3:-8080}

exec ffmpeg -re -stream_loop -1 -i "$INPUT" \
  -c copy -f mpegts -listen 1 "http://$BIND_IP:$PORT/stream.ts"

