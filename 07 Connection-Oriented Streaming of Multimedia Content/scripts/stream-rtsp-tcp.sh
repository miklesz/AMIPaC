#!/usr/bin/env sh
# RTSP/TCP sender for exercise 07.
# Requires mediamtx as RTSP broker running on BROKER_IP:PORT.
# Start the broker first:  mediamtx
# Then run this script.
set -eu

if [ "${1-}" = "" ]; then
  echo "Usage: $0 INPUT [BROKER_IP] [PORT] [PATH]" >&2
  echo "  Requires mediamtx (or compatible RTSP broker) running on BROKER_IP:PORT." >&2
  exit 1
fi

INPUT=$1
BROKER_IP=${2:-127.0.0.1}
PORT=${3:-8554}
PATH_PART=${4:-live}

exec ffmpeg -re -stream_loop -1 -i "$INPUT" \
  -c copy -f rtsp -rtsp_transport tcp \
  "rtsp://$BROKER_IP:$PORT/$PATH_PART"

