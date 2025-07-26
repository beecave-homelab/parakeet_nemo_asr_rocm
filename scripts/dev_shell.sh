#!/usr/bin/env bash
# Drop into a bash shell inside the running parakeet-asr container
# Usage: ./scripts/dev_shell.sh
set -euo pipefail

CONTAINER="parakeet-asr-rocm"
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  echo "Container ${CONTAINER} not running. Starting via docker compose..." >&2
  docker compose up -d
fi

docker exec -it "${CONTAINER}" bash
