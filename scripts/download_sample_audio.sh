#!/usr/bin/env bash
# Download a small public-domain WAV into data/samples/sample.wav
set -euo pipefail
mkdir -p data/samples
curl -L -o data/samples/sample.wav \
  https://github.com/python/cpython/raw/main/Lib/test/support/data/whattup.wav

echo "Saved sample audio to data/samples/sample.wav"
