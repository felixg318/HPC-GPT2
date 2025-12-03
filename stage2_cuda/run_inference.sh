#!/usr/bin/env bash
set -euo pipefail

# Build and run stage2 CUDA inference. Pass any extra args to the binary.
make -s inference
./inference "$@"
