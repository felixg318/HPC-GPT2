#!/usr/bin/env bash
set -euo pipefail

# Build and run stage3 MPI+CUDA inference. Pass any extra args to the binary.
make -s inference
./inference "$@"
