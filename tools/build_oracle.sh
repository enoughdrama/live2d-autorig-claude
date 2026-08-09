#!/bin/sh
# Build the moc3 validation oracle against the official Cubism Core.
# Core lib+header live in reference/core/ (see README for how to fetch them).
set -e
cd "$(dirname "$0")/.."
cc -O2 -o build/validate tools/validate.c \
   -Ireference/core reference/core/libLive2DCubismCore.a
echo "built build/validate"
