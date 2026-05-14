#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python}"

echo "[pipeline] repo root: ${REPO_ROOT}"
echo "[pipeline] src dir:   ${SCRIPT_DIR}"

cd "${REPO_ROOT}"

echo "[1/4] RGBD_image_generator.py (source)"
"${PYTHON}" src/RGBD_image_generator.py

echo "[1/4] RGBD_image_generator.py (target)"
"${PYTHON}" src/RGBD_image_generator.py --is-target

echo "[2/4] build_segmented_pointclouds.py"
"${PYTHON}" src/build_segmented_pointclouds.py "$@"

cd "${SCRIPT_DIR}"

echo "[3/4] test_OAR.py"
"${PYTHON}" test_OAR.py

echo "[4/4] find_correspondence.py"
"${PYTHON}" find_correspondence.py

echo "[pipeline] done."
