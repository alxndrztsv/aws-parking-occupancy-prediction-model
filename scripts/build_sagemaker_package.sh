#!/usr/bin/env bash
# Build the SageMaker training archive.
#
# The archive root must contain train.py, requirements.txt, src/ and configs/,
# because step_functions.tf points sagemaker_submit_directory at it and
# sagemaker_program is just "train.py".
#
# Usage: scripts/build_sagemaker_package.sh [output.tar.gz]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/dist/train.tar.gz}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/src/models/estimators" "$STAGE/configs"

cp "$ROOT/sagemaker/train.py" "$STAGE/train.py"
cp "$ROOT/sagemaker/requirements.txt" "$STAGE/requirements.txt"
cp "$ROOT/configs/models.yaml" "$STAGE/configs/models.yaml"
cp "$ROOT/src/__init__.py" "$STAGE/src/__init__.py"
cp "$ROOT/src/config.py" "$STAGE/src/config.py"
cp "$ROOT"/src/models/*.py "$STAGE/src/models/"
cp "$ROOT"/src/models/estimators/*.py "$STAGE/src/models/estimators/"

# Remove any stale __pycache__
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} +

for required in src/__init__.py src/models/__init__.py src/models/estimators/__init__.py; do
    if [ ! -f "$STAGE/$required" ]; then
        echo "missing $required - 'import src...' would fail inside the container" >&2
        exit 1
    fi
done

mkdir -p "$(dirname "$OUT")"
tar -czf "$OUT" -C "$STAGE" .

echo "built $OUT ($(du -h "$OUT" | cut -f1))"
tar -tzf "$OUT" | sort
