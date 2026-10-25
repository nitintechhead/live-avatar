#!/usr/bin/env bash
set -euo pipefail
ROOT="${HALLOLIVE_ROOT:-/workspace/Hallo-Live}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
test -d "$ROOT/hallolive" || { echo "ERROR: $ROOT does not look like Hallo-Live (hallolive/ missing)"; exit 1; }
test -f "$ROOT/inference.py" || { echo "ERROR: $ROOT/inference.py missing"; exit 1; }
mkdir -p "$ROOT/stream_demo/static"
cp "$SRC_DIR/server.py" "$ROOT/stream_demo/server.py"
cp "$SRC_DIR/static/index.html" "$ROOT/stream_demo/static/index.html"
cp "$SRC_DIR/README.md" "$ROOT/stream_demo/README.md"
touch "$ROOT/stream_demo/__init__.py"
echo "Installed files under $ROOT/stream_demo/"
echo
echo "Next:"
echo "  cd $ROOT"
echo "  source hallo-venv/bin/activate"
echo "  python -m pip install fastapi 'uvicorn[standard]'"
echo "  export HALLOLIVE_MODEL_DIR='/exact/model/root'"
echo "  export HALLOLIVE_CHECKPOINT='/exact/path/to/hallolive_dit.pt'"
echo "  export HALLOLIVE_CONFIG='$ROOT/configs/dual_stream_dmd_5B.yaml'"
echo "  export HALLOLIVE_MAX_PROMPT_CHARS='THE_VERIFIED_INTEGER_LIMIT'"
echo "  python -m uvicorn stream_demo.server:app --host 0.0.0.0 --port 8090"
