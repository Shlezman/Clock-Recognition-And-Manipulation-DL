#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# train_gpu.sh — Run ViT training with guaranteed CUDA torch.
#
# Usage:
#   bash analog_clock/vit_clock_gen/train_gpu.sh [train.py args...]
#
# What it does:
#   1. Resolves the project root and activates the uv venv.
#   2. Prints CUDA diagnostics (device count, driver version).
#   3. Aborts with a clear message if no CUDA device is visible.
#   4. Delegates to train.py with any extra args you pass.
#
# The script does NOT call `uv run` so there is no auto-sync that could
# revert the GPU torch wheels to the CPU-only PyPI build.
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON=".venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "❌  .venv not found. Run 'uv sync' from the project root first."
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ViT Clock Generation — GPU Training Launcher"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Print CUDA diagnostics
"$PYTHON" - <<'EOF'
import torch, sys
print(f"  Python      : {sys.version.split()[0]}")
print(f"  PyTorch     : {torch.__version__}")
print(f"  CUDA avail  : {torch.cuda.is_available()}")
print(f"  CUDA version: {torch.version.cuda or 'N/A'}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}       : {p.name}  ({p.total_memory / 1e9:.1f} GB)")
else:
    print()
    print("  ⚠️  torch.cuda.is_available() = False")
    print("  The installed torch may be the CPU-only build.")
    print("  Re-install the CUDA build and re-run uv sync:")
    print("    uv sync")
    print("  (pyproject.toml now pins torch to the cu121 index on Linux)")
    sys.exit(1)
EOF

echo ""
echo "  Launching: $PYTHON analog_clock/vit_clock_gen/train.py $*"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

exec "$PYTHON" analog_clock/vit_clock_gen/train.py "$@"
