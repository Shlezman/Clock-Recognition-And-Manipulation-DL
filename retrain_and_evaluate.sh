#!/usr/bin/env bash
# ============================================================================
# retrain_and_evaluate.sh
#
# Fresh-server bootstrap: validate GPU, install deps (with CUDA PyTorch),
# retrain both GAN models, (optionally) retrain the time-recognition CNN,
# then execute the full-pipeline notebook to inspect the new outputs.
#
# Usage:
#   ./retrain_and_evaluate.sh                  # full run (20k samples, 100 epochs)
#   ./retrain_and_evaluate.sh --quick          # smoke-test (500 samples, 10 epochs)
#   ./retrain_and_evaluate.sh --skip-install   # skip environment setup
#   ./retrain_and_evaluate.sh --skip-cnn       # skip time-recognition CNN training
#   ./retrain_and_evaluate.sh --device cuda    # force CUDA device
#   ./retrain_and_evaluate.sh --batch-size 16  # larger batch for big GPUs
#
# Prerequisites:
#   - Python 3.10+ installed
#   - uv installed (curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - (NVIDIA GPU) CUDA toolkit + drivers installed
# ============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────
SAMPLES=20000
EPOCHS=100
BATCH_SIZE=8
CNN_EPOCHS=60
CNN_SAMPLES=20000
SKIP_INSTALL=false
SKIP_CNN=false
QUICK=false
DATA_DIR="./dataset"
OUTPUT_DIR="./output"
LOG_DIR="./logs"
DEVICE=""          # empty = auto-detect in Python
CUDA_TORCH="auto"  # auto / force / skip

# ── Parse CLI flags ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)        QUICK=true; shift ;;
        --skip-install) SKIP_INSTALL=true; shift ;;
        --skip-cnn)     SKIP_CNN=true; shift ;;
        --samples)      SAMPLES="$2"; shift 2 ;;
        --epochs)       EPOCHS="$2"; shift 2 ;;
        --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
        --data-dir)     DATA_DIR="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --device)       DEVICE="$2"; shift 2 ;;
        --cuda-torch)   CUDA_TORCH="$2"; shift 2 ;;
        -h|--help)
            head -23 "$0" | grep -E '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if $QUICK; then
    SAMPLES=500
    EPOCHS=10
    BATCH_SIZE=4
    CNN_EPOCHS=5
    CNN_SAMPLES=1000
fi

# ── Resolve project root (directory this script lives in) ─────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Setup log dir ────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
MAIN_LOG="${LOG_DIR}/retrain_$(date +%Y%m%d_%H%M%S).log"

# Tee all output to both console and log file
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "============================================================"
echo "  Clock GAN Retraining Pipeline"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Project root:  $SCRIPT_DIR"
echo "  Samples:       $SAMPLES"
echo "  Epochs:        $EPOCHS"
echo "  Batch size:    $BATCH_SIZE"
echo "  Device:        ${DEVICE:-auto-detect}"
echo "  Data dir:      $DATA_DIR"
echo "  Output dir:    $OUTPUT_DIR"
echo "  Log file:      $MAIN_LOG"
if $QUICK; then
    echo "  Mode:          QUICK (smoke-test)"
fi
echo ""

# Helper: resolve the python runner
run_py() {
    if command -v uv &>/dev/null; then
        uv run python "$@"
    else
        python3 "$@"
    fi
}

run_jupyter() {
    if command -v uv &>/dev/null; then
        uv run jupyter "$@"
    else
        jupyter "$@"
    fi
}

# ============================================================================
# Step 0: NVIDIA / GPU Pre-flight
# ============================================================================
step_gpu_preflight() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Step 0/6: GPU Pre-flight Check"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Check nvidia-smi
    if command -v nvidia-smi &>/dev/null; then
        echo ""
        echo "nvidia-smi output:"
        echo "─────────────────────────────────────────────────"
        nvidia-smi
        echo "─────────────────────────────────────────────────"
        echo ""

        # Parse GPU info
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
        GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        CUDA_VER_DRIVER=$(nvidia-smi | grep -oP 'CUDA Version: \K[\d.]+' || echo "unknown")

        echo "GPU:            $GPU_NAME"
        echo "VRAM:           $GPU_MEM"
        echo "Driver:         $GPU_DRIVER"
        echo "CUDA (driver):  $CUDA_VER_DRIVER"

        # Check for common issues
        GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
        if [[ -n "$GPU_MEM_MB" ]] && (( GPU_MEM_MB < 4000 )); then
            echo "⚠️  WARNING: GPU has only ${GPU_MEM_MB}MB VRAM. Consider reducing --batch-size"
        fi

        # Check GPU utilization (should be low before training)
        GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
        if [[ -n "$GPU_UTIL" ]] && (( GPU_UTIL > 50 )); then
            echo "⚠️  WARNING: GPU utilization is ${GPU_UTIL}% — another process may be using the GPU"
        fi

        # Check memory usage
        GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
        if [[ -n "$GPU_MEM_USED" ]] && (( GPU_MEM_USED > 1000 )); then
            echo "⚠️  WARNING: ${GPU_MEM_USED}MB GPU memory already in use"
        fi
    else
        echo "nvidia-smi not found"
        if [[ "$DEVICE" == "cuda" ]]; then
            echo "❌ ERROR: --device cuda specified but nvidia-smi not found"
            echo "   Install NVIDIA drivers: https://docs.nvidia.com/cuda/"
            exit 1
        fi
    fi

    # Check NVCC
    if command -v nvcc &>/dev/null; then
        NVCC_VER=$(nvcc --version | grep -oP 'release \K[\d.]+' || echo "unknown")
        echo "CUDA toolkit:   $NVCC_VER (nvcc)"
    else
        echo "nvcc:           not found (CUDA toolkit not in PATH)"
        if command -v nvidia-smi &>/dev/null; then
            echo "  NOTE: CUDA toolkit not needed for PyTorch — PyTorch bundles its own CUDA runtime"
        fi
    fi

    # Disk space check
    DISK_FREE=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')
    echo "Disk free:      ${DISK_FREE}G"
    if (( DISK_FREE < 10 )); then
        echo "⚠️  WARNING: Low disk space (${DISK_FREE}G). Dataset generation needs ~5-10G"
    fi
    if (( DISK_FREE < 2 )); then
        echo "❌ ERROR: Critically low disk space"
        exit 1
    fi

    # RAM check
    if command -v free &>/dev/null; then
        RAM_AVAIL=$(free -g | awk '/Mem:/{print $7}')
        echo "RAM available:  ${RAM_AVAIL}G"
        if (( RAM_AVAIL < 4 )); then
            echo "⚠️  WARNING: Low available RAM (${RAM_AVAIL}G)"
        fi
    fi

    echo ""
    echo "✅ GPU pre-flight complete"
}

# ============================================================================
# Step 1: Environment Setup
# ============================================================================
step_env() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Step 1/6: Environment Setup"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if $SKIP_INSTALL; then
        echo "⏭  Skipping install (--skip-install)"
        # Still verify torch is importable
        run_py -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')" || {
            echo "❌ ERROR: PyTorch not importable. Run without --skip-install"
            exit 1
        }
        return
    fi

    # Determine if we need CUDA PyTorch
    NEED_CUDA_TORCH=false
    if [[ "$CUDA_TORCH" == "force" ]]; then
        NEED_CUDA_TORCH=true
    elif [[ "$CUDA_TORCH" == "auto" ]] && command -v nvidia-smi &>/dev/null; then
        NEED_CUDA_TORCH=true
    fi

    if command -v uv &>/dev/null; then
        echo "📦 Installing dependencies with uv..."
        uv sync

        if $NEED_CUDA_TORCH; then
            echo "🔧 Installing CUDA PyTorch (cu124)..."
            uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
        fi
    else
        echo "📦 uv not found — falling back to pip..."
        python3 -m pip install --upgrade pip

        if $NEED_CUDA_TORCH; then
            echo "🔧 Installing CUDA PyTorch (cu124)..."
            python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
        fi

        python3 -m pip install -e ".[dev]"
    fi

    # Verify installation
    echo ""
    echo "Environment verification:"
    run_py -c "
import torch
import sys
print(f'  Python:       {sys.version.split()[0]}')
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA avail:   {torch.cuda.is_available()}')
print(f'  CUDA version: {torch.version.cuda or \"N/A\"}')
print(f'  cuDNN:        {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else \"N/A\"}')
print(f'  MPS avail:    {torch.backends.mps.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}:        {p.name} ({p.total_mem / 1e9:.1f} GB)')
    # Quick sanity
    a = torch.randn(128, 128, device='cuda')
    b = a @ a
    torch.cuda.synchronize()
    print(f'  CUDA test:    PASSED (matmul on GPU)')
elif not torch.cuda.is_available() and '$NEED_CUDA_TORCH' == 'true':
    print('  ⚠️  WARNING: CUDA PyTorch was installed but torch.cuda.is_available() = False')
    print('     Check: nvidia-smi, CUDA drivers, and PyTorch CUDA version compatibility')
"

    echo ""
    echo "✅ Environment ready"
}

# ============================================================================
# Step 2: Retrain Sketch-cGAN
# ============================================================================
step_sketch() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Step 2/6: Retrain Sketch-cGAN  (${SAMPLES} samples, ${EPOCHS} epochs)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    DEVICE_FLAG=""
    if [[ -n "$DEVICE" ]]; then
        DEVICE_FLAG="--device $DEVICE"
    fi

    run_py -m analog_clock.GAN.retrain \
        --model sketch \
        --samples "$SAMPLES" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --data-dir "$DATA_DIR" \
        $DEVICE_FLAG

    SKETCH_WEIGHTS="analog_clock/GAN/sketch/generator_${EPOCHS}.pth"
    if [[ -f "$SKETCH_WEIGHTS" ]]; then
        WEIGHT_SIZE=$(du -h "$SKETCH_WEIGHTS" | cut -f1)
        echo "✅ Sketch-cGAN weights saved: $SKETCH_WEIGHTS ($WEIGHT_SIZE)"
    else
        echo "⚠️  Expected weights not found at $SKETCH_WEIGHTS"
    fi
}

# ============================================================================
# Step 3: Retrain Inpainting GAN
# ============================================================================
step_inpainting() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Step 3/6: Retrain Inpainting GAN  (${SAMPLES} samples, ${EPOCHS} epochs)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    DEVICE_FLAG=""
    if [[ -n "$DEVICE" ]]; then
        DEVICE_FLAG="--device $DEVICE"
    fi

    run_py -m analog_clock.GAN.retrain \
        --model inpainting \
        --samples "$SAMPLES" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --data-dir "$DATA_DIR" \
        $DEVICE_FLAG

    INPAINT_WEIGHTS="analog_clock/GAN/inpainting/inpaint_gen_${EPOCHS}.pth"
    if [[ -f "$INPAINT_WEIGHTS" ]]; then
        WEIGHT_SIZE=$(du -h "$INPAINT_WEIGHTS" | cut -f1)
        echo "✅ Inpainting GAN weights saved: $INPAINT_WEIGHTS ($WEIGHT_SIZE)"
    else
        echo "⚠️  Expected weights not found at $INPAINT_WEIGHTS"
    fi
}

# ============================================================================
# Step 4: Retrain Time-Recognition CNN  (optional)
# ============================================================================
step_cnn() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Step 4/6: Retrain Time-Recognition CNN"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if $SKIP_CNN; then
        echo "⏭  Skipping CNN (--skip-cnn)"
        return
    fi

    CNN_DIR="analog_clock/time_recognition_cnn"
    CNN_DATA="${DATA_DIR}/time_cnn"

    # Generate synthetic mask dataset
    echo "🎨 Generating ${CNN_SAMPLES} synthetic clock-hand masks..."
    run_py "${CNN_DIR}/dataset_generator.py" \
        --n_samples "$CNN_SAMPLES" \
        --output_dir "$CNN_DATA"

    # Train the CNN
    echo "🧠 Training ClockHandCNN for ${CNN_EPOCHS} epochs..."
    (cd "$CNN_DIR" && run_py train.py \
        --data_dir "../../${CNN_DATA}" \
        --epochs "$CNN_EPOCHS" \
        --batch_size 64)

    if [[ -f "${CNN_DIR}/clock_hand_cnn_best.pth" ]]; then
        echo "✅ Time-Recognition CNN weights saved: ${CNN_DIR}/clock_hand_cnn_best.pth"
    fi
}

# ============================================================================
# Step 5: Execute Full Pipeline Notebook
# ============================================================================
step_notebook() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Step 5/6: Execute full-pipeline.ipynb"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    mkdir -p "$OUTPUT_DIR"

    echo "📓 Executing notebook (this may take a few minutes)..."
    run_jupyter nbconvert \
        --to notebook \
        --execute \
        --ExecutePreprocessor.timeout=600 \
        --ExecutePreprocessor.kernel_name=python3 \
        --output-dir="$OUTPUT_DIR" \
        --output="full-pipeline-executed.ipynb" \
        full-pipeline.ipynb \
    && echo "✅ Executed notebook saved: ${OUTPUT_DIR}/full-pipeline-executed.ipynb" \
    || echo "⚠️  Notebook execution had errors — check ${OUTPUT_DIR}/full-pipeline-executed.ipynb for details"

    # Also export an HTML version for easy viewing
    run_jupyter nbconvert \
        --to html \
        --no-input \
        "${OUTPUT_DIR}/full-pipeline-executed.ipynb" \
        --output="full-pipeline-results.html" \
    && echo "🌐 HTML report: ${OUTPUT_DIR}/full-pipeline-results.html" \
    || true
}

# ============================================================================
# Step 6: Summary
# ============================================================================
step_summary() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Done! Here's where to find everything:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  Model Weights:"
    echo "    Sketch-cGAN:       analog_clock/GAN/sketch/generator_${EPOCHS}.pth"
    echo "    Inpainting GAN:    analog_clock/GAN/inpainting/inpaint_gen_${EPOCHS}.pth"
    if ! $SKIP_CNN; then
    echo "    Time-Recog CNN:    analog_clock/time_recognition_cnn/clock_hand_cnn_best.pth"
    fi
    echo ""
    echo "  Training Artefacts (${DATA_DIR}/):"
    echo "    Checkpoints:       ${DATA_DIR}/generator_*.pth, inpaint_gen_*.pth"
    echo "    Sample grids:      ${DATA_DIR}/samples_epoch_*.png, inpaint_samples_*.png"
    echo "    Training logs:     ${DATA_DIR}/sketch_cgan_training_log.csv"
    echo "                       ${DATA_DIR}/inpainting_training_log.csv"
    echo "    Python logs:       ${DATA_DIR}/retrain_sketch_*.log"
    echo "                       ${DATA_DIR}/retrain_inpainting_*.log"
    echo "    TensorBoard:       ${DATA_DIR}/tensorboard_sketch/"
    echo "                       ${DATA_DIR}/tensorboard_inpainting/"
    echo ""
    echo "  Pipeline Results (${OUTPUT_DIR}/):"
    echo "    Executed notebook: ${OUTPUT_DIR}/full-pipeline-executed.ipynb"
    echo "    HTML report:       ${OUTPUT_DIR}/full-pipeline-results.html"
    echo ""
    echo "  Shell log:           $MAIN_LOG"
    echo ""
    echo "  Quick commands:"
    echo "    tensorboard --logdir ${DATA_DIR}                         # training curves"
    echo "    jupyter notebook ${OUTPUT_DIR}/full-pipeline-executed.ipynb  # results"
    echo ""
}

# ============================================================================
# Run all steps
# ============================================================================
SECONDS=0

step_gpu_preflight
step_env
step_sketch
step_inpainting
step_cnn
step_notebook
step_summary

elapsed=$SECONDS
printf "⏱  Total time: %dh %dm %ds\n" $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
echo "Pipeline finished at $(date '+%Y-%m-%d %H:%M:%S')"
