#!/usr/bin/env bash
# ============================================================================
# retrain_and_evaluate.sh
#
# Fresh-server bootstrap: install deps, retrain both GAN models,
# (optionally) retrain the time-recognition CNN, then execute the
# full-pipeline notebook so you can inspect the new outputs.
#
# Usage:
#   ./retrain_and_evaluate.sh                  # full run (20k samples, 100 epochs)
#   ./retrain_and_evaluate.sh --quick          # smoke-test (500 samples, 10 epochs)
#   ./retrain_and_evaluate.sh --skip-install   # skip environment setup
#   ./retrain_and_evaluate.sh --skip-cnn       # skip time-recognition CNN training
#   ./retrain_and_evaluate.sh --gpu            # force CUDA (default: auto-detect)
#
# Prerequisites:
#   - Python 3.10+ installed
#   - uv installed (curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - (optional) NVIDIA GPU with CUDA for faster training
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

# ── Parse CLI flags ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)       QUICK=true; shift ;;
        --skip-install) SKIP_INSTALL=true; shift ;;
        --skip-cnn)    SKIP_CNN=true; shift ;;
        --samples)     SAMPLES="$2"; shift 2 ;;
        --epochs)      EPOCHS="$2"; shift 2 ;;
        --batch-size)  BATCH_SIZE="$2"; shift 2 ;;
        --data-dir)    DATA_DIR="$2"; shift 2 ;;
        --gpu)         export CUDA_VISIBLE_DEVICES=0; shift ;;
        -h|--help)
            head -20 "$0" | grep -E '^#' | sed 's/^# \?//'
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
    echo "⚡ Quick mode: ${SAMPLES} samples, ${EPOCHS} epochs"
fi

# ── Resolve project root (directory this script lives in) ─────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "📂 Project root: $SCRIPT_DIR"

# ============================================================================
# Step 1: Environment Setup
# ============================================================================
step_env() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Step 1/5: Environment Setup"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if $SKIP_INSTALL; then
        echo "⏭  Skipping install (--skip-install)"
        return
    fi

    # Check for uv, fall back to pip
    if command -v uv &>/dev/null; then
        echo "📦 Installing dependencies with uv..."
        uv sync
        # All subsequent python/pytest calls go through uv run
        RUN_PREFIX="uv run"
    else
        echo "📦 uv not found — falling back to pip..."
        python3 -m pip install --upgrade pip
        python3 -m pip install -e ".[dev]"
        RUN_PREFIX="python3 -m"
    fi

    # Verify torch is importable
    $RUN_PREFIX python -c "import torch; print(f'PyTorch {torch.__version__}  |  CUDA: {torch.cuda.is_available()}  |  MPS: {torch.backends.mps.is_available()}')"
    echo "✅ Environment ready"
}

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
# Step 2: Retrain Sketch-cGAN
# ============================================================================
step_sketch() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Step 2/5: Retrain Sketch-cGAN  (${SAMPLES} samples, ${EPOCHS} epochs)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    run_py -m analog_clock.GAN.retrain \
        --model sketch \
        --samples "$SAMPLES" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --data-dir "$DATA_DIR"

    SKETCH_WEIGHTS="analog_clock/GAN/sketch/generator_${EPOCHS}.pth"
    if [[ -f "$SKETCH_WEIGHTS" ]]; then
        echo "✅ Sketch-cGAN weights saved: $SKETCH_WEIGHTS"
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
    echo "  Step 3/5: Retrain Inpainting GAN  (${SAMPLES} samples, ${EPOCHS} epochs)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    run_py -m analog_clock.GAN.retrain \
        --model inpainting \
        --samples "$SAMPLES" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --data-dir "$DATA_DIR"

    INPAINT_WEIGHTS="analog_clock/GAN/inpainting/inpaint_gen_${EPOCHS}.pth"
    if [[ -f "$INPAINT_WEIGHTS" ]]; then
        echo "✅ Inpainting GAN weights saved: $INPAINT_WEIGHTS"
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
    echo "  Step 4/5: Retrain Time-Recognition CNN"
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
    echo "  Step 5/5: Execute full-pipeline.ipynb"
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
    echo ""
    echo "  Pipeline Results (${OUTPUT_DIR}/):"
    echo "    Executed notebook: ${OUTPUT_DIR}/full-pipeline-executed.ipynb"
    echo "    HTML report:       ${OUTPUT_DIR}/full-pipeline-results.html"
    echo "    Output GIFs:       (look inside the notebook output cells)"
    echo ""
    echo "  Quick commands to explore results:"
    echo "    jupyter notebook ${OUTPUT_DIR}/full-pipeline-executed.ipynb"
    echo "    open ${OUTPUT_DIR}/full-pipeline-results.html  # macOS"
    echo ""
}

# ============================================================================
# Run all steps
# ============================================================================
SECONDS=0

step_env
step_sketch
step_inpainting
step_cnn
step_notebook
step_summary

elapsed=$SECONDS
printf "⏱  Total time: %dh %dm %ds\n" $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
