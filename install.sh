#!/usr/bin/env bash
# ==============================================================================
# Project Sandwich One-Line Installer & System Initializer
# High-Throughput Ultra-Context (128k) Inference Engine for Consumer Hardware
# ==============================================================================

set -e

INSTALL_DIR="${HOME}/.local/bin"
SANDWICH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SANDWICH_ROOT}/build"

echo "🥪 [Project Sandwich] Initializing installation..."

# 1. Verify Architecture & CPU Features
echo "🔍 [Hardware Probe] Verifying CPU & GPU capabilities..."
if grep -q "avx2" /proc/cpuinfo; then
    echo "  -> AVX2 SIMD: Supported ✅"
else
    echo "  -> Warning: AVX2 not detected. Performance may be degraded."
fi

if grep -q "avx_vnni\|vnni" /proc/cpuinfo; then
    echo "  -> AVX-VNNI Matrix Acceleration: Supported ✅"
fi

if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    echo "  -> Dedicated GPU: ${GPU_NAME} (${VRAM_MB} MB) ✅"
else
    echo "  -> Note: No NVIDIA GPU detected. Operating in pure CPU AVX2-VNNI mode."
fi

# 2. Ensure Install Directory Exists
mkdir -p "${INSTALL_DIR}"

# 3. Create the Unified 'sandwich' CLI Wrapper
WRAPPER_PATH="${INSTALL_DIR}/sandwich"
cat << 'EOF' > "${WRAPPER_PATH}"
#!/usr/bin/env bash
SANDWICH_BIN_DIR="/home/sandeep/dev/sandwich/sandwich-cpp/build/bin"
CLI_BIN="${SANDWICH_BIN_DIR}/llama-cli"
SERVER_BIN="${SANDWICH_BIN_DIR}/llama-server"

case "$1" in
    run)
        shift
        exec "${CLI_BIN}" --sandwich-auto "$@"
        ;;
    serve)
        shift
        exec "${SERVER_BIN}" --sandwich-auto "$@"
        ;;
    bench)
        shift
        exec python3 /home/sandeep/dev/sandwich/sandwich-cpp/sustained_stress_test_of_sandwich_cpp.py "$@"
        ;;
    niah)
        shift
        exec python3 /home/sandeep/dev/sandwich/sandwich-cpp/tests/test_niah_retrieval.py "$@"
        ;;
    status)
        echo "🥪 [Project Sandwich System Telemetry]"
        nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv
        free -h
        ;;
    *)
        echo "🥪 Project Sandwich: Sub-8GB Ultra-Context (128k) Inference Engine"
        echo "Usage: sandwich <command> [options]"
        echo ""
        echo "Commands:"
        echo "  run    -m <model> -c 131072 -p \"Prompt\"   (Run local inference with auto-dispatcher)"
        echo "  serve  -m <model> --port 8080              (Launch OpenAI-compatible API server)"
        echo "  bench                                     (Execute full 128k sustained benchmark)"
        echo "  niah                                      (Run Needle-In-A-Haystack retrieval test)"
        echo "  status                                    (Inspect VRAM, DDR5, and GPU telemetry)"
        echo ""
        exit 1
        ;;
esac
EOF

chmod +x "${WRAPPER_PATH}"

# 4. Symlink direct binary aliases
ln -sf "${BUILD_DIR}/bin/llama-cli" "${INSTALL_DIR}/sandwich-cli"
ln -sf "${BUILD_DIR}/bin/llama-server" "${INSTALL_DIR}/sandwich-server"

echo ""
echo "✅ [Project Sandwich] Installation complete!"
echo "  -> Command:   ${WRAPPER_PATH}"
echo "  -> CLI Alias: ${INSTALL_DIR}/sandwich-cli"
echo "  -> Server:    ${INSTALL_DIR}/sandwich-server"
echo ""
echo "Try running: sandwich status"
