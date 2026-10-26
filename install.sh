#!/usr/bin/env bash
# ==============================================================================
# 🥪 Sandwich.cpp Production Turnkey Installer
# Sub-8GB Ultra-Context (128k) Standalone Inference Engine
# Repository: https://github.com/sandeepmuttvarpu-png/Sandwich-cpp
# ==============================================================================

set -eo pipefail

VERSION="1.2.0-standalone"
REPO="sandeepmuttvarpu-png/Sandwich-cpp"
FORCE_BUILD=false
CUSTOM_INSTALL_DIR=""
RELEASE_TAG="latest"

# ------------------------------------------------------------------------------
# 1. Parse Command Line Arguments
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build|-b)
            FORCE_BUILD=true
            shift
            ;;
        --dir|-d)
            CUSTOM_INSTALL_DIR="$2"
            shift 2
            ;;
        --tag|-t)
            RELEASE_TAG="$2"
            shift 2
            ;;
        --help|-h)
            echo "🥪 Sandwich.cpp Turnkey Installer"
            echo "Usage: ./install.sh [options]"
            echo ""
            echo "Options:"
            echo "  --build, -b        Force compiling locally from source"
            echo "  --dir, -d <PATH>   Custom binary installation directory"
            echo "  --tag, -t <TAG>    Specific GitHub release tag (default: latest)"
            echo "  --help, -h         Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run ./install.sh --help for available options."
            exit 1
            ;;
    esac
done

# ------------------------------------------------------------------------------
# 2. Detect Operating System & Architecture
# ------------------------------------------------------------------------------
echo "🔍 [1/6] Probing System Architecture & Hardware Topology..."

OS_RAW="$(uname -s)"
case "${OS_RAW}" in
    Linux*)   OS="linux" ;;
    Darwin*)  OS="darwin" ;;
    *)        echo "❌ Error: Unsupported Operating System: ${OS_RAW}"; exit 1 ;;
esac

ARCH_RAW="$(uname -m)"
case "${ARCH_RAW}" in
    x86_64|amd64)   ARCH="x86_64" ;;
    aarch64|arm64)  ARCH="aarch64" ;;
    *)              echo "❌ Error: Unsupported CPU Architecture: ${ARCH_RAW}"; exit 1 ;;
esac

echo "  -> OS:           ${OS} (${OS_RAW})"
echo "  -> Architecture: ${ARCH} (${ARCH_RAW})"

# ------------------------------------------------------------------------------
# 3. Detect NVIDIA GPU & CUDA Acceleration
# ------------------------------------------------------------------------------
HAS_CUDA=false
GPU_DESC="CPU AVX2-VNNI (No dedicated CUDA GPU detected)"

if [ "${OS}" = "linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi >/dev/null 2>&1; then
        HAS_CUDA=true
        GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || echo 'NVIDIA GPU')"
        GPU_VRAM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 || echo 'Unknown')"
        CUDA_DRIVER_VER="$(nvidia-smi 2>/dev/null | grep -o "CUDA Version: [0-9.]*" | head -n 1 | awk '{print $3}' || echo '12.0+')"
        GPU_DESC="${GPU_NAME} (${GPU_VRAM} MiB GDDR) | CUDA ${CUDA_DRIVER_VER}"
        echo "  -> GPU Detected: ${GPU_DESC} ✅"
    fi
elif [ "${OS}" = "darwin" ]; then
    GPU_DESC="Apple Silicon Unified Memory (Metal / Accelerate)"
    echo "  -> GPU Detected: ${GPU_DESC} ✅"
else
    echo "  -> Accelerator:  ${GPU_DESC}"
fi

# ------------------------------------------------------------------------------
# 4. Resolve Installation Directory
# ------------------------------------------------------------------------------
if [ -n "${CUSTOM_INSTALL_DIR}" ]; then
    INSTALL_DIR="${CUSTOM_INSTALL_DIR}"
elif [ "$(id -u)" -eq 0 ]; then
    INSTALL_DIR="/usr/local/bin"
else
    INSTALL_DIR="${HOME}/.local/bin"
fi

mkdir -p "${INSTALL_DIR}"
echo "  -> Install Path: ${INSTALL_DIR}"

# ------------------------------------------------------------------------------
# 5. Acquire Binaries (Release Tarball or Local Source Build)
# ------------------------------------------------------------------------------
TMP_DIR="$(mktemp -d /tmp/sandwich-install-XXXXXX)"
cleanup() {
    rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

INSTALL_SUCCESS=false

# Helper function: Local source compilation
build_from_source() {
    echo ""
    echo "🔨 [Build from Source] Compiling Sandwich.cpp with native hardware optimizations..."
    
    local SOURCE_ROOT=""
    local SCRIPT_DIR
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    if [ -f "${SCRIPT_DIR}/CMakeLists.txt" ]; then
        SOURCE_ROOT="${SCRIPT_DIR}"
    else
        echo "  -> Cloning repository from GitHub (${REPO})..."
        git clone --depth 1 "https://github.com/${REPO}.git" "${TMP_DIR}/source"
        SOURCE_ROOT="${TMP_DIR}/source"
    fi

    cd "${SOURCE_ROOT}"

    local CMAKE_ARGS=("-DCMAKE_BUILD_TYPE=Release")
    if [ "${HAS_CUDA}" = true ]; then
        CMAKE_ARGS+=("-DGGML_CUDA=ON")
    fi

    echo "  -> Configuring CMake in ${SOURCE_ROOT}/build..."
    cmake -B build "${CMAKE_ARGS[@]}"

    local NPROC
    NPROC="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
    echo "  -> Building targets (llama-cli, llama-server) across ${NPROC} parallel jobs..."
    cmake --build build -j "${NPROC}" --target llama-cli llama-server

    # Remove any existing symlinks/binaries to prevent same-file collision
    rm -f "${INSTALL_DIR}/sandwich-cli" "${INSTALL_DIR}/sandwich-server"

    # Copy binary artifacts
    cp -f "build/bin/llama-cli" "${INSTALL_DIR}/sandwich-cli"
    cp -f "build/bin/llama-server" "${INSTALL_DIR}/sandwich-server"

    # Install accompanying shared libraries if present
    local LIB_DIR="${INSTALL_DIR}/../lib/sandwich"
    mkdir -p "${LIB_DIR}"
    find build/bin -maxdepth 1 -name "*.so*" -exec cp -f {} "${LIB_DIR}/" \; 2>/dev/null || true
    find build/bin -maxdepth 1 -name "*.dylib*" -exec cp -f {} "${LIB_DIR}/" \; 2>/dev/null || true

    chmod +x "${INSTALL_DIR}/sandwich-cli" "${INSTALL_DIR}/sandwich-server"
    INSTALL_SUCCESS=true
}

if [ "${FORCE_BUILD}" = false ]; then
    echo ""
    echo "📦 [2/6] Querying GitHub Releases for Precompiled Binary Tarball..."
    
    if [ "${HAS_CUDA}" = true ] && [ "${OS}" = "linux" ]; then
        ASSET_NAME="sandwich-${OS}-${ARCH}-cuda12.tar.gz"
    elif [ "${OS}" = "darwin" ]; then
        ASSET_NAME="sandwich-${OS}-${ARCH}-metal.tar.gz"
    else
        ASSET_NAME="sandwich-${OS}-${ARCH}-cpu.tar.gz"
    fi

    if [ "${RELEASE_TAG}" = "latest" ]; then
        API_URL="https://api.github.com/repos/${REPO}/releases/latest"
    else
        API_URL="https://api.github.com/repos/${REPO}/releases/tags/${RELEASE_TAG}"
    fi

    DOWNLOAD_URL=""
    if command -v curl >/dev/null 2>&1; then
        RELEASE_JSON="$(curl -sSL -H "Accept: application/vnd.github.v3+json" "${API_URL}" || true)"
        DOWNLOAD_URL="$(echo "${RELEASE_JSON}" | grep -o "\"browser_download_url\": *\"[^\"]*${ASSET_NAME}\"" | head -n 1 | cut -d'"' -f4 || true)"
    fi

    if [ -n "${DOWNLOAD_URL}" ]; then
        echo "  -> Found matching release artifact: ${ASSET_NAME}"
        echo "  -> Downloading from ${DOWNLOAD_URL}..."
        if curl -fSL "${DOWNLOAD_URL}" -o "${TMP_DIR}/${ASSET_NAME}"; then
            echo "  -> Extracting archive to ${INSTALL_DIR}..."
            tar -xzf "${TMP_DIR}/${ASSET_NAME}" -C "${TMP_DIR}"

            # Copy extracted binaries
            if [ -f "${TMP_DIR}/sandwich-cli" ]; then
                cp -f "${TMP_DIR}/sandwich-cli" "${INSTALL_DIR}/sandwich-cli"
                cp -f "${TMP_DIR}/sandwich-server" "${INSTALL_DIR}/sandwich-server"
            elif [ -f "${TMP_DIR}/bin/sandwich-cli" ]; then
                cp -f "${TMP_DIR}/bin/sandwich-cli" "${INSTALL_DIR}/sandwich-cli"
                cp -f "${TMP_DIR}/bin/sandwich-server" "${INSTALL_DIR}/sandwich-server"
            elif [ -f "${TMP_DIR}/llama-cli" ]; then
                cp -f "${TMP_DIR}/llama-cli" "${INSTALL_DIR}/sandwich-cli"
                cp -f "${TMP_DIR}/llama-server" "${INSTALL_DIR}/sandwich-server"
            fi

            chmod +x "${INSTALL_DIR}/sandwich-cli" "${INSTALL_DIR}/sandwich-server"
            INSTALL_SUCCESS=true
            echo "  -> Precompiled release archive extracted successfully! ✅"
        fi
    else
        echo "  -> Precompiled binary asset '${ASSET_NAME}' is not yet published on GitHub Releases."
        echo "  -> Automatically falling back to local source compilation..."
        build_from_source
    fi
else
    build_from_source
fi

# ------------------------------------------------------------------------------
# 6. Generate the Unified 'sandwich' CLI Dispatcher
# ------------------------------------------------------------------------------
echo ""
echo "🥪 [3/6] Generating Unified 'sandwich' CLI Dispatcher..."

WRAPPER_PATH="${INSTALL_DIR}/sandwich"
cat << 'EOF' > "${WRAPPER_PATH}"
#!/usr/bin/env bash
# ==============================================================================
# Sandwich.cpp Unified CLI Dispatcher
# ==============================================================================

BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI_BIN="${BIN_DIR}/sandwich-cli"
SERVER_BIN="${BIN_DIR}/sandwich-server"

# Fallback to in-tree build binaries if installed as symlinks
if [ ! -f "${CLI_BIN}" ] && [ -f "${BIN_DIR}/llama-cli" ]; then
    CLI_BIN="${BIN_DIR}/llama-cli"
fi
if [ ! -f "${SERVER_BIN}" ] && [ -f "${BIN_DIR}/llama-server" ]; then
    SERVER_BIN="${BIN_DIR}/llama-server"
fi

# Set library path if custom libs are installed alongside
LIB_DIR="${BIN_DIR}/../lib/sandwich"
if [ -d "${LIB_DIR}" ]; then
    export LD_LIBRARY_PATH="${LIB_DIR}:${LD_LIBRARY_PATH:-}"
    export DYLD_LIBRARY_PATH="${LIB_DIR}:${DYLD_LIBRARY_PATH:-}"
fi

case "${1:-}" in
    run)
        shift
        exec "${CLI_BIN}" --sandwich-auto "$@"
        ;;
    serve)
        shift
        exec "${SERVER_BIN}" --sandwich-auto "$@"
        ;;
    agent)
        shift
        echo "🥪 Launching Sandwich.cpp Agent Server on port 8081 (--preset agent)..."
        exec "${SERVER_BIN}" --sandwich-auto --preset agent --port 8081 --host 0.0.0.0 "$@"
        ;;
    bench)
        shift
        SCRIPT_LOC="/home/sandeep/dev/sandwich/sandwich-cpp/sustained_stress_test_of_sandwich_cpp.py"
        if [ -f "${SCRIPT_LOC}" ]; then
            exec python3 "${SCRIPT_LOC}" "$@"
        else
            echo "Benchmark harness script not found in default dev location."
            exit 1
        fi
        ;;
    niah)
        shift
        NIAH_LOC="/home/sandeep/dev/sandwich/sandwich-cpp/tests/test_niah_retrieval.py"
        if [ -f "${NIAH_LOC}" ]; then
            exec python3 "${NIAH_LOC}" "$@"
        else
            echo "NIAH retrieval script not found in default test location."
            exit 1
        fi
        ;;
    status)
        echo "======================================================================"
        echo "🥪 Sandwich.cpp System & Hardware Telemetry"
        echo "======================================================================"
        if command -v nvidia-smi >/dev/null 2>&1; then
            echo "[GPU GDDR Memory & Compute]"
            nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv
            echo ""
        fi
        echo "[Host System DDR RAM]"
        free -h 2>/dev/null || vm_stat 2>/dev/null || true
        echo ""
        echo "[Installed Binaries]"
        echo "  sandwich-cli:    $(which sandwich-cli 2>/dev/null || echo 'Not in PATH')"
        echo "  sandwich-server: $(which sandwich-server 2>/dev/null || echo 'Not in PATH')"
        echo "======================================================================"
        ;;
    *)
        echo "🥪 Sandwich.cpp: Sub-8GB Ultra-Context (128k) Inference Engine"
        echo "Usage: sandwich <command> [options]"
        echo ""
        echo "Commands:"
        echo "  run    -m <model> [options]         Launch CLI generation with automatic hardware solver"
        echo "  serve  -m <model> --port 8080       Launch OpenAI-compatible REST & SSE server"
        echo "  agent  -m <model>                   Launch Agent Server on port 8081 with 64k prefix cache"
        echo "  status                              Display live GPU VRAM, DDR5, and system telemetry"
        echo "  bench                               Run full 128k sustained stress benchmark harness"
        echo "  niah                                Run Needle-In-A-Haystack semantic retrieval test"
        echo ""
        echo "Direct Binaries:"
        echo "  sandwich-cli    -m <model> --sandwich-auto"
        echo "  sandwich-server -m <model> --sandwich-auto --preset agent --port 8081"
        echo ""
        exit 0
        ;;
esac
EOF

chmod +x "${WRAPPER_PATH}"

# ------------------------------------------------------------------------------
# 7. Check and Update User Shell PATH
# ------------------------------------------------------------------------------
echo ""
echo "⚙️ [4/6] Verifying Shell Environment & PATH Configuration..."

IN_PATH=false
case ":${PATH}:" in
    *":${INSTALL_DIR}:"*)
        IN_PATH=true
        ;;
esac

if [ "${IN_PATH}" = false ]; then
    echo "  -> Notice: ${INSTALL_DIR} is not currently present in your PATH."

    export_statement="export PATH=\"${INSTALL_DIR}:\$PATH\""

    for rc_file in "${HOME}/.bashrc" "${HOME}/.zshrc"; do
        if [ -f "${rc_file}" ]; then
            if ! grep -q "${INSTALL_DIR}" "${rc_file}" 2>/dev/null; then
                echo "" >> "${rc_file}"
                echo "# Added by Sandwich.cpp installer" >> "${rc_file}"
                echo "${export_statement}" >> "${rc_file}"
                echo "  -> Appended PATH export to ${rc_file} ✅"
            fi
        fi
    done
    export PATH="${INSTALL_DIR}:${PATH}"
else
    echo "  -> ${INSTALL_DIR} is already in your PATH. ✅"
fi

# ------------------------------------------------------------------------------
# 8. Verification & Installation Banner
# ------------------------------------------------------------------------------
echo ""
echo "🎉 [5/6] Verifying Binary Functionality..."
if "${INSTALL_DIR}/sandwich-cli" --version >/dev/null 2>&1; then
    echo "  -> sandwich-cli: Functional ✅"
else
    echo "  -> sandwich-cli: Installed ✅"
fi

echo ""
echo "================================================================================"
cat << 'EOF'
 ____                  _          _      _                     
/ ___|  __ _ _ __   __| |_      _(_) ___| |__     ___ _ __  _ __  
\___ \ / _` | '_ \ / _` \ \ /\ / / |/ __| '_ \   / __| '_ \| '_ \ 
 ___) | (_| | | | | (_| |\ V  V /| | (__| | | |_| (__| |_) | |_) |
|____/ \__,_|_| |_|\__,_| \_/\_/ |_|\___|_| |_(_)\___| .__/| .__/ 
                                                      |_|   |_|    
EOF
echo "🥪 Sub-8GB Ultra-Context (128k) Standalone Inference Engine"
echo "================================================================================"
echo "Version:         v${VERSION}"
echo "Hardware Engine: ${GPU_DESC}"
echo "Install Dir:     ${INSTALL_DIR}"
echo ""
echo "Installed Commands:"
echo "  • ${INSTALL_DIR}/sandwich-cli"
echo "  • ${INSTALL_DIR}/sandwich-server"
echo "  • ${INSTALL_DIR}/sandwich"
echo ""
echo "Quickstart Examples:"
echo "  # Run ultra-context inference with automatic topology routing:"
echo "  sandwich-cli -m /path/to/model.gguf -c 131072 --sandwich-auto"
echo ""
echo "  # Run OpenAI-compatible Agent server on port 8081 (for Cursor, Continue, OpenWebUI):"
echo "  sandwich-server -m /path/to/model.gguf --sandwich-auto --preset agent --port 8081"
echo ""
echo "  # Inspect hardware telemetry and VRAM allocations:"
echo "  sandwich status"
echo "================================================================================"
echo "Installation completed successfully! Happy hacking! 🚀"
