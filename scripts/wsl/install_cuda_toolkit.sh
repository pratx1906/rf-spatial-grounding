#!/usr/bin/env bash
# Install the CUDA toolkit (nvcc + headers) inside WSL2 Ubuntu-24.04. Run as root:
#   wsl -d Ubuntu-24.04 -u root -- bash scripts/wsl/install_cuda_toolkit.sh
# vLLM / FlashInfer JIT-compile kernels at runtime and need nvcc. Uses NVIDIA's WSL-Ubuntu repo, which ships
# the toolkit WITHOUT a Linux GPU driver (WSL uses the Windows driver; never install a driver inside WSL).
# Version matches torch's CUDA build (torch 2.13.0+cu130 -> 13.0).
set -euo pipefail
CUDA_MM="${CUDA_MM:-13-0}"
# Default: only what vLLM / FlashInfer JIT compilation needs (compiler, headers, core dev libs), ~1/3 of the full
# toolkit download. CUDA_FULL=1 installs the whole toolkit (Nsight, samples, all libs).
PKGS=(cuda-nvcc cuda-cudart-dev cuda-cccl cuda-crt cuda-nvrtc-dev cuda-driver-dev cuda-profiler-api
      libcublas-dev libcurand-dev libcusparse-dev libcusolver-dev)

export DEBIAN_FRONTEND=noninteractive
if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
    tmp=$(mktemp -d)
    curl -fsSL -o "$tmp/cuda-keyring.deb" \
        https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i "$tmp/cuda-keyring.deb"
fi
apt-get update -qq
if [ "${CUDA_FULL:-0}" = "1" ]; then
    apt-get install -y -qq "cuda-toolkit-${CUDA_MM}"
else
    apt-get install -y -qq "${PKGS[@]/%/-${CUDA_MM}}"
fi

ln -sfn "/usr/local/cuda-${CUDA_MM/-/.}" /usr/local/cuda
/usr/local/cuda/bin/nvcc --version | tail -2
