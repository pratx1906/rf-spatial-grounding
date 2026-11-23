"""Process environment for GPU scripts. Import before torch/vLLM.

- HF_HOME: model cache inside the project folder (D-007).
- CUDA_HOME / PATH: vLLM and FlashInfer JIT-compile kernels and need nvcc (installed by
  scripts/wsl/install_cuda_toolkit.sh at /usr/local/cuda). WSL's default PATH does not include it.
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

os.environ.setdefault("HF_HOME", str(REPO / ".cache" / "huggingface"))

# FlashInfer JIT-builds kernels with ninja, which cannot handle spaces in paths, and this venv lives under
# ".../Pratyush Dixit/..." (D-015). vLLM uses prebuilt FLASH_ATTN for attention; only its FlashInfer sampler needs
# JIT, so fall back to the PyTorch sampler. Remove once the venv path has no spaces.
if " " in sys.prefix:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

_cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
if (_cuda / "bin" / "nvcc").exists():
    os.environ["CUDA_HOME"] = str(_cuda)
    if str(_cuda / "bin") not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = f"{_cuda / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
