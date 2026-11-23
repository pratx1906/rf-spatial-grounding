#!/usr/bin/env bash
# Create the Linux-side Python environment for the agent stack (run inside WSL2 as the normal user).
# Everything project-specific stays inside the repo folder:
#   .venv-wsl/          Linux venv (separate from any Windows venv)
#   .cache/huggingface  model downloads
#   data/               site products and run outputs
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${AKSHA_VENV:-$REPO/.venv-wsl}"
export PATH="$HOME/.local/bin:$PATH"
export UV_LINK_MODE=copy            # uv cache (ext4) and venv (/mnt/c) are on different filesystems
export HF_HOME="$REPO/.cache/huggingface"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p "$REPO/data" "$HF_HOME"

[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

LOCK="$REPO/requirements/wsl.lock"
if [ -f "$LOCK" ] && [ "${AKSHA_UNLOCKED:-0}" != "1" ]; then
    # Reproducible install of the exact versions verified in docs/ENV_REPORT.md (SPEC 0.8, 14).
    uv pip install -r "$LOCK"
else
    # Fresh resolve (AKSHA_UNLOCKED=1): vLLM pins a matching torch build (CUDA 12.8+ is needed for sm_120).
    # Re-verify with `make env-check`, then refresh the lock: uv pip freeze > requirements/wsl.lock
    uv pip install vllm
    uv pip install peft trl datasets accelerate bitsandbytes \
        numpy shapely duckdb zarr pydantic pyyaml pyarrow psutil nvidia-ml-py pytest
fi

python - <<'EOF'
import torch, vllm
print("torch", torch.__version__, "cuda", torch.version.cuda, "vllm", vllm.__version__)
EOF
