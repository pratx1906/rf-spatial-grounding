#!/usr/bin/env bash
# Start the demo: model server (GPU, base Qwen3-4B + the SceneScout SFT v1.1 adapter) + gateway/UI with the live
# Blender world. Then open http://localhost:8080 in a Windows browser (Chrome or Edge).
#   bash scripts/wsl/demo.sh          (from the repo root, inside WSL)   or   make demo
# Logs: runs/serve_llm.log, runs/gateway.log. Stop: bash scripts/wsl/demo.sh stop
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=.venv-wsl/bin/python
export PATH="/usr/local/cuda/bin:$PATH"

if [ "${1:-}" = "stop" ]; then
    pkill -f "serve.gateway"; pkill -f "serve.llm"; pkill -f "vllm serve"; echo "stopped"; exit 0
fi

[ -f checkpoints/sft_v11/adapter_model.safetensors ] || $PY scripts/bigfiles.py join   # fresh clone: rebuild >100 MB files
[ -f data/stub_site/meta.yaml ] || $PY -m sim.stub_site
[ -f data/demo_tasks.json ] || $PY scripts/make_demo_tasks.py

if curl -s -m 2 http://127.0.0.1:8001/v1/models | grep -q scenescout; then
    echo "model server already running"
else
    echo "starting model server (about 3 minutes)..."
    nohup $PY -m serve.llm --lora sft=checkpoints/sft_v11 --spec-ngram 4 > runs/serve_llm.log 2>&1 &
    for _ in $(seq 1 300); do
        curl -s -m 2 http://127.0.0.1:8001/v1/models | grep -q '"sft"' && break
        if ! pgrep -f "serve.llm" > /dev/null; then echo "model server failed; see runs/serve_llm.log"; exit 1; fi
        sleep 2
    done
fi

pkill -f "serve.gateway" 2>/dev/null
SCENESCOUT_SERVED_MODEL=sft nohup $PY -m serve.gateway --port 8080 > runs/gateway.log 2>&1 &
for _ in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8080/api/health | grep -q ok && break; sleep 1; done
echo "SceneScout is up: open http://localhost:8080"
