# Run inside WSL from the repo root. GPU-heavy targets take .gpu_lock themselves.
SHELL := /bin/bash
VENV  ?= $(or $(AKSHA_VENV),.venv-wsl)
PY    := $(VENV)/bin/python
export HF_HOME := $(CURDIR)/.cache/huggingface
export CUDA_HOME ?= /usr/local/cuda
# venv bin first (tests and subprocesses find `python`), then the CUDA toolkit (nvcc for vLLM/FlashInfer JIT)
# (no $(abspath): make splits on spaces and the repo path contains one)
VENV_BIN := $(if $(filter /%,$(VENV)),$(VENV),$(CURDIR)/$(VENV))/bin
export PATH := $(VENV_BIN):$(CUDA_HOME)/bin:$(PATH)

.PHONY: setup cuda env-check bench-m0 bench rl-dryrun demo demo-stop demo-check stub smoke budget test lock monitor

setup:            ## create the Linux venv from the lockfile and install the stack
	bash scripts/wsl/setup_env.sh

cuda:             ## install the CUDA toolkit inside WSL (needs root)
	sudo bash scripts/wsl/install_cuda_toolkit.sh

env-check:        ## M0 smoke test -> docs/ENV_REPORT.md
	$(PY) scripts/env_check.py

bench-m0:         ## M0 measurements (tokens/s for the 4B model) -> runs/m0/measurements.json
	$(PY) scripts/bench_m0.py

bench: bench-m0   ## SPEC 15.4 name; the latency/throughput benches (bench/) arrive in M6

demo:             ## start model server + UI, then open http://localhost:8080
	bash scripts/wsl/demo.sh

demo-stop:        ## stop the demo servers
	bash scripts/wsl/demo.sh stop

demo-check:       ## rehearse every demo question with the live model and score it
	$(PY) scripts/demo_check.py --repeats 2

stub:             ## (re)generate data/stub_site and the demo task list
	$(PY) -m sim.stub_site && $(PY) scripts/make_demo_tasks.py

rl-dryrun:        ## M0: 2-step GRPO dry run (TRL, 4B + LoRA, colocated vLLM, toy tool env)
	$(PY) scripts/rl_dryrun_trl.py --steps 2

smoke: env-check  ## SPEC 15.4: env check now; + 3 scripted episodes on stub_site once M1 exists

budget:           ## disk estimate for configured precompute jobs
	$(PY) scripts/disk_budget.py

test:             ## unit tests (more suites are added per milestone)
	$(PY) -m pytest -q

lock:             ## show who holds the GPU
	$(PY) scripts/gpu_lock.py status

monitor:          ## log GPU/RAM every 5 s until Ctrl-C
	$(PY) scripts/gpu_monitor.py --out runs/monitor_$$(date +%Y%m%d_%H%M%S)
