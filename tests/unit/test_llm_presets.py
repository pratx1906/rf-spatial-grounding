"""serve.llm presets for the P2 size ladder: the default command is unchanged; other models get their parser/template."""

import json

import pytest

pytest.importorskip("pynvml")


def _old_default(util):
    # the vLLM command serve.llm built before the ladder presets (default flags)
    return ["vllm", "serve", "Qwen/Qwen3-4B-Instruct-2507", "--port", "8001", "--host", "127.0.0.1",
            "--served-model-name", "scenescout", "--dtype", "bfloat16", "--max-model-len", "15360",
            "--gpu-memory-utilization", str(util), "--enable-prefix-caching",
            "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--generation-config", "vllm"]


def test_default_command_unchanged():
    from serve.llm import build_cmd, parse_args

    assert build_cmd(parse_args([]), 0.7) == _old_default(0.7)
    a = parse_args(["--lora", "sft=checkpoints/sft_v11"])
    assert build_cmd(a, 0.7) == _old_default(0.7) + ["--enable-lora", "--max-lora-rank", "64", "--lora-modules",
                                                     "sft=checkpoints/sft_v11"]


def test_presets():
    from serve.llm import build_cmd, parse_args, preset_for

    assert preset_for("Qwen/Qwen3-4B-Instruct-2507")["chat_template_kwargs"] is None
    assert preset_for("stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ")["chat_template_kwargs"] is None
    assert preset_for("Qwen/Qwen3-8B")["chat_template_kwargs"] == {"enable_thinking": False}
    snap = "/repo/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    assert preset_for(snap)["chat_template_kwargs"] == {"enable_thinking": False}
    assert preset_for("/c/hub/models--openai--gpt-oss-20b/snapshots/6cee5e8")["tool_call_parser"] == "openai"
    assert preset_for("/c/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/abc")["chat_template_kwargs"] is None
    g = preset_for("openai/gpt-oss-20b")
    assert g["tool_call_parser"] == "openai" and g["reasoning_parser"] == "openai_gptoss"

    cmd = build_cmd(parse_args(["--model", "Qwen/Qwen3-8B"]), 0.8)
    assert json.loads(cmd[cmd.index("--default-chat-template-kwargs") + 1]) == {"enable_thinking": False}
    assert cmd[cmd.index("--tool-call-parser") + 1] == "hermes" and "--reasoning-parser" not in cmd

    cmd = build_cmd(parse_args(["--model", "openai/gpt-oss-20b", "--max-model-len", "20480"]), 0.8)
    assert cmd[cmd.index("--tool-call-parser") + 1] == "openai"
    assert cmd[cmd.index("--reasoning-parser") + 1] == "openai_gptoss"
    assert "--default-chat-template-kwargs" not in cmd and cmd[cmd.index("--max-model-len") + 1] == "20480"


def test_overrides():
    from serve.llm import build_cmd, parse_args

    a = parse_args(["--model", "x/m", "--quantization", "compressed-tensors", "--tool-call-parser", "qwen3_xml",
                    "--gpu-memory-utilization", "0.6", "--chat-template-kwargs", '{"enable_thinking": false}',
                    "--dtype", "float16"])
    assert a.gpu_util == 0.6
    cmd = build_cmd(a, a.gpu_util)
    assert cmd[cmd.index("--quantization") + 1] == "compressed-tensors"
    assert cmd[cmd.index("--tool-call-parser") + 1] == "qwen3_xml" and cmd[cmd.index("--dtype") + 1] == "float16"
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.6"
    assert "--default-chat-template-kwargs" in cmd
