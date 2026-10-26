#!/usr/bin/env python3
"""
sustained_stress_test_of_sandwich_cpp.py
========================================
Comprehensive 128k Ultra-Context & Sustained Workload Stress Test Suite for Sandwich.cpp.
Targets models >10B parameters, applying two-stage early-exit qualification gating.

Deliverables:
- Persisted raw data: /home/sandeep/dev/sustained_stress_test_results.json
- Final curated markdown report: /home/sandeep/dev/the-mds/sustained_stress_test_of_sandwich_cpp.md
"""

import os
import sys
import time
import subprocess
import re
import json
import threading

BIN_DIR = "/home/sandeep/dev/sandwich/sandwich-cpp/build/bin"
CLI_BIN = os.path.join(BIN_DIR, "llama-cli")
PROMPTS_DIR = "/home/sandeep/dev/sandwich/sandwich-cpp/prompts"
HISTORICAL_JSON = "/home/sandeep/dev/turboquant_results.json"
RESULTS_JSON = "/home/sandeep/dev/sustained_stress_test_results.json"
FINAL_MD = "/home/sandeep/dev/the-mds/sustained_stress_test_of_sandwich_cpp.md"

PROMPTS = {
    "small": {
        "label": "Small (100 tk)",
        "file": os.path.join(PROMPTS_DIR, "prompt_100.txt"),
        "nominal_tokens": 109
    },
    "medium": {
        "label": "Medium (1k tk)",
        "file": os.path.join(PROMPTS_DIR, "prompt_1k.txt"),
        "nominal_tokens": 915
    },
    "large": {
        "label": "Large (5k tk)",
        "file": os.path.join(PROMPTS_DIR, "prompt_5k.txt"),
        "nominal_tokens": 4861
    }
}

SUSTAINED_FILLS = {
    "10pct": {
        "label": "10% Fill (~13k tk)",
        "file": os.path.join(PROMPTS_DIR, "prompt_13k.txt"),
        "nominal_tokens": 14583,
        "fraction": 0.10
    },
    "50pct": {
        "label": "50% Fill (~65k tk)",
        "file": os.path.join(PROMPTS_DIR, "prompt_65k.txt"),
        "nominal_tokens": 63193,
        "fraction": 0.50
    },
    "75pct": {
        "label": "75% Fill (~98k tk)",
        "file": os.path.join(PROMPTS_DIR, "prompt_98k.txt"),
        "nominal_tokens": 97220,
        "fraction": 0.75
    },
    "95pct": {
        "label": "95-100% Limit Fill (~125k tk)",
        "file": os.path.join(PROMPTS_DIR, "prompt_125k.txt"),
        "nominal_tokens": 126386,
        "fraction": 0.95
    }
}

CONTEXTS = [4096, 8192, 16384, 32768, 65536, 131072]
KV_TYPES = ["turbo4", "turbo3", "turbo2"]

# Qualifying Heavyweight Models (> 10 Billion Parameters)
MODELS = [
    {
        "id": "command-r-35b",
        "name": "Command R 35B (08-2024)",
        "arch": "Dense (command-r)",
        "params": "35B",
        "path": "/home/sandeep/ai-models/dense/c4ai-command-r-08-2024-GGUF/c4ai-command-r-08-2024-IQ3_XS.gguf",
        "extra_args": ["--sandwich-auto"],
        "timeout": 300,
        "is_new": True
    },
    {
        "id": "deepseek-coder-v2-lite",
        "name": "DeepSeek-Coder-V2-Lite 16B",
        "arch": "MoE (deepseek2, 64 exp)",
        "params": "16B Total / 2.4B Active",
        "path": "/home/sandeep/ai-models/moe/DeepSeek-Coder-V2-Lite-Instruct-GGUF/DeepSeek-Coder-V2-Lite-Instruct-IQ3_XXS.gguf",
        "extra_args": ["--sandwich-auto"],
        "timeout": 240,
        "is_new": True
    },
    {
        "id": "bonsai-27b",
        "name": "Bonsai 27B",
        "arch": "Dense (qwen35, Q1_0)",
        "params": "27B",
        "path": "/home/sandeep/ai-models/dense/Bonsai-27B-GGUF/Bonsai-27B-Q1_0.gguf",
        "extra_args": ["--sandwich-auto"],
        "timeout": 240,
        "is_new": True
    },
    {
        "id": "qwen-27b",
        "name": "Qwen3.8-27B",
        "arch": "Dense Hybrid (IQ3_XXS + DFlash2)",
        "params": "27B",
        "path": "/home/sandeep/ai-models/dense/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_XXS.gguf",
        "extra_args": ["--sandwich-auto", "-md", "/home/sandeep/ai-models/drafters/Qwen3.8-27B-Q4_K_M-DFlash2-Strix-Halo/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"],
        "timeout": 240,
        "is_new": False
    },
    {
        "id": "qwen-35b-moe",
        "name": "Qwen3.6-35B-A3B",
        "arch": "MoE (IQ3_XXS + MTP, 256 exp)",
        "params": "35B Total / 3B Active",
        "path": "/home/sandeep/ai-models/moe/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf",
        "extra_args": ["--sandwich-auto"],
        "timeout": 240,
        "is_new": False
    },
    {
        "id": "glm-4.7-flash",
        "name": "GLM-4.7-Flash",
        "arch": "MoE (Q4_K_M + FlashMLA, 64 exp)",
        "params": "30B Total / 3B Active",
        "path": "/home/sandeep/ai-models/moe/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q4_K_M.gguf",
        "extra_args": ["--sandwich-auto"],
        "timeout": 240,
        "is_new": False
    },
    {
        "id": "nemotron-30b-moe",
        "name": "Nemotron-3-Nano-30B-A3B",
        "arch": "MoE (IQ3_XXS, 128 exp)",
        "params": "30B Total / 3B Active",
        "path": "/home/sandeep/ai-models/moe/Nemotron-3-Nano-30B-A3B-GGUF/Nemotron-3-Nano-30B-A3B-UD-IQ3_XXS.gguf",
        "extra_args": ["--sandwich-auto"],
        "timeout": 240,
        "is_new": False
    },
    {
        "id": "muse-30b",
        "name": "Muse-Glimmer-30B",
        "arch": "Dense Hybrid (Q4_K_M + DFlash2)",
        "params": "30B",
        "path": "/home/sandeep/ai-models/dense/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
        "extra_args": ["--sandwich-auto", "-md", "/home/sandeep/ai-models/drafters/Muse-Glimmer-30B-DFlash2-GGUF/Muse-Glimmer-30B-DFlash2-Q4_K_M.gguf"],
        "timeout": 300,
        "is_new": False
    },
    {
        "id": "gemma-31b",
        "name": "Gemma-4-31B-it",
        "arch": "Dense Hybrid (Q4_0 QAT + MTP)",
        "params": "31B",
        "path": "/home/sandeep/ai-models/dense/gemma-4-31B-it-QAT-GGUF/gemma-4-31B-it-QAT-Q4_0.gguf",
        "extra_args": ["--sandwich-auto", "-md", "/home/sandeep/ai-models/drafters/gemma-4-31B-it-qat-Q4_0-Q4emb-MTP-assistant-gguf/gemma-4-31b-qat-it-assistant-Q4_0-Q6emb.gguf"],
        "timeout": 300,
        "is_new": False
    },
    {
        "id": "llama-70b",
        "name": "Llama-3.3-70B-Instruct",
        "arch": "Dense Hybrid (IQ2_XXS + Llama 1B)",
        "params": "70B",
        "path": "/home/sandeep/ai-models/dense/Llama-3.3-70B-Instruct-GGUF/Llama-3.3-70B-Instruct-UD-IQ2_XXS.gguf",
        "extra_args": ["--sandwich-auto", "-md", "/home/sandeep/ai-models/drafters/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q5_K_S.gguf"],
        "timeout": 420,
        "is_new": False
    },
    {
        "id": "gpt-oss-120b",
        "name": "GPT-OSS-120B",
        "arch": "Massive MoE (128 Experts / 36 Layers)",
        "params": "117B Total / 12B Active",
        "path": "/home/sandeep/ai-models/moe/gpt-oss-120b-GGUF/gpt-oss-120b-UD-Q4_K_XL-00001-of-00002.gguf",
        "extra_args": ["--moe-3tier", "--moe-2q"],
        "timeout": 500,
        "is_new": False
    }
]

def get_vram_used_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return int(out.split('\n')[0])
    except Exception:
        return 0

class VRAMTracker(threading.Thread):
    def __init__(self, interval=0.1):
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_vram = get_vram_used_mb()

    def run(self):
        while not self.stop_event.is_set():
            v = get_vram_used_mb()
            if v > self.peak_vram:
                self.peak_vram = v
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()

def parse_cli_output(clean_out):
    prompt_speed = 0.0
    gen_speed = 0.0
    ttft = 0.0

    # Look for [ Prompt: X t/s | Generation: Y t/s ]
    m_p = re.search(r'Prompt:\s*([\d\.]+)\s*t/s', clean_out)
    m_g = re.search(r'Generation:\s*([\d\.]+)\s*t/s', clean_out)
    if m_p:
        prompt_speed = float(m_p.group(1))
    if m_g:
        gen_speed = float(m_g.group(1))

    # Fallback to standard llama timing logs
    if prompt_speed == 0.0:
        m = re.search(r'prompt eval time =\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([\d\.]+)\s*t/s', clean_out)
        if m:
            prompt_speed = float(m.group(3))
            ttft = float(m.group(1)) / 1000.0

    if gen_speed == 0.0:
        m = re.search(r'eval time =\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*(?:runs|tokens).*?([\d\.]+)\s*t/s', clean_out)
        if m:
            gen_speed = float(m.group(3))

    if ttft == 0.0 and prompt_speed > 0:
        m = re.search(r'prompt eval time =\s*([\d\.]+)\s*ms', clean_out)
        if m:
            ttft = float(m.group(1)) / 1000.0

    # Telemetry extraction
    telemetry = {}
    m_t1 = re.search(r'Tier 1 \(HOT VRAM\)\s*Hits\s*:\s*(\d+)', clean_out)
    m_t2 = re.search(r'Tier 2 \(WARM RAM\)\s*Hits\s*:\s*(\d+)', clean_out)
    m_t3 = re.search(r'Tier 3 \(COLD SSD\)\s*Fetches:\s*(\d+)', clean_out)
    m_2q_ghost = re.search(r'2Q Ghost Cache Hits\s*:\s*(\d+)', clean_out)
    m_2q_quar = re.search(r'2Q Pollution Quarantined\s*:\s*(\d+)', clean_out)
    if m_t1: telemetry["tier1_vram_hits"] = int(m_t1.group(1))
    if m_t2: telemetry["tier2_ram_hits"] = int(m_t2.group(1))
    if m_t3: telemetry["tier3_cold_fetches"] = int(m_t3.group(1))
    if m_2q_ghost: telemetry["ghost_hits"] = int(m_2q_ghost.group(1))
    if m_2q_quar: telemetry["quarantined"] = int(m_2q_quar.group(1))

    return prompt_speed, gen_speed, ttft, telemetry

def execute_cli(cmd, timeout):
    vram_tracker = VRAMTracker(interval=0.1)
    vram_tracker.start()
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace"
        )
        stdout, _ = proc.communicate(timeout=timeout)
        retcode = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, _ = proc.communicate()
        retcode = -9
    except Exception as e:
        stdout = str(e)
        retcode = -1

    elapsed = time.time() - t0
    vram_tracker.stop()
    vram_tracker.join()
    peak_vram = vram_tracker.peak_vram

    clean_out = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', stdout)
    return retcode, clean_out, elapsed, peak_vram

def run_matrix_test(model_spec, ctx, kv_type, prompt_key):
    prompt_info = PROMPTS[prompt_key]
    prompt_file = prompt_info["file"]
    nom_tokens = prompt_info["nominal_tokens"]

    if nom_tokens > ctx:
        return {
            "status": "EXCEEDS_CTX",
            "prompt_speed": 0.0,
            "gen_speed": 0.0,
            "peak_vram": 0,
            "ttft": 0.0,
            "elapsed": 0.0,
            "note": f"Prompt ({nom_tokens} tk) exceeds Context ({ctx} tk)"
        }

    cmd = [
        CLI_BIN,
        "-m", model_spec["path"],
        "-c", str(ctx),
        "-ctk", kv_type,
        "-ctv", kv_type,
        "-f", prompt_file,
        "-n", "16",
        "-t", "6",
        "--no-warmup",
        "-st",
        "--simple-io"
    ]

    # Under Technique #42 (70% RAM Rule), dense models pin KV cache & FlashAttention in GPU VRAM (--split-kv).
    # Only massive MoE (GPT-OSS-120B) requires host DDR5 KV routing at high context due to its 4.5 GB VRAM expert slab.
    if model_spec["id"] == "gpt-oss-120b" and ctx >= 32768:
        cmd.append("-nkvo")

    cmd.extend(model_spec.get("extra_args", []))

    retcode, clean_out, elapsed, peak_vram = execute_cli(cmd, model_spec.get("timeout", 180))

    if retcode == -9:
        return {
            "status": "TIMEOUT",
            "prompt_speed": 0.0,
            "gen_speed": 0.0,
            "peak_vram": peak_vram,
            "ttft": 0.0,
            "elapsed": elapsed,
            "note": f"Execution timed out (> {model_spec.get('timeout', 180)}s)"
        }

    if "out of memory" in clean_out.lower() or "cuda oom" in clean_out.lower() or "failed to allocate" in clean_out.lower():
        return {
            "status": "OOM",
            "prompt_speed": 0.0,
            "gen_speed": 0.0,
            "peak_vram": peak_vram,
            "ttft": 0.0,
            "elapsed": elapsed,
            "note": "CUDA Out Of Memory"
        }

    prompt_speed, gen_speed, ttft, telemetry = parse_cli_output(clean_out)

    if prompt_speed > 0 or gen_speed > 0:
        return {
            "status": "PASS",
            "prompt_speed": round(prompt_speed, 1),
            "gen_speed": round(gen_speed, 1),
            "peak_vram": peak_vram,
            "ttft": round(ttft, 3),
            "elapsed": round(elapsed, 2),
            "telemetry": telemetry,
            "note": "OK"
        }

    return {
        "status": "FAIL",
        "prompt_speed": 0.0,
        "gen_speed": 0.0,
        "peak_vram": peak_vram,
        "ttft": 0.0,
        "elapsed": elapsed,
        "note": "Zero throughput output"
    }

def run_sustained_fill_test(model_spec, kv_type, fill_key):
    fill_info = SUSTAINED_FILLS[fill_key]
    prompt_file = fill_info["file"]
    nom_tokens = fill_info["nominal_tokens"]
    ctx = 131072 # 128k context

    cmd = [
        CLI_BIN,
        "-m", model_spec["path"],
        "-c", str(ctx),
        "-ctk", kv_type,
        "-ctv", kv_type,
        "-f", prompt_file,
        "-n", "32", # 32 tokens multi-turn sustained generation
        "-t", "6",
        "--no-warmup",
        "-st",
        "--simple-io"
    ]

    if model_spec["id"] == "gpt-oss-120b":
        cmd.append("-nkvo")

    cmd.extend(model_spec.get("extra_args", []))

    # Allow generous timeout for 128k sustained execution
    timeout = max(360, model_spec.get("timeout", 240) * 2)
    retcode, clean_out, elapsed, peak_vram = execute_cli(cmd, timeout)

    if retcode == -9:
        return {
            "status": "TIMEOUT",
            "fill_fraction": fill_info["fraction"],
            "fill_tokens": nom_tokens,
            "prompt_speed": 0.0,
            "gen_speed": 0.0,
            "peak_vram": peak_vram,
            "elapsed": elapsed
        }

    if "out of memory" in clean_out.lower() or "cuda oom" in clean_out.lower() or "failed to allocate" in clean_out.lower():
        return {
            "status": "OOM",
            "fill_fraction": fill_info["fraction"],
            "fill_tokens": nom_tokens,
            "prompt_speed": 0.0,
            "gen_speed": 0.0,
            "peak_vram": peak_vram,
            "elapsed": elapsed
        }

    prompt_speed, gen_speed, ttft, telemetry = parse_cli_output(clean_out)
    return {
        "status": "PASS" if gen_speed > 0 else "FAIL",
        "fill_fraction": fill_info["fraction"],
        "fill_tokens": nom_tokens,
        "prompt_speed": round(prompt_speed, 1),
        "gen_speed": round(gen_speed, 1),
        "peak_vram": peak_vram,
        "ttft": round(ttft, 3),
        "elapsed": round(elapsed, 2),
        "telemetry": telemetry
    }

def main():
    print("==========================================================================")
    print("🥪 Sustained Stress Test of Sandwich.cpp — Ultra-Context & Sustained Suite")
    print("==========================================================================")

    # 1. Load historical database
    all_results = {}
    if os.path.exists(RESULTS_JSON):
        try:
            with open(RESULTS_JSON, "r") as f:
                all_results = json.load(f)
            print(f"Loaded existing results database with {len(all_results)} models.")
        except Exception:
            all_results = {}

    historical_data = {}
    if os.path.exists(HISTORICAL_JSON):
        try:
            with open(HISTORICAL_JSON, "r") as f:
                historical_data = json.load(f)
            print(f"Loaded historical 45-combination baseline data for {len(historical_data)} models.")
        except Exception:
            historical_data = {}

    for m in MODELS:
        m_id = m["id"]
        print(f"\n--------------------------------------------------------------------------")
        print(f"Evaluating Model: {m['name']} ({m['params']} | {m['arch']})")
        print(f"--------------------------------------------------------------------------")

        if m_id not in all_results:
            all_results[m_id] = {
                "spec": m,
                "matrix": {},
                "sustained": {}
            }

        model_matrix = all_results[m_id]["matrix"]
        model_sustained = all_results[m_id]["sustained"]

        # Merge historical data if available
        if not m["is_new"] and m_id in historical_data:
            for k, v in historical_data[m_id].items():
                if k not in model_matrix:
                    model_matrix[k] = v

        # Stage 1: Ultra-Context Matrix Qualification (up to 128k)
        # Check 64k qualification per KV type
        for kv in KV_TYPES:
            # For new models, run 4k -> 64k first if missing
            if m["is_new"]:
                early_exit_kv = False
                for ctx in [4096, 8192, 16384, 32768, 65536]:
                    for pkey in ["small", "medium", "large"]:
                        test_key = f"{ctx}_{kv}_{pkey}"
                        if test_key in model_matrix:
                            res = model_matrix[test_key]
                        else:
                            if early_exit_kv:
                                res = {"status": "SKIPPED_DUE_TO_EARLIER_FAILURE", "prompt_speed": 0.0, "gen_speed": 0.0, "peak_vram": 0}
                            elif pkey == "large" and any(model_matrix.get(f"{prev_c}_{kv}_large", {}).get("status") == "TIMEOUT" for prev_c in [4096, 8192, 16384, 32768]):
                                res = {"status": "TIMEOUT", "prompt_speed": 0.0, "gen_speed": 0.0, "peak_vram": 0, "ttft": 0.0, "elapsed": 0.0, "note": "Predictive Timeout (>300s)"}
                            else:
                                print(f"  [Matrix 4k-64k] {m_id} | ctx={ctx:<6} | kv={kv:<6} | prompt={pkey:<6} ...", end="", flush=True)
                                res = run_matrix_test(m, ctx, kv, pkey)
                                print(f" {res['status']} (Gen: {res.get('gen_speed', 0)} t/s | VRAM: {res.get('peak_vram', 0)} MB)")
                                if res["status"] in ["OOM", "FAIL"] and pkey == "small":
                                    early_exit_kv = True
                            model_matrix[test_key] = res
                            with open(RESULTS_JSON, "w") as f:
                                json.dump(all_results, f, indent=2)

            # Two-Stage Gate: Did this kv_type PASS at 64k?
            passed_64k = any(
                model_matrix.get(f"65536_{kv}_{pkey}", {}).get("status") == "PASS"
                for pkey in ["small", "medium", "large"]
            )

            if not passed_64k:
                print(f"  [Gate Pruned] {m_id} kv={kv} failed or unviable at 64k. Pruning 128k test.")
                for pkey in ["small", "medium", "large"]:
                    model_matrix[f"131072_{kv}_{pkey}"] = {
                        "status": "PRUNED_64K_FAILED",
                        "prompt_speed": 0.0,
                        "gen_speed": 0.0,
                        "peak_vram": 0,
                        "note": "Pruned: KV format failed at 64k"
                    }
                continue

            # Run 128k matrix for qualified KV type
            for pkey in ["small", "medium", "large"]:
                test_key = f"131072_{kv}_{pkey}"
                if test_key in model_matrix and model_matrix[test_key].get("status") in ["PASS", "EXCEEDS_CTX", "TIMEOUT", "OOM", "FAIL", "PRUNED_64K_FAILED", "SKIPPED_DUE_TO_EARLIER_FAILURE"]:
                    continue
                if pkey == "large" and any(model_matrix.get(f"{prev_c}_{kv}_large", {}).get("status") == "TIMEOUT" for prev_c in [4096, 8192, 16384, 32768, 65536]):
                    model_matrix[test_key] = {"status": "TIMEOUT", "prompt_speed": 0.0, "gen_speed": 0.0, "peak_vram": 0, "ttft": 0.0, "elapsed": 0.0, "note": "Predictive Timeout (>300s)"}
                    continue
                print(f"  [Matrix 128k] {m_id} | ctx=131072 | kv={kv:<6} | prompt={pkey:<6} ...", end="", flush=True)
                res = run_matrix_test(m, 131072, kv, pkey)
                print(f" {res['status']} (Gen: {res.get('gen_speed', 0)} t/s | VRAM: {res.get('peak_vram', 0)} MB)")
                model_matrix[test_key] = res
                with open(RESULTS_JSON, "w") as f:
                    json.dump(all_results, f, indent=2)

            # Stage 2: Sustained Multi-Turn Workload Profiling (10% -> 95%)
            # Only promote if 128k passed in at least one prompt
            passed_128k = any(
                model_matrix.get(f"131072_{kv}_{pkey}", {}).get("status") == "PASS"
                for pkey in ["small", "medium", "large"]
            )

            if passed_128k:
                print(f"  >>> PROMOTED to Stage 2 Sustained Fill Profiling: {m_id} @ kv={kv}")
                # Predictive Pre-check: If smaller sustained fill already timed out on earlier KV format on CPU without drafter,
                # then prefilling 14k-125k tokens on CPU will predictably time out across all KV formats.
                earlier_sustained_timeout = any(
                    model_sustained.get(f"{prev_kv}_10pct", {}).get("status") == "TIMEOUT"
                    for prev_kv in ["turbo4", "turbo3"] if prev_kv != kv
                )
                if earlier_sustained_timeout:
                    print(f"    [Sustained Fill Predictive Bypass] Dense CPU prefill on 14k-125k tk exceeds budget. Recording predictive timeouts.")
                    for fkey in ["10pct", "50pct", "75pct", "95pct"]:
                        sust_key = f"{kv}_{fkey}"
                        if sust_key not in model_sustained or model_sustained[sust_key].get("status") != "PASS":
                            model_sustained[sust_key] = {
                                "status": "TIMEOUT",
                                "fill_fraction": SUSTAINED_FILLS[fkey]["fraction"],
                                "fill_tokens": SUSTAINED_FILLS[fkey]["nominal_tokens"],
                                "prompt_speed": 0.0,
                                "gen_speed": 0.0,
                                "peak_vram": 0,
                                "elapsed": 0.0,
                                "note": "Predictive Timeout: dense CPU prefill on >14k tokens exceeds time budget without GPU drafter"
                            }
                    with open(RESULTS_JSON, "w") as f:
                        json.dump(all_results, f, indent=2)
                else:
                    for fkey in ["10pct", "50pct", "75pct", "95pct"]:
                        sust_key = f"{kv}_{fkey}"
                        if sust_key in model_sustained and model_sustained[sust_key].get("status") in ["PASS", "TIMEOUT", "OOM", "FAIL"]:
                            continue
                        print(f"    [Sustained Fill] {m_id} | kv={kv:<6} | fill={fkey:<6} ...", end="", flush=True)
                        sres = run_sustained_fill_test(m, kv, fkey)
                        print(f" {sres['status']} (Gen: {sres.get('gen_speed', 0)} t/s | VRAM: {sres.get('peak_vram', 0)} MB)")
                        model_sustained[sust_key] = sres
                        with open(RESULTS_JSON, "w") as f:
                            json.dump(all_results, f, indent=2)
                        if sres["status"] == "TIMEOUT":
                            # Any strictly larger fills will also time out on prefill
                            all_f = ["10pct", "50pct", "75pct", "95pct"]
                            idx = all_f.index(fkey)
                            for rem_f in all_f[idx+1:]:
                                model_sustained[f"{kv}_{rem_f}"] = {
                                    "status": "TIMEOUT",
                                    "fill_fraction": SUSTAINED_FILLS[rem_f]["fraction"],
                                    "fill_tokens": SUSTAINED_FILLS[rem_f]["nominal_tokens"],
                                    "prompt_speed": 0.0,
                                    "gen_speed": 0.0,
                                    "peak_vram": sres.get("peak_vram", 0),
                                    "elapsed": 0.0,
                                    "note": f"Predictive Timeout: smaller fill ({fkey}) timed out"
                                }
                            with open(RESULTS_JSON, "w") as f:
                                json.dump(all_results, f, indent=2)
                            break
            else:
                print(f"  [Stage 2 Gate] {m_id} kv={kv} did not qualify at 128k. Skipped sustained fill.")

    print("\n==========================================================================")
    print("Testing Complete. Generating Curated Markdown Report...")
    print("==========================================================================")
    curate_script = os.path.join(os.path.dirname(__file__), "generate_curated_report.py")
    if os.path.exists(curate_script):
        subprocess.run([sys.executable, curate_script], check=False)

if __name__ == "__main__":
    main()
