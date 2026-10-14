#!/usr/bin/env python3
import subprocess
import time
import re
import os
import sys
import json
import threading

MODELS = [
    # MoE Models
    {
        "name": "LFM2-8B-A1B (Quantum MoE + Recurrent)",
        "type": "MoE",
        "path": "/home/sandeep/ai-models/moe/LFM2-8B-A1B-Instruct-Quantum-IQ1C-Qwen3.6-35B-A3B-DISTILL-i1-GGUF/LFM2-8B-A1B-Instruct-Quantum-IQ1C-Qwen3.6-35B-A3B-DISTILL.i1-IQ2_XXS.gguf",
        "tokens": 48
    },
    {
        "name": "GLM-4.7-Flash (MoE + FlashMLA)",
        "type": "MoE",
        "path": "/home/sandeep/ai-models/moe/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q4_K_M.gguf",
        "tokens": 48
    },
    {
        "name": "Qwen3.6-35B-A3B (MoE)",
        "type": "MoE",
        "path": "/home/sandeep/ai-models/moe/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf",
        "tokens": 48
    },
    {
        "name": "Nemotron-3-Nano-30B-A3B (MoE)",
        "type": "MoE",
        "path": "/home/sandeep/ai-models/moe/Nemotron-3-Nano-30B-A3B-GGUF/Nemotron-3-Nano-30B-A3B-UD-IQ3_XXS.gguf",
        "tokens": 48
    },
    # Dense Models
    {
        "name": "NVIDIA-Nemotron-3-Nano-4B",
        "type": "Dense",
        "path": "/home/sandeep/ai-models/dense/NVIDIA-Nemotron-3-Nano-4B-GGUF/NVIDIA-Nemotron-3-Nano-4B-Q6_K.gguf",
        "tokens": 48
    },
    {
        "name": "Dolphin-2.8-Mistral-7B",
        "type": "Dense",
        "path": "/home/sandeep/ai-models/dense/dolphin-2.8-mistral-7b-v02-GGUF/dolphin-2.8-mistral-7b-v02-Q4_K_S.gguf",
        "tokens": 48
    },
    {
        "name": "Qwen3.8-27B (IQ3_XXS + DFlash2)",
        "type": "Dense",
        "path": "/home/sandeep/ai-models/dense/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_XXS.gguf",
        "tokens": 48
    },
    {
        "name": "Muse-Glimmer-30B (Q4_K_M + DFlash2)",
        "type": "Dense",
        "path": "/home/sandeep/ai-models/dense/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
        "tokens": 48
    },
    {
        "name": "Gemma-4-31B-it (Q4_0 + MTP Assistant)",
        "type": "Dense",
        "path": "/home/sandeep/ai-models/dense/gemma-4-31B-it-QAT-GGUF/gemma-4-31B-it-QAT-Q4_0.gguf",
        "tokens": 48
    },
    {
        "name": "Llama-3.3-70B-Instruct (IQ2_XXS + Llama-3.2-1B)",
        "type": "Dense",
        "path": "/home/sandeep/ai-models/dense/Llama-3.3-70B-Instruct-GGUF/Llama-3.3-70B-Instruct-UD-IQ2_XXS.gguf",
        "tokens": 48
    },
    # Standalone Drafter Model
    {
        "name": "Llama-3.2-1B-Instruct (Q5_K_S)",
        "type": "Dense Drafter",
        "path": "/home/sandeep/ai-models/drafters/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q5_K_S.gguf",
        "tokens": 48
    },
    # Embedding Model
    {
        "name": "Nomic-Embed-Text-v1.5 (F16)",
        "type": "Embedding",
        "path": "/home/sandeep/ai-models/embed/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.f16.gguf",
        "tokens": 0
    }
]

def get_vram_used_mb():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL).decode().strip()
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

def run_benchmark(item):
    name = item["name"]
    mtype = item["type"]
    path = item["path"]
    tokens = item["tokens"]

    print(f"\n=======================================================", flush=True)
    print(f"▶ BENCHMARKING: {name} [{mtype}]", flush=True)
    print(f"  Path: {path}", flush=True)

    if not os.path.exists(path):
        print(f"  [ERROR] Model file not found!", flush=True)
        return None

    if mtype == "Embedding":
        cmd = [
            "./build/bin/llama-embedding",
            "-m", path,
            "-p", "Explain quantum computing in three concise bullet points.",
            "-ngl", "99"
        ]
    else:
        cmd = [
            "./build/bin/llama-cli",
            "-m", path,
            "--sandwich-auto",
            "-p", "Explain quantum computing in three concise bullet points.",
            "-n", str(tokens),
            "--no-warmup",
            "-st"
        ]

    vram_tracker = VRAMTracker(interval=0.1)
    vram_tracker.start()

    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        stdout, _ = proc.communicate(timeout=180)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, _ = proc.communicate()
        returncode = -9
        print("  [ERROR] Process timed out after 180s!", flush=True)
    except Exception as e:
        print(f"  [ERROR] Subprocess error: {e}", flush=True)
        returncode = -1
        stdout = ""

    total_time = time.time() - t0
    vram_tracker.stop()
    vram_tracker.join()
    peak_vram = vram_tracker.peak_vram

    full_output = stdout
    # Strip ANSI escape sequences for reliable regex matching
    clean_output = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', full_output)
    success = (returncode == 0)

    # Extract metrics
    prompt_speed = 0.0
    gen_speed = 0.0
    layers = "Auto"
    spec_mode = "None"
    topology = mtype

    if mtype == "Embedding":
        topology = "Embedding"
        spec_mode = "Direct Offload (-ngl 99)"
        layers = "All GPU"
        prompt_speed = 12.0 / max(total_time, 0.001)
        gen_speed = 0.0
    else:
        m_prompt = re.search(r"Prompt:\s*([\d\.]+)\s*t/s\s*\|\s*Generation:\s*([\d\.]+)\s*t/s", clean_output)
        if m_prompt:
            prompt_speed = float(m_prompt.group(1))
            gen_speed = float(m_prompt.group(2))

        if "MIXTURE OF EXPERTS" in clean_output:
            topology = "MoE"
            spec_mode = "MoE Fast-Path (-ngl 99)"
            layers = "99"
        elif "DENSE TRANSFORMER" in clean_output:
            topology = "Dense"
            if "Speculative Drafter:" in clean_output:
                m_draft = re.search(r"Speculative Drafter:\s*([^\n]+)", clean_output)
                dname = os.path.basename(m_draft.group(1).strip()) if m_draft else "GGUF Drafter"
                spec_mode = f"Spec Drafter ({dname})"
            elif "N-Gram Lookup Speculation" in clean_output:
                spec_mode = "Zero-VRAM N-Gram (n=12, m=48)"
            else:
                spec_mode = "Direct Dense"

            m_layers = re.search(r"n_gpu_layers\s*=\s*(\d+)", clean_output)
            if m_layers:
                layers = m_layers.group(1)

    print(f"  Status: {'SUCCESS' if success else 'FAILED (code ' + str(returncode) + ')'}", flush=True)
    print(f"  Topology: {topology} | Mode: {spec_mode} | Offloaded Layers: {layers}", flush=True)
    if mtype != "Embedding":
        print(f"  Prompt Speed:     {prompt_speed:.1f} t/s", flush=True)
        print(f"  Generation Speed: {gen_speed:.1f} t/s", flush=True)
    else:
        print(f"  Latency:          {total_time*1000.0:.1f} ms", flush=True)
    print(f"  Peak VRAM:        {peak_vram} MB", flush=True)
    print(f"  Wall Time:        {total_time:.2f}s", flush=True)

    res = {
        "name": name,
        "type": mtype,
        "success": success,
        "topology": topology,
        "spec_mode": spec_mode,
        "layers": layers,
        "prompt_speed": prompt_speed,
        "gen_speed": gen_speed,
        "peak_vram": peak_vram,
        "total_time": total_time
    }

    # Append to intermediate json
    try:
        cur_results = []
        if os.path.exists("benchmark_results.json"):
            with open("benchmark_results.json", "r") as f:
                cur_results = json.load(f)
        cur_results.append(res)
        with open("benchmark_results.json", "w") as f:
            json.dump(cur_results, f, indent=2)
    except Exception:
        pass

    return res

def main():
    print("=" * 70)
    print("🥪 PROJECT SANDWICH.CPP DENSE & MOE SYSTEM-WIDE BENCHMARK SUITE")
    print("Hardware: RTX 5050 Mobile (8GB GDDR6) + Intel i7-13620H (6 P-cores pinned)")
    print("=" * 70, flush=True)

    if os.path.exists("benchmark_results.json"):
        os.remove("benchmark_results.json")

    results = []
    for item in MODELS:
        res = run_benchmark(item)
        if res:
            results.append(res)
        time.sleep(2) # Cooldown & VRAM release

    print("\n\n" + "=" * 90)
    print("📊 COMPLETE BENCHMARK & STRESS TEST RESULTS")
    print("=" * 90)
    header = f"| Model | Type | GPU Layers | Speculative / Dispatch Mode | Prompt (t/s) | Gen (t/s) | Peak VRAM | Status |"
    sep = f"|---|---|---|---|---|---|---|---|"
    print(header)
    print(sep)
    for r in results:
        status_str = "✅ PASS" if r["success"] else "❌ FAIL"
        row = f"| {r['name']} | {r['type']} | {r['layers']} | {r['spec_mode']} | {r['prompt_speed']:.1f} | {r['gen_speed']:.1f} | {r['peak_vram']} MB | {status_str} |"
        print(row)
    print("=" * 90)

if __name__ == "__main__":
    main()

