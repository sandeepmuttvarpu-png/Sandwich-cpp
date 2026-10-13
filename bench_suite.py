#!/usr/bin/env python3
import subprocess
import time
import re
import os
import sys

MODELS = [
    # MoE Models
    {
        "name": "GLM-4.7-Flash (MoE)",
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
        "name": "Nemotron-3-Nano-4B",
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
        "name": "Qwen3.8-27B (IQ3_XXS)",
        "type": "Dense",
        "path": "/home/sandeep/ai-models/dense/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_XXS.gguf",
        "tokens": 48
    },
    {
        "name": "Muse-Glimmer-30B (Q4_K_M)",
        "type": "Dense",
        "path": "/home/sandeep/ai-models/dense/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
        "tokens": 48
    },
    {
        "name": "Gemma-4-31B-it (Q4_0)",
        "type": "Dense",
        "path": "/home/sandeep/ai-models/dense/gemma-4-31B-it-QAT-GGUF/gemma-4-31B-it-QAT-Q4_0.gguf",
        "tokens": 48
    },
    {
        "name": "Llama-3.3-70B-Instruct (IQ2_XXS)",
        "type": "Dense",
        "path": "/home/sandeep/ai-models/dense/Llama-3.3-70B-Instruct-GGUF/Llama-3.3-70B-Instruct-UD-IQ2_XXS.gguf",
        "tokens": 48
    }
]

def get_vram_used_mb():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).decode().strip()
        return int(out.split('\n')[0])
    except Exception:
        return 0

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

    cmd = [
        "./build/bin/llama-cli",
        "-m", path,
        "--sandwich-auto",
        "-p", "Explain quantum computing in three concise bullet points.",
        "-n", str(tokens),
        "--no-warmup",
        "-st"
    ]

    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    output_lines = []
    peak_vram = get_vram_used_mb()

    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            output_lines.append(line)
            # Sample VRAM
            v = get_vram_used_mb()
            if v > peak_vram:
                peak_vram = v

    proc.wait()
    total_time = time.time() - t0

    full_output = "".join(output_lines)
    success = (proc.returncode == 0)

    # Extract metrics
    prompt_speed = 0.0
    gen_speed = 0.0
    layers = "Auto"
    spec_mode = "None"
    topology = "Dense"

    m_prompt = re.search(r"Prompt:\s*([\d\.]+)\s*t/s\s*\|\s*Generation:\s*([\d\.]+)\s*t/s", full_output)
    if m_prompt:
        prompt_speed = float(m_prompt.group(1))
        gen_speed = float(m_prompt.group(2))

    if "MIXTURE OF EXPERTS" in full_output:
        topology = "MoE"
        spec_mode = "MoE Fast-Path (-ngl 99)"
        layers = "99"
    elif "DENSE TRANSFORMER" in full_output:
        topology = "Dense"
        if "Speculative Drafter:" in full_output:
            m_draft = re.search(r"Speculative Drafter:\s*([^\n]+)", full_output)
            dname = os.path.basename(m_draft.group(1).strip()) if m_draft else "GGUF Drafter"
            spec_mode = f"Spec Drafter ({dname})"
        elif "N-Gram Lookup Speculation" in full_output:
            spec_mode = "Zero-VRAM N-Gram (n=12, m=48)"
        else:
            spec_mode = "Direct Dense"

        m_layers = re.search(r"n_gpu_layers\s*=\s*(\d+)", full_output)
        if m_layers:
            layers = m_layers.group(1)

    print(f"  Status: {'SUCCESS' if success else 'FAILED (code ' + str(proc.returncode) + ')'}", flush=True)
    print(f"  Topology: {topology} | Mode: {spec_mode} | Offloaded Layers: {layers}", flush=True)
    print(f"  Prompt Speed:     {prompt_speed:.1f} t/s", flush=True)
    print(f"  Generation Speed: {gen_speed:.1f} t/s", flush=True)
    print(f"  Peak VRAM:        {peak_vram} MB", flush=True)
    print(f"  Wall Time:        {total_time:.2f}s", flush=True)

    return {
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

def main():
    print("=" * 70)
    print("🥪 PROJECT SANDWICH.CPP DENSE & MOE SYSTEM-WIDE BENCHMARK SUITE")
    print("Hardware: RTX 5050 Mobile (8GB GDDR6) + Intel i7-13620H (6 P-cores pinned)")
    print("=" * 70, flush=True)

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
