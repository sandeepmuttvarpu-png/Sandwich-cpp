#!/usr/bin/env python3
import subprocess
import time
import re
import os
import sys
import json
import threading

MODEL_PATH = "/home/sandeep/ai-models/unsloth/gpt-oss-120b-GGUF/gpt-oss-120b-UD-Q4_K_XL-00001-of-00002.gguf"
BIN_PATH = "/home/sandeep/dev/sandwich/sandwich-cpp/build/bin/llama-cli"
DOC_PATH = "/home/sandeep/dev/sandwich/sandwich-cpp/docs/benchmarks.md"
PROMPT = "Explain the key differences between monolithic kernels, microkernels, and hybrid kernels in modern operating systems."
N_PREDICT = 32
CTX_SIZE = 2048
THREADS = 6

CONFIGS = [
    {
        "name": "ngl 0 (Pure CPU)",
        "ngl": 0,
        "extra_args": ["-ngl", "0"],
        "desc": "Pure CPU execution (Host RAM + NVMe paging)"
    },
    {
        "name": "ngl 12 (Partial Offload 1/3)",
        "ngl": 12,
        "extra_args": ["-ngl", "12", "-cmoe"],
        "desc": "12 layers attention on GPU, 24 on CPU, MoE on CPU"
    },
    {
        "name": "ngl 24 (Partial Offload 2/3)",
        "ngl": 24,
        "extra_args": ["-ngl", "24", "-cmoe"],
        "desc": "24 layers attention on GPU, 12 on CPU, MoE on CPU"
    },
    {
        "name": "ngl 37 (Full Attention Offload)",
        "ngl": 37,
        "extra_args": ["-ngl", "37", "-cmoe"],
        "desc": "All 36 layers attention on GPU, MoE on CPU"
    },
    {
        "name": "Sandwich-Auto (Dynamic MoE Cache)",
        "ngl": 37,
        "extra_args": ["--sandwich-auto"],
        "desc": "Auto GPU offload + Dynamic MoE cache in VRAM"
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

def get_ram_swap_info():
    try:
        mem = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    mem[parts[0].strip()] = int(parts[1].strip().split()[0])
        total_ram = mem.get("MemTotal", 0) / 1024 / 1024
        avail_ram = mem.get("MemAvailable", 0) / 1024 / 1024
        used_ram = total_ram - avail_ram
        total_swap = mem.get("SwapTotal", 0) / 1024 / 1024
        free_swap = mem.get("SwapFree", 0) / 1024 / 1024
        used_swap = total_swap - free_swap
        return used_ram, used_swap
    except Exception:
        return 0.0, 0.0

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

def run_test(cfg):
    print(f"\n=======================================================", flush=True)
    print(f"▶ RUNNING: {cfg['name']}", flush=True)
    print(f"  Description: {cfg['desc']}", flush=True)

    cmd = [
        BIN_PATH,
        "-m", MODEL_PATH,
        "-t", str(THREADS),
        "-c", str(CTX_SIZE),
        "-fa", "on",
        "-n", str(N_PREDICT),
        "--no-warmup",
        "-st",
        "--simple-io",
        "-p", PROMPT
    ] + cfg["extra_args"]

    print(f"  Command: {' '.join(cmd)}", flush=True)

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
        stdout, _ = proc.communicate(timeout=300)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, _ = proc.communicate()
        returncode = -9
        print("  [ERROR] Timed out after 300s!", flush=True)
    except Exception as e:
        print(f"  [ERROR] Execution failed: {e}", flush=True)
        returncode = -1
        stdout = ""

    elapsed = time.time() - t0
    vram_tracker.stop()
    vram_tracker.join()
    peak_vram = vram_tracker.peak_vram
    used_ram, used_swap = get_ram_swap_info()

    # Clean ANSI
    clean_out = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', stdout)
    
    prompt_speed = 0.0
    gen_speed = 0.0

    # Pattern 1: [ Prompt: X t/s | Generation: Y t/s ]
    m_st = re.search(r"Prompt:\s*([\d\.]+)\s*t/s\s*\|\s*Generation:\s*([\d\.]+)\s*t/s", clean_out)
    if m_st:
        prompt_speed = float(m_st.group(1))
        gen_speed = float(m_st.group(2))
    else:
        # Pattern 2: llama.cpp standard timings
        m_pp = re.search(r"prompt eval time\s*=.*?([\d\.]+)\s*tokens per second", clean_out)
        if m_pp:
            prompt_speed = float(m_pp.group(1))
        m_tg = re.search(r"eval time\s*=.*?([\d\.]+)\s*tokens per second", clean_out)
        if m_tg:
            gen_speed = float(m_tg.group(1))

    status = "SUCCESS" if returncode == 0 and gen_speed > 0 else f"FAILED (code {returncode})"
    print(f"  Result: {status}", flush=True)
    print(f"  Prompt Speed: {prompt_speed:.2f} tok/s | Generation: {gen_speed:.2f} tok/s", flush=True)
    print(f"  Peak VRAM: {peak_vram} MiB | Total Latency: {elapsed:.2f}s", flush=True)

    return {
        "name": cfg["name"],
        "ngl": cfg["ngl"],
        "desc": cfg["desc"],
        "extra_args": " ".join(cfg["extra_args"]),
        "status": status,
        "prompt_speed": prompt_speed,
        "gen_speed": gen_speed,
        "peak_vram": peak_vram,
        "used_ram_gb": round(used_ram, 1),
        "used_swap_gb": round(used_swap, 1),
        "elapsed_sec": round(elapsed, 2)
    }

def main():
    print("🚀 Starting GPT-OSS 120B Benchmark Suite...", flush=True)
    results = []
    for cfg in CONFIGS:
        res = run_test(cfg)
        results.append(res)
        time.sleep(2)

    # Save results json
    json_path = "/home/sandeep/dev/sandwich/sandwich-cpp/gpt_oss_120b_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results to {json_path}", flush=True)

    # Print summary table
    print("\n" + "="*80, flush=True)
    print("🏆 FINAL BENCHMARK RESULTS: GPT-OSS 120B (Unsloth UD-Q4_K_XL, 59 GB)", flush=True)
    print("="*80, flush=True)
    header = f"{'Configuration':<32} | {'Prompt (tok/s)':<14} | {'Decode (tok/s)':<14} | {'Peak VRAM':<11} | {'Time (s)':<8}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in results:
        print(f"{r['name']:<32} | {r['prompt_speed']:<14.2f} | {r['gen_speed']:<14.2f} | {r['peak_vram']:<7} MiB | {r['elapsed_sec']:<8.1f}", flush=True)
    print("="*80, flush=True)

    # Update benchmarks.md
    update_benchmarks_doc(results)

def update_benchmarks_doc(results):
    if not os.path.exists(DOC_PATH):
        print(f"[WARN] {DOC_PATH} not found, skipping doc update.", flush=True)
        return

    with open(DOC_PATH, "r") as f:
        content = f.read()

    # Generate new markdown section
    table_lines = [
        "### D. GPT-OSS 120B-A3B MoE (`UD-Q4_K_XL`) — 58.7 GB (128 Experts / 36 Layers)",
        "",
        "Flagship 120B Mixture of Experts model evaluated on RTX 5050 Laptop GPU (8GB VRAM) + 24GB DDR5 Host RAM + NVMe swap.",
        "",
        "| Configuration | GPU Offload (`ngl`) | Memory Mode | Prompt Speed (PP) | Decode Speed (TG) | Peak VRAM | Status |",
        "|---|---|---|---|---|---|---|"
    ]

    for r in results:
        cfg_name = r['name']
        ngl = r['ngl']
        mem_mode = r['desc']
        pp = f"**{r['prompt_speed']:.2f} tok/s**" if r['prompt_speed'] > 0 else "N/A"
        tg = f"**{r['gen_speed']:.2f} tok/s**" if r['gen_speed'] > 0 else "N/A"
        vram = f"{r['peak_vram']} MiB"
        status = "✅ Passed" if "SUCCESS" in r['status'] else "❌ Failed"
        table_lines.append(f"| **{cfg_name}** | {ngl} | {mem_mode} | {pp} | {tg} | {vram} | {status} |")

    table_lines.append("")
    new_section = "\n".join(table_lines)

    # Insert under Section 1 (Mixture of Experts)
    anchor = "#### C. GLM-4.7 Flash 30B-A3B MoE"
    if anchor in content:
        # Find the next section divider (---)
        anchor_idx = content.find(anchor)
        divider_idx = content.find("\n---\n", anchor_idx)
        if divider_idx != -1:
            updated_content = content[:divider_idx] + "\n\n" + new_section + content[divider_idx:]
            with open(DOC_PATH, "w") as f:
                f.write(updated_content)
            print(f"Successfully integrated results into {DOC_PATH}!", flush=True)
            return

    # Fallback append
    with open(DOC_PATH, "a") as f:
        f.write("\n\n" + new_section)
    print(f"Appended results to {DOC_PATH}!", flush=True)

if __name__ == "__main__":
    main()
