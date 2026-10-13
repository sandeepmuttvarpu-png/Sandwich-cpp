#!/usr/bin/env python3
import subprocess
import time
import urllib.request
import json
import re
import os
import signal

def test_model_acceptance(model_path, model_name, drafter_name):
    print(f"\n=======================================================", flush=True)
    print(f"▶ MEASURING ACCEPTANCE RATE: {model_name}", flush=True)
    print(f"  Drafter: {drafter_name}", flush=True)
    print(f"  Path: {model_path}", flush=True)

    port = 8999
    cmd = [
        "./build/bin/llama-server",
        "-m", model_path,
        "--sandwich-auto",
        "--port", str(port),
        "-lv", "3"
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    server_ready = False
    log_lines = []
    t_start = time.time()

    # Wait for server ready
    while time.time() - t_start < 120:
        line = proc.stdout.readline()
        if line:
            log_lines.append(line)
            if "HTTP server listening" in line or "listening on http" in line or "model loaded" in line:
                server_ready = True
                break
        if proc.poll() is not None:
            break

    if not server_ready:
        print("  [ERROR] Server failed to start within timeout", flush=True)
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
        return None

    print("  Server ready! Sending completion request...", flush=True)
    time.sleep(1)

    # Send completion request
    req_data = json.dumps({
        "prompt": "Explain quantum computing in three concise bullet points.",
        "n_predict": 48,
        "temperature": 0.0
    }).encode('utf-8')

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=req_data,
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp_body = resp.read().decode('utf-8')
            resp_json = json.loads(resp_body)
    except Exception as e:
        print(f"  [ERROR] Request failed: {e}", flush=True)
        proc.terminate()
        proc.wait()
        return None

    # Wait a moment for slot logging
    time.sleep(2)
    proc.send_signal(signal.SIGINT)

    # Read remaining logs
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            log_lines.append(line)

    full_log = "".join(log_lines)

    # Search for acceptance stats
    # Example: draft acceptance = 0.54000 (   27 accepted /    50 generated), mean len =  2.35
    m_acc = re.search(r"draft acceptance\s*=\s*([\d\.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated\)", full_log)
    m_mean = re.search(r"mean len\s*=\s*([\d\.]+)", full_log)

    acc_rate = 0.0
    n_acc = 0
    n_gen = 0
    mean_len = 0.0

    if m_acc:
        acc_rate = float(m_acc.group(1)) * 100.0
        n_acc = int(m_acc.group(2))
        n_gen = int(m_acc.group(3))
    if m_mean:
        mean_len = float(m_mean.group(1))

    # Also search for statistics in common_speculative_print_stats
    # statistics draft: #gen tokens = ..., #acc tokens = ...
    m_stat = re.search(r"#gen tokens\s*=\s*(\d+),\s*#acc tokens\s*=\s*(\d+)", full_log)
    if m_stat and n_gen == 0:
        n_gen = int(m_stat.group(1))
        n_acc = int(m_stat.group(2))
        if n_gen > 0:
            acc_rate = (n_acc / n_gen) * 100.0

    print(f"  Accepted Tokens:  {n_acc} / {n_gen}", flush=True)
    print(f"  Acceptance Rate:  {acc_rate:.2f}%", flush=True)
    if mean_len > 0:
        print(f"  Mean Accept Len:  {mean_len:.2f} tokens/iter", flush=True)

    return {
        "model": model_name,
        "drafter": drafter_name,
        "n_accepted": n_acc,
        "n_generated": n_gen,
        "acceptance_rate": acc_rate,
        "mean_len": mean_len,
        "log_sample": [l for l in log_lines if "acceptance" in l or "speculative" in l.lower()][:5]
    }

if __name__ == "__main__":
    targets = [
        (
            "/home/sandeep/ai-models/dense/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
            "Muse-Glimmer-30B (Dense)",
            "Muse-Glimmer-30B-DFlash2 (GGUF Drafter, N=5)"
        ),
        (
            "/home/sandeep/ai-models/dense/Llama-3.3-70B-Instruct-GGUF/Llama-3.3-70B-Instruct-UD-IQ2_XXS.gguf",
            "Llama-3.3-70B (Dense)",
            "Llama-3.2-1B-Instruct (GGUF Drafter, N=5)"
        ),
        (
            "/home/sandeep/ai-models/dense/gemma-4-31B-it-QAT-GGUF/gemma-4-31B-it-QAT-Q4_0.gguf",
            "Gemma-4-31B-it (Dense)",
            "gemma-4-31b-assistant-Q4_0-Q6emb (MTP Drafter, N=5)"
        ),
        (
            "/home/sandeep/ai-models/dense/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_XXS.gguf",
            "Qwen3.8-27B (Dense)",
            "Qwen3.8-27B-DFlash2-Q4_K_M (DFlash2 Drafter, N=5)"
        )
    ]

    results = []
    for path, mname, dname in targets:
        res = test_model_acceptance(path, mname, dname)
        if res:
            results.append(res)
        time.sleep(2)

    print("\n\n" + "=" * 80)
    print("📊 SPECULATIVE DRAFTER ACCEPTANCE RATE REPORT")
    print("=" * 80)
    print(f"| Target Model | Drafter Type / Model | Accepted / Drafted | Acceptance Rate | Mean Acc Length |")
    print(f"|---|---|---|---|---|")
    for r in results:
        print(f"| {r['model']} | {r['drafter']} | {r['n_accepted']} / {r['n_generated']} | {r['acceptance_rate']:.1f}% | {r['mean_len']:.2f} tok/iter |")
    print("=" * 80)
