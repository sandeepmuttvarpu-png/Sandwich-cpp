#!/usr/bin/env python3
"""
test_niah_retrieval.py
======================
Technique #44: Needle-In-A-Haystack (NIAH) Long-Context Retrieval Verification Engine.
Empirically tests semantic fidelity and retrieval accuracy across context tiers (8k -> 128k)
and context depths (0% -> 100%) under TurboQuant compression (turbo4, turbo3, turbo2).
"""

import os
import sys
import subprocess
import time
import json
import re

BIN_DIR = "/home/sandeep/dev/sandwich/sandwich-cpp/build/bin"
CLI_BIN = os.path.join(BIN_DIR, "llama-cli")
OUTPUT_JSON = "/home/sandeep/dev/niah_retrieval_results.json"

SECRET_NEEDLE = "The secret verification passkey for Project Sandwich is: [SANDWICH-128K-RETRIEVAL-VERIFIED]."
QUESTION = "What is the secret verification passkey for Project Sandwich? Reply only with the passkey enclosed in brackets."
TARGET_KEY = "SANDWICH-128K-RETRIEVAL-VERIFIED"

# Haystack filler text (repeating technical context from sandwich-cpp architecture)
HAYSTACK_BASE = (
    "In high-performance heterogeneous inference systems, model weights and activation tensors are partitioned "
    "across host DDR5 RAM and device VRAM. Tensor Core execution leverages FlashAttention with PolarQuant Lloyd-Max "
    "transformations to minimize quantization noise. Dynamic offloading heuristics dynamically reserve VRAM headroom "
    "for the display compositor and speculative drafter heads while maximizing arithmetic throughput across SIMD pipelines. "
)

def generate_haystack_prompt(target_tokens, depth_fraction):
    """Generates a synthetic prompt of exact token size with the needle placed at depth_fraction."""
    # Approximate ~4 chars per token
    target_chars = target_tokens * 4
    base_chars = len(HAYSTACK_BASE)
    repeats = (target_chars // base_chars) + 1
    full_text = HAYSTACK_BASE * repeats
    full_text = full_text[:target_chars]

    insert_pos = int(len(full_text) * depth_fraction)
    needle_text = f"\n\n*** IMPORTANT SYSTEM RECORD: {SECRET_NEEDLE} ***\n\n"
    
    haystack_with_needle = full_text[:insert_pos] + needle_text + full_text[insert_pos:]
    final_prompt = f"{haystack_with_needle}\n\nQuestion: {QUESTION}\nAnswer:"
    return final_prompt

def run_niah_test(model_path, ctx, kv_type, depth_pct, extra_args=None):
    if extra_args is None:
        extra_args = ["--sandwich-auto"]

    depth_frac = depth_pct / 100.0
    prompt_text = generate_haystack_prompt(ctx - 256, depth_frac)
    
    prompt_file = f"/tmp/niah_prompt_{ctx}_{depth_pct}.txt"
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt_text)

    cmd = [
        CLI_BIN,
        "-m", model_path,
        "-c", str(ctx),
        "-ctk", kv_type,
        "-ctv", kv_type,
        "-f", prompt_file,
        "-n", "32",
        "-t", "6",
        "--no-warmup",
        "-st",
        "--simple-io"
    ]
    cmd.extend(extra_args)

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300,
            errors="replace"
        )
        out = proc.stdout
        elapsed = time.time() - t0
        found = TARGET_KEY in out
        status = "PASS" if found else "FAIL"
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        status = "TIMEOUT"
        found = False
        out = ""
    except Exception as e:
        elapsed = time.time() - t0
        status = f"ERR: {str(e)[:30]}"
        found = False
        out = ""

    if os.path.exists(prompt_file):
        os.remove(prompt_file)

    return {
        "ctx": ctx,
        "kv": kv_type,
        "depth": depth_pct,
        "status": status,
        "found": found,
        "elapsed": round(elapsed, 2)
    }

def print_ascii_heatmap(results_matrix, ctx_tiers, depth_tiers):
    print("\n========================================================")
    print(" 📍 Needle-In-A-Haystack (NIAH) Empirical Retrieval Heatmap")
    print("========================================================")
    header = " Context \\ Depth | " + " | ".join(f"{d:>4}%" for d in depth_tiers) + " |"
    print(header)
    print("-" * len(header))
    for c in ctx_tiers:
        row = f" {c:>6} tokens  | "
        for d in depth_tiers:
            res = results_matrix.get((c, d), " . ")
            symbol = " 🟩 " if res == "PASS" else (" 🟥 " if res == "FAIL" else " ⏳ ")
            row += f"{symbol}| "
        print(row)
    print("========================================================\n")

if __name__ == "__main__":
    print("Technique #44: NIAH Retrieval Verification Engine initialized.")
