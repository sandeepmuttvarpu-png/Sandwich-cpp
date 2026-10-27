#!/usr/bin/env python3
"""
Sandwich.cpp OpenAI-Compatible Streaming Chat Example.

Connects to a running `sandwich-server` instance and streams tokens in real-time,
measuring Time-To-First-Token (TTFT) and token generation velocity.
"""

import sys
import time
from typing import Generator

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] 'openai' package not found. Please install via: pip install openai")
    sys.exit(1)


def stream_response(client: OpenAI, prompt: str, model: str = "sandwich-model") -> None:
    print(f"\n[Prompt]: {prompt}")
    print("[Response]: ", end="", flush=True)

    t_start = time.perf_counter()
    first_token_received = False
    token_count = 0

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful, ultra-fast AI coding assistant powered by Sandwich.cpp."},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        temperature=0.7,
        max_tokens=512,
    )

    for chunk in response:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            if not first_token_received:
                ttft = (time.perf_counter() - t_start) * 1000.0
                first_token_received = True
            token_count += 1
            print(delta, end="", flush=True)

    t_total = time.perf_counter() - t_start
    gen_time = t_total - (ttft / 1000.0) if first_token_received else t_total
    gen_tps = (token_count / gen_time) if gen_time > 0 else 0.0

    print("\n" + "-" * 50)
    print(f"Tokens Generated: {token_count}")
    print(f"Time to First Token (TTFT): {ttft:.2f} ms")
    print(f"Generation Speed: {gen_tps:.2f} tok/s")
    print("-" * 50 + "\n")


def main():
    base_url = "http://localhost:8081/v1"
    api_key = "sandwich"

    print("=" * 60)
    print("🥪 Sandwich.cpp High-Performance Streaming Client")
    print(f"Connecting to: {base_url}")
    print("=" * 60)

    client = OpenAI(base_url=base_url, api_key=api_key)

    try:
        models = client.models.list()
        active_model = models.data[0].id if models.data else "sandwich-model"
        print(f"✓ Connected successfully. Active model: {active_model}\n")
    except Exception as e:
        print(f"[!] Note: Could not query /v1/models ({e}). Proceeding with default endpoint...")
        active_model = "sandwich-model"

    prompt = sys.argv[1] if len(sys.argv) > 1 else "Explain Walsh-Hadamard Transform in 3 concise bullet points."
    stream_response(client, prompt, model=active_model)


if __name__ == "__main__":
    main()
