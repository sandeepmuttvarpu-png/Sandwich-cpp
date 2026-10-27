#!/usr/bin/env python3
"""
Sandwich.cpp Structured Output & Schema Extraction Client.

Demonstrates requesting structured JSON output and schema validation using
an OpenAI-compatible client connected to `sandwich-server`.
"""

import json
import sys

try:
    from openai import OpenAI
    from pydantic import BaseModel, Field
except ImportError:
    print("[ERROR] Required packages not found. Install via: pip install openai pydantic")
    sys.exit(1)


class HardwareOptimizationResult(BaseModel):
    model_name: str = Field(description="Name or parameter count of the LLM")
    context_tokens: int = Field(description="Target context window in tokens")
    optimal_kv_cache: str = Field(description="Selected TurboQuant cache format: turbo4, turbo3, or turbo2")
    vram_saved_percentage: float = Field(description="Estimated percentage of VRAM saved compared to FP16")
    recommendation: str = Field(description="High-level hardware deployment advice")


def main():
    base_url = "http://localhost:8081/v1"
    api_key = "sandwich"

    client = OpenAI(base_url=base_url, api_key=api_key)

    prompt = (
        "Analyze deployment of Qwen3.6-35B MoE at 64k context on an 8GB NVIDIA RTX 5050 Mobile laptop GPU. "
        "Recommend optimal Sandwich.cpp configuration parameters."
    )

    print("=" * 60)
    print("🥪 Sandwich.cpp Structured JSON Schema Client")
    print(f"Prompt: {prompt}")
    print("=" * 60)

    try:
        response = client.beta.chat.completions.parse(
            model="sandwich-model",
            messages=[
                {
                    "role": "system",
                    "content": "You are a hardware-aware ML systems advisor. Always respond with valid JSON adhering strictly to the schema.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format=HardwareOptimizationResult,
            temperature=0.2,
        )

        result: HardwareOptimizationResult = response.choices[0].message.parsed
        print("\n✓ Structured Response Received Successfully:\n")
        print(json.dumps(result.model_dump(), indent=2))

    except Exception as e:
        print(f"\n[!] Fallback to standard JSON schema payload (Server response or connection error: {e})")
        # Direct raw json_object completion
        res = client.chat.completions.create(
            model="sandwich-model",
            messages=[
                {"role": "system", "content": "Return valid JSON adhering to: model_name, context_tokens, optimal_kv_cache, recommendation."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        print(res.choices[0].message.content)


if __name__ == "__main__":
    main()
