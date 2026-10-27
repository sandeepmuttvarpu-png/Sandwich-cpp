# 🥪 Sandwich.cpp Client SDK Examples

This directory provides standalone, plug-and-play Python examples demonstrating integration with the **Sandwich.cpp OpenAI-Compatible HTTP Server** (`sandwich-server`).

---

## 1. Prerequisites

Install the standard client dependencies:

```bash
pip install openai pydantic
```

---

## 2. Launch the Sandwich Server

Start `sandwich-server` on `localhost:8081` with automatic memory topology and agent presets:

```bash
sandwich-server \
  -m /path/to/your-model.gguf \
  --sandwich-auto \
  --preset agent \
  --port 8081 \
  --host 0.0.0.0
```

---

## 3. Running the Examples

### A. High-Throughput Token Streaming (`streaming_chat.py`)
Streams completion tokens in real-time and computes Time-To-First-Token (TTFT) and decode speed (tok/s):

```bash
python examples/sandwich-client/streaming_chat.py "Explain 3-tier MoE memory hierarchy."
```

### B. Structured Pydantic JSON Output (`structured_output.py`)
Forces the model to conform strictly to a validated Pydantic schema using structured outputs:

```bash
python examples/sandwich-client/structured_output.py
```

---

## 4. Environment Variables

You can configure endpoint and auth via standard environment variables:

```bash
export OPENAI_BASE_URL="http://localhost:8081/v1"
export OPENAI_API_KEY="sandwich"
```
