# 🔌 Sandwich.cpp Client Integration Guide

Sandwich.cpp provides a native, OpenAI-compatible HTTP REST and Server-Sent Events (SSE) streaming API server (`sandwich-server`). Designed specifically for local multi-turn agentic coding, RAG workflows, and interactive chat, it features hardware-aware topology routing, sub-millisecond Radix prefix caching, and TurboQuant memory compression.

---

## 1. Quickstart: Launching the Agent Server

The recommended production deployment command uses `--sandwich-auto` and `--preset agent` on port `8081`:

```bash
sandwich-server \
  -m /path/to/your-model.gguf \
  --sandwich-auto \
  --preset agent \
  --port 8081 \
  --host 0.0.0.0
```

### What `--preset agent` + `--sandwich-auto` activates:
- **Zero-Latency Prefix Caching:** Engages Technique #45 (Radix-Tree RadixAttention) with a 4GB persistent prefix cache, slashing multi-turn TTFT by **$200\times$** (down to 0.05 seconds).
- **Automated Memory Topology:** Applies the **70% RAM Rule** (Technique #42) for massive dense models and **3-Tier MoE Caching** (Technique #40) with 2Q Ghost Cache (Technique #41) for MoE architectures.
- **TurboQuant KV Cache:** Auto-routes KV tensors to `turbo4` (VRAM) or `turbo2` (extreme long-context squeeze).
- **Agent Stop Tokens:** Registers `<|im_end|>`, `<|endoftext|>`, `[TOOL_CALLS]`, and `\``` ` as default stop boundaries.
- **Continuous Batching:** Pre-configures 4 parallel execution slots (`-np 4`) and context sliding window shifts.

---

## 2. API Endpoints

| Endpoint | Method | Description |
|:---|:---:|:---|
| `http://localhost:8081/v1/chat/completions` | `POST` | OpenAI-compatible streaming SSE & non-streaming chat completions |
| `http://localhost:8081/v1/completions` | `POST` | Raw text completions |
| `http://localhost:8081/v1/models` | `GET` | List available models |
| `http://localhost:8081/health` | `GET` | Server readiness probe |

---

## 3. Client Integrations

### 3.1 Cursor IDE

1. Open **Cursor Settings** $\to$ **Features** $\to$ **Models**.
2. Under **OpenAI API Key**, enter any placeholder string:
   ```text
   sandwich-local
   ```
3. Click **Override OpenAI Base URL** and enter:
   ```text
   http://localhost:8081/v1
   ```
4. Under **Model Names**, add the identifier of your loaded model (e.g. `sandwich-local`, `qwen3.6-35b`, or `deepseek-coder`).
5. Toggle off other cloud models to enforce local routing.

---

### 3.2 Continue.dev

Add the following configuration block to your `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "Sandwich Local Agent",
      "provider": "openai",
      "model": "sandwich-local",
      "apiBase": "http://localhost:8081/v1",
      "apiKey": "sandwich-local",
      "contextLength": 65536,
      "completionOptions": {
        "temperature": 0.2,
        "topP": 0.95
      }
    }
  ],
  "tabAutocompleteModel": {
    "title": "Sandwich Fast Tab",
    "provider": "openai",
    "model": "sandwich-local",
    "apiBase": "http://localhost:8081/v1",
    "apiKey": "sandwich-local"
  }
}
```

---

### 3.3 OpenWebUI

#### Option A: Docker Environment Variable
When running OpenWebUI via Docker:
```bash
docker run -d -p 3000:8080 \
  -e OPENAI_API_BASE_URL="http://host.docker.internal:8081/v1" \
  -e OPENAI_API_KEY="sandwich-local" \
  --add-host=host.docker.internal:host-gateway \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

#### Option B: In-App Admin Settings
1. Log in to **OpenWebUI** as Admin $\to$ **Settings** $\to$ **Connections**.
2. Under **OpenAI API**, set:
   - **URL:** `http://127.0.0.1:8081/v1`
   - **Key:** `sandwich-local`
3. Click the verify icon. OpenWebUI will detect the loaded model immediately.

---

### 3.4 LibreChat

Add a custom OpenAI endpoint definition to your `librechat.yaml`:

```yaml
version: 1.1.5

endpoints:
  custom:
    - name: "Sandwich.cpp"
      apiKey: "sandwich-local"
      baseURL: "http://host.docker.internal:8081/v1"
      models:
        default: ["sandwich-local"]
        fetch: true
      titleConvo: true
      titleModel: "sandwich-local"
      summarize: false
      summaryModel: "sandwich-local"
      forcePrompt: false
      modelDisplayLabel: "Sandwich 128k Local Engine"
```

---

## 4. Verification via cURL (SSE Streaming)

To verify real-time Server-Sent Events (SSE) token delivery:

```bash
curl -N http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sandwich-local" \
  -d '{
    "model": "sandwich-local",
    "messages": [
      {"role": "system", "content": "You are a concise engineering assistant."},
      {"role": "user", "content": "Explain Asymmetric Split-KV in two bullet points."}
    ],
    "temperature": 0.2,
    "stream": true
  }'
```

Expected response starts streaming tokens instantly:
```text
data: {"choices":[{"delta":{"content":"•"}}]}
data: {"choices":[{"delta":{"content":" 100%"}}]}
data: {"choices":[{"delta":{"content":" of"}}]}
...
data: [DONE]
```

---

## 5. Ready-to-Run Python SDK Examples

Executable client scripts are available under [`examples/sandwich-client/`](../examples/sandwich-client/):
- **[`streaming_chat.py`](../examples/sandwich-client/streaming_chat.py):** Real-time streaming chat with automatic TTFT and tokens/sec telemetry.
- **[`structured_output.py`](../examples/sandwich-client/structured_output.py):** Pydantic structured JSON schema output extraction.

