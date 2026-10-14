/**
 * 🥪 Sandwich.cpp - Web UI & SSE Streaming Chat Engine
 * Vanilla JS Client for OpenAI-Compatible Local Inference
 */

(() => {
  // --- DOM Elements ---
  const chatHistory = document.getElementById('chat-history');
  const chatViewport = document.getElementById('chat-viewport');
  const chatForm = document.getElementById('chat-form');
  const userInput = document.getElementById('user-input');
  const sendBtn = document.getElementById('send-btn');
  const stopBtn = document.getElementById('stop-btn');
  const clearChatBtn = document.getElementById('clear-chat-btn');
  const hudToggleBtn = document.getElementById('hud-toggle-btn');
  const telemetryHud = document.getElementById('telemetry-hud');
  const activeModelName = document.getElementById('active-model-name');
  const speedIndicator = document.getElementById('speed-indicator');
  const typingIndicator = document.getElementById('typing-indicator');
  const systemPromptInput = document.getElementById('system-prompt-input');

  // --- State ---
  let messages = [];
  let abortController = null;
  let isGenerating = false;
  let activeModel = 'sandwich-model';

  // --- Initializer ---
  async function init() {
    loadSavedSettings();
    setupEventListeners();
    await fetchActiveModel();
  }

  // --- Fetch Active Model from Server ---
  async function fetchActiveModel() {
    try {
      const resp = await fetch('/v1/models');
      if (resp.ok) {
        const data = await resp.json();
        if (data && data.data && data.data.length > 0) {
          activeModel = data.data[0].id || 'sandwich-local';
          const shortName = activeModel.split('/').pop().replace(/\.gguf$/i, '');
          activeModelName.textContent = shortName;
          activeModelName.title = activeModel;
          return;
        }
      }
    } catch (e) {
      console.warn('[Sandwich UI] /v1/models query failed, falling back to /props:', e);
    }

    try {
      const respProps = await fetch('/props');
      if (respProps.ok) {
        const props = await respProps.json();
        if (props.default_generation_settings && props.default_generation_settings.model) {
          activeModel = props.default_generation_settings.model;
          const shortName = activeModel.split('/').pop().replace(/\.gguf$/i, '');
          activeModelName.textContent = shortName;
          return;
        }
      }
    } catch (e) {
      console.warn('[Sandwich UI] /props query failed:', e);
    }

    activeModelName.textContent = 'Sandwich.cpp Engine';
  }

  // --- Event Listeners ---
  function setupEventListeners() {
    // Auto-resize input textarea
    userInput.addEventListener('input', () => {
      userInput.style.height = 'auto';
      userInput.style.height = Math.min(userInput.scrollHeight, 180) + 'px';
      sendBtn.disabled = !userInput.value.trim();
    });

    // Enter to submit (Shift+Enter for newline)
    userInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!isGenerating && userInput.value.trim()) {
          chatForm.requestSubmit();
        }
      }
    });

    // Submit form
    chatForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const text = userInput.value.trim();
      if (!text || isGenerating) return;
      sendMessage(text);
    });

    // Stop Generation
    stopBtn.addEventListener('click', () => {
      if (abortController) {
        abortController.abort();
        abortController = null;
      }
      setGeneratingState(false);
    });

    // Clear Chat
    clearChatBtn.addEventListener('click', () => {
      if (confirm('Clear chat history?')) {
        messages = [];
        saveHistory();
        renderHistory();
      }
    });

    // HUD Toggle
    hudToggleBtn.addEventListener('click', () => {
      const isHidden = telemetryHud.classList.toggle('hidden');
      hudToggleBtn.classList.toggle('active', !isHidden);
    });

    // Persist system prompt
    systemPromptInput.addEventListener('change', () => {
      localStorage.setItem('sandwich_system_prompt', systemPromptInput.value.trim());
    });
  }

  // --- Send Message & SSE Stream ---
  async function sendMessage(userText) {
    // Hide hero if present
    const hero = document.getElementById('welcome-hero');
    if (hero) hero.remove();

    // Append User Message
    messages.push({ role: 'user', content: userText });
    appendMessageElement('user', userText);

    // Reset input
    userInput.value = '';
    userInput.style.height = 'auto';
    sendBtn.disabled = true;

    // Prepare assistant placeholder element
    const assistantMsg = { role: 'assistant', content: '' };
    messages.push(assistantMsg);
    const bubbleEl = appendMessageElement('assistant', '');

    setGeneratingState(true);
    abortController = new AbortController();

    const tStart = performance.now();
    let tokenCount = 0;

    // Prepare payload
    const systemPrompt = systemPromptInput.value.trim();
    const payloadMessages = [];
    if (systemPrompt) {
      payloadMessages.push({ role: 'system', content: systemPrompt });
    }
    // Include last 10 messages for context
    const recentMessages = messages.slice(-11, -1);
    payloadMessages.push(...recentMessages);

    try {
      const response = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify({
          model: activeModel,
          messages: payloadMessages,
          stream: true,
          temperature: 0.7
        }),
        signal: abortController.signal
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep partial line

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          const dataStr = trimmed.slice(5).trim();
          if (dataStr === '[DONE]') break;

          try {
            const parsed = JSON.parse(dataStr);
            const delta = parsed.choices?.[0]?.delta?.content;
            if (delta) {
              assistantMsg.content += delta;
              tokenCount++;
              updateBubbleContent(bubbleEl, assistantMsg.content);
              scrollToBottom();

              const elapsedSec = (performance.now() - tStart) / 1000;
              if (elapsedSec > 0.5) {
                const tokPerSec = (tokenCount / elapsedSec).toFixed(1);
                speedIndicator.textContent = `Streaming: ${tokPerSec} t/s`;
              }
            }
          } catch (err) {
            // Partial JSON chunk; continue
          }
        }
      }

      const totalTime = ((performance.now() - tStart) / 1000).toFixed(2);
      const finalSpeed = tokenCount > 0 ? (tokenCount / Math.max(0.01, totalTime)).toFixed(1) : 0;
      speedIndicator.textContent = `Generated ${tokenCount} tokens in ${totalTime}s (${finalSpeed} t/s)`;

    } catch (err) {
      if (err.name === 'AbortError') {
        speedIndicator.textContent = 'Generation stopped by user.';
      } else {
        console.error('[Sandwich UI] SSE stream error:', err);
        assistantMsg.content += `\n\n*(Error: ${err.message})*`;
        updateBubbleContent(bubbleEl, assistantMsg.content);
        speedIndicator.textContent = `Error: ${err.message}`;
      }
    } finally {
      setGeneratingState(false);
      abortController = null;
      saveHistory();
    }
  }

  // --- UI Helpers ---
  function setGeneratingState(generating) {
    isGenerating = generating;
    if (generating) {
      sendBtn.classList.add('hidden');
      stopBtn.classList.remove('hidden');
      typingIndicator.classList.remove('hidden');
    } else {
      stopBtn.classList.add('hidden');
      sendBtn.classList.remove('hidden');
      typingIndicator.classList.add('hidden');
    }
    scrollToBottom();
  }

  function appendMessageElement(role, content) {
    const msgEl = document.createElement('div');
    msgEl.className = `chat-message ${role}`;

    const header = document.createElement('div');
    header.className = 'message-header';
    header.innerHTML = role === 'user' 
      ? '<span class="user-badge">● User</span>' 
      : '<span class="ai-badge">● Sandwich.cpp AI</span>';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = formatMarkdown(content);

    msgEl.appendChild(header);
    msgEl.appendChild(bubble);
    chatHistory.appendChild(msgEl);
    scrollToBottom();

    return bubble;
  }

  function updateBubbleContent(bubbleEl, rawMarkdown) {
    bubbleEl.innerHTML = formatMarkdown(rawMarkdown);
  }

  function scrollToBottom() {
    chatViewport.scrollTop = chatViewport.scrollHeight;
  }

  // Simple Markdown Parser (Safe escaping + code blocks + bold/italics)
  function formatMarkdown(text) {
    if (!text) return '';
    // Escape HTML
    let escaped = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // Code blocks ```code```
    escaped = escaped.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
      return `<pre><code>${code.trim()}</code></pre>`;
    });

    // Inline code `code`
    escaped = escaped.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold **text**
    escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Italic *text*
    escaped = escaped.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Line breaks
    escaped = escaped.replace(/\n/g, '<br>');

    // Restore pre tag linebreaks
    escaped = escaped.replace(/<pre><code>([\s\S]*?)<\/code><\/pre>/g, (m, inner) => {
      return `<pre><code>${inner.replace(/<br>/g, '\n')}</code></pre>`;
    });

    return escaped;
  }

  // --- Persistence ---
  function loadSavedSettings() {
    const savedPrompt = localStorage.getItem('sandwich_system_prompt');
    if (savedPrompt && systemPromptInput) {
      systemPromptInput.value = savedPrompt;
    }

    const savedHistory = localStorage.getItem('sandwich_chat_history');
    if (savedHistory) {
      try {
        messages = JSON.parse(savedHistory);
        renderHistory();
      } catch (e) {
        messages = [];
      }
    }
  }

  function saveHistory() {
    try {
      localStorage.setItem('sandwich_chat_history', JSON.stringify(messages));
    } catch (e) {
      // Storage full
    }
  }

  function renderHistory() {
    chatHistory.innerHTML = '';
    if (messages.length === 0) {
      // Re-insert welcome hero
      const hero = document.createElement('div');
      hero.id = 'welcome-hero';
      hero.className = 'hero-screen';
      hero.innerHTML = `
        <div class="hero-icon">🥪</div>
        <h1>Sandwich<span class="accent">.cpp</span> Workstation</h1>
        <p class="hero-subtitle">High-performance local inference powered by O(1) LRU MoE caching, FlashMLA, and TurboQuant compression.</p>
        <div class="hero-features">
          <div class="feature-card">
            <span class="feat-icon">⚡</span>
            <h3>Dynamic MoE Caching</h3>
            <p>Execute 30B–35B MoE architectures locally on consumer 8GB GPUs without PCIe stalls.</p>
          </div>
          <div class="feature-card">
            <span class="feat-icon">🚀</span>
            <h3>Speculative Decoding</h3>
            <p>Paired DFlash2, MTP assistant, and small GGUF drafters for up to 3.5× faster generation.</p>
          </div>
          <div class="feature-card">
            <span class="feat-icon">💾</span>
            <h3>TurboQuant KV</h3>
            <p>75% smaller KV cache footprint with Walsh-Hadamard transform acceleration.</p>
          </div>
        </div>
      `;
      chatHistory.appendChild(hero);
      return;
    }

    for (const msg of messages) {
      if (msg.role === 'user' || msg.role === 'assistant') {
        appendMessageElement(msg.role, msg.content);
      }
    }
  }

  // Run initialization
  document.addEventListener('DOMContentLoaded', init);
})();
