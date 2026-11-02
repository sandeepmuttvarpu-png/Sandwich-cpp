// sandwich.cpp - Interactive Server Web UI
// Clean, Lightweight, 100% Functional Architecture

(function() {
  'use strict';

  // --- Constants & Storage Keys ---
  const STORAGE_SESSIONS_KEY = 'sandwich_sessions_v3';
  const STORAGE_ACTIVE_ID_KEY = 'sandwich_active_session_id_v3';
  const STORAGE_PARAMS_KEY = 'sandwich_sampling_params_v3';
  const STORAGE_SYSPROMPT_KEY = 'sandwich_system_prompt_v3';

  const DEFAULT_PARAMS = {
    temperature: 0.7,
    top_p: 0.9,
    min_p: 0.05,
    repeat_penalty: 1.10,
    max_tokens: 2048,
    system_prompt: 'You are a helpful, respectful, and honest assistant.'
  };

  // --- DOM Elements ---
  const sidebar = document.getElementById('sessions-sidebar');
  const sidebarToggleBtn = document.getElementById('sidebar-toggle-btn');
  const sessionsList = document.getElementById('sessions-list');
  const sessionSearchInput = document.getElementById('session-search-input');
  const newChatNavBtn = document.getElementById('new-chat-nav-btn');
  const newChatSidebarBtn = document.getElementById('new-chat-sidebar-btn');
  const clearChatBtn = document.getElementById('clear-chat-btn');
  const clearAllSessionsBtn = document.getElementById('clear-all-sessions-btn');
  const activeModelName = document.getElementById('active-model-name');

  // Context Inspector Elements
  const contextInspectorBtn = document.getElementById('context-inspector-btn');
  const navContextLabel = document.getElementById('nav-context-label');
  const composerContextPill = document.getElementById('composer-context-pill');
  const tokenCounter = document.getElementById('token-counter');
  const contextModal = document.getElementById('context-modal');
  const closeContextModal = document.getElementById('close-context-modal');
  const closeContextModalBtn = document.getElementById('close-context-modal-btn');
  const refreshContextBtn = document.getElementById('refresh-context-btn');
  const eraseKvSlotBtn = document.getElementById('erase-kv-slot-btn');
  const inspectorBarFill = document.getElementById('inspector-bar-fill');
  const inspectorTokensUsed = document.getElementById('inspector-tokens-used');
  const inspectorTokensTotal = document.getElementById('inspector-tokens-total');
  const inspectorPct = document.getElementById('inspector-pct');
  const inspectorModelName = document.getElementById('inspector-model-name');
  const inspectorKvPolicy = document.getElementById('inspector-kv-policy');
  const inspectorSlotState = document.getElementById('inspector-slot-state');
  const inspectorVramUsage = document.getElementById('inspector-vram-usage');
  const inspectorSlotsTbody = document.getElementById('inspector-slots-tbody');

  // Parameters Modal Elements
  const paramsToggleBtn = document.getElementById('params-toggle-btn');
  const paramsModal = document.getElementById('params-modal');
  const closeParamsModal = document.getElementById('close-params-modal');
  const resetParamsBtn = document.getElementById('reset-params-btn');
  const saveParamsBtn = document.getElementById('save-params-btn');
  const paramTemp = document.getElementById('param-temp');
  const paramTempVal = document.getElementById('param-temp-val');
  const paramTopP = document.getElementById('param-topp');
  const paramTopPVal = document.getElementById('param-topp-val');
  const paramMinP = document.getElementById('param-minp');
  const paramMinPVal = document.getElementById('param-minp-val');
  const paramRepPen = document.getElementById('param-reppen');
  const paramRepPenVal = document.getElementById('param-reppen-val');
  const paramMaxTok = document.getElementById('param-maxtok');
  const paramMaxTokVal = document.getElementById('param-maxtok-val');
  const systemPromptInput = document.getElementById('system-prompt-input');

  // Chat Elements
  const chatHistory = document.getElementById('chat-history');
  const chatForm = document.getElementById('chat-form');
  const userInput = document.getElementById('user-input');
  const sendBtn = document.getElementById('send-btn');
  const stopBtn = document.getElementById('stop-btn');
  const speedIndicator = document.getElementById('speed-indicator');
  const promptCharCount = document.getElementById('prompt-char-count');

  // --- State ---
  let sessions = [];
  let activeSessionId = null;
  let activeModel = 'sandwich.cpp';
  let isGenerating = false;
  let abortController = null;
  let samplingParams = { ...DEFAULT_PARAMS };
  let systemMaxContext = 8192;
  let systemCachedTokens = 0;

  // --- Initializer ---
  async function init() {
    loadSettings();
    loadSessions();
    setupEventListeners();
    await fetchLiveContext();
    renderCurrentSession();
  }

  // --- Local Storage Management ---
  function loadSettings() {
    try {
      const saved = localStorage.getItem(STORAGE_PARAMS_KEY);
      if (saved) {
        samplingParams = { ...DEFAULT_PARAMS, ...JSON.parse(saved) };
      }
      const savedSys = localStorage.getItem(STORAGE_SYSPROMPT_KEY);
      if (savedSys) {
        samplingParams.system_prompt = savedSys;
      }
    } catch (e) {
      samplingParams = { ...DEFAULT_PARAMS };
    }
    syncParamsModalInputs();
  }

  function saveSettings() {
    try {
      localStorage.setItem(STORAGE_PARAMS_KEY, JSON.stringify(samplingParams));
      localStorage.setItem(STORAGE_SYSPROMPT_KEY, samplingParams.system_prompt || '');
    } catch (e) {
      console.warn('Failed to save settings:', e);
    }
  }

  function syncParamsModalInputs() {
    paramTemp.value = samplingParams.temperature;
    paramTempVal.textContent = Number(samplingParams.temperature).toFixed(2);
    paramTopP.value = samplingParams.top_p;
    paramTopPVal.textContent = Number(samplingParams.top_p).toFixed(2);
    paramMinP.value = samplingParams.min_p;
    paramMinPVal.textContent = Number(samplingParams.min_p).toFixed(2);
    paramRepPen.value = samplingParams.repeat_penalty;
    paramRepPenVal.textContent = Number(samplingParams.repeat_penalty).toFixed(2);
    paramMaxTok.value = samplingParams.max_tokens;
    paramMaxTokVal.textContent = samplingParams.max_tokens;
    systemPromptInput.value = samplingParams.system_prompt || '';
  }

  function loadSessions() {
    try {
      const data = localStorage.getItem(STORAGE_SESSIONS_KEY);
      sessions = data ? JSON.parse(data) : [];
      activeSessionId = localStorage.getItem(STORAGE_ACTIVE_ID_KEY);
    } catch (e) {
      sessions = [];
      activeSessionId = null;
    }

    if (!Array.isArray(sessions) || sessions.length === 0) {
      createNewSession('Default Chat');
    } else {
      const exists = sessions.some(s => s.id === activeSessionId);
      if (!exists) {
        activeSessionId = sessions[0].id;
      }
    }
    renderSessionsList();
  }

  function saveSessions() {
    try {
      localStorage.setItem(STORAGE_SESSIONS_KEY, JSON.stringify(sessions));
      if (activeSessionId) {
        localStorage.setItem(STORAGE_ACTIVE_ID_KEY, activeSessionId);
      }
    } catch (e) {
      console.warn('Failed to persist sessions:', e);
    }
  }

  function getActiveSession() {
    return sessions.find(s => s.id === activeSessionId) || sessions[0];
  }

  function createNewSession(title = 'New Conversation') {
    const newSession = {
      id: 'sess_' + Date.now() + '_' + Math.random().toString(36).substring(2, 7),
      title: title,
      createdAt: Date.now(),
      messages: []
    };
    sessions.unshift(newSession);
    activeSessionId = newSession.id;
    saveSessions();
    renderSessionsList();
    renderCurrentSession();
    userInput.focus();
    return newSession;
  }

  // --- Real Live Context & Slot Inspector ---
  async function fetchLiveContext() {
    try {
      // 1. Fetch props
      const propsRes = await fetch('/props', { signal: AbortSignal.timeout(3000) }).catch(() => null);
      if (propsRes && propsRes.ok) {
        const props = await propsRes.json();
        if (props.default_generation_settings && props.default_generation_settings.n_ctx) {
          systemMaxContext = props.default_generation_settings.n_ctx;
        }
        if (props.default_generation_settings && props.default_generation_settings.model) {
          const rawPath = props.default_generation_settings.model;
          activeModel = rawPath.split('/').pop().replace(/\.gguf$/i, '');
          activeModelName.textContent = activeModel;
          inspectorModelName.textContent = activeModel;
        }
      }

      // 2. Fetch models
      const modelsRes = await fetch('/v1/models', { signal: AbortSignal.timeout(3000) }).catch(() => null);
      if (modelsRes && modelsRes.ok) {
        const modelsData = await modelsRes.json();
        if (modelsData.data && modelsData.data.length > 0) {
          const m = modelsData.data[0].id;
          activeModel = m.split('/').pop().replace(/\.gguf$/i, '');
          activeModelName.textContent = activeModel;
          inspectorModelName.textContent = activeModel;
        }
      }

      // 3. Fetch slots
      let slotTokens = 0;
      let slotState = 'IDLE';
      const slotsRes = await fetch('/slots', { signal: AbortSignal.timeout(3000) }).catch(() => null);
      if (slotsRes && slotsRes.ok) {
        const slotsData = await slotsRes.json();
        if (Array.isArray(slotsData) && slotsData.length > 0) {
          inspectorSlotsTbody.innerHTML = '';
          slotsData.forEach(slot => {
            const isProc = slot.is_processing || false;
            const nPast = slot.n_past || 0;
            const nCtx = slot.n_ctx || systemMaxContext;
            slotTokens += nPast;
            if (isProc) slotState = 'PROCESSING';

            const tr = document.createElement('tr');
            tr.innerHTML = `
              <td class="font-mono">#${slot.id}</td>
              <td><span class="badge ${isProc ? 'badge-warning' : 'badge-success'}">${isProc ? 'RUNNING' : 'IDLE'}</span></td>
              <td class="font-mono font-bold">${nPast.toLocaleString()} tk</td>
              <td class="font-mono">${nCtx.toLocaleString()} tk</td>
              <td>
                <button type="button" class="btn-table-action" onclick="window.eraseSlot(${slot.id})">Erase</button>
              </td>
            `;
            inspectorSlotsTbody.appendChild(tr);
          });
        }
      }

      // 4. Fetch VRAM info if available
      const vramRes = await fetch('/vram', { signal: AbortSignal.timeout(3000) }).catch(() => null);
      if (vramRes && vramRes.ok) {
        const vramData = await vramRes.json();
        if (vramData.total && vramData.free) {
          const used = Math.round((vramData.total - vramData.free) / (1024 * 1024));
          const total = Math.round(vramData.total / (1024 * 1024));
          inspectorVramUsage.textContent = `${used} / ${total} MiB`;
        }
      } else {
        inspectorVramUsage.textContent = 'CUDA 12.0 / RTX 5050';
      }

      systemCachedTokens = slotTokens;
      updateContextDisplays(slotTokens, systemMaxContext, slotState);
    } catch (err) {
      console.warn('fetchLiveContext notice:', err);
      updateContextDisplays(0, systemMaxContext, 'IDLE');
    }
  }

  function updateContextDisplays(used, total, state) {
    const pct = total > 0 ? Math.min(100, Math.round((used / total) * 100)) : 0;
    
    // Header Pill
    navContextLabel.textContent = `Context: ${used.toLocaleString()} / ${total.toLocaleString()} tk (${pct}%)`;

    // Composer Mini-Pill
    tokenCounter.textContent = `${used.toLocaleString()} / ${total.toLocaleString()} tk (${pct}%)`;

    // Modal Inspector
    inspectorTokensUsed.textContent = `Used: ${used.toLocaleString()} tk`;
    inspectorTokensTotal.textContent = `Limit: ${total.toLocaleString()} tk`;
    inspectorPct.textContent = `${pct}%`;
    inspectorBarFill.style.width = `${pct}%`;
    inspectorSlotState.textContent = state;

    if (pct > 85) {
      inspectorBarFill.style.backgroundColor = '#ef4444';
    } else if (pct > 65) {
      inspectorBarFill.style.backgroundColor = '#f59e0b';
    } else {
      inspectorBarFill.style.backgroundColor = '#10b981';
    }
  }

  window.eraseSlot = async function(slotId) {
    try {
      const res = await fetch(`/slots/${slotId}?action=erase`, { method: 'POST' });
      if (res.ok) {
        await fetchLiveContext();
      }
    } catch (e) {
      console.warn('Failed to erase slot:', e);
    }
  };

  // --- Rendering UI ---
  function renderSessionsList(query = '') {
    sessionsList.innerHTML = '';
    const filtered = query.trim() 
      ? sessions.filter(s => s.title.toLowerCase().includes(query.toLowerCase()))
      : sessions;

    filtered.forEach(sess => {
      const item = document.createElement('div');
      item.className = `session-item ${sess.id === activeSessionId ? 'active' : ''}`;
      item.onclick = () => {
        activeSessionId = sess.id;
        saveSessions();
        renderSessionsList();
        renderCurrentSession();
      };

      const title = document.createElement('span');
      title.className = 'session-title';
      title.textContent = sess.title || 'Conversation';

      const delBtn = document.createElement('button');
      delBtn.className = 'session-del-btn';
      delBtn.innerHTML = '&times;';
      delBtn.title = 'Delete Chat';
      delBtn.onclick = (e) => {
        e.stopPropagation();
        deleteSession(sess.id);
      };

      item.appendChild(title);
      item.appendChild(delBtn);
      sessionsList.appendChild(item);
    });
  }

  function deleteSession(id) {
    if (sessions.length <= 1) {
      sessions = [];
      createNewSession('Default Chat');
      return;
    }
    sessions = sessions.filter(s => s.id !== id);
    if (activeSessionId === id) {
      activeSessionId = sessions[0].id;
    }
    saveSessions();
    renderSessionsList();
    renderCurrentSession();
  }

  function renderCurrentSession() {
    chatHistory.innerHTML = '';
    const sess = getActiveSession();

    if (!sess.messages || sess.messages.length === 0) {
      renderWelcomeScreen();
      return;
    }

    sess.messages.forEach((msg, idx) => {
      appendMessageElement(msg.role, msg.content, idx);
    });

    scrollToBottom();
  }

  function renderWelcomeScreen() {
    const welcome = document.createElement('div');
    welcome.className = 'welcome-box';
    welcome.innerHTML = `
      <div class="welcome-logo">sandwich<span class="accent">.cpp</span></div>
      <p class="welcome-text">High-performance inference engine for consumer-grade hardware.</p>
      <div class="welcome-features">
        <div class="feature-chip">⚡ TurboQuant KV Cache</div>
        <div class="feature-chip">🔥 MoE 3-Tier Dynamic Memory</div>
        <div class="feature-chip">🎯 P-Core Thread Scheduler</div>
      </div>
    `;
    chatHistory.appendChild(welcome);
  }

  function appendMessageElement(role, content, msgIdx = -1) {
    const msgEl = document.createElement('div');
    msgEl.className = `chat-message ${role}`;
    msgEl.dataset.idx = msgIdx;

    const header = document.createElement('div');
    header.className = 'message-header';
    header.innerHTML = role === 'user' 
      ? '<span class="user-badge">You</span>' 
      : '<span class="ai-badge">sandwich.cpp</span>';

    const copyBtn = document.createElement('button');
    copyBtn.className = 'msg-copy-btn';
    copyBtn.textContent = 'Copy';
    copyBtn.onclick = () => {
      navigator.clipboard.writeText(content);
      copyBtn.textContent = 'Copied!';
      setTimeout(() => copyBtn.textContent = 'Copy', 1500);
    };
    header.appendChild(copyBtn);

    const body = document.createElement('div');
    body.className = 'message-content';
    body.innerHTML = formatMarkdown(content);

    msgEl.appendChild(header);
    msgEl.appendChild(body);
    chatHistory.appendChild(msgEl);

    // Attach copy buttons to generated code blocks
    body.querySelectorAll('pre').forEach(pre => {
      const codeBtn = document.createElement('button');
      codeBtn.className = 'code-copy-btn';
      codeBtn.textContent = 'Copy code';
      codeBtn.onclick = () => {
        const text = pre.querySelector('code') ? pre.querySelector('code').innerText : pre.innerText;
        navigator.clipboard.writeText(text);
        codeBtn.textContent = 'Copied!';
        setTimeout(() => codeBtn.textContent = 'Copy code', 1500);
      };
      pre.appendChild(codeBtn);
    });

    return msgEl;
  }

  function formatMarkdown(text) {
    if (!text) return '';
    let escaped = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // Fenced code blocks
    escaped = escaped.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, function(_, lang, code) {
      return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
    });

    // Inline code
    escaped = escaped.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold & italic
    escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    escaped = escaped.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Line breaks
    escaped = escaped.replace(/\n/g, '<br>');

    return escaped;
  }

  function scrollToBottom() {
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }

  // --- Chat Submission & Streaming ---
  async function handleSendMessage() {
    const rawText = userInput.value.trim();
    if (!rawText || isGenerating) return;

    const sess = getActiveSession();

    // Remove welcome screen if present
    const welcome = chatHistory.querySelector('.welcome-box');
    if (welcome) welcome.remove();

    // 1. Add User Message
    const userMsg = { role: 'user', content: rawText };
    sess.messages.push(userMsg);
    appendMessageElement('user', rawText, sess.messages.length - 1);

    userInput.value = '';
    userInput.style.height = 'auto';
    promptCharCount.textContent = '0 chars';
    scrollToBottom();

    // 2. Prepare Assistant Bubble
    const assistantMsg = { role: 'assistant', content: '' };
    sess.messages.push(assistantMsg);
    const bubbleEl = appendMessageElement('assistant', '', sess.messages.length - 1);
    const contentEl = bubbleEl.querySelector('.message-content');

    setGeneratingState(true);
    abortController = new AbortController();

    const tStart = performance.now();
    let tokenCount = 0;

    // Build payload messages
    const payloadMessages = [];
    const activeSysPrompt = samplingParams.system_prompt || '';
    if (activeSysPrompt) {
      payloadMessages.push({ role: 'system', content: activeSysPrompt });
    }
    sess.messages.slice(0, -1).forEach(m => {
      payloadMessages.push({ role: m.role, content: m.content });
    });

    try {
      speedIndicator.textContent = 'Generating...';
      const response = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: abortController.signal,
        body: JSON.stringify({
          messages: payloadMessages,
          stream: true,
          temperature: parseFloat(samplingParams.temperature),
          top_p: parseFloat(samplingParams.top_p),
          min_p: parseFloat(samplingParams.min_p),
          repeat_penalty: parseFloat(samplingParams.repeat_penalty),
          max_tokens: parseInt(samplingParams.max_tokens, 10)
        })
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // preserve unparsed fragment

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
              contentEl.innerHTML = formatMarkdown(assistantMsg.content);
              scrollToBottom();

              const elapsedSec = (performance.now() - tStart) / 1000;
              if (elapsedSec > 0.5) {
                const liveTps = (tokenCount / elapsedSec).toFixed(1);
                speedIndicator.textContent = `${liveTps} t/s (${tokenCount} tokens)`;
              }
            }
          } catch (pe) {
            // ignore non-json SSE lines
          }
        }
      }

      const tTotalSec = (performance.now() - tStart) / 1000;
      const finalTps = tTotalSec > 0 ? (tokenCount / tTotalSec).toFixed(1) : '0';
      speedIndicator.textContent = `Completed: ${finalTps} t/s (${tokenCount} tokens in ${tTotalSec.toFixed(2)}s)`;

      // Auto title session from first prompt
      if (sess.messages.length <= 2) {
        sess.title = rawText.length > 28 ? rawText.substring(0, 28) + '...' : rawText;
        renderSessionsList();
      }
      saveSessions();
      await fetchLiveContext();

    } catch (err) {
      if (err.name === 'AbortError') {
        speedIndicator.textContent = 'Generation stopped by user.';
      } else {
        speedIndicator.textContent = `Error: ${err.message}`;
        contentEl.innerHTML = `<span class="text-danger font-mono">Error: ${err.message}</span>`;
      }
    } finally {
      setGeneratingState(false);
      abortController = null;
      saveSessions();
    }
  }

  function setGeneratingState(running) {
    isGenerating = running;
    sendBtn.classList.toggle('hidden', running);
    stopBtn.classList.toggle('hidden', !running);
    userInput.disabled = running;
    if (!running) {
      userInput.focus();
    }
  }

  // --- Event Listeners Setup ---
  function setupEventListeners() {
    // Sidebar toggle
    sidebarToggleBtn.addEventListener('click', () => {
      sidebar.classList.toggle('collapsed');
    });

    // Session creation & clearing
    newChatNavBtn.addEventListener('click', () => createNewSession());
    newChatSidebarBtn.addEventListener('click', () => createNewSession());

    clearChatBtn.addEventListener('click', () => {
      if (confirm('Clear messages in this conversation?')) {
        const sess = getActiveSession();
        sess.messages = [];
        saveSessions();
        renderCurrentSession();
      }
    });

    clearAllSessionsBtn.addEventListener('click', () => {
      if (confirm('Delete all conversations and start fresh?')) {
        sessions = [];
        createNewSession('Default Chat');
      }
    });

    sessionSearchInput.addEventListener('input', (e) => {
      renderSessionsList(e.target.value);
    });

    // Context Inspector Modals
    contextInspectorBtn.addEventListener('click', async () => {
      contextModal.classList.remove('hidden');
      await fetchLiveContext();
    });
    composerContextPill.addEventListener('click', async () => {
      contextModal.classList.remove('hidden');
      await fetchLiveContext();
    });
    closeContextModal.addEventListener('click', () => {
      contextModal.classList.add('hidden');
    });
    closeContextModalBtn.addEventListener('click', () => {
      contextModal.classList.add('hidden');
    });
    refreshContextBtn.addEventListener('click', async () => {
      refreshContextBtn.disabled = true;
      await fetchLiveContext();
      refreshContextBtn.disabled = false;
    });
    eraseKvSlotBtn.addEventListener('click', async () => {
      if (confirm('Erase Slot 0 KV cache allocation now?')) {
        await window.eraseSlot(0);
        await fetchLiveContext();
      }
    });

    // Parameters Modal
    paramsToggleBtn.addEventListener('click', () => {
      syncParamsModalInputs();
      paramsModal.classList.remove('hidden');
    });
    closeParamsModal.addEventListener('click', () => {
      paramsModal.classList.add('hidden');
    });
    resetParamsBtn.addEventListener('click', () => {
      samplingParams = { ...DEFAULT_PARAMS };
      syncParamsModalInputs();
    });
    saveParamsBtn.addEventListener('click', () => {
      samplingParams.temperature = parseFloat(paramTemp.value);
      samplingParams.top_p = parseFloat(paramTopP.value);
      samplingParams.min_p = parseFloat(paramMinP.value);
      samplingParams.repeat_penalty = parseFloat(paramRepPen.value);
      samplingParams.max_tokens = parseInt(paramMaxTok.value, 10);
      samplingParams.system_prompt = systemPromptInput.value.trim();
      saveSettings();
      paramsModal.classList.add('hidden');
    });

    // Real-time slider labels
    paramTemp.addEventListener('input', () => paramTempVal.textContent = Number(paramTemp.value).toFixed(2));
    paramTopP.addEventListener('input', () => paramTopPVal.textContent = Number(paramTopP.value).toFixed(2));
    paramMinP.addEventListener('input', () => paramMinPVal.textContent = Number(paramMinP.value).toFixed(2));
    paramRepPen.addEventListener('input', () => paramRepPenVal.textContent = Number(paramRepPen.value).toFixed(2));
    paramMaxTok.addEventListener('input', () => paramMaxTokVal.textContent = paramMaxTok.value);

    // Stop button
    stopBtn.addEventListener('click', () => {
      if (abortController) {
        abortController.abort();
      }
    });

    // Chat form input handling
    sendBtn.addEventListener('click', handleSendMessage);
    userInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
      }
    });

    userInput.addEventListener('input', () => {
      // Auto-grow textarea
      userInput.style.height = 'auto';
      userInput.style.height = Math.min(userInput.scrollHeight, 180) + 'px';
      promptCharCount.textContent = `${userInput.value.length} chars`;
    });

    // Close modals on escape key or clicking outside
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        contextModal.classList.add('hidden');
        paramsModal.classList.add('hidden');
      }
    });

    // Periodically update live context every 10 seconds
    setInterval(fetchLiveContext, 10000);
  }

  // Run on page load
  document.addEventListener('DOMContentLoaded', init);
})();
