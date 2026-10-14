/**
 * 🥪 Sandwich.cpp - Web UI & Telemetry Client
 * Features:
 *  - Sessions Dashboard with Multiple Chat Sessions (Create, Switch, Rename, Delete, Search)
 *  - Vanilla llama.cpp Prompt Input Box with Multimodal File Attachments
 *  - Drag & Drop, Clipboard Image Paste, and Microphone Voice Recording
 *  - Dynamic Model Picker and Reasoning (Thinking) Toggle
 *  - SSE Streaming Chat Engine with Token/sec Telemetry
 */

(() => {
  'use strict';

  // --- Constants & Storage Keys ---
  const STORAGE_SESSIONS_KEY = 'sandwich_sessions_v1';
  const STORAGE_ACTIVE_ID_KEY = 'sandwich_active_session_id';
  const STORAGE_SYSPROMPT_KEY = 'sandwich_system_prompt';

  // --- DOM Elements ---
  // Top Navbar
  const sidebarToggleBtn = document.getElementById('sidebar-toggle-btn');
  const newChatNavBtn = document.getElementById('new-chat-nav-btn');
  const hudToggleBtn = document.getElementById('hud-toggle-btn');
  const clearChatBtn = document.getElementById('clear-chat-btn');
  const activeModelName = document.getElementById('active-model-name');

  // Sidebar Dashboard
  const sessionsSidebar = document.getElementById('sessions-sidebar');
  const newChatSidebarBtn = document.getElementById('new-chat-sidebar-btn');
  const sessionSearchInput = document.getElementById('session-search-input');
  const sessionsList = document.getElementById('sessions-list');
  const clearAllSessionsBtn = document.getElementById('clear-all-sessions-btn');

  // Telemetry HUD
  const telemetryHud = document.getElementById('telemetry-hud');

  // Chat Viewport & History
  const chatViewport = document.getElementById('chat-viewport');
  const chatHistory = document.getElementById('chat-history');
  const dropOverlay = document.getElementById('drop-overlay');
  const typingIndicator = document.getElementById('typing-indicator');

  // Composer Form & Inputs
  const chatForm = document.getElementById('chat-form');
  const userInput = document.getElementById('user-input');
  const attachmentsTray = document.getElementById('attachments-tray');
  const addAttachmentBtn = document.getElementById('add-attachment-btn');
  const attachmentMenu = document.getElementById('attachment-menu');
  const micBtn = document.getElementById('mic-btn');
  const recordingTimer = document.getElementById('recording-timer');
  const reasoningToggleBtn = document.getElementById('reasoning-toggle-btn');
  const modelPickerBtn = document.getElementById('model-picker-btn');
  const selectedModelLabel = document.getElementById('selected-model-label');
  const modelPickerDropdown = document.getElementById('model-picker-dropdown');
  const tokenCounter = document.getElementById('token-counter');
  const sendBtn = document.getElementById('send-btn');
  const stopBtn = document.getElementById('stop-btn');
  const speedIndicator = document.getElementById('speed-indicator');

  // Hidden File Inputs
  const fileInputAll = document.getElementById('file-input-all');
  const fileInputImages = document.getElementById('file-input-images');
  const fileInputAudio = document.getElementById('file-input-audio');

  // Menu items in attachment dropdown
  const menuUploadFiles = document.getElementById('menu-upload-files');
  const menuUploadImages = document.getElementById('menu-upload-images');
  const menuUploadAudio = document.getElementById('menu-upload-audio');
  const menuSystemPrompt = document.getElementById('menu-system-prompt');

  // System Prompt Modal
  const systemPromptModal = document.getElementById('system-prompt-modal');
  const closeSyspromptModal = document.getElementById('close-sysprompt-modal');
  const systemPromptInput = document.getElementById('system-prompt-input');
  const saveSyspromptBtn = document.getElementById('save-sysprompt-btn');
  const resetSyspromptBtn = document.getElementById('reset-sysprompt-btn');

  // --- State Variables ---
  let sessions = [];
  let activeSessionId = null;
  let activeModel = 'sandwich-model';
  let availableModels = [];
  let currentAttachments = []; // Array of { id, name, type, mimeType, size, dataUrl, textContent }
  let isGenerating = false;
  let abortController = null;
  let isReasoningEnabled = false;

  // Audio Recording State
  let mediaRecorder = null;
  let audioChunks = [];
  let recordingStartTime = 0;
  let recordingInterval = null;

  // --- Initializer ---
  async function init() {
    loadSessionsFromStorage();
    setupEventListeners();
    await fetchModels();
    renderSessionsList();
    renderCurrentSession();
    updateTokenCounter();
  }

  // =========================================================================
  // Sessions Management (Dashboard)
  // =========================================================================

  function createNewSession(title = 'New Conversation') {
    const newSession = {
      id: 'sess_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6),
      title: title,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: [],
      systemPrompt: localStorage.getItem(STORAGE_SYSPROMPT_KEY) || ''
    };
    sessions.unshift(newSession);
    activeSessionId = newSession.id;
    saveSessionsToStorage();
    renderSessionsList();
    renderCurrentSession();
    userInput.focus();
    return newSession;
  }

  function getActiveSession() {
    return sessions.find(s => s.id === activeSessionId) || sessions[0];
  }

  function switchSession(sessionId) {
    if (activeSessionId === sessionId) return;
    if (isGenerating && abortController) {
      abortController.abort();
      setGeneratingState(false);
    }
    activeSessionId = sessionId;
    localStorage.setItem(STORAGE_ACTIVE_ID_KEY, activeSessionId);
    currentAttachments = [];
    renderAttachmentsTray();
    renderSessionsList();
    renderCurrentSession();
    userInput.focus();
  }

  function renameSession(sessionId, newTitle) {
    const sess = sessions.find(s => s.id === sessionId);
    if (sess) {
      sess.title = newTitle.trim() || 'Untitled Session';
      sess.updatedAt = Date.now();
      saveSessionsToStorage();
      renderSessionsList();
    }
  }

  function deleteSession(sessionId, event) {
    if (event) event.stopPropagation();
    if (sessions.length <= 1) {
      if (confirm('Delete this conversation and start fresh?')) {
        sessions = [];
        createNewSession();
      }
      return;
    }

    if (confirm('Delete this session permanently?')) {
      sessions = sessions.filter(s => s.id !== sessionId);
      if (activeSessionId === sessionId) {
        activeSessionId = sessions[0].id;
      }
      saveSessionsToStorage();
      renderSessionsList();
      renderCurrentSession();
    }
  }

  function clearAllSessions() {
    if (confirm('Are you sure you want to delete all saved sessions? This cannot be undone.')) {
      if (isGenerating && abortController) abortController.abort();
      sessions = [];
      localStorage.removeItem(STORAGE_SESSIONS_KEY);
      localStorage.removeItem(STORAGE_ACTIVE_ID_KEY);
      createNewSession();
    }
  }

  function loadSessionsFromStorage() {
    try {
      const stored = localStorage.getItem(STORAGE_SESSIONS_KEY);
      if (stored) {
        sessions = JSON.parse(stored);
      }
    } catch (e) {
      console.warn('[Sandwich Sessions] Failed to parse stored sessions:', e);
      sessions = [];
    }

    if (!Array.isArray(sessions) || sessions.length === 0) {
      createNewSession('Welcome to Sandwich.cpp');
    } else {
      const savedActiveId = localStorage.getItem(STORAGE_ACTIVE_ID_KEY);
      if (savedActiveId && sessions.some(s => s.id === savedActiveId)) {
        activeSessionId = savedActiveId;
      } else {
        activeSessionId = sessions[0].id;
      }
    }
  }

  function saveSessionsToStorage() {
    try {
      localStorage.setItem(STORAGE_SESSIONS_KEY, JSON.stringify(sessions));
      localStorage.setItem(STORAGE_ACTIVE_ID_KEY, activeSessionId);
    } catch (e) {
      console.warn('[Sandwich Sessions] Storage error (quota exceeded):', e);
    }
  }

  function renderSessionsList(filterQuery = '') {
    sessionsList.innerHTML = '';
    const query = (filterQuery || sessionSearchInput.value || '').trim().toLowerCase();

    const filtered = query 
      ? sessions.filter(s => s.title.toLowerCase().includes(query))
      : sessions;

    if (filtered.length === 0) {
      const emptyEl = document.createElement('div');
      emptyEl.style.padding = '20px 10px';
      emptyEl.style.textAlign = 'center';
      emptyEl.style.color = 'var(--text-muted)';
      emptyEl.style.fontSize = '0.8rem';
      emptyEl.textContent = 'No matching sessions';
      sessionsList.appendChild(emptyEl);
      return;
    }

    filtered.forEach(sess => {
      const item = document.createElement('div');
      item.className = `session-item ${sess.id === activeSessionId ? 'active' : ''}`;
      item.onclick = () => switchSession(sess.id);

      const info = document.createElement('div');
      info.className = 'session-info';

      const title = document.createElement('span');
      title.className = 'session-title';
      title.textContent = sess.title || 'Untitled Session';

      const time = document.createElement('span');
      time.className = 'session-time';
      time.textContent = formatTimestamp(sess.updatedAt || sess.createdAt);

      info.appendChild(title);
      info.appendChild(time);

      const actions = document.createElement('div');
      actions.className = 'session-actions';

      const editBtn = document.createElement('button');
      editBtn.className = 'session-action-btn';
      editBtn.title = 'Rename Session';
      editBtn.innerHTML = '✏️';
      editBtn.onclick = (e) => {
        e.stopPropagation();
        const newTitle = prompt('Rename session to:', sess.title);
        if (newTitle !== null) renameSession(sess.id, newTitle);
      };

      const delBtn = document.createElement('button');
      delBtn.className = 'session-action-btn delete-btn';
      delBtn.title = 'Delete Session';
      delBtn.innerHTML = '🗑️';
      delBtn.onclick = (e) => deleteSession(sess.id, e);

      actions.appendChild(editBtn);
      actions.appendChild(delBtn);

      item.appendChild(info);
      item.appendChild(actions);
      sessionsList.appendChild(item);
    });
  }

  function formatTimestamp(timestamp) {
    if (!timestamp) return '';
    const diffMs = Date.now() - timestamp;
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHours = Math.floor(diffMin / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffSec < 60) return 'Just now';
    if (diffMin < 60) return `${diffMin}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays}d ago`;
    return new Date(timestamp).toLocaleDateString([], { month: 'short', day: 'numeric' });
  }

  function renderCurrentSession() {
    chatHistory.innerHTML = '';
    const sess = getActiveSession();
    if (!sess || !sess.messages || sess.messages.length === 0) {
      renderWelcomeHero();
      return;
    }

    sess.messages.forEach(msg => {
      appendMessageElement(msg.role, msg.content, msg.attachments);
    });
    scrollToBottom();
  }

  function renderWelcomeHero() {
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
  }

  // =========================================================================
  // Vanilla llama.cpp Prompt Input Box & File Attachments
  // =========================================================================

  function handleFileSelection(files) {
    if (!files || files.length === 0) return;
    const fileList = Array.from(files);

    fileList.forEach(file => {
      const fileId = 'att_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5);
      const isImage = file.type.startsWith('image/');
      const isAudio = file.type.startsWith('audio/');
      const isText = file.type.startsWith('text/') || 
                     /\.(md|txt|json|csv|py|cpp|c|h|hpp|js|ts|html|css|xml|yml|yaml|log)$/i.test(file.name);

      const attachmentObj = {
        id: fileId,
        name: file.name,
        type: isImage ? 'image' : isAudio ? 'audio' : isText ? 'text' : 'file',
        mimeType: file.type || 'application/octet-stream',
        size: file.size,
        dataUrl: null,
        textContent: null
      };

      if (isImage || isAudio) {
        const reader = new FileReader();
        reader.onload = (e) => {
          attachmentObj.dataUrl = e.target.result;
          currentAttachments.push(attachmentObj);
          renderAttachmentsTray();
          userInput.focus();
        };
        reader.readAsDataURL(file);
      } else if (isText) {
        const reader = new FileReader();
        reader.onload = (e) => {
          attachmentObj.textContent = e.target.result;
          currentAttachments.push(attachmentObj);
          renderAttachmentsTray();
          userInput.focus();
        };
        reader.readAsText(file);
      } else {
        // Generic binary or PDF file
        const reader = new FileReader();
        reader.onload = (e) => {
          attachmentObj.dataUrl = e.target.result;
          currentAttachments.push(attachmentObj);
          renderAttachmentsTray();
          userInput.focus();
        };
        reader.readAsDataURL(file);
      }
    });

    // Reset inputs so same file can be re-selected if removed
    fileInputAll.value = '';
    fileInputImages.value = '';
    fileInputAudio.value = '';
  }

  function removeAttachment(attachmentId) {
    currentAttachments = currentAttachments.filter(a => a.id !== attachmentId);
    renderAttachmentsTray();
    updateSubmitButtonState();
  }

  function renderAttachmentsTray() {
    attachmentsTray.innerHTML = '';
    if (currentAttachments.length === 0) {
      attachmentsTray.classList.add('hidden');
      return;
    }

    attachmentsTray.classList.remove('hidden');

    currentAttachments.forEach(att => {
      const chip = document.createElement('div');
      chip.className = 'attachment-chip';

      if (att.type === 'image' && att.dataUrl) {
        const img = document.createElement('img');
        img.src = att.dataUrl;
        img.alt = att.name;
        chip.appendChild(img);
      } else {
        const iconSpan = document.createElement('span');
        iconSpan.className = 'chip-icon';
        iconSpan.textContent = att.type === 'audio' ? '🎵' : att.type === 'text' ? '📝' : '📄';
        chip.appendChild(iconSpan);
      }

      const meta = document.createElement('div');
      meta.className = 'chip-meta';

      const nameSpan = document.createElement('span');
      nameSpan.className = 'chip-name';
      nameSpan.textContent = att.name;
      nameSpan.title = att.name;

      const sizeSpan = document.createElement('span');
      sizeSpan.className = 'chip-size';
      sizeSpan.textContent = formatBytes(att.size);

      meta.appendChild(nameSpan);
      meta.appendChild(sizeSpan);
      chip.appendChild(meta);

      const removeBtn = document.createElement('button');
      removeBtn.className = 'chip-remove-btn';
      removeBtn.type = 'button';
      removeBtn.innerHTML = '&times;';
      removeBtn.title = 'Remove attachment';
      removeBtn.onclick = () => removeAttachment(att.id);

      chip.appendChild(removeBtn);
      attachmentsTray.appendChild(chip);
    });

    updateSubmitButtonState();
  }

  function formatBytes(bytes, decimals = 1) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  }

  function updateTokenCounter() {
    const text = userInput.value;
    const chars = text.length;
    const estTokens = Math.ceil(chars / 4);
    tokenCounter.textContent = `${chars} chars • ~${estTokens} tok`;
    updateSubmitButtonState();
  }

  function updateSubmitButtonState() {
    const hasText = userInput.value.trim().length > 0;
    const hasAttachments = currentAttachments.length > 0;
    sendBtn.disabled = !hasText && !hasAttachments;
  }

  // =========================================================================
  // Audio Voice Recording (Microphone)
  // =========================================================================

  async function toggleAudioRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      stopAudioRecording();
    } else {
      startAudioRecording();
    }
  }

  async function startAudioRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert('Audio recording is not supported in this browser.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks = [];
      mediaRecorder = new MediaRecorder(stream);

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        const reader = new FileReader();
        reader.onload = (e) => {
          const fileId = 'att_rec_' + Date.now();
          currentAttachments.push({
            id: fileId,
            name: `Voice-Recording-${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}.webm`,
            type: 'audio',
            mimeType: 'audio/webm',
            size: audioBlob.size,
            dataUrl: e.target.result,
            textContent: null
          });
          renderAttachmentsTray();
        };
        reader.readAsDataURL(audioBlob);

        // Stop all media tracks to release microphone
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      micBtn.classList.add('recording');
      recordingTimer.classList.remove('hidden');
      recordingStartTime = Date.now();

      recordingInterval = setInterval(() => {
        const elapsedSec = Math.floor((Date.now() - recordingStartTime) / 1000);
        const m = Math.floor(elapsedSec / 60);
        const s = elapsedSec % 60;
        recordingTimer.textContent = `${m}:${s < 10 ? '0' : ''}${s}`;
      }, 500);

    } catch (err) {
      console.error('[Sandwich UI] Microphone access denied or failed:', err);
      alert('Unable to access microphone: ' + err.message);
    }
  }

  function stopAudioRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
    }
    micBtn.classList.remove('recording');
    recordingTimer.classList.add('hidden');
    recordingTimer.textContent = '0:00';
    if (recordingInterval) {
      clearInterval(recordingInterval);
      recordingInterval = null;
    }
  }

  // =========================================================================
  // Models Fetcher & Switcher
  // =========================================================================

  async function fetchModels() {
    try {
      const resp = await fetch('/v1/models');
      if (resp.ok) {
        const data = await resp.json();
        if (data && data.data && data.data.length > 0) {
          availableModels = data.data.map(m => m.id);
          activeModel = availableModels[0];
          updateModelLabels(activeModel);
          renderModelPickerDropdown();
          return;
        }
      }
    } catch (e) {
      console.warn('[Sandwich UI] /v1/models query failed:', e);
    }

    try {
      const respProps = await fetch('/props');
      if (respProps.ok) {
        const props = await respProps.json();
        if (props.default_generation_settings && props.default_generation_settings.model) {
          activeModel = props.default_generation_settings.model;
          availableModels = [activeModel];
          updateModelLabels(activeModel);
          renderModelPickerDropdown();
          return;
        }
      }
    } catch (e) {
      console.warn('[Sandwich UI] /props query failed:', e);
    }

    activeModel = 'Sandwich.cpp Engine';
    availableModels = [activeModel];
    updateModelLabels(activeModel);
    renderModelPickerDropdown();
  }

  function updateModelLabels(modelId) {
    const shortName = modelId.split('/').pop().replace(/\.gguf$/i, '');
    activeModelName.textContent = shortName;
    activeModelName.title = modelId;
    selectedModelLabel.textContent = shortName;
    selectedModelLabel.title = modelId;
  }

  function renderModelPickerDropdown() {
    modelPickerDropdown.innerHTML = '';
    availableModels.forEach(m => {
      const item = document.createElement('button');
      item.type = 'button';
      item.className = 'menu-item';
      const short = m.split('/').pop().replace(/\.gguf$/i, '');
      item.innerHTML = `<span class="menu-icon">${m === activeModel ? '✓' : '•'}</span><span class="menu-text">${short}</span>`;
      item.onclick = () => {
        activeModel = m;
        updateModelLabels(m);
        modelPickerDropdown.classList.add('hidden');
      };
      modelPickerDropdown.appendChild(item);
    });
  }

  // =========================================================================
  // Send Message & Streaming Pipeline
  // =========================================================================

  async function handleSendMessage() {
    const rawText = userInput.value.trim();
    if ((!rawText && currentAttachments.length === 0) || isGenerating) return;

    // Retrieve active session
    const sess = getActiveSession();
    if (!sess) return;

    // Auto-name session from first message
    if (sess.messages.length === 0) {
      const autoTitle = rawText ? rawText.slice(0, 32) + (rawText.length > 32 ? '...' : '') : 'File Analysis';
      sess.title = autoTitle;
      renderSessionsList();
    }

    // Hide welcome hero if present
    const hero = document.getElementById('welcome-hero');
    if (hero) hero.remove();

    // Prepare message attachments clone
    const messageAttachments = [...currentAttachments];

    // User Message Object
    const userMsg = {
      role: 'user',
      content: rawText,
      attachments: messageAttachments
    };
    sess.messages.push(userMsg);
    appendMessageElement('user', rawText, messageAttachments);

    // Reset Composer
    userInput.value = '';
    userInput.style.height = 'auto';
    currentAttachments = [];
    renderAttachmentsTray();
    updateTokenCounter();

    // Assistant Placeholder Message
    const assistantMsg = {
      role: 'assistant',
      content: ''
    };
    sess.messages.push(assistantMsg);
    const bubbleEl = appendMessageElement('assistant', '');

    setGeneratingState(true);
    abortController = new AbortController();

    const tStart = performance.now();
    let tokenCount = 0;

    // Construct Payload for /v1/chat/completions
    const payloadMessages = [];

    // 1. System Prompt (Session override or global)
    const activeSysPrompt = sess.systemPrompt || localStorage.getItem(STORAGE_SYSPROMPT_KEY);
    if (activeSysPrompt) {
      payloadMessages.push({ role: 'system', content: activeSysPrompt });
    }

    // 2. Prior History (last 8 messages for memory efficiency)
    const prior = sess.messages.slice(0, -2).slice(-8);
    for (const msg of prior) {
      payloadMessages.push(formatPayloadMessage(msg));
    }

    // 3. Current User Message (formatted with attachments)
    payloadMessages.push(formatPayloadMessage(userMsg));

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
        buffer = lines.pop();

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
              if (elapsedSec > 0.4) {
                const tokPerSec = (tokenCount / elapsedSec).toFixed(1);
                speedIndicator.textContent = `Streaming: ${tokPerSec} t/s`;
              }
            }
          } catch (e) {
            // Partial JSON chunk
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
      sess.updatedAt = Date.now();
      saveSessionsToStorage();
      renderSessionsList();
      setGeneratingState(false);
      abortController = null;
      userInput.focus();
    }
  }

  function formatPayloadMessage(msg) {
    if (msg.role !== 'user' || !msg.attachments || msg.attachments.length === 0) {
      return { role: msg.role, content: msg.content };
    }

    // Check for images
    const images = msg.attachments.filter(a => a.type === 'image' && a.dataUrl);
    const textFiles = msg.attachments.filter(a => a.textContent);

    let prefixText = '';
    textFiles.forEach(tf => {
      prefixText += `--- File: ${tf.name} ---\n${tf.textContent}\n\n`;
    });

    const combinedText = (prefixText + (msg.content || '')).trim();

    if (images.length > 0) {
      const parts = [];
      if (combinedText) {
        parts.push({ type: 'text', text: combinedText });
      }
      images.forEach(img => {
        parts.push({
          type: 'image_url',
          image_url: { url: img.dataUrl }
        });
      });
      return { role: 'user', content: parts };
    }

    return { role: 'user', content: combinedText };
  }

  // =========================================================================
  // DOM Message Bubbles Rendering
  // =========================================================================

  function appendMessageElement(role, content, attachments = []) {
    const msgEl = document.createElement('div');
    msgEl.className = `chat-message ${role}`;

    const header = document.createElement('div');
    header.className = 'message-header';
    header.innerHTML = role === 'user' 
      ? '<span class="user-badge">● User</span>' 
      : '<span class="ai-badge">● Sandwich.cpp AI</span>';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    // Render attachments if present
    if (attachments && attachments.length > 0) {
      const attsContainer = document.createElement('div');
      attsContainer.className = 'bubble-attachments';
      attachments.forEach(att => {
        if (att.type === 'image' && att.dataUrl) {
          const img = document.createElement('img');
          img.className = 'bubble-attachment-thumb';
          img.src = att.dataUrl;
          img.alt = att.name;
          attsContainer.appendChild(img);
        } else {
          const filePill = document.createElement('div');
          filePill.className = 'bubble-attachment-file';
          filePill.innerHTML = `<span>${att.type === 'audio' ? '🎵' : '📄'}</span><span>${escapeHtml(att.name)}</span>`;
          attsContainer.appendChild(filePill);
        }
      });
      bubble.appendChild(attsContainer);
    }

    const textEl = document.createElement('div');
    textEl.className = 'bubble-text';
    textEl.innerHTML = formatMarkdown(content);
    bubble.appendChild(textEl);

    msgEl.appendChild(header);
    msgEl.appendChild(bubble);
    chatHistory.appendChild(msgEl);
    scrollToBottom();

    return textEl;
  }

  function updateBubbleContent(bubbleTextEl, rawMarkdown) {
    bubbleTextEl.innerHTML = formatMarkdown(rawMarkdown);
  }

  function scrollToBottom() {
    chatViewport.scrollTop = chatViewport.scrollHeight;
  }

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
  }

  // Markdown Formatting
  function formatMarkdown(text) {
    if (!text) return '';
    let escaped = escapeHtml(text);

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

  function escapeHtml(text) {
    if (!text) return '';
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  // =========================================================================
  // Event Listeners & Interactions
  // =========================================================================

  function setupEventListeners() {
    // 1. Sidebar Toggle
    sidebarToggleBtn.addEventListener('click', () => {
      sessionsSidebar.classList.toggle('collapsed');
    });

    // Keyboard shortcut Ctrl+B or Cmd+B for sidebar
    window.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'b') {
        e.preventDefault();
        sessionsSidebar.classList.toggle('collapsed');
      }
    });

    // 2. New Chat Buttons
    newChatNavBtn.addEventListener('click', () => createNewSession());
    newChatSidebarBtn.addEventListener('click', () => createNewSession());

    // 3. Search Sessions
    sessionSearchInput.addEventListener('input', (e) => {
      renderSessionsList(e.target.value);
    });

    // 4. Clear All Sessions
    clearAllSessionsBtn.addEventListener('click', clearAllSessions);

    // 5. Clear Current Chat
    clearChatBtn.addEventListener('click', () => {
      const sess = getActiveSession();
      if (sess && confirm('Clear messages in this conversation?')) {
        sess.messages = [];
        sess.updatedAt = Date.now();
        saveSessionsToStorage();
        renderCurrentSession();
      }
    });

    // 6. Telemetry HUD Toggle
    hudToggleBtn.addEventListener('click', () => {
      const isHidden = telemetryHud.classList.toggle('hidden');
      hudToggleBtn.classList.toggle('active', !isHidden);
    });

    // 7. Auto-Resize Input Textarea & Token Counting
    userInput.addEventListener('input', () => {
      userInput.style.height = 'auto';
      userInput.style.height = Math.max(52, Math.min(userInput.scrollHeight, 300)) + 'px';
      updateTokenCounter();
    });

    // 8. Keydown Handling: Enter to Send, Shift+Enter for Newline
    userInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!sendBtn.disabled && !isGenerating) {
          handleSendMessage();
        }
      }
    });

    // 9. Form Submission
    chatForm.addEventListener('submit', (e) => {
      e.preventDefault();
      handleSendMessage();
    });

    // 10. Stop Generation Button
    stopBtn.addEventListener('click', () => {
      if (abortController) {
        abortController.abort();
        abortController = null;
      }
      setGeneratingState(false);
    });

    // 11. Add Attachment Menu Toggle
    addAttachmentBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      attachmentMenu.classList.toggle('hidden');
    });

    // Close dropdown menus when clicking outside
    document.addEventListener('click', () => {
      attachmentMenu.classList.add('hidden');
      modelPickerDropdown.classList.add('hidden');
    });

    // Dropdown items
    menuUploadFiles.addEventListener('click', () => fileInputAll.click());
    menuUploadImages.addEventListener('click', () => fileInputImages.click());
    menuUploadAudio.addEventListener('click', () => fileInputAudio.click());
    menuSystemPrompt.addEventListener('click', () => openSystemPromptModal());

    // File input changes
    fileInputAll.addEventListener('change', (e) => handleFileSelection(e.target.files));
    fileInputImages.addEventListener('change', (e) => handleFileSelection(e.target.files));
    fileInputAudio.addEventListener('change', (e) => handleFileSelection(e.target.files));

    // 12. Voice Recording Mic Button
    micBtn.addEventListener('click', toggleAudioRecording);

    // 13. Reasoning (Thinking) Toggle
    reasoningToggleBtn.addEventListener('click', () => {
      isReasoningEnabled = !isReasoningEnabled;
      reasoningToggleBtn.classList.toggle('active', isReasoningEnabled);
    });

    // 14. Model Picker Dropdown
    modelPickerBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      modelPickerDropdown.classList.toggle('hidden');
    });

    // 15. Drag & Drop Handling
    let dragCounter = 0;
    window.addEventListener('dragenter', (e) => {
      e.preventDefault();
      dragCounter++;
      dropOverlay.classList.remove('hidden');
    });

    window.addEventListener('dragleave', (e) => {
      e.preventDefault();
      dragCounter--;
      if (dragCounter <= 0) {
        dragCounter = 0;
        dropOverlay.classList.add('hidden');
      }
    });

    window.addEventListener('dragover', (e) => {
      e.preventDefault();
    });

    window.addEventListener('drop', (e) => {
      e.preventDefault();
      dragCounter = 0;
      dropOverlay.classList.add('hidden');
      if (e.dataTransfer && e.dataTransfer.files) {
        handleFileSelection(e.dataTransfer.files);
      }
    });

    // 16. Clipboard Paste Handling (Pasting Screenshots/Images)
    window.addEventListener('paste', (e) => {
      if (e.clipboardData && e.clipboardData.items) {
        const items = e.clipboardData.items;
        for (let i = 0; i < items.length; i++) {
          if (items[i].type.indexOf('image') !== -1) {
            const blob = items[i].getAsFile();
            if (blob) {
              handleFileSelection([blob]);
              e.preventDefault();
              break;
            }
          }
        }
      }
    });

    // 17. System Prompt Modal Controls
    closeSyspromptModal.addEventListener('click', () => systemPromptModal.classList.add('hidden'));
    systemPromptModal.addEventListener('click', (e) => {
      if (e.target === systemPromptModal) systemPromptModal.classList.add('hidden');
    });

    saveSyspromptBtn.addEventListener('click', () => {
      const val = systemPromptInput.value.trim();
      const sess = getActiveSession();
      if (sess) {
        sess.systemPrompt = val;
        saveSessionsToStorage();
      }
      localStorage.setItem(STORAGE_SYSPROMPT_KEY, val);
      systemPromptModal.classList.add('hidden');
    });

    resetSyspromptBtn.addEventListener('click', () => {
      systemPromptInput.value = '';
    });
  }

  function openSystemPromptModal() {
    const sess = getActiveSession();
    systemPromptInput.value = (sess && sess.systemPrompt) || localStorage.getItem(STORAGE_SYSPROMPT_KEY) || '';
    systemPromptModal.classList.remove('hidden');
    systemPromptInput.focus();
  }

  // Run initialization on load
  document.addEventListener('DOMContentLoaded', init);
})();
