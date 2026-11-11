#pragma once

namespace sandwich {

inline const char * get_cockpit_html() {
    return R"HTML(<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sandwich.cpp - Visual Inference Cockpit</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #07090E;
            --bg-card: rgba(15, 23, 42, 0.75);
            --bg-card-hover: rgba(30, 41, 59, 0.85);
            --border-glow: rgba(56, 189, 248, 0.2);
            --border-subtle: rgba(255, 255, 255, 0.08);
            --text-main: #F1F5F9;
            --text-muted: #94A3B8;
            --cyan-glow: #06B6D4;
            --emerald-glow: #10B981;
            --violet-glow: #8B5CF6;
            --rose-glow: #F43F5E;
            --amber-glow: #F59E0B;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        body {
            background-color: var(--bg-base);
            background-image: 
                radial-gradient(at 0% 0%, rgba(6, 182, 212, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.12) 0px, transparent 50%),
                radial-gradient(at 50% 100%, rgba(16, 185, 129, 0.08) 0px, transparent 50%);
            color: var(--text-main);
            min-height: 100vh;
            padding-bottom: 60px;
            overflow-x: hidden;
        }

        .mono {
            font-family: 'Fira Code', monospace;
        }

        /* Top Header */
        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 32px;
            border-bottom: 1px solid var(--border-subtle);
            backdrop-filter: blur(16px);
            background: rgba(7, 9, 14, 0.8);
            position: sticky;
            top: 0;
            z-index: 50;
        }

        .brand-logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-badge {
            background: linear-gradient(135deg, #06B6D4, #8B5CF6);
            color: white;
            font-weight: 800;
            font-size: 13px;
            padding: 4px 10px;
            border-radius: 6px;
            letter-spacing: 1px;
            box-shadow: 0 0 15px rgba(6, 182, 212, 0.4);
        }

        .brand-title {
            font-size: 20px;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(to right, #FFFFFF, #94A3B8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header-meta {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .status-pill {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: var(--emerald-glow);
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            background-color: var(--emerald-glow);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        /* Container Layout */
        .container {
            max-width: 1440px;
            margin: 0 auto;
            padding: 24px 32px;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        /* Card Styles */
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 22px;
            backdrop-filter: blur(12px);
            transition: all 0.2s ease;
            position: relative;
            overflow: hidden;
        }

        .card:hover {
            border-color: var(--border-glow);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
        }

        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 18px;
        }

        .card-title {
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Hero Odometer Board */
        .odometer-card {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.9) 0%, rgba(8, 12, 22, 0.95) 100%);
            border: 1px solid rgba(6, 182, 212, 0.3);
            box-shadow: 0 0 35px rgba(6, 182, 212, 0.08);
        }

        .odometer-grid {
            display: grid;
            grid-template-columns: 2fr 1fr 1fr 1fr;
            gap: 24px;
            align-items: center;
        }

        @media (max-width: 1024px) {
            .odometer-grid {
                grid-template-columns: 1fr 1fr;
            }
        }
        @media (max-width: 640px) {
            .odometer-grid {
                grid-template-columns: 1fr;
            }
        }

        .odometer-digits-container {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .odometer-display {
            display: flex;
            gap: 4px;
            background: #000000;
            padding: 8px 12px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: inset 0 2px 8px rgba(0,0,0,0.8);
            width: fit-content;
        }

        .digit-box {
            background: linear-gradient(180deg, #1A202C 0%, #0D1117 50%, #161B22 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            width: 38px;
            height: 52px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 32px;
            font-weight: 700;
            color: #38BDF8;
            text-shadow: 0 0 12px rgba(56, 189, 248, 0.6);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.2), 0 2px 4px rgba(0,0,0,0.5);
            position: relative;
        }

        .digit-box::after {
            content: '';
            position: absolute;
            left: 0;
            right: 0;
            top: 50%;
            height: 1px;
            background: rgba(0, 0, 0, 0.6);
            box-shadow: 0 1px 0 rgba(255, 255, 255, 0.05);
        }

        .stat-metric {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .stat-metric .val {
            font-size: 26px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }

        .stat-metric .sub {
            font-size: 12px;
            color: var(--text-muted);
        }

        .stat-money {
            color: #34D399;
            text-shadow: 0 0 15px rgba(52, 211, 153, 0.3);
        }

        .badge-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 8px;
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(245, 158, 11, 0.05));
            border: 1px solid rgba(245, 158, 11, 0.4);
            color: #FBBF24;
            font-weight: 600;
            font-size: 13px;
        }

        /* 2-Column Dashboard Grid */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }

        @media (max-width: 1024px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Silicon HUD & Progress Bars */
        .gauge-item {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-bottom: 16px;
        }

        .gauge-labels {
            display: flex;
            justify-content: space-between;
            font-size: 13px;
        }

        .progress-track {
            height: 10px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 9999px;
            overflow: hidden;
            position: relative;
        }

        .progress-bar-vram {
            height: 100%;
            background: linear-gradient(90deg, #06B6D4, #3B82F6, #8B5CF6);
            border-radius: 9999px;
            transition: width 0.5s ease;
            box-shadow: 0 0 10px rgba(6, 182, 212, 0.5);
        }

        .progress-bar-ram {
            height: 100%;
            background: linear-gradient(90deg, #10B981, #F59E0B);
            border-radius: 9999px;
            transition: width 0.5s ease;
        }

        .progress-bar-guard-line {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 2px;
            background: #EF4444;
            z-index: 10;
            box-shadow: 0 0 6px #EF4444;
        }

        /* MoE Heatmap */
        .moe-grid-container {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .moe-legend {
            display: flex;
            gap: 16px;
            font-size: 11px;
            color: var(--text-muted);
            margin-bottom: 6px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .legend-dot {
            width: 10px;
            height: 10px;
            border-radius: 2px;
        }

        .moe-blocks-grid {
            display: grid;
            grid-template-columns: repeat(16, 1fr);
            gap: 4px;
            padding: 8px;
            background: rgba(0, 0, 0, 0.4);
            border-radius: 8px;
            border: 1px solid var(--border-subtle);
        }

        .expert-block {
            aspect-ratio: 1;
            border-radius: 3px;
            background: #1E293B;
            transition: all 0.2s ease;
            cursor: pointer;
            position: relative;
        }

        .expert-block.tier1 {
            background: #6366F1;
            box-shadow: 0 0 6px rgba(99, 102, 241, 0.6);
        }

        .expert-block.tier2 {
            background: #F59E0B;
            box-shadow: 0 0 4px rgba(245, 158, 11, 0.4);
        }

        .expert-block.tier3 {
            background: #0284C7;
            box-shadow: 0 0 4px rgba(2, 132, 199, 0.4);
        }

        .expert-block:hover {
            transform: scale(1.3);
            z-index: 20;
        }

        /* SGLang Token Stream Visualizer */
        .token-stream-box {
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 14px;
            min-height: 120px;
            max-height: 200px;
            overflow-y: auto;
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            align-content: flex-start;
        }

        .stream-token {
            display: inline-flex;
            align-items: center;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 13px;
            transition: all 0.2s;
        }

        .token-accepted {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #34D399;
            box-shadow: 0 0 8px rgba(16, 185, 129, 0.2);
        }

        .token-rejected {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #F87171;
            text-decoration: line-through;
            opacity: 0.6;
        }

        .token-target {
            background: rgba(56, 189, 248, 0.15);
            border: 1px solid rgba(56, 189, 248, 0.4);
            color: #38BDF8;
        }

        /* Playground Panel */
        .playground-card {
            grid-column: 1 / -1;
        }

        .prompt-input-group {
            display: flex;
            gap: 12px;
            margin-bottom: 12px;
        }

        .prompt-input {
            flex: 1;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 12px 16px;
            color: white;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
        }

        .prompt-input:focus {
            border-color: var(--cyan-glow);
        }

        .run-btn {
            background: linear-gradient(135deg, #06B6D4, #3B82F6);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: 0 4px 15px rgba(6, 182, 212, 0.3);
        }

        .run-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(6, 182, 212, 0.5);
        }

        .run-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .playground-output {
            background: #000000;
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 16px;
            min-height: 100px;
            max-height: 250px;
            overflow-y: auto;
            font-size: 14px;
            line-height: 1.6;
            color: #E2E8F0;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>

    <!-- Header -->
    <header>
        <div class="brand-logo">
            <span class="brand-badge">SANDWICH</span>
            <span class="brand-title">Visual Inference Cockpit</span>
        </div>
        <div class="header-meta">
            <div class="status-pill">
                <div class="pulse-dot"></div>
                <span id="cockpit-status-text">CALIBRATED & READY</span>
            </div>
            <div class="mono" style="font-size: 12px; color: var(--text-muted);" id="header-time">--:--:--</div>
        </div>
    </header>

    <div class="container">

        <!-- Hero Row: Lifetime Token-Board (Odometer) -->
        <div class="card odometer-card">
            <div class="card-header">
                <div class="card-title">
                    <span>⚡ LIFETIME TOKEN-BOARD & COMPUTE ODOMETER</span>
                </div>
                <div class="badge-pill" id="milestone-badge">
                    <span>🏆</span> <span id="badge-name">Novice Explorer</span>
                </div>
            </div>
            <div class="odometer-grid">
                <div class="odometer-digits-container">
                    <div style="font-size: 11px; color: var(--cyan-glow); font-weight: 700; letter-spacing: 1px; text-transform: uppercase;">
                        Cumulative Lifetime Tokens Generated
                    </div>
                    <div class="odometer-display mono" id="odometer-digits">
                        <!-- Digits injected dynamically -->
                    </div>
                    <div style="font-size: 11px; color: var(--text-muted);">
                        Persisted in <span class="mono" style="color: #64748B;">~/.config/sandwich/token_odometer.json</span>
                    </div>
                </div>

                <div class="stat-metric">
                    <div class="sub">Frontier API Dollars Saved</div>
                    <div class="val stat-money mono" id="cost-saved-val">$0.00</div>
                    <div class="sub" style="font-size: 11px;">Blended Frontier Rate ($2.50/M)</div>
                </div>

                <div class="stat-metric">
                    <div class="sub">Active Session Tokens</div>
                    <div class="val mono" style="color: #38BDF8;" id="session-tokens-val">0</div>
                    <div class="sub" id="session-prompt-val">0 prompt tokens</div>
                </div>

                <div class="stat-metric">
                    <div class="sub">Total System Sessions</div>
                    <div class="val mono" style="color: #A855F7;" id="sessions-count-val">1</div>
                    <div class="sub" id="lifetime-compute-time">0h 0m compute</div>
                </div>
            </div>
        </div>

        <!-- 2-Column Grid: Silicon HUD & vLLM Heatmap -->
        <div class="dashboard-grid">

            <!-- Card 1: Strata Silicon & Guardrail HUD -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <span>🛡️ SILICON & GUARDRAIL HUD (STRATA INVARIANTS)</span>
                    </div>
                    <span class="mono" style="font-size: 12px; color: var(--cyan-glow);">RTX 5050 Mobile</span>
                </div>

                <!-- VRAM Bar -->
                <div class="gauge-item">
                    <div class="gauge-labels">
                        <span>VRAM Usage (DRM Connector Guard: 7,785 MB)</span>
                        <span class="mono" id="vram-label">0 / 7,800 MB (0%)</span>
                    </div>
                    <div class="progress-track">
                        <div class="progress-bar-vram" id="vram-bar" style="width: 0%;"></div>
                        <div class="progress-bar-guard-line" style="left: 95.5%;" title="Wayland / cuBLAS Safety Cushion"></div>
                    </div>
                </div>

                <!-- RAM Bar -->
                <div class="gauge-item">
                    <div class="gauge-labels">
                        <span>Host DDR5 RAM (Technique #42: 70% Max Cap)</span>
                        <span class="mono" id="ram-label">70% Enforced</span>
                    </div>
                    <div class="progress-track">
                        <div class="progress-bar-ram" id="ram-bar" style="width: 45%;"></div>
                        <div class="progress-bar-guard-line" style="left: 70%;" title="Max Safe Host RAM Cap (70%)"></div>
                    </div>
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 14px;">
                    <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px;">
                        <div style="font-size: 11px; color: var(--text-muted);">Dynamic Free VRAM</div>
                        <div class="mono" style="font-size: 16px; font-weight: 700; color: #38BDF8;" id="free-vram-val">7,705 MB</div>
                    </div>
                    <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px;">
                        <div style="font-size: 11px; color: var(--text-muted);">Compute Share Split</div>
                        <div class="mono" style="font-size: 16px; font-weight: 700; color: #10B981;" id="compute-share-val">100% GPU</div>
                    </div>
                </div>
            </div>

            <!-- Card 2: vLLM-Style 3-Tier MoE Memory Heatmap -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <span>🧠 3-TIER MoE MEMORY HIERARCHY (vLLM HEATMAP)</span>
                    </div>
                    <span class="mono" style="font-size: 12px; color: #10B981;" id="moe-hit-rate">Hit Rate: 98.4%</span>
                </div>

                <div class="moe-grid-container">
                    <div class="moe-legend">
                        <div class="legend-item"><div class="legend-dot" style="background: #6366F1;"></div> Tier 1: VRAM Hot</div>
                        <div class="legend-item"><div class="legend-dot" style="background: #F59E0B;"></div> Tier 2: DDR5 Warm Pool</div>
                        <div class="legend-item"><div class="legend-dot" style="background: #0284C7;"></div> Tier 3: Direct NVMe Fetch</div>
                    </div>

                    <!-- 4x16 Expert Block Matrix -->
                    <div class="moe-blocks-grid" id="moe-blocks">
                        <!-- Populated by JS -->
                    </div>

                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: var(--text-muted); margin-top: 10px;">
                        <span>io_uring Engine: <strong style="color: #38BDF8;" id="uring-status">Active</strong></span>
                        <span>Ghost Hits: <strong style="color: #A855F7;" id="ghost-hits-val">0</strong></span>
                        <span>Warm Pool: <strong style="color: #F59E0B;" id="warm-pool-val">0 MB</strong></span>
                    </div>
                </div>
            </div>

        </div>

        <!-- Speculative Visualizer Row -->
        <div class="card">
            <div class="card-header">
                <div class="card-title">
                    <span>✨ SGLang SPECULATIVE STREAM VISUALIZER</span>
                </div>
                <div style="display: flex; gap: 16px;">
                    <span class="mono" style="font-size: 12px; color: var(--cyan-glow);" id="spec-acceptance-rate">Acceptance: 0.0%</span>
                    <span class="mono" style="font-size: 12px; color: #10B981;" id="spec-speedup">Speedup: 1.00x</span>
                </div>
            </div>

            <div class="token-stream-box mono" id="spec-token-stream">
                <span class="stream-token token-target">System initialized. Ready for token verification stream...</span>
            </div>
        </div>

        <!-- Interactive Playground -->
        <div class="card playground-card">
            <div class="card-header">
                <div class="card-title">
                    <span>⚡ LIVE INFERENCE PLAYGROUND & TEST BENCH</span>
                </div>
                <span class="mono" style="font-size: 12px; color: var(--text-muted);">Real-Time SSE Streaming</span>
            </div>

            <div class="prompt-input-group">
                <input type="text" class="prompt-input" id="test-prompt-input" placeholder="Enter prompt to evaluate model with Sandwich techniques..." value="Explain why 3-Tier memory offloading with speculative decoding is optimal for LLMs in 2 sentences.">
                <button class="run-btn" id="run-inference-btn" onclick="runTestInference()">Run Inference</button>
            </div>

            <div class="playground-output mono" id="playground-output">Model response will stream here...</div>
        </div>

    </div>

    <script>
        // Clock
        function updateClock() {
            const now = new Date();
            document.getElementById('header-time').textContent = now.toLocaleTimeString();
        }
        setInterval(updateClock, 1000);
        updateClock();

        // Initialize MoE Grid Blocks
        const moeGrid = document.getElementById('moe-blocks');
        const numBlocks = 64;
        for (let i = 0; i < numBlocks; i++) {
            const block = document.createElement('div');
            block.className = 'expert-block';
            block.id = `expert-${i}`;
            block.title = `Expert ${i}`;
            // initial tier pattern
            if (i % 5 === 0) block.classList.add('tier1');
            else if (i % 3 === 0) block.classList.add('tier2');
            else block.classList.add('tier3');
            moeGrid.appendChild(block);
        }

        // Render Mechanical Odometer Digits
        function renderOdometer(val) {
            const container = document.getElementById('odometer-digits');
            const str = String(val).padStart(8, '0');
            container.innerHTML = '';
            for (let char of str) {
                const box = document.createElement('div');
                box.className = 'digit-box';
                box.textContent = char;
                container.appendChild(box);
            }
        }
        renderOdometer(0);

        // Telemetry Poller
        async function fetchTelemetry() {
            try {
                const res = await fetch('/v1/sandwich/telemetry');
                if (!res.ok) return;
                const data = await res.json();

                // 1. Token-Board
                if (data.token_board) {
                    const tb = data.token_board;
                    renderOdometer(tb.lifetime_tokens || 0);
                    document.getElementById('cost-saved-val').textContent = `$${(tb.cost_saved_usd || 0).toFixed(4)}`;
                    document.getElementById('session-tokens-val').textContent = (tb.session_tokens || 0).toLocaleString();
                    document.getElementById('session-prompt-val').textContent = `${(tb.session_prompt_tokens || 0).toLocaleString()} prompt tokens`;
                    document.getElementById('sessions-count-val').textContent = tb.total_sessions || 1;
                    
                    const computeSecs = Math.floor((tb.lifetime_compute_ms || 0) / 1000);
                    const hrs = Math.floor(computeSecs / 3600);
                    const mins = Math.floor((computeSecs % 3600) / 60);
                    document.getElementById('lifetime-compute-time').textContent = `${hrs}h ${mins}m compute`;

                    if (tb.milestone_badge) {
                        document.getElementById('badge-name').textContent = tb.milestone_badge;
                    }
                }

                // 2. Hardware HUD
                if (data.hardware) {
                    const hw = data.hardware;
                    const used = hw.vram_used_mb || 0;
                    const total = hw.vram_guard_mb || 7800;
                    const pct = Math.min(100, Math.round((used / total) * 100));
                    document.getElementById('vram-bar').style.width = `${pct}%`;
                    document.getElementById('vram-label').textContent = `${used} / ${total} MB (${pct}%)`;
                    document.getElementById('free-vram-val').textContent = `${hw.vram_free_mb || 0} MB`;
                }

                // 3. MoE 3-Tier
                if (data.moe_3tier) {
                    const moe = data.moe_3tier;
                    if (moe.hit_rate_pct !== undefined) {
                        document.getElementById('moe-hit-rate').textContent = `Hit Rate: ${moe.hit_rate_pct.toFixed(1)}%`;
                    }
                    if (moe.io_uring_active !== undefined) {
                        document.getElementById('uring-status').textContent = moe.io_uring_active ? 'Active (O_DIRECT)' : 'Standard';
                    }
                    document.getElementById('ghost-hits-val').textContent = (moe.ghost_cache_hits || 0).toLocaleString();
                    document.getElementById('warm-pool-val').textContent = `${moe.warm_pool_size_mb || 0} MB`;

                    // Animate a random subset of blocks to represent active expert activations
                    if (moe.enabled) {
                        for (let k = 0; k < 4; k++) {
                            const idx = Math.floor(Math.random() * numBlocks);
                            const el = document.getElementById(`expert-${idx}`);
                            if (el) {
                                el.style.transform = 'scale(1.2)';
                                setTimeout(() => { el.style.transform = ''; }, 300);
                            }
                        }
                    }
                }

                // 4. Speculative
                if (data.speculative) {
                    const spec = data.speculative;
                    const ratePct = ((spec.acceptance_rate || 0) * 100).toFixed(1);
                    document.getElementById('spec-acceptance-rate').textContent = `Acceptance: ${ratePct}%`;
                    document.getElementById('spec-speedup').textContent = `Speedup: ${(spec.speedup || 1.0).toFixed(2)}x`;
                }

            } catch (err) {
                console.error("Telemetry fetch error:", err);
            }
        }

        setInterval(fetchTelemetry, 600);
        fetchTelemetry();

        // Live Playground Inference with Streaming SSE
        async function runTestInference() {
            const input = document.getElementById('test-prompt-input');
            const btn = document.getElementById('run-inference-btn');
            const output = document.getElementById('playground-output');
            const streamBox = document.getElementById('spec-token-stream');

            const prompt = input.value.trim();
            if (!prompt) return;

            btn.disabled = true;
            btn.textContent = 'Generating...';
            output.textContent = '';
            streamBox.innerHTML = '';

            try {
                const response = await fetch('/completion', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prompt: prompt,
                        n_predict: 80,
                        stream: true
                    })
                });

                if (!response.ok) {
                    output.textContent = `Error: Server returned status ${response.status}`;
                    btn.disabled = false;
                    btn.textContent = 'Run Inference';
                    return;
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop(); // preserve remainder

                    for (const line of lines) {
                        const trimmed = line.trim();
                        if (!trimmed || !trimmed.startsWith('data:')) continue;
                        const dataStr = trimmed.slice(5).trim();
                        if (dataStr === '[DONE]') break;

                        try {
                            const parsed = JSON.parse(dataStr);
                            const content = parsed.content || '';
                            if (content) {
                                output.textContent += content;
                                output.scrollTop = output.scrollHeight;

                                // Visualizer token
                                const tokSpan = document.createElement('span');
                                tokSpan.className = 'stream-token token-accepted';
                                tokSpan.textContent = content;
                                streamBox.appendChild(tokSpan);
                                streamBox.scrollTop = streamBox.scrollHeight;
                            }
                        } catch (e) {}
                    }
                }
            } catch (err) {
                output.textContent += `\n[Inference Error: ${err.message}]`;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Run Inference';
                fetchTelemetry(); // refresh odometer immediately
            }
        }
    </script>
</body>
</html>)HTML";
}

} // namespace sandwich
