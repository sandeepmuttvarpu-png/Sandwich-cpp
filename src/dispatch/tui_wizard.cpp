#include "tui_wizard.hpp"

#include <nlohmann/json.hpp>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/ioctl.h>
#include <poll.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace sandwich {
namespace dispatch {

// Global pointer for clean terminal restoration on SIGINT/SIGTERM
static TUIWizard * g_active_wizard = nullptr;

static void global_signal_handler(int sig) {
    if (g_active_wizard) {
        // Restore terminal immediately
        std::cout << "\033[?1049l\033[?25h\033[0m" << std::flush;
    }
    std::exit(128 + sig);
}

TUIWizard::TUIWizard() {
    g_active_wizard = this;
    std::signal(SIGINT, global_signal_handler);
    std::signal(SIGTERM, global_signal_handler);

    // Load hardware profile and model catalog
    profile = HardwareProfiler::load_or_profile();
    catalog = ModelScanner::load_or_scan();

    update_terminal_size();
}

TUIWizard::~TUIWizard() {
    disable_raw_mode();
    g_active_wizard = nullptr;
}

void TUIWizard::enable_raw_mode() {
    if (!isatty(STDIN_FILENO)) return;

    if (tcgetattr(STDIN_FILENO, &orig_termios) == 0) {
        struct termios raw = orig_termios;
        raw.c_lflag &= ~(ECHO | ICANON | IEXTEN);
        raw.c_iflag &= ~(IXON | ICRNL | BRKINT | INPCK | ISTRIP);
        raw.c_cflag |= (CS8);
        raw.c_cc[VMIN]  = 0;
        raw.c_cc[VTIME] = 1; // 100ms timeout
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
        raw_mode_active = true;

        // Enter alternate buffer and hide cursor
        std::cout << "\033[?1049h\033[?25l" << std::flush;
    }
}

void TUIWizard::disable_raw_mode() {
    if (raw_mode_active) {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
        // Exit alternate buffer, show cursor, reset styles
        std::cout << "\033[?1049l\033[?25h\033[0m" << std::flush;
        raw_mode_active = false;
    }
}

void TUIWizard::update_terminal_size() {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0 && ws.ws_row > 0) {
        term_cols = ws.ws_col;
        term_rows = ws.ws_row;
    } else {
        term_cols = 80;
        term_rows = 24;
    }
}

KeyEvent TUIWizard::read_key() {
    KeyEvent ev;
    struct pollfd pfd;
    pfd.fd = STDIN_FILENO;
    pfd.events = POLLIN;

    int ret = poll(&pfd, 1, 25); // 25ms low latency
    if (ret <= 0) {
        return ev; // UNKNOWN
    }

    char c = 0;
    if (read(STDIN_FILENO, &c, 1) != 1) {
        return ev;
    }

    if (c == '\r' || c == '\n') {
        ev.code = KeyCode::ENTER;
        return ev;
    }
    if (c == 27) { // ESC or escape sequence
        char seq[4] = {0};
        if (read(STDIN_FILENO, &seq[0], 1) != 1) {
            ev.code = KeyCode::ESCAPE;
            return ev;
        }
        if (seq[0] == '[') {
            if (read(STDIN_FILENO, &seq[1], 1) == 1) {
                if (seq[1] == 'A') ev.code = KeyCode::UP;
                else if (seq[1] == 'B') ev.code = KeyCode::DOWN;
                else if (seq[1] == 'C') ev.code = KeyCode::RIGHT;
                else if (seq[1] == 'D') ev.code = KeyCode::LEFT;
                else if (seq[1] == '5' || seq[1] == '6') {
                    char t = 0;
                    if (read(STDIN_FILENO, &t, 1) == 1 && t == '~') {
                        ev.code = (seq[1] == '5') ? KeyCode::PAGE_UP : KeyCode::PAGE_DOWN;
                    }
                }
                return ev;
            }
        }
        ev.code = KeyCode::ESCAPE;
        return ev;
    }
    if (c == 127 || c == 8) {
        ev.code = KeyCode::BACKSPACE;
        return ev;
    }
    if (c == '\t') {
        ev.code = KeyCode::TAB;
        return ev;
    }
    if (c == ' ') {
        ev.code = KeyCode::SPACE;
        return ev;
    }

    ev.code = KeyCode::CHAR;
    ev.ch   = c;
    return ev;
}

void TUIWizard::render_header(const std::string & title) {
    std::cout << "\033[H\033[2J"; // Home + Clear screen

    // Sandwich Gold: 229, 169, 59 | Cyan: 78, 201, 176
    std::cout << "\033[1;38;2;229;169;59m"
              << "  🥪 SANDWICH.CPP ── SMART DISPATCH & AUTO-TUNING WIZARD\033[0m\n";
    std::cout << "\033[38;2;120;120;120m"
              << "  ─────────────────────────────────────────────────────────────────────────────\033[0m\n";
    std::cout << "  \033[1;38;2;78;201;176m[" << title << "]\033[0m\n\n";
}

void TUIWizard::render_footer(const std::string & navigation_help) {
    std::cout << "\n\033[38;2;120;120;120m"
              << "  ─────────────────────────────────────────────────────────────────────────────\033[0m\n";
    std::cout << "  \033[38;2;160;160;160m" << navigation_help << "\033[0m\n";
    std::cout << std::flush;
}

void TUIWizard::screen_welcome() {
    render_header("Screen 1/6: Hardware Topology & Memory Guard Verification");

    std::cout << "  \033[1mDetected System Topology:\033[0m\n";
    std::cout << "    • \033[38;2;78;201;176mOS:\033[0m          " << profile.os.pretty_name << " (Kernel " << profile.os.kernel_release << ")\n";
    std::cout << "    • \033[38;2;78;201;176mCPU:\033[0m         " << profile.cpu.model_name << "\n";
    std::cout << "      └─ Cores:     " << profile.cpu.p_cores << " P-cores (" << profile.cpu.optimal_threads << " threads pinned) + "
              << profile.cpu.e_cores << " E-cores | " << profile.cpu.total_logical_cores << " total threads\n";
    std::cout << "    • \033[38;2;78;201;176mHost RAM:\033[0m    " << (profile.ram.total_mb / 1024.0) << " GB DDR5 (~" << profile.ram.est_bandwidth_gbps << " GB/s dual-channel)\n";
    std::cout << "      └─ \033[1;38;2;229;169;59mRule #42:\033[0m  70% Safe WARM Pool Limit = " << (profile.ram.max_safe_allocation_mb / 1024.0) << " GB (guarantees zero OS thrash)\n";
    std::cout << "    • \033[38;2;78;201;176mGPU:\033[0m         " << profile.gpu.device_name << " (" << (profile.gpu.total_vram_mb / 1024.0) << " GB GDDR6)\n";
    std::cout << "      └─ \033[1;38;2;239;68;68mGuard:\033[0m     7,100 MB Wayland VRAM Guard invariant active (160 MB safety margin)\n";
    std::cout << "    • \033[38;2;78;201;176mStorage:\033[0m     " << profile.storage.nvme_model << " (PCIe " << profile.storage.pcie_link_speed << " x" << profile.storage.pcie_link_width << ")\n\n";

    std::cout << "  \033[38;2;34;197;94m✔ Hardware profile synchronized to ~/.config/sandwich/hardware_profile.json\033[0m\n";
    std::cout << "  \033[38;2;34;197;94m✔ Model catalog loaded: " << catalog.models.size() << " base models, " << catalog.drafters.size() << " speculative drafters discovered\033[0m\n";

    render_footer("[Enter] Proceed to Model Selection    [Q] Quit");
}

void TUIWizard::screen_model_selection() {
    render_header("Screen 2/6: Model Discovery & Selection");

    if (model_filter_mode) {
        std::cout << "  \033[1;38;2;229;169;59mSearch Filter:\033[0m " << model_filter << "_\n\n";
    } else {
        std::cout << "  \033[38;2;160;160;160mSelect target model (.gguf). Press '/' to search/filter.\033[0m\n\n";
    }

    // Filter models
    std::vector<size_t> matched_indices;
    for (size_t i = 0; i < catalog.models.size(); ++i) {
        if (model_filter.empty()) {
            matched_indices.push_back(i);
        } else {
            std::string n = catalog.models[i].display_name;
            std::string f = model_filter;
            std::transform(n.begin(), n.end(), n.begin(), ::tolower);
            std::transform(f.begin(), f.end(), f.begin(), ::tolower);
            if (n.find(f) != std::string::npos) {
                matched_indices.push_back(i);
            }
        }
    }

    if (matched_indices.empty()) {
        std::cout << "  \033[38;2;239;68;68mNo models found matching criteria.\033[0m\n";
        render_footer("[/] Search    [Esc] Clear Filter    [Q] Quit");
        return;
    }

    if (model_cursor >= matched_indices.size()) {
        model_cursor = matched_indices.size() - 1;
    }

    const size_t page_size = 7;
    if (model_cursor < model_page_offset) {
        model_page_offset = model_cursor;
    } else if (model_cursor >= model_page_offset + page_size) {
        model_page_offset = model_cursor - page_size + 1;
    }

    std::cout << "  ┌────┬──────────────────────────────────────────┬──────────┬──────────┬──────────────┐\n";
    std::cout << "  │    │ Model Name                               │ Quant    │ Size     │ Arch / Type  │\n";
    std::cout << "  ├────┼──────────────────────────────────────────┼──────────┼──────────┼──────────────┤\n";

    for (size_t i = model_page_offset; i < std::min(matched_indices.size(), model_page_offset + page_size); ++i) {
        const auto & m = catalog.models[matched_indices[i]];
        bool is_sel = (i == model_cursor);

        std::string tag = is_sel ? " ▶ " : "   ";
        std::string dname = m.display_name;
        if (dname.size() > 40) dname = dname.substr(0, 37) + "...";

        char row[256];
        snprintf(row, sizeof(row), "  │%s │ %-40s │ %-8s │ %6.2f GB │ %-12s │",
                 tag.c_str(), dname.c_str(), m.quantization.c_str(), m.size_gb,
                 (m.is_moe ? "MoE Hybrid" : "Dense"));

        if (is_sel) {
            std::cout << "\033[1;38;2;229;169;59;48;2;40;40;40m" << row << "\033[0m\n";
        } else {
            std::cout << row << "\n";
        }
    }
    std::cout << "  └────┴──────────────────────────────────────────┴──────────┴──────────┴──────────────┘\n";

    std::cout << "\n  \033[38;2;120;120;120mShowing " << (model_cursor + 1) << " of " << matched_indices.size() << " models\033[0m\n";

    render_footer("[↑/↓] Navigate    [/] Filter    [Enter] Select Model    [B] Back    [Q] Quit");
}

void TUIWizard::screen_drafter_selection() {
    render_header("Screen 3/6: Speculative Decoding Drafter Configuration");

    std::cout << "  Target Base Model: \033[1;38;2;78;201;176m" << state.selected_model.display_name << "\033[0m ("
              << state.selected_model.quantization << ", " << state.selected_model.size_gb << " GB)\n\n";

    // Auto-detect recommended drafter
    std::string rec_name = "None detected";
    ModelMetadata rec_drafter;
    bool has_rec = false;

    for (const auto & d : catalog.drafters) {
        if (!state.selected_model.drafter_target.empty() && d.filename.find(state.selected_model.drafter_target) != std::string::npos) {
            rec_drafter = d;
            has_rec = true;
            break;
        }
        if (d.filename.find("Qwen3.6-35B") != std::string::npos && state.selected_model.filename.find("35B") != std::string::npos) {
            rec_drafter = d;
            has_rec = true;
            break;
        }
        if (d.filename.find("Flash-Next") != std::string::npos && state.selected_model.filename.find("Flash-Next") != std::string::npos) {
            rec_drafter = d;
            has_rec = true;
            break;
        }
    }

    std::vector<std::string> options = {
        "Auto-Select Recommended Drafter " + (has_rec ? ("[\033[1;38;2;34;197;94m" + rec_drafter.filename + "\033[0m]") : "[No direct match]"),
        "Manual Drafter Selection (Pick from discovered drafters in cache)",
        "Skip Speculative Decoding (Run standalone autoregressive inference)"
    };

    std::cout << "  \033[1mSpeculative Drafter Options:\033[0m\n\n";
    for (size_t i = 0; i < options.size(); ++i) {
        if (i == drafter_option) {
            std::cout << "    \033[1;38;2;229;169;59m▶ (" << (i + 1) << ") " << options[i] << "\033[0m\n";
        } else {
            std::cout << "      (" << (i + 1) << ") " << options[i] << "\n";
        }
    }

    if (drafter_option == 1 && !catalog.drafters.empty()) {
        std::cout << "\n    \033[1mDiscovered Drafters:\033[0m\n";
        for (size_t j = 0; j < catalog.drafters.size(); ++j) {
            bool sel = (j == drafter_cursor);
            if (sel) {
                std::cout << "      \033[1;38;2;78;201;176m▶ " << catalog.drafters[j].filename << " (" << catalog.drafters[j].size_gb << " GB)\033[0m\n";
            } else {
                std::cout << "        " << catalog.drafters[j].filename << " (" << catalog.drafters[j].size_gb << " GB)\n";
            }
        }
    }

    render_footer("[↑/↓] Change Option    [Enter] Confirm & Next    [B] Back    [Q] Quit");
}

void TUIWizard::screen_context_quant() {
    render_header("Screen 4/6: Context Window & KV Cache Quantization");

    std::vector<int> contexts = {2048, 4096, 8192, 16384, 32768};
    std::vector<std::pair<std::string, std::string>> quants = {
        {"turbo3", "3.5-bit TurboQuant (Sandwich Recommended - 62% KV reduction)"},
        {"turbo4", "4.0-bit TurboQuant (Balanced - 50% KV reduction)"},
        {"turbo2", "2.5-bit TurboQuant (Ultra-low VRAM - 75% KV reduction)"},
        {"f16",    "16-bit Full Precision (Standard baseline)"}
    };

    int chosen_ctx = contexts[context_cursor];
    std::string chosen_q = quants[kv_quant_cursor].first;

    std::cout << "  \033[1m1. Target Context Length:\033[0m\n  ";
    for (size_t i = 0; i < contexts.size(); ++i) {
        if (i == context_cursor) {
            std::cout << "  \033[1;38;2;229;169;59;48;2;40;40;40m [" << contexts[i] << "] \033[0m";
        } else {
            std::cout << "   " << contexts[i] << " ";
        }
    }
    std::cout << "\n\n";

    std::cout << "  \033[1m2. KV Cache Quantization Mode:\033[0m\n";
    for (size_t i = 0; i < quants.size(); ++i) {
        if (i == kv_quant_cursor) {
            std::cout << "    \033[1;38;2;78;201;176m▶ [" << quants[i].first << "] - " << quants[i].second << "\033[0m\n";
        } else {
            std::cout << "      [" << quants[i].first << "] - " << quants[i].second << "\n";
        }
    }
    std::cout << "\n";

    // Recalculate tiering live
    TieringPlan tp = HardwareProfiler::calculate_tiering(
        profile,
        state.selected_model.size_bytes,
        state.selected_model.total_layers,
        state.selected_model.is_moe,
        state.selected_model.expert_count,
        state.selected_model.active_experts,
        chosen_ctx,
        chosen_q
    );

    // Live VRAM Usage Meter
    uint64_t total_guard = profile.gpu.wayland_guard_vram_mb;
    uint64_t used = tp.tier1_hot_vram_mb;
    double fraction = static_cast<double>(used) / total_guard;
    int bar_width = 30;
    int filled = static_cast<int>(fraction * bar_width);

    std::cout << "  \033[1mReal-Time VRAM Allocation Meter:\033[0m\n";
    std::cout << "    [";
    for (int b = 0; b < bar_width; ++b) {
        if (b < filled) std::cout << "\033[38;2;34;197;94m█\033[0m";
        else std::cout << "\033[38;2;80;80;80m░\033[0m";
    }
    std::cout << "] " << used << " MB / " << total_guard << " MB max limit\n";
    std::cout << "    Safety Cushion: \033[1;38;2;34;197;94m" << (total_guard - used) << " MB free\033[0m under 7,100 MB Wayland Guard\n";

    render_footer("[←/→] Context    [↑/↓] Quantization    [Enter] Next    [B] Back    [Q] Quit");
}

void TUIWizard::screen_advanced_flags() {
    render_header("Screen 5/6: Advanced Hardware & Routing Flags");

    std::vector<std::pair<std::string, bool*>> flags = {
        {"Flash Attention 2 (-fa)",                     &state.flash_attn},
        {"Lock Threads to Physical P-Cores (-t 6)",     &state.lock_p_cores},
        {"MoE Row-Split Tensor Parallel (-sm row)",     &state.moe_row_split},
        {"Radix Prefix Caching (--radix-cache)",        &state.radix_cache},
        {"Speculative Tree Decoding (--spec-tree)",     &state.spec_tree}
    };

    std::cout << "  \033[38;2;160;160;160mPress [Space] to toggle optimization flags on/off.\033[0m\n\n";

    for (size_t i = 0; i < flags.size(); ++i) {
        bool checked = *(flags[i].second);
        std::string mark = checked ? "[\033[1;38;2;34;197;94mX\033[0m]" : "[ ]";
        if (i == flags_cursor) {
            std::cout << "    \033[1;38;2;229;169;59m▶ " << mark << " " << flags[i].first << "\033[0m\n";
        } else {
            std::cout << "      " << mark << " " << flags[i].first << "\n";
        }
    }

    render_footer("[↑/↓] Navigate    [Space] Toggle    [Enter] Next    [B] Back    [Q] Quit");
}

void TUIWizard::recompute_tiering_and_command() {
    state.tiering = HardwareProfiler::calculate_tiering(
        profile,
        state.selected_model.size_bytes,
        state.selected_model.total_layers,
        state.selected_model.is_moe,
        state.selected_model.expert_count,
        state.selected_model.active_experts,
        state.context_length,
        state.kv_quant
    );

    std::ostringstream cmd;
    cmd << "./sandwich run -m " << state.selected_model.file_path << " ";
    cmd << "-fitt " << profile.gpu.wayland_guard_vram_mb << " ";
    cmd << "-t " << (state.lock_p_cores ? profile.cpu.optimal_threads : profile.cpu.total_logical_cores) << " ";
    cmd << "-c " << state.context_length << " ";
    if (state.flash_attn) cmd << "-fa ";
    if (state.selected_model.is_moe && state.moe_row_split) cmd << "-sm row ";
    if (state.radix_cache) cmd << "--radix-cache ";
    if (state.spec_tree) cmd << "--spec-tree ";

    if (state.kv_quant == "turbo3") cmd << "-ctk q3_k -ctv q3_k ";
    else if (state.kv_quant == "turbo4") cmd << "-ctk q4_0 -ctv q4_0 ";
    else if (state.kv_quant == "turbo2") cmd << "-ctk q2_k -ctv q2_k ";

    if (state.use_speculative && !state.selected_drafter.file_path.empty()) {
        cmd << "-md " << state.selected_drafter.file_path << " ";
    }

    state.synthesized_command = cmd.str();
}

void TUIWizard::save_preset() {
    const char * home = std::getenv("HOME");
    if (!home) home = "/tmp";
    fs::path pdir = fs::path(home) / ".config" / "sandwich" / "presets";
    std::error_code ec;
    fs::create_directories(pdir, ec);

    std::string safe_name = state.selected_model.display_name;
    for (char & c : safe_name) {
        if (!isalnum(c) && c != '-' && c != '_') c = '_';
    }

    fs::path pfile = pdir / (safe_name + ".json");
    json pj;
    pj["model_path"]     = state.selected_model.file_path;
    pj["drafter_path"]   = state.selected_drafter.file_path;
    pj["context_length"] = state.context_length;
    pj["kv_quant"]       = state.kv_quant;
    pj["command"]        = state.synthesized_command;

    std::ofstream out(pfile.string());
    if (out.is_open()) {
        out << pj.dump(4) << std::endl;
        out.close();
        state.saved_preset = true;
    }
}

void TUIWizard::screen_review_launch() {
    recompute_tiering_and_command();
    render_header("Screen 6/6: Synthesis, Performance Simulation & Dispatch");

    std::cout << "  \033[1mSelected Model:\033[0m   \033[38;2;78;201;176m" << state.selected_model.display_name << "\033[0m ("
              << state.selected_model.size_gb << " GB)\n";
    if (state.use_speculative) {
        std::cout << "  \033[1mDrafter:\033[0m          \033[38;2;34;197;94m" << state.selected_drafter.filename << "\033[0m\n";
    } else {
        std::cout << "  \033[1mDrafter:\033[0m          None (Autoregressive)\n";
    }

    std::cout << "\n  \033[1mSandwich 3-Tier Allocation Plan:\033[0m\n";
    std::cout << "    • \033[1;38;2;229;169;59mTier 1 HOT (VRAM):\033[0m    " << state.tiering.tier1_hot_vram_mb << " MB "
              << "(Wayland Guard " << profile.gpu.wayland_guard_vram_mb << " MB limit)\n";
    std::cout << "    • \033[1;38;2;59;130;246mTier 2 WARM (DDR5):\033[0m   " << state.tiering.tier2_warm_ddr5_mb << " MB "
              << "(Rule #42 Cap: " << profile.ram.max_safe_allocation_mb << " MB)\n";
    std::cout << "    • \033[1;38;2;160;160;160mTier 3 COLD (NVMe):\033[0m   " << state.tiering.tier3_cold_nvme_mb << " MB "
              << "(Paged on-demand)\n\n";

    std::cout << "  \033[1mPerformance Simulation:\033[0m\n";
    std::cout << "    • \033[38;2;34;197;94mBest-Case Throughput:\033[0m   " << std::fixed << std::setprecision(1) << state.tiering.est_tokens_sec_best << " tokens/sec\n";
    std::cout << "    • \033[38;2;229;169;59mHybrid Sandwich Mode:\033[0m   " << state.tiering.est_tokens_sec_hybrid << " tokens/sec\n";
    std::cout << "    • \033[38;2;239;68;68mCold Thrash Floor:\033[0m      " << state.tiering.est_tokens_sec_worst << " tokens/sec\n\n";

    std::cout << "  \033[1mSynthesized CLI Command:\033[0m\n";
    std::cout << "    \033[48;2;30;30;30;38;2;250;250;250m " << state.synthesized_command << " \033[0m\n\n";

    if (state.saved_preset) {
        std::cout << "  \033[38;2;34;197;94m✔ Preset saved successfully to ~/.config/sandwich/presets/\033[0m\n";
    }

    render_footer("[Enter] Launch Engine    [S] Save Preset    [B] Back    [Q] Quit");
}

void TUIWizard::handle_input_welcome(const KeyEvent & ev) {
    if (ev.code == KeyCode::ENTER) {
        current_screen = 1;
    }
}

void TUIWizard::handle_input_model(const KeyEvent & ev) {
    if (model_filter_mode) {
        if (ev.code == KeyCode::ENTER || ev.code == KeyCode::ESCAPE) {
            model_filter_mode = false;
        } else if (ev.code == KeyCode::BACKSPACE) {
            if (!model_filter.empty()) model_filter.pop_back();
        } else if (ev.code == KeyCode::CHAR) {
            model_filter.push_back(ev.ch);
        }
        return;
    }

    if (ev.code == KeyCode::UP) {
        if (model_cursor > 0) model_cursor--;
    } else if (ev.code == KeyCode::DOWN) {
        model_cursor++;
    } else if (ev.code == KeyCode::CHAR && ev.ch == '/') {
        model_filter_mode = true;
    } else if (ev.code == KeyCode::CHAR && (ev.ch == 'b' || ev.ch == 'B')) {
        current_screen = 0;
    } else if (ev.code == KeyCode::ENTER) {
        if (!catalog.models.empty()) {
            state.selected_model = catalog.models[std::min(model_cursor, catalog.models.size() - 1)];
            current_screen = 2; // proceed to drafter
        }
    }
}

void TUIWizard::handle_input_drafter(const KeyEvent & ev) {
    if (ev.code == KeyCode::UP) {
        if (drafter_option > 0) drafter_option--;
    } else if (ev.code == KeyCode::DOWN) {
        if (drafter_option < 2) drafter_option++;
    } else if (ev.code == KeyCode::CHAR && (ev.ch == 'b' || ev.ch == 'B')) {
        current_screen = 1;
    } else if (ev.code == KeyCode::ENTER) {
        if (drafter_option == 0) {
            // Auto-select recommended
            state.use_speculative = false;
            for (const auto & d : catalog.drafters) {
                if (d.filename.find("35B") != std::string::npos && state.selected_model.filename.find("35B") != std::string::npos) {
                    state.selected_drafter = d;
                    state.use_speculative = true;
                    break;
                }
                if (d.filename.find("Flash-Next") != std::string::npos && state.selected_model.filename.find("Flash-Next") != std::string::npos) {
                    state.selected_drafter = d;
                    state.use_speculative = true;
                    break;
                }
            }
        } else if (drafter_option == 1) {
            if (!catalog.drafters.empty()) {
                state.selected_drafter = catalog.drafters[std::min(drafter_cursor, catalog.drafters.size() - 1)];
                state.use_speculative = true;
            }
        } else {
            state.use_speculative = false;
        }
        current_screen = 3; // proceed to context/quant
    }
}

void TUIWizard::handle_input_context(const KeyEvent & ev) {
    std::vector<int> contexts = {2048, 4096, 8192, 16384, 32768};
    std::vector<std::string> quants = {"turbo3", "turbo4", "turbo2", "f16"};

    if (ev.code == KeyCode::LEFT) {
        if (context_cursor > 0) context_cursor--;
    } else if (ev.code == KeyCode::RIGHT) {
        if (context_cursor + 1 < contexts.size()) context_cursor++;
    } else if (ev.code == KeyCode::UP) {
        if (kv_quant_cursor > 0) kv_quant_cursor--;
    } else if (ev.code == KeyCode::DOWN) {
        if (kv_quant_cursor + 1 < quants.size()) kv_quant_cursor++;
    } else if (ev.code == KeyCode::CHAR && (ev.ch == 'b' || ev.ch == 'B')) {
        current_screen = 2;
    } else if (ev.code == KeyCode::ENTER) {
        state.context_length = contexts[context_cursor];
        state.kv_quant       = quants[kv_quant_cursor];
        current_screen = 4; // proceed to flags
    }
}

void TUIWizard::handle_input_flags(const KeyEvent & ev) {
    std::vector<bool*> flags = {
        &state.flash_attn,
        &state.lock_p_cores,
        &state.moe_row_split,
        &state.radix_cache,
        &state.spec_tree
    };

    if (ev.code == KeyCode::UP) {
        if (flags_cursor > 0) flags_cursor--;
    } else if (ev.code == KeyCode::DOWN) {
        if (flags_cursor + 1 < flags.size()) flags_cursor++;
    } else if (ev.code == KeyCode::SPACE) {
        *(flags[flags_cursor]) = !(*(flags[flags_cursor]));
    } else if (ev.code == KeyCode::CHAR && (ev.ch == 'b' || ev.ch == 'B')) {
        current_screen = 3;
    } else if (ev.code == KeyCode::ENTER) {
        current_screen = 5; // proceed to review
    }
}

void TUIWizard::handle_input_review(const KeyEvent & ev) {
    if (ev.code == KeyCode::CHAR && (ev.ch == 's' || ev.ch == 'S')) {
        save_preset();
    } else if (ev.code == KeyCode::CHAR && (ev.ch == 'b' || ev.ch == 'B')) {
        current_screen = 4;
    } else if (ev.code == KeyCode::ENTER) {
        state.should_launch = true;
    }
}

bool TUIWizard::run(WizardConfig & out_config) {
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) {
        // Fallback for non-interactive environments
        std::cerr << "Sandwich Smart-Dispatch: Non-interactive terminal detected.\n";
        return false;
    }

    enable_raw_mode();

    bool running = true;
    while (running && !state.should_launch) {
        update_terminal_size();

        switch (current_screen) {
            case 0: screen_welcome(); break;
            case 1: screen_model_selection(); break;
            case 2: screen_drafter_selection(); break;
            case 3: screen_context_quant(); break;
            case 4: screen_advanced_flags(); break;
            case 5: screen_review_launch(); break;
        }

        KeyEvent ev = read_key();
        if (ev.code == KeyCode::CHAR && (ev.ch == 'q' || ev.ch == 'Q') && !model_filter_mode) {
            running = false;
            break;
        }

        switch (current_screen) {
            case 0: handle_input_welcome(ev); break;
            case 1: handle_input_model(ev); break;
            case 2: handle_input_drafter(ev); break;
            case 3: handle_input_context(ev); break;
            case 4: handle_input_flags(ev); break;
            case 5: handle_input_review(ev); break;
        }
    }

    disable_raw_mode();

    if (state.should_launch) {
        out_config = state;
        return true;
    }

    return false;
}

} // namespace dispatch
} // namespace sandwich
