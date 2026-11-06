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
#include <chrono>
#include <ctime>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/statvfs.h>
#include <poll.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace sandwich {
namespace dispatch {

// ANSI Theme Color Palette
static const char * CLR_RESET      = "\033[0m";
static const char * CLR_BOLD       = "\033[1m";
static const char * CLR_GOLD       = "\033[1;38;2;229;169;59m";  // Sandwich Gold
static const char * CLR_LIGHT_YEL  = "\033[1;38;2;254;240;138m"; // Bright highlight yellow
static const char * CLR_CYAN       = "\033[1;38;2;78;201;176m";
static const char * CLR_GREEN      = "\033[1;38;2;34;197;94m";
static const char * CLR_RED        = "\033[1;38;2;239;68;68m";
static const char * CLR_GRAY       = "\033[38;2;140;140;140m";
static const char * CLR_DARK_GRAY  = "\033[38;2;80;80;80m";
static const char * BG_GOLD        = "\033[48;2;229;169;59m\033[30m\033[1m";
static const char * BG_YEL         = "\033[48;2;254;240;138m\033[30m\033[1m";

static std::string repeat_utf8(const std::string & s, int count) {
    if (count <= 0) return "";
    std::string res;
    res.reserve(s.size() * count);
    for (int i = 0; i < count; ++i) res += s;
    return res;
}

// Global pointer for clean terminal restoration
static TUIWizard * g_active_wizard = nullptr;

static void global_signal_handler(int sig) {
    if (g_active_wizard) {
        std::cout << "\033[?1049l\033[?25h\033[0m" << std::flush;
    }
    std::exit(128 + sig);
}

TUIWizard::TUIWizard() {
    g_active_wizard = this;
    std::signal(SIGINT, global_signal_handler);
    std::signal(SIGTERM, global_signal_handler);

    profile = HardwareProfiler::load_or_profile();
    catalog = ModelScanner::load_or_scan();

    update_terminal_size();

    // Default configuration state
    state.context_length  = 4096;
    state.kv_quant        = "turbo4";
    state.flash_attn      = true;
    state.lock_p_cores    = true;
    state.moe_row_split   = true;
    state.radix_cache     = true;
    state.turbo_kvarn     = true;
    state.dynamic_yarn    = true;
    state.wayland_guard   = true;

    // Default to first model
    if (!catalog.models.empty()) {
        state.selected_model = catalog.models[0];
        // Auto-pair drafter if matching name found
        for (size_t i = 0; i < catalog.drafters.size(); ++i) {
            std::string mf = catalog.models[0].filename;
            std::string df = catalog.drafters[i].filename;
            std::transform(mf.begin(), mf.end(), mf.begin(), ::tolower);
            std::transform(df.begin(), df.end(), df.begin(), ::tolower);
            if ((df.find("qwen3.6-35b") != std::string::npos && mf.find("qwen3.6-35b") != std::string::npos) ||
                (df.find("qwen3.8") != std::string::npos && mf.find("qwen3.8") != std::string::npos)) {
                drafter_cursor = i + 1;
                state.selected_drafter = catalog.drafters[i];
                state.use_speculative = true;
                break;
            }
        }
    }

    log_event("Console loaded. Discovered " + std::to_string(catalog.models.size()) + " models.");
    log_event("Hardware: " + profile.cpu.model_name + " | " + profile.gpu.device_name);

    recompute_tiering_and_command();
}

TUIWizard::~TUIWizard() {
    stop_hardware_monitor();
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
        raw.c_cc[VTIME] = 1;
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
        raw_mode_active = true;

        // Enter alternate buffer, hide cursor, clear screen
        std::cout << "\033[?1049h\033[?25l\033[2J\033[H" << std::flush;
    }
}

void TUIWizard::disable_raw_mode() {
    if (raw_mode_active) {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
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
        term_cols = 100;
        term_rows = 24;
    }
}

void TUIWizard::log_event(const std::string & msg) {
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
    localtime_r(&tt, &tm_buf);

    char time_str[16];
    std::strftime(time_str, sizeof(time_str), "%H:%M:%S", &tm_buf);

    event_logs.push_back("[" + std::string(time_str) + "] " + msg);
    if (event_logs.size() > 40) {
        event_logs.erase(event_logs.begin());
    }
}

void TUIWizard::start_hardware_monitor() {
    monitor_running = true;
    monitor_thread = std::thread([this]() {
        while (monitor_running) {
            update_hardware_stats();
            stats_updated = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    });
}

void TUIWizard::stop_hardware_monitor() {
    if (monitor_running) {
        monitor_running = false;
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }
    }
}

void TUIWizard::update_hardware_stats() {
    LiveHardwareStats st;

    // RAM via /proc/meminfo
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo.is_open()) {
        std::string line;
        uint64_t total_kb = 0, avail_kb = 0;
        while (std::getline(meminfo, line)) {
            if (line.rfind("MemTotal:", 0) == 0) {
                sscanf(line.c_str(), "MemTotal: %lu kB", &total_kb);
            } else if (line.rfind("MemAvailable:", 0) == 0) {
                sscanf(line.c_str(), "MemAvailable: %lu kB", &avail_kb);
            }
        }
        st.ram_total_mb = total_kb / 1024;
        st.ram_free_mb  = avail_kb / 1024;
    }

    // Storage via statvfs on /home
    struct statvfs svfs;
    if (statvfs("/home", &svfs) == 0) {
        double total_bytes = (double)svfs.f_blocks * svfs.f_frsize;
        double free_bytes  = (double)svfs.f_bavail * svfs.f_frsize;
        st.disk_total_gb   = total_bytes / (1024.0 * 1024.0 * 1024.0);
        st.disk_free_gb    = free_bytes  / (1024.0 * 1024.0 * 1024.0);
    }

    // CPU Temp via thermal_zone0
    std::ifstream thm("/sys/class/thermal/thermal_zone0/temp");
    if (thm.is_open()) {
        int millic = 0;
        if (thm >> millic) st.cpu_temp_c = millic / 1000;
    }

    // GPU via nvidia-smi
    FILE * fp = popen("nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null", "r");
    if (fp) {
        char buf[128];
        if (fgets(buf, sizeof(buf), fp)) {
            unsigned int used = 0, total = 8151, temp = 36, util = 0;
            if (sscanf(buf, "%u, %u, %u, %u", &used, &total, &temp, &util) >= 3) {
                st.gpu_used_mb  = used;
                st.gpu_total_mb = total;
                st.gpu_temp_c   = temp;
                st.gpu_util_pct = util;
            }
        }
        pclose(fp);
    }

    std::lock_guard<std::mutex> lock(stats_mutex);
    live_stats = st;
}

KeyEvent TUIWizard::read_key() {
    KeyEvent ev;
    struct pollfd pfd;
    pfd.fd = STDIN_FILENO;
    pfd.events = POLLIN;

    int ret = poll(&pfd, 1, 100); // 100ms polling
    if (ret <= 0) return ev;

    char c;
    if (read(STDIN_FILENO, &c, 1) != 1) return ev;

    if (c == 27) {
        char seq[5];
        if (read(STDIN_FILENO, &seq[0], 1) != 1) {
            ev.code = KeyCode::ESCAPE;
            return ev;
        }
        if (read(STDIN_FILENO, &seq[1], 1) != 1) {
            ev.code = KeyCode::ESCAPE;
            return ev;
        }

        if (seq[0] == '[') {
            if (seq[1] >= '0' && seq[1] <= '9') {
                if (read(STDIN_FILENO, &seq[2], 1) == 1 && seq[2] == '~') {
                    switch (seq[1]) {
                        case '5': ev.code = KeyCode::PAGE_UP;   return ev;
                        case '6': ev.code = KeyCode::PAGE_DOWN; return ev;
                        case '1': ev.code = KeyCode::F5;        return ev;
                    }
                }
            } else {
                switch (seq[1]) {
                    case 'A': ev.code = KeyCode::UP;      return ev;
                    case 'B': ev.code = KeyCode::DOWN;    return ev;
                    case 'C': ev.code = KeyCode::RIGHT;   return ev;
                    case 'D': ev.code = KeyCode::LEFT;    return ev;
                    case 'Z': ev.code = KeyCode::BACKTAB; return ev;
                }
            }
        } else if (seq[0] == 'O' && seq[1] == 'P') {
            ev.code = KeyCode::F1;
            return ev;
        }
        ev.code = KeyCode::ESCAPE;
        return ev;
    } else if (c == '\r' || c == '\n') {
        ev.code = KeyCode::ENTER;
    } else if (c == '\t') {
        ev.code = KeyCode::TAB;
    } else if (c == ' ') {
        ev.code = KeyCode::SPACE;
    } else if (c == 127 || c == '\b') {
        ev.code = KeyCode::BACKSPACE;
    } else {
        ev.code = KeyCode::CHAR;
        ev.ch = c;
    }
    return ev;
}

void TUIWizard::recompute_tiering_and_command() {
    if (!catalog.models.empty() && model_cursor < catalog.models.size()) {
        state.selected_model = catalog.models[model_cursor];
    }

    if (drafter_cursor > 0 && drafter_cursor <= catalog.drafters.size()) {
        state.selected_drafter = catalog.drafters[drafter_cursor - 1];
        state.use_speculative = true;
    } else {
        state.use_speculative = false;
    }

    static const int ctx_options[] = {2048, 4096, 8192, 16384, 32768, 65536};
    state.context_length = ctx_options[context_cursor % 6];

    static const char * kv_options[] = {"turbo4", "turbo-kvarn", "turbo3", "turbo2", "f16"};
    state.kv_quant = kv_options[kv_quant_cursor % 5];

    // Compute tiering
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

    // Build synthesized command line with 100% valid Sandwich flags
    std::ostringstream cmd;
    cmd << "./sandwich run -m " << state.selected_model.file_path;

    if (state.use_speculative && !state.selected_drafter.file_path.empty()) {
        cmd << " -md " << state.selected_drafter.file_path;
    }

    cmd << " -c " << state.context_length;

    if (state.kv_quant == "turbo-kvarn") {
        cmd << " -ctk turbo4 -ctv turbo4 --turbo-kvran";
    } else {
        cmd << " -ctk " << state.kv_quant << " -ctv " << state.kv_quant;
    }

    if (state.flash_attn)    cmd << " -fa";
    if (state.lock_p_cores)  cmd << " -t " << profile.cpu.optimal_threads;
    if (state.moe_row_split && state.selected_model.is_moe) cmd << " -sm row";
    if (state.radix_cache)   cmd << " --radix-cache";
    if (state.wayland_guard) cmd << " -fitt " << WAYLAND_VRAM_GUARD_MB;
    if (state.dynamic_yarn)  cmd << " --dynamic-yarn";

    state.synthesized_command = cmd.str();

    // Update single-line hint string
    switch (active_section) {
        case ActiveSection::MODELS:
            last_hovered_hint = "Selected Model: " + state.selected_model.display_name + " ──▶ [Enter] Launches immediately";
            break;
        case ActiveSection::DRAFTERS:
            if (state.use_speculative) {
                last_hovered_hint = "Speculative Drafter: " + state.selected_drafter.filename + " ──▶ Adds: -md " + state.selected_drafter.filename;
            } else {
                last_hovered_hint = "Single-Token Decoding (No drafter) ──▶ [Enter] Launches";
            }
            break;
        case ActiveSection::CONTEXT_QUANT:
            if (cq_subfocus == 0) {
                last_hovered_hint = "Context Window: " + std::to_string(state.context_length) + " tokens ──▶ Adds: -c " + std::to_string(state.context_length);
            } else {
                if (state.kv_quant == "turbo-kvarn") {
                    last_hovered_hint = "KV Quant: Turbo-KVarN Fused Hybrid ──▶ Adds: -ctk turbo4 -ctv turbo4 --turbo-kvran";
                } else {
                    last_hovered_hint = "KV Quant: " + state.kv_quant + " ──▶ Adds: -ctk " + state.kv_quant + " -ctv " + state.kv_quant;
                }
            }
            break;
        case ActiveSection::OPTIMIZATION_FLAGS: {
            switch (flags_cursor) {
                case 0: last_hovered_hint = "Flash Attention 2: Hardware-accelerated attention ──▶ Adds: -fa"; break;
                case 1: last_hovered_hint = "Turbo-KVarN Hybrid: 4-bit Polar quantization ──▶ Adds: --turbo-kvran"; break;
                case 2: last_hovered_hint = "Dynamic YaRN: Context retention scaling ──▶ Adds: --dynamic-yarn"; break;
                case 3: last_hovered_hint = "P-Core Affinity: Pin to 6 physical CPU cores ──▶ Adds: -t 6"; break;
                case 4: last_hovered_hint = "MoE Row-Split: Expert memory tiering (VRAM/RAM) ──▶ Adds: -sm row"; break;
                case 5: last_hovered_hint = "Wayland Guard: Reserve 850 MB VRAM for compositor ──▶ Adds: -fitt 7100"; break;
            }
            break;
        }
        default:
            break;
    }
}

void TUIWizard::save_preset() {
    std::string preset_dir = std::string(getenv("HOME") ? getenv("HOME") : ".") + "/.config/sandwich/presets";
    fs::create_directories(preset_dir);
    std::string out_path = preset_dir + "/latest.json";

    json j;
    j["model"]         = state.selected_model.file_path;
    j["drafter"]       = state.use_speculative ? state.selected_drafter.file_path : "";
    j["context"]       = state.context_length;
    j["kv_quant"]      = state.kv_quant;
    j["flash_attn"]    = state.flash_attn;
    j["lock_p_cores"]  = state.lock_p_cores;
    j["moe_row_split"] = state.moe_row_split;
    j["radix_cache"]   = state.radix_cache;
    j["turbo_kvarn"]   = state.turbo_kvarn;
    j["dynamic_yarn"]  = state.dynamic_yarn;
    j["wayland_guard"] = state.wayland_guard;
    j["command"]       = state.synthesized_command;

    std::ofstream ofs(out_path);
    if (ofs.is_open()) {
        ofs << j.dump(4);
        log_event("Preset saved: latest.json");
    }
}

void TUIWizard::render_dashboard() {
    update_terminal_size();

    std::ostringstream out;
    out << "\033[H"; // Cursor to top-left

    int left_w = static_cast<int>(term_cols * 0.68);
    int right_w = term_cols - left_w - 2;
    if (left_w < 46) left_w = 46;
    if (right_w < 24) right_w = 24;

    int content_h = term_rows - 4; // leave 3 rows at bottom for Hint, Command, Status
    if (content_h < 14) content_h = 14;

    render_header(out);
    render_left_panel(out, 2, 1, left_w, content_h);
    render_right_panel(out, 2, left_w + 2, right_w, content_h);
    render_footer(out);

    if (help_overlay_active) {
        int box_w = std::min(68, term_cols - 4);
        int box_h = 15;
        int top = (term_rows - box_h) / 2;
        int left = (term_cols - box_w) / 2;

        out << "\033[" << top << ";" << left << "H" << CLR_GOLD << "╔" << repeat_utf8("═", box_w - 2) << "╗" << CLR_RESET;
        out << "\033[" << top + 1 << ";" << left << "H" << CLR_GOLD << "║ " << CLR_LIGHT_YEL << "SANDWICH.CPP CONTROLS GUIDE" << std::string(box_w - 30, ' ') << CLR_GOLD << "║" << CLR_RESET;
        out << "\033[" << top + 2 << ";" << left << "H" << CLR_GOLD << "╠" << repeat_utf8("═", box_w - 2) << "╣" << CLR_RESET;

        const char * help_lines[] = {
            "  [Enter] / [F5]        Launch Sandwich Engine with current settings",
            "  [↑] / [↓]             Browse models / navigate flags",
            "  [←] / [→]             Change Context length or KV Quant",
            "  [Tab] / [Shift+Tab]   Switch between Model, Drafter, Context, Flags",
            "  [Space]               Toggle optimization checkbox",
            "  [/]                   Quick-search filter for model catalog",
            "  [R]                   Rescan local GGUF models on disk",
            "  [S]                   Save current setup to ~/.config/sandwich/presets",
            "  [Esc] / [Q]           Exit Sandwich Console"
        };

        for (size_t i = 0; i < 9; ++i) {
            std::string hl = help_lines[i];
            if (hl.size() < (size_t)(box_w - 4)) hl += std::string(box_w - 4 - hl.size(), ' ');
            out << "\033[" << top + 3 + i << ";" << left << "H" << CLR_GOLD << "║ " << CLR_RESET << hl << CLR_GOLD << " ║" << CLR_RESET;
        }

        out << "\033[" << top + 12 << ";" << left << "H" << CLR_GOLD << "╠" << repeat_utf8("═", box_w - 2) << "╣" << CLR_RESET;
        out << "\033[" << top + 13 << ";" << left << "H" << CLR_GOLD << "║ " << CLR_LIGHT_YEL << "  Press any key or [Esc] to return... " << std::string(box_w - 40, ' ') << CLR_GOLD << "║" << CLR_RESET;
        out << "\033[" << top + 14 << ";" << left << "H" << CLR_GOLD << "╚" << repeat_utf8("═", box_w - 2) << "╝" << CLR_RESET;
    }

    std::cout << out.str() << std::flush;
}

void TUIWizard::render_header(std::ostringstream & out) {
    out << "\033[1;1H" << CLR_GOLD << " 🥪 SANDWICH.CPP ── SMART-DISPATCH "
        << CLR_GRAY << "│ " << CLR_CYAN << profile.cpu.model_name << " "
        << CLR_GRAY << "│ " << CLR_LIGHT_YEL << profile.gpu.device_name
        << CLR_RESET << "\033[K";
}

void TUIWizard::render_left_panel(std::ostringstream & out, int start_row, int start_col, int width, int height) {
    (void)height;
    int r = start_row;

    auto print_box_header = [&](const std::string & title, bool active) {
        out << "\033[" << r++ << ";" << start_col << "H";
        const char * clr = active ? CLR_GOLD : CLR_DARK_GRAY;
        const char * tclr = active ? CLR_LIGHT_YEL : CLR_GRAY;
        out << clr << "┌─[" << tclr << " " << title << " " << clr << "]─"
            << repeat_utf8("─", std::max(0, width - (int)title.size() - 8)) << "┐" << CLR_RESET << "\033[K";
    };

    auto print_box_footer = [&](bool active) {
        out << "\033[" << r++ << ";" << start_col << "H";
        const char * clr = active ? CLR_GOLD : CLR_DARK_GRAY;
        out << clr << "└" << repeat_utf8("─", std::max(0, width - 2)) << "┘" << CLR_RESET << "\033[K";
    };

    // ─────────────────────────────────────────────────────────────
    // 1. MODEL SELECTION BOX
    // ─────────────────────────────────────────────────────────────
    bool act_models = (active_section == ActiveSection::MODELS);
    print_box_header(act_models ? "1. MODELS (Active: [Enter] Runs)" : "1. MODELS", act_models);

    // Search bar
    out << "\033[" << r++ << ";" << start_col << "H" << (act_models ? CLR_GOLD : CLR_DARK_GRAY) << "│ " << CLR_RESET;
    if (model_filter_mode) {
        out << CLR_LIGHT_YEL << "Search [/]: " << CLR_BOLD << model_filter << "█" << CLR_RESET;
    } else if (!model_filter.empty()) {
        out << CLR_CYAN << "Filter: '" << model_filter << "' ([/] edit, [Esc] clear)" << CLR_RESET;
    } else {
        out << CLR_GRAY << "Press [/] to search " << catalog.models.size() << " models | [Enter] Launch" << CLR_RESET;
    }
    out << "\033[" << (r - 1) << ";" << (start_col + width - 1) << "H" << (act_models ? CLR_GOLD : CLR_DARK_GRAY) << "│" << CLR_RESET << "\033[K";

    // Model list
    std::vector<size_t> matched_idx;
    for (size_t i = 0; i < catalog.models.size(); ++i) {
        if (model_filter.empty()) {
            matched_idx.push_back(i);
        } else {
            std::string n = catalog.models[i].display_name;
            std::string f = model_filter;
            std::transform(n.begin(), n.end(), n.begin(), ::tolower);
            std::transform(f.begin(), f.end(), f.begin(), ::tolower);
            if (n.find(f) != std::string::npos) matched_idx.push_back(i);
        }
    }

    const size_t page_sz = 3; // Keep compact to fit 24 rows
    if (matched_idx.empty()) {
        out << "\033[" << r++ << ";" << start_col << "H" << (act_models ? CLR_GOLD : CLR_DARK_GRAY) << "│ "
            << CLR_RED << "No models matching filter" << CLR_RESET;
        out << "\033[" << (r - 1) << ";" << (start_col + width - 1) << "H" << (act_models ? CLR_GOLD : CLR_DARK_GRAY) << "│" << CLR_RESET << "\033[K";
    } else {
        if (model_cursor >= matched_idx.size()) model_cursor = matched_idx.size() - 1;
        if (model_cursor < model_page_offset) model_page_offset = model_cursor;
        else if (model_cursor >= model_page_offset + page_sz) model_page_offset = model_cursor - page_sz + 1;

        for (size_t p = 0; p < page_sz; ++p) {
            size_t idx = model_page_offset + p;
            out << "\033[" << r++ << ";" << start_col << "H" << (act_models ? CLR_GOLD : CLR_DARK_GRAY) << "│ " << CLR_RESET;
            if (idx < matched_idx.size()) {
                const auto & m = catalog.models[matched_idx[idx]];
                bool is_cur = (idx == model_cursor);
                std::string dname = m.display_name;
                if (dname.size() > 32) dname = dname.substr(0, 29) + "...";

                if (is_cur && act_models) {
                    out << BG_YEL << " ▶ " << std::left << std::setw(32) << dname
                        << " [" << std::setw(8) << m.quantization << "] "
                        << std::fixed << std::setprecision(1) << std::setw(5) << m.size_gb << "GB "
                        << (m.is_moe ? "[MoE]" : "[Dense]") << " " << CLR_RESET;
                } else if (is_cur) {
                    out << CLR_LIGHT_YEL << " ● " << std::left << std::setw(32) << dname
                        << " [" << std::setw(8) << m.quantization << "] "
                        << std::fixed << std::setprecision(1) << std::setw(5) << m.size_gb << "GB "
                        << (m.is_moe ? "[MoE]" : "[Dense]") << CLR_RESET;
                } else {
                    out << CLR_GRAY << "   " << std::left << std::setw(32) << dname
                        << " [" << std::setw(8) << m.quantization << "] "
                        << std::fixed << std::setprecision(1) << std::setw(5) << m.size_gb << "GB "
                        << (m.is_moe ? "[MoE]" : "[Dense]") << CLR_RESET;
                }
            }
            out << "\033[" << (r - 1) << ";" << (start_col + width - 1) << "H" << (act_models ? CLR_GOLD : CLR_DARK_GRAY) << "│" << CLR_RESET << "\033[K";
        }
    }
    print_box_footer(act_models);

    // ─────────────────────────────────────────────────────────────
    // 2. SPECULATIVE DRAFTER
    // ─────────────────────────────────────────────────────────────
    bool act_d = (active_section == ActiveSection::DRAFTERS);
    print_box_header(act_d ? "2. SPECULATIVE DRAFTER (Active)" : "2. SPECULATIVE DRAFTER", act_d);
    out << "\033[" << r++ << ";" << start_col << "H" << (act_d ? CLR_GOLD : CLR_DARK_GRAY) << "│ " << CLR_RESET;

    bool is_d0 = (drafter_cursor == 0);
    if (is_d0 && act_d) {
        out << BG_YEL << "[ None ]" << CLR_RESET << " ";
    } else if (is_d0) {
        out << CLR_LIGHT_YEL << "[● None]" << CLR_RESET << " ";
    } else {
        out << CLR_GRAY << "[ None ]" << CLR_RESET << " ";
    }

    for (size_t di = 0; di < std::min((size_t)2, catalog.drafters.size()); ++di) {
        bool is_cur_d = (drafter_cursor == di + 1);
        std::string fn = catalog.drafters[di].filename;
        if (fn.size() > 22) fn = fn.substr(0, 19) + "..";
        if (is_cur_d && act_d) {
            out << BG_YEL << "[ " << fn << " ]" << CLR_RESET << " ";
        } else if (is_cur_d) {
            out << CLR_LIGHT_YEL << "[● " << fn << "]" << CLR_RESET << " ";
        } else {
            out << CLR_GRAY << "[ " << fn << " ]" << CLR_RESET << " ";
        }
    }
    out << "\033[" << (r - 1) << ";" << (start_col + width - 1) << "H" << (act_d ? CLR_GOLD : CLR_DARK_GRAY) << "│" << CLR_RESET << "\033[K";
    print_box_footer(act_d);

    // ─────────────────────────────────────────────────────────────
    // 3. CONTEXT & KV QUANTIZATION
    // ─────────────────────────────────────────────────────────────
    bool act_cq = (active_section == ActiveSection::CONTEXT_QUANT);
    print_box_header(act_cq ? "3. CONTEXT & KV QUANT (Active: [←/→] Select)" : "3. CONTEXT & KV QUANT", act_cq);

    // Context line
    out << "\033[" << r++ << ";" << start_col << "H" << (act_cq ? CLR_GOLD : CLR_DARK_GRAY) << "│ " << CLR_RESET;
    out << CLR_BOLD << "Ctx:   " << CLR_RESET;
    static const int ctx_vals[] = {2048, 4096, 8192, 16384, 32768, 65536};
    for (size_t ci = 0; ci < 6; ++ci) {
        bool is_ctx = (ci == context_cursor);
        bool focused = (act_cq && cq_subfocus == 0 && is_ctx);
        if (focused) out << BG_YEL << "[" << ctx_vals[ci] << "]" << CLR_RESET << " ";
        else if (is_ctx) out << CLR_LIGHT_YEL << "[●" << ctx_vals[ci] << "]" << CLR_RESET << " ";
        else out << CLR_GRAY << "[" << ctx_vals[ci] << "]" << CLR_RESET << " ";
    }
    out << "\033[" << (r - 1) << ";" << (start_col + width - 1) << "H" << (act_cq ? CLR_GOLD : CLR_DARK_GRAY) << "│" << CLR_RESET << "\033[K";

    // KV Quant line
    out << "\033[" << r++ << ";" << start_col << "H" << (act_cq ? CLR_GOLD : CLR_DARK_GRAY) << "│ " << CLR_RESET;
    out << CLR_BOLD << "Quant: " << CLR_RESET;
    static const char * kv_names[] = {"turbo4 (4-bit)", "turbo-kvran (Fused)", "turbo3 (3-bit)", "f16"};
    for (size_t ki = 0; ki < 4; ++ki) {
        bool is_kv = (ki == kv_quant_cursor);
        bool focused = (act_cq && cq_subfocus == 1 && is_kv);
        if (focused) out << BG_YEL << "[" << kv_names[ki] << "]" << CLR_RESET << " ";
        else if (is_kv) out << CLR_LIGHT_YEL << "[●" << kv_names[ki] << "]" << CLR_RESET << " ";
        else out << CLR_GRAY << "[" << kv_names[ki] << "]" << CLR_RESET << " ";
    }
    out << "\033[" << (r - 1) << ";" << (start_col + width - 1) << "H" << (act_cq ? CLR_GOLD : CLR_DARK_GRAY) << "│" << CLR_RESET << "\033[K";
    print_box_footer(act_cq);

    // ─────────────────────────────────────────────────────────────
    // 4. OPTIMIZATION FLAGS (Compact 2-Column Grid)
    // ─────────────────────────────────────────────────────────────
    bool act_f = (active_section == ActiveSection::OPTIMIZATION_FLAGS);
    print_box_header(act_f ? "4. OPTIMIZATION FLAGS (Active: [Space] Toggle)" : "4. OPTIMIZATION FLAGS", act_f);

    struct FlagItem {
        bool & val;
        const char * name;
        const char * flag;
    };
    FlagItem flag_items[] = {
        {state.flash_attn,    "Flash Attention 2", "-fa"},
        {state.turbo_kvarn,   "Turbo-KVarN Hybrid", "--turbo-kvran"},
        {state.dynamic_yarn,  "Dynamic YaRN Scaling", "--dynamic-yarn"},
        {state.lock_p_cores,  "Pin 6 Intel P-Cores", "-t 6"},
        {state.moe_row_split, "MoE Row-Split Tiering", "-sm row"},
        {state.wayland_guard, "7.1GB Wayland Guard", "-fitt 7100"}
    };

    for (size_t row_i = 0; row_i < 3; ++row_i) {
        out << "\033[" << r++ << ";" << start_col << "H" << (act_f ? CLR_GOLD : CLR_DARK_GRAY) << "│ " << CLR_RESET;

        // Col 1
        size_t fi1 = row_i * 2;
        bool cur1 = (act_f && flags_cursor == fi1);
        std::string m1 = flag_items[fi1].val ? "[✔] " : "[ ] ";
        if (cur1) out << BG_YEL << "▶ " << m1 << flag_items[fi1].name << " " << CLR_RESET;
        else out << (flag_items[fi1].val ? CLR_GREEN : CLR_GRAY) << m1 << CLR_RESET << CLR_GRAY << std::left << std::setw(20) << flag_items[fi1].name << CLR_RESET;

        out << "   ";

        // Col 2
        size_t fi2 = row_i * 2 + 1;
        bool cur2 = (act_f && flags_cursor == fi2);
        std::string m2 = flag_items[fi2].val ? "[✔] " : "[ ] ";
        if (cur2) out << BG_YEL << "▶ " << m2 << flag_items[fi2].name << " " << CLR_RESET;
        else out << (flag_items[fi2].val ? CLR_GREEN : CLR_GRAY) << m2 << CLR_RESET << CLR_GRAY << std::left << std::setw(20) << flag_items[fi2].name << CLR_RESET;

        out << "\033[" << (r - 1) << ";" << (start_col + width - 1) << "H" << (act_f ? CLR_GOLD : CLR_DARK_GRAY) << "│" << CLR_RESET << "\033[K";
    }
    print_box_footer(act_f);
}

void TUIWizard::render_right_panel(std::ostringstream & out, int start_row, int start_col, int width, int height) {
    int r = start_row;
    int half_h = (height - 2) / 2;

    // ─────────────────────────────────────────────────────────────
    // UPPER HALF: ACTIVITY & DIAGNOSTIC LOG
    // ─────────────────────────────────────────────────────────────
    out << "\033[" << r++ << ";" << start_col << "H" << CLR_GOLD
        << "┌─[ " << CLR_LIGHT_YEL << "ACTIVITY LOG" << CLR_GOLD << " ]"
        << repeat_utf8("─", std::max(0, width - 16)) << "┐" << CLR_RESET << "\033[K";

    int log_lines = std::max(3, half_h - 2);
    int start_log_idx = (int)event_logs.size() - log_lines;
    if (start_log_idx < 0) start_log_idx = 0;

    for (int i = 0; i < log_lines; ++i) {
        out << "\033[" << r++ << ";" << start_col << "H" << CLR_GOLD << "│ " << CLR_RESET;
        int idx = start_log_idx + i;
        if (idx < (int)event_logs.size()) {
            std::string el = event_logs[idx];
            if (el.size() > (size_t)(width - 4)) el = el.substr(0, width - 7) + "...";
            out << CLR_GRAY << el << CLR_RESET;
        }
        out << "\033[" << (r - 1) << ";" << (start_col + width - 1) << "H" << CLR_GOLD << "│" << CLR_RESET << "\033[K";
    }
    out << "\033[" << r++ << ";" << start_col << "H" << CLR_GOLD << "└" << repeat_utf8("─", std::max(0, width - 2)) << "┘" << CLR_RESET << "\033[K";

    // ─────────────────────────────────────────────────────────────
    // LOWER HALF: REAL-TIME SYSTEM HARDWARE MONITOR
    // ─────────────────────────────────────────────────────────────
    LiveHardwareStats st;
    {
        std::lock_guard<std::mutex> lock(stats_mutex);
        st = live_stats;
    }

    out << "\033[" << r++ << ";" << start_col << "H" << CLR_GOLD
        << "┌─[ " << CLR_LIGHT_YEL << "SYSTEM MONITOR" << CLR_GOLD << " ]"
        << repeat_utf8("─", std::max(0, width - 18)) << "┐" << CLR_RESET << "\033[K";

    auto print_stat_line = [&](const std::string & label, const std::string & val, const char * clr) {
        out << "\033[" << r++ << ";" << start_col << "H" << CLR_GOLD << "│ " << CLR_RESET;
        out << CLR_BOLD << std::left << std::setw(11) << label << CLR_RESET << ": "
            << clr << val << CLR_RESET;
        out << "\033[" << (r - 1) << ";" << (start_col + width - 1) << "H" << CLR_GOLD << "│" << CLR_RESET << "\033[K";
    };

    print_stat_line("GPU", profile.gpu.device_name, CLR_LIGHT_YEL);

    char vram_buf[64];
    snprintf(vram_buf, sizeof(vram_buf), "%lu MB / %lu MB (%d°C)",
             (unsigned long)st.gpu_used_mb, (unsigned long)st.gpu_total_mb, st.gpu_temp_c);
    print_stat_line("VRAM", vram_buf, CLR_CYAN);

    print_stat_line("Guard", "7,100 MB Limit (Safe)", CLR_GREEN);

    char ram_buf[64];
    snprintf(ram_buf, sizeof(ram_buf), "%.1f GB Free / %.1f GB Total",
             st.ram_free_mb / 1024.0, st.ram_total_mb / 1024.0);
    print_stat_line("RAM", ram_buf, CLR_CYAN);

    char warm_buf[64];
    snprintf(warm_buf, sizeof(warm_buf), "%.1f GB (70%% Rule #42)",
             profile.ram.max_safe_allocation_mb / 1024.0);
    print_stat_line("WARM Pool", warm_buf, CLR_GREEN);

    char cpu_buf[64];
    snprintf(cpu_buf, sizeof(cpu_buf), "%d P-cores pinned @ %d°C",
             profile.cpu.p_cores, st.cpu_temp_c);
    print_stat_line("CPU", cpu_buf, CLR_LIGHT_YEL);

    char disk_buf[64];
    snprintf(disk_buf, sizeof(disk_buf), "%.0f GB Free (NVMe)", st.disk_free_gb);
    print_stat_line("Storage", disk_buf, CLR_GRAY);

    out << "\033[" << r++ << ";" << start_col << "H" << CLR_GOLD << "└" << repeat_utf8("─", std::max(0, width - 2)) << "┘" << CLR_RESET << "\033[K";
}

void TUIWizard::render_footer(std::ostringstream & out) {
    // Row term_rows - 2: Single-Line Clean Hint Bar (outside the box layout)
    out << "\033[" << (term_rows - 2) << ";1H"
        << CLR_LIGHT_YEL << "💡 Hint: " << CLR_RESET << last_hovered_hint
        << "\033[K";

    // Row term_rows - 1: Clean Command Preview Line
    std::string cmd = state.synthesized_command;
    if (cmd.size() > (size_t)(term_cols - 12)) cmd = cmd.substr(0, term_cols - 15) + "...";
    out << "\033[" << (term_rows - 1) << ";1H"
        << CLR_GOLD << "▶ Command: " << CLR_RESET << cmd
        << "\033[K";

    // Row term_rows: Clean Status / Hotkey bar
    out << "\033[" << term_rows << ";1H"
        << BG_GOLD << " [Enter] Launch Selected Model " << CLR_RESET << " "
        << CLR_GOLD << "[↑/↓] Browse Models  [Tab] Switch Section  [Space] Toggle  [/] Search  [Esc] Quit" << CLR_RESET
        << "\033[K";
}

void TUIWizard::handle_input(const KeyEvent & ev) {
    if (help_overlay_active) {
        help_overlay_active = false;
        return;
    }

    if (model_filter_mode) {
        if (ev.code == KeyCode::ENTER || ev.code == KeyCode::ESCAPE) {
            model_filter_mode = false;
            log_event("Filter: '" + model_filter + "'");
        } else if (ev.code == KeyCode::BACKSPACE) {
            if (!model_filter.empty()) model_filter.pop_back();
        } else if (ev.code == KeyCode::CHAR) {
            model_filter.push_back(ev.ch);
        }
        recompute_tiering_and_command();
        return;
    }

    switch (ev.code) {
        case KeyCode::TAB:
            active_section = static_cast<ActiveSection>((static_cast<int>(active_section) + 1) % static_cast<int>(ActiveSection::COUNT));
            break;
        case KeyCode::BACKTAB:
            active_section = static_cast<ActiveSection>((static_cast<int>(active_section) - 1 + static_cast<int>(ActiveSection::COUNT)) % static_cast<int>(ActiveSection::COUNT));
            break;
        case KeyCode::F1:
            help_overlay_active = true;
            break;
        case KeyCode::F5:
            log_event("Launch command dispatched (F5).");
            state.should_launch = true;
            return;
        case KeyCode::ENTER:
            // PRESSING ENTER LAUNCHES THE ENGINE DIRECTLY!
            log_event("Launch command dispatched (Enter).");
            state.should_launch = true;
            return;
        case KeyCode::CHAR:
            if (ev.ch == 'h' || ev.ch == 'H' || ev.ch == '?') {
                help_overlay_active = true;
            } else if (ev.ch == 'q' || ev.ch == 'Q') {
                state.should_launch = false;
            } else if (ev.ch == '/') {
                model_filter_mode = true;
                model_filter.clear();
            } else if (ev.ch == 'r' || ev.ch == 'R') {
                log_event("Rescanning local GGUF models...");
                catalog = ModelScanner::run_full_scan();
                log_event("Rescan done: " + std::to_string(catalog.models.size()) + " models found.");
            } else if (ev.ch == 's' || ev.ch == 'S') {
                save_preset();
            }
            break;
        case KeyCode::UP:
            if (active_section == ActiveSection::MODELS) {
                if (model_cursor > 0) model_cursor--;
            } else if (active_section == ActiveSection::OPTIMIZATION_FLAGS) {
                if (flags_cursor >= 2) flags_cursor -= 2;
            } else if (active_section == ActiveSection::CONTEXT_QUANT) {
                cq_subfocus = 0;
            }
            break;
        case KeyCode::DOWN:
            if (active_section == ActiveSection::MODELS) {
                if (model_cursor + 1 < catalog.models.size()) model_cursor++;
            } else if (active_section == ActiveSection::OPTIMIZATION_FLAGS) {
                if (flags_cursor + 2 < 6) flags_cursor += 2;
            } else if (active_section == ActiveSection::CONTEXT_QUANT) {
                cq_subfocus = 1;
            }
            break;
        case KeyCode::LEFT:
            if (active_section == ActiveSection::DRAFTERS) {
                if (drafter_cursor > 0) drafter_cursor--;
            } else if (active_section == ActiveSection::CONTEXT_QUANT) {
                if (cq_subfocus == 0 && context_cursor > 0) context_cursor--;
                else if (cq_subfocus == 1 && kv_quant_cursor > 0) kv_quant_cursor--;
            } else if (active_section == ActiveSection::OPTIMIZATION_FLAGS) {
                if (flags_cursor % 2 == 1) flags_cursor--;
            }
            break;
        case KeyCode::RIGHT:
            if (active_section == ActiveSection::DRAFTERS) {
                if (drafter_cursor < catalog.drafters.size()) drafter_cursor++;
            } else if (active_section == ActiveSection::CONTEXT_QUANT) {
                if (cq_subfocus == 0 && context_cursor < 5) context_cursor++;
                else if (cq_subfocus == 1 && kv_quant_cursor < 3) kv_quant_cursor++;
            } else if (active_section == ActiveSection::OPTIMIZATION_FLAGS) {
                if (flags_cursor % 2 == 0 && flags_cursor + 1 < 6) flags_cursor++;
            }
            break;
        case KeyCode::SPACE:
            if (active_section == ActiveSection::OPTIMIZATION_FLAGS) {
                switch (flags_cursor) {
                    case 0: state.flash_attn    = !state.flash_attn;    log_event(std::string("Flash Attention: ") + (state.flash_attn ? "ON" : "OFF")); break;
                    case 1: state.turbo_kvarn   = !state.turbo_kvarn;   log_event(std::string("Turbo-KVarN: ") + (state.turbo_kvarn ? "ON" : "OFF")); break;
                    case 2: state.dynamic_yarn  = !state.dynamic_yarn;  log_event(std::string("Dynamic YaRN: ") + (state.dynamic_yarn ? "ON" : "OFF")); break;
                    case 3: state.lock_p_cores  = !state.lock_p_cores;  log_event(std::string("P-Core Affinity: ") + (state.lock_p_cores ? "ON" : "OFF")); break;
                    case 4: state.moe_row_split = !state.moe_row_split; log_event(std::string("MoE Row-Split: ") + (state.moe_row_split ? "ON" : "OFF")); break;
                    case 5: state.wayland_guard = !state.wayland_guard; log_event(std::string("Wayland Guard: ") + (state.wayland_guard ? "ON" : "OFF")); break;
                }
            } else if (active_section == ActiveSection::MODELS) {
                log_event("Selected model: " + catalog.models[model_cursor].display_name);
            }
            break;
        default:
            break;
    }

    recompute_tiering_and_command();
}

bool TUIWizard::run(WizardConfig & out_config) {
    enable_raw_mode();
    start_hardware_monitor();

    bool running = true;
    bool need_redraw = true;

    while (running) {
        if (need_redraw) {
            render_dashboard();
            need_redraw = false;
        }

        KeyEvent ev = read_key();
        if (ev.code != KeyCode::UNKNOWN) {
            if (ev.code == KeyCode::ESCAPE && !help_overlay_active && !model_filter_mode) {
                running = false;
                state.should_launch = false;
                break;
            }

            if (ev.code == KeyCode::CHAR && (ev.ch == 'q' || ev.ch == 'Q') && !model_filter_mode && !help_overlay_active) {
                running = false;
                state.should_launch = false;
                break;
            }

            handle_input(ev);
            need_redraw = true;

            if (state.should_launch) {
                running = false;
                break;
            }
        }

        if (stats_updated.exchange(false)) {
            need_redraw = true;
        }
    }

    stop_hardware_monitor();
    disable_raw_mode();

    out_config = state;
    return state.should_launch;
}

} // namespace dispatch
} // namespace sandwich
