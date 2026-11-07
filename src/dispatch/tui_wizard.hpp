#pragma once

#include "hardware_profiler.hpp"
#include "model_scanner.hpp"

#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <sstream>
#include <termios.h>

namespace sandwich {
namespace dispatch {

enum class KeyCode {
    UNKNOWN = 0,
    ENTER,
    ESCAPE,
    BACKSPACE,
    TAB,
    BACKTAB,
    SPACE,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    PAGE_UP,
    PAGE_DOWN,
    F1,
    F5,
    CHAR
};

struct KeyEvent {
    KeyCode code = KeyCode::UNKNOWN;
    char    ch   = 0;
};

struct WizardConfig {
    ModelMetadata   selected_model;
    ModelMetadata   selected_drafter;
    bool            use_speculative       = false;
    int             context_length        = 65536;
    std::string     kv_quant              = "turbo4";
    bool            flash_attn            = true;
    bool            lock_p_cores          = true;
    bool            moe_row_split         = true;
    bool            radix_cache           = true;
    bool            spec_tree             = false;
    bool            turbo_kvarn           = true;
    bool            dynamic_yarn          = true;
    bool            wayland_guard         = true;
    bool            attn_sinks            = false;
    bool            split_kv              = false;
    bool            warm_pool             = true;
    bool            io_uring              = true;
    TieringPlan     tiering;
    std::string     synthesized_command;
    bool            should_launch         = false;
    bool            saved_preset          = false;
};

enum class ActiveSection {
    MODELS = 0,
    DRAFTERS,
    CONTEXT_QUANT,
    OPTIMIZATION_FLAGS,
    COUNT
};

struct LiveHardwareStats {
    uint64_t gpu_used_mb   = 0;
    uint64_t gpu_total_mb  = 8151;
    int      gpu_temp_c    = 36;
    int      gpu_util_pct  = 0;
    uint64_t ram_free_mb   = 0;
    uint64_t ram_total_mb  = 23646;
    int      cpu_temp_c    = 45;
    double   disk_free_gb  = 0.0;
    double   disk_total_gb = 0.0;
};

class TUIWizard {
public:
    TUIWizard();
    ~TUIWizard();

    // Runs the unified split-pane interactive TUI
    // Returns true if user chose to Launch, populating out_config
    bool run(WizardConfig & out_config);

    static void on_terminal_resize();

private:
    HardwareProfile profile;
    ModelCatalog    catalog;
    WizardConfig    state;
    int             term_rows = 24;
    int             term_cols = 100;

    // Terminal Raw Mode Management
    struct termios  orig_termios;
    bool            raw_mode_active = false;
    static std::atomic<bool> s_sigwinch_received;

    void enable_raw_mode();
    void disable_raw_mode();
    void update_terminal_size();
    KeyEvent read_key();

    // Split-Pane Layout Renderer
    void render_dashboard();
    void render_header(std::ostringstream & out);
    void render_footer(std::ostringstream & out);

    // Sub-panel renderers
    void render_left_panel(std::ostringstream & out, int start_row, int start_col, int width, int height);
    void render_right_panel(std::ostringstream & out, int start_row, int start_col, int width, int height);

    // Helpers
    std::string format_progress_bar(float fraction, int bar_width, const char * fill_clr);
    std::string smart_truncate(const std::string & text, size_t max_len);

    // Event & Input Handlers
    void handle_input(const KeyEvent & ev);
    void recompute_tiering_and_command();
    void save_preset();
    void log_event(const std::string & msg);

    // State
    ActiveSection active_section = ActiveSection::MODELS;
    bool help_overlay_active = false;

    // Models navigation
    size_t model_cursor = 0;
    size_t model_page_offset = 0;
    std::string model_filter;
    bool model_filter_mode = false;

    // Drafters navigation (horizontal scrolling)
    size_t drafter_cursor = 0; // 0=None, 1..N drafters
    size_t drafter_page_offset = 0;

    // Context & Quant navigation (horizontal scrolling + custom)
    size_t context_cursor = 5; // default: 5 -> 65536 (64k)
    size_t context_page_offset = 0;
    bool   context_custom_mode = false;
    std::string context_custom_str = "65536";

    size_t kv_quant_cursor = 0; // 0..7
    size_t kv_quant_page_offset = 0;
    int    cq_subfocus = 0; // 0 = context row, 1 = quant row

    // Optimization Flags navigation (2-column scrollable grid: 12 flags)
    size_t flags_cursor = 0; // 0..11 flags
    size_t flags_row_offset = 0;
    std::string last_hovered_hint;

    // Activity Log
    std::vector<std::string> event_logs;

    // Background Hardware Monitor Thread
    std::atomic<bool> monitor_running{false};
    std::atomic<bool> stats_updated{false};
    std::thread monitor_thread;
    std::mutex stats_mutex;
    LiveHardwareStats live_stats;
    void start_hardware_monitor();
    void stop_hardware_monitor();
    void update_hardware_stats();
};

} // namespace dispatch
} // namespace sandwich
