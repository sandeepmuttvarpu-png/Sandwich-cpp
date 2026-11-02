#pragma once

#include "hardware_profiler.hpp"
#include "model_scanner.hpp"

#include <string>
#include <vector>
#include <memory>
#include <termios.h>

namespace sandwich {
namespace dispatch {

enum class KeyCode {
    UNKNOWN = 0,
    ENTER,
    ESCAPE,
    BACKSPACE,
    TAB,
    SPACE,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    PAGE_UP,
    PAGE_DOWN,
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
    int             context_length        = 4096;
    std::string     kv_quant              = "turbo3";
    bool            flash_attn            = true;
    bool            lock_p_cores          = true;
    bool            moe_row_split         = true;
    bool            radix_cache           = true;
    bool            spec_tree             = false;
    TieringPlan     tiering;
    std::string     synthesized_command;
    bool            should_launch         = false;
    bool            saved_preset          = false;
};

class TUIWizard {
public:
    TUIWizard();
    ~TUIWizard();

    // Runs the interactive TUI wizard flow
    // Returns true if user chose to Launch, populating out_config
    bool run(WizardConfig & out_config);

private:
    HardwareProfile profile;
    ModelCatalog    catalog;
    WizardConfig    state;
    int             current_screen = 0; // 0=Welcome, 1=Model, 2=Drafter, 3=Context, 4=Flags, 5=Review
    int             term_rows = 24;
    int             term_cols = 80;

    // Terminal Raw Mode Management
    struct termios  orig_termios;
    bool            raw_mode_active = false;

    void enable_raw_mode();
    void disable_raw_mode();
    void update_terminal_size();
    KeyEvent read_key();

    // Screen Renderers
    void render();
    void render_header(const std::string & title);
    void render_footer(const std::string & navigation_help);

    void screen_welcome();
    void screen_model_selection();
    void screen_drafter_selection();
    void screen_context_quant();
    void screen_advanced_flags();
    void screen_review_launch();

    // Navigation and state handlers
    void handle_input_welcome(const KeyEvent & ev);
    void handle_input_model(const KeyEvent & ev);
    void handle_input_drafter(const KeyEvent & ev);
    void handle_input_context(const KeyEvent & ev);
    void handle_input_flags(const KeyEvent & ev);
    void handle_input_review(const KeyEvent & ev);

    // Helpers
    void recompute_tiering_and_command();
    void save_preset();

    // Paging / Selection State
    size_t model_cursor = 0;
    size_t model_page_offset = 0;
    std::string model_filter;
    bool model_filter_mode = false;

    size_t drafter_option = 0; // 0=Recommended, 1=Manual, 2=Skip
    size_t drafter_cursor = 0;

    size_t context_cursor = 1; // index into {2048, 4096, 8192, 16384, 32768}
    size_t kv_quant_cursor = 0; // index into {turbo3, turbo4, turbo2, f16}

    size_t flags_cursor = 0;
};

} // namespace dispatch
} // namespace sandwich
