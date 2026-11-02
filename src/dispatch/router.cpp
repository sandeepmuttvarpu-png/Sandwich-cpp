#include "router.hpp"
#include "hardware_profiler.hpp"
#include "model_scanner.hpp"
#include "tui_wizard.hpp"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <cstring>
#include <unistd.h>

// Forward declaration of engine execution entry point
int llama_cli(int argc, char ** argv);

namespace sandwich {
namespace dispatch {

bool Router::is_manual_cli_invocation(int argc, char ** argv) {
    if (argc <= 1) return false;

    // Check if any argument is a standard inference flag
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model" ||
            arg == "-ngl" || arg == "--n-gpu-layers" ||
            arg == "-fitt" || arg == "--fit-target" ||
            arg == "-t" || arg == "--threads" ||
            arg == "-c" || arg == "--ctx-size" ||
            arg == "-p" || arg == "--prompt" ||
            arg == "-i" || arg == "--interactive" ||
            arg == "run" || arg == "cli" || arg == "serve" || arg == "bench") {
            return true;
        }
    }
    return false;
}

void Router::print_help(const char * progname) {
    std::cout << "\033[1;38;2;229;169;59m🥪 Sandwich.cpp ── Next-Generation Independent Inference Engine\033[0m\n\n";
    std::cout << "\033[1mUSAGE:\033[0m\n";
    std::cout << "  " << progname << "                  Launch Interactive Smart-Dispatch TUI Wizard\n";
    std::cout << "  " << progname << " [flags...]       Run direct CLI inference (bypasses TUI)\n\n";

    std::cout << "\033[1mAUTO-DISPATCH & PROFILING COMMANDS:\033[0m\n";
    std::cout << "  --tune, --profile           Run hardware profiling and display JSON diagnostics\n";
    std::cout << "  --scan-models               Force rescan of local GGUF models into cache\n";
    std::cout << "  --tui                       Explicitly launch the TUI Wizard\n";
    std::cout << "  -h, --help                  Show this help overview\n\n";

    std::cout << "\033[1mCORE INFERENCE FLAGS (Bypass Mode):\033[0m\n";
    std::cout << "  -m, --model <path>          Path to GGUF model file\n";
    std::cout << "  -md, --model-draft <path>   Path to speculative decoding drafter GGUF\n";
    std::cout << "  -fitt <mb>                  Target VRAM budget in MB (Enforces 7,100 MB Guard)\n";
    std::cout << "  -t, --threads <n>           Number of compute threads (Optimal: physical P-cores)\n";
    std::cout << "  -c, --ctx-size <n>          Context window size (default: 4096)\n";
    std::cout << "  -fa, --flash-attn           Enable Flash Attention 2\n";
    std::cout << "  -sm row                     MoE row-split cache tiering across VRAM/RAM/NVMe\n";
    std::cout << "  -ctk, -ctv <type>           KV cache quantization (e.g. q3_k for TurboQuant 3)\n";
}

int Router::run_standalone_profiling(bool force) {
    std::cout << "\033[1;38;2;78;201;176mExecuting Sandwich Hardware Profiler...\033[0m\n";
    HardwareProfile p = HardwareProfiler::run_profiling(force);
    std::cout << p.to_json() << std::endl;
    std::cout << "\n\033[38;2;34;197;94m✔ Hardware profile stored at " << HardwareProfiler::get_profile_path() << "\033[0m\n";
    return 0;
}

int Router::run_standalone_scan(const std::vector<std::string> & paths) {
    std::cout << "\033[1;38;2;78;201;176mScanning filesystem for GGUF models and speculative drafters...\033[0m\n";
    ModelCatalog cat = ModelScanner::run_full_scan(paths);

    std::cout << "\n\033[1mDiscovered Base Models (" << cat.models.size() << "):\033[0m\n";
    for (const auto & m : cat.models) {
        std::cout << "  • " << std::left << std::setw(42) << m.display_name
                  << " [" << std::setw(8) << m.quantization << "] "
                  << std::fixed << std::setprecision(2) << std::setw(6) << m.size_gb << " GB  "
                  << (m.is_moe ? "(MoE)" : "(Dense)") << "\n";
    }

    std::cout << "\n\033[1mDiscovered Speculative Drafters (" << cat.drafters.size() << "):\033[0m\n";
    for (const auto & d : cat.drafters) {
        std::cout << "  • " << std::left << std::setw(42) << d.filename
                  << " [" << std::setw(8) << d.quantization << "] "
                  << std::fixed << std::setprecision(2) << std::setw(6) << d.size_gb << " GB\n";
    }

    std::cout << "\n\033[38;2;34;197;94m✔ Model catalog written to " << ModelScanner::get_cache_path() << "\033[0m\n";
    return 0;
}

int Router::route(int argc, char ** argv) {
    // Case 1: Standalone Profiler flag
    if (argc >= 2 && (std::strcmp(argv[1], "--tune") == 0 || std::strcmp(argv[1], "--profile") == 0)) {
        return run_standalone_profiling(true);
    }

    // Case 2: Standalone Model Scanner flag
    if (argc >= 2 && std::strcmp(argv[1], "--scan-models") == 0) {
        std::vector<std::string> custom_paths;
        for (int i = 2; i < argc; ++i) {
            custom_paths.push_back(argv[i]);
        }
        return run_standalone_scan(custom_paths);
    }

    // Case 3: Help
    if (argc >= 2 && (std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0)) {
        print_help(argv[0]);
        return 0;
    }

    // Case 4: Manual CLI flags provided -> Bypass TUI directly
    if (is_manual_cli_invocation(argc, argv)) {
        // If user ran: sandwich run -m ... skip 'run' token
        if (argc >= 2 && std::strcmp(argv[1], "run") == 0) {
            return llama_cli(argc - 1, argv + 1);
        }
        return llama_cli(argc, argv);
    }

    // Case 5: Interactive TUI Wizard (default for argc == 1 or --tui)
    if (argc == 1 || (argc == 2 && std::strcmp(argv[1], "--tui") == 0)) {
        if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) {
            // Non-interactive terminal: print help
            print_help(argv[0]);
            return 0;
        }

        WizardConfig config;
        TUIWizard wizard;
        bool launched = wizard.run(config);

        if (launched && config.should_launch) {
            std::cout << "\n\033[1;38;2;34;197;94mLaunching Sandwich Engine with Auto-Dispatched Configuration...\033[0m\n";
            std::cout << "\033[38;2;160;160;160mCommand: " << config.synthesized_command << "\033[0m\n\n";

            // Parse synthesized command line into argc/argv
            std::vector<std::string> args;
            std::istringstream iss(config.synthesized_command);
            std::string token;
            while (iss >> token) {
                args.push_back(token);
            }

            std::vector<char*> c_args;
            c_args.push_back(argv[0]); // progname
            // Skip argv[0] from synthesized command (it starts with ./sandwich run)
            size_t start_idx = (args.size() > 2 && args[0] == "./sandwich" && args[1] == "run") ? 2 : 1;
            for (size_t i = start_idx; i < args.size(); ++i) {
                c_args.push_back(const_cast<char*>(args[i].c_str()));
            }

            return llama_cli(static_cast<int>(c_args.size()), c_args.data());
        }

        return 0;
    }

    // Fallback: print help
    print_help(argv[0]);
    return 1;
}

} // namespace dispatch
} // namespace sandwich
