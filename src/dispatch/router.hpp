#pragma once

#include <string>
#include <vector>

namespace sandwich {
namespace dispatch {

class Router {
public:
    // Main routing entry point for sandwich binary
    static int route(int argc, char ** argv);

    // Prints CLI help and command overview
    static void print_help(const char * progname);

    // Runs standalone hardware profiling and prints diagnostic JSON
    static int run_standalone_profiling(bool force);

    // Runs standalone model scan and prints discovery table
    static int run_standalone_scan(const std::vector<std::string> & paths);

private:
    static bool is_manual_cli_invocation(int argc, char ** argv);
};

} // namespace dispatch
} // namespace sandwich
