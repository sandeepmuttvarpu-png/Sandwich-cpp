#include "router.hpp"

int main(int argc, char ** argv) {
    return sandwich::dispatch::Router::route(argc, argv);
}
