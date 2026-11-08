set(TARGET sandwich-dispatch)

add_library(${TARGET} STATIC
    hardware_profiler.cpp
    model_scanner.cpp
    ram_estimator.cpp
    memory_watermark.cpp
    prefetch_scheduler.cpp
    ram_cache_layer.cpp
    h2o_cache_compressor.cpp
    paged_block_allocator.cpp
    tui_wizard.cpp
    router.cpp
)

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tools/cli
    ${PROJECT_SOURCE_DIR}/tools/server
    ${PROJECT_SOURCE_DIR}/common
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${TARGET} PUBLIC
    nlohmann
    llama-cli-impl
    llama-common
    llama
    ${CMAKE_THREAD_LIBS_INIT}
)

target_compile_features(${TARGET} PUBLIC cxx_std_17)

# Unified Sandwich binary (Smart-Dispatch TUI + CLI)
add_executable(sandwich main.cpp)
target_link_libraries(sandwich PRIVATE ${TARGET})
target_compile_features(sandwich PRIVATE cxx_std_17)

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS sandwich RUNTIME)
endif()
