add_library(ggml-iqk STATIC
    ggml-iqk.cpp
    ggml-iqk.h
    iqk-internal.h
    iqk-generic.cpp
    iqk-avx2.cpp
    iqk-avx512.cpp
)

target_include_directories(ggml-iqk PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_target_properties(ggml-iqk PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

if (MSVC)
    set_source_files_properties(iqk-avx2.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")
    set_source_files_properties(iqk-avx512.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX512")
else()
    include(CheckCXXCompilerFlag)

    # Check AVX2 flags
    check_cxx_compiler_flag("-mavx2 -mfma" COMPILER_SUPPORTS_AVX2)
    if (COMPILER_SUPPORTS_AVX2)
        check_cxx_compiler_flag("-mavx2 -mfma -mavxvnni" COMPILER_SUPPORTS_AVX_VNNI)
        if (COMPILER_SUPPORTS_AVX_VNNI)
            set_source_files_properties(iqk-avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -mavxvnni")
        else()
            set_source_files_properties(iqk-avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
        endif()
    endif()

    # Check AVX-512 flags
    check_cxx_compiler_flag("-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni" COMPILER_SUPPORTS_AVX512_VNNI)
    if (COMPILER_SUPPORTS_AVX512_VNNI)
        set_source_files_properties(iqk-avx512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni")
    else()
        check_cxx_compiler_flag("-mavx512f -mavx512bw -mavx512vl" COMPILER_SUPPORTS_AVX512)
        if (COMPILER_SUPPORTS_AVX512)
            set_source_files_properties(iqk-avx512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512bw -mavx512vl")
        endif()
    endif()
endif()
