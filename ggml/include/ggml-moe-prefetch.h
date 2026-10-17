#pragma once

#include "ggml.h"

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Mapping registry: records mmap-backed weight base addresses to verify eligibility
void ggml_moe_prefetch_register_mapping(const void * base, size_t size);
void ggml_moe_prefetch_unregister_mapping(const void * base);

// Thread pool management: n_threads > 0 spawns workers, <= 0 terminates pool
void ggml_moe_prefetch_set_n_threads(int n_threads);
bool ggml_moe_prefetch_enabled(void);

// Epoch counter for deduplication within a scheduler compute step
void ggml_moe_prefetch_new_epoch(void);

// Asynchronously prefetch selected expert weight slices for an MoE node
void ggml_moe_prefetch_node(const struct ggml_tensor * node);

// Asynchronously prefetch a specific expert slice (by memory address and size)
void ggml_moe_prefetch_slice(const void * addr, size_t size);

// Prefetch a whole tensor's mapped pages asynchronously (e.g. prompt lookahead)
void ggml_moe_prefetch_tensor(const struct ggml_tensor * w);

// Wait for all pending asynchronous prefetch tasks for tensor w to finish
void ggml_moe_prefetch_wait(const struct ggml_tensor * w);

// Advise Linux kernel with MADV_COLD on the pages of tensor w after prompt processing
void ggml_moe_prefetch_cold(const struct ggml_tensor * w);

// Early-Lookahead Router: register layer descriptors for speculative expert prefetching
void ggml_moe_prefetch_register_moe_layer(
    int32_t il,
    const struct ggml_tensor * gate_inp,
    const struct ggml_tensor * up_exps,
    const struct ggml_tensor * gate_exps,
    const struct ggml_tensor * down_exps,
    int64_t n_expert,
    int64_t n_expert_used
);

void ggml_moe_prefetch_clear_moe_layers(void);

// Triggers concurrent lookahead router evaluation and expert prefetching for Layer L+1
// while the GPU is executing Layer L
void ggml_moe_prefetch_lookahead_step(int32_t current_il, const int32_t * current_ids, size_t n_ids);

// Kernel-entry hook invoked when evaluating MoE operations
void ggml_moe_prefetch_kernel_hook(const struct ggml_tensor * node, int ith);

#ifdef __cplusplus
}
#endif
