#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    uint32_t n_outputs_max_per_seq;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    int32_t  nextn_layer_offset = 0;

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool split_kv;
    bool flash_attn_native_quants;
    bool sched_async_cpu;
    bool moe_hot_only;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool fused_lid;          // use fused lightning indexer
    bool auto_flid;
    bool fused_dsv4_hc_pre;
    bool fused_dsv4_hc_comb;
    bool fused_dsv4_hc_post;
    bool auto_fhc;
    bool moe_3tier;
    bool moe_2q;
    bool moe_gds;
    bool latent_mla;
    bool prefill_head_alias = true; // [Technique #55] Borrow output head VRAM for prefill buffers
    bool kva_prefill        = false;// [Technique #56] KVA Late-Layer Recurrence Prefill
    int32_t kva_start_layer = -1;   // [Technique #56] KVA starting layer
    bool dma_batch          = true; // [Technique #58] Batched DMA Engine
    bool moe_disjoint_guard = true; // [Technique #57] Disjoint Peer Cache Guard
    bool moe_exchange_rot   = true; // [Technique #59] Exchange Pointer Rotation
    bool msg_boundary_cache = true; // [Technique #60] Message Boundary Delta Caching
    bool dynamic_yarn       = true; // [Technique #64] Dynamic YaRN Auto-Frequency Scaling
    bool unrotated_kv       = true; // [Technique #66] Unrotated KV Staging (O(1) zero-copy sliding context shifts)
    bool dsv41_mode         = false;// DeepSeek V4.1 Flash native mode
    int32_t dsv41_experts   = 5;    // DeepSeek V4.1 routed experts per word
    bool no_perf;
    bool warmup;             // TODO: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;
    bool training;           // set by llama_opt_init()

    size_t moe_cache_size;
    size_t moe_warm_ram_bytes;
    float  moe_mass_thresh;

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer

    enum llama_context_type ctx_type;
    enum llama_rope_scaling_type rope_scaling_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    llama_context * ctx_other;
};
