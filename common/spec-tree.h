#pragma once

#include "llama.h"
#include "common.h"

#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <algorithm>
#include <cmath>

// ============================================================================
// Phase 7: EAGLE3 / P-EAGLE Speculative Tree Decoding Subsystem
// Architecture: Tree-Causal Attention Masking, Multi-Branch Drafting,
// Parallel Batched Verification, and Longest-Path KV Pruning.
// Strict Clean-Room Provenance: Pure mathematical implementation for sandwich.cpp
// ============================================================================

// A single candidate token node in the speculative tree
struct common_spec_tree_node {
    int32_t     index;       // 0-based index in the flattened 1D batch (0 .. N-1)
    llama_token token;       // Candidate token ID
    int32_t     parent;      // Parent node index in tree (-1 for direct children of root)
    int32_t     depth;       // Tree depth level (0 = direct child of root, 1, 2, ...)
    llama_pos   pos;         // Sequence position (pos0 + 1 + depth)
    float       prob;        // Draft probability / confidence
    std::vector<int32_t> children; // Child node indices in flattened tree
};

// Flattened speculative token tree topology
struct common_spec_tree {
    std::vector<common_spec_tree_node> nodes;
    int32_t max_depth = 3;    // Maximum tree depth (levels of speculation)
    int32_t max_width = 2;    // Maximum branching factor per node
    float   p_split   = 0.15f;// Confidence threshold to trigger a secondary branch
    float   p_min     = 0.05f;// Minimum confidence to admit a candidate node

    void clear() {
        nodes.clear();
    }

    size_t size() const {
        return nodes.size();
    }

    bool empty() const {
        return nodes.empty();
    }

    // Check if node 'ancestor' is an ancestor of 'descendant' (or the exact same node)
    bool is_ancestor(int32_t ancestor_idx, int32_t descendant_idx) const {
        if (ancestor_idx == descendant_idx) {
            return true;
        }
        int32_t curr = descendant_idx;
        while (curr >= 0 && curr < (int32_t)nodes.size()) {
            if (curr == ancestor_idx) {
                return true;
            }
            curr = nodes[curr].parent;
        }
        return false;
    }

    // Trace the path from root down to node_idx (inclusive)
    std::vector<int32_t> get_path(int32_t node_idx) const {
        std::vector<int32_t> path;
        int32_t curr = node_idx;
        while (curr >= 0 && curr < (int32_t)nodes.size()) {
            path.push_back(curr);
            curr = nodes[curr].parent;
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    // Return all leaf node indices in the tree
    std::vector<int32_t> get_leaves() const {
        std::vector<int32_t> leaves;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i].children.empty()) {
                leaves.push_back((int32_t)i);
            }
        }
        return leaves;
    }

    // Total number of leaves (equivalent to independent candidate paths)
    size_t n_paths() const {
        return get_leaves().size();
    }
};

// Result of target model tree verification
struct common_spec_tree_result {
    std::vector<llama_token> accepted_tokens;   // Accepted draft tokens along the best path
    llama_token              target_token = -1; // Target model's newly predicted token at end of path
    int32_t                  best_leaf_node = -1;// Tree node index of last accepted draft token (-1 if none)
    int32_t                  n_accepted = 0;    // Number of accepted draft tokens
    int32_t                  total_tokens_emitted = 1; // n_accepted + 1 (including novel target token)
    std::vector<int32_t>     accepted_node_indices; // Indices of accepted nodes in the tree
};

// Phase 7 Real-time Performance & Verification Telemetry
struct common_spec_tree_stats {
    uint64_t n_drafts_total   = 0; // Total tree draft tokens proposed
    uint64_t n_accepted_total = 0; // Total draft tokens accepted
    uint64_t n_forward_passes = 0; // Total target model verification forward passes
    uint64_t n_emitted_total  = 0; // Total tokens emitted to user

    double acceptance_rate() const {
        return n_drafts_total > 0 ? (double)n_accepted_total / (double)n_drafts_total : 0.0;
    }

    // Key Metric: Accepted Tokens Per Forward Pass (Target: > 2.5 on dense models)
    double tokens_per_pass() const {
        return n_forward_passes > 0 ? (double)n_emitted_total / (double)n_forward_passes : 1.0;
    }

    void reset() {
        n_drafts_total   = 0;
        n_accepted_total = 0;
        n_forward_passes = 0;
        n_emitted_total  = 0;
    }
};

// Phase 10 / Technique #31: Adaptive Spec-Tree PID Controller
struct common_spec_tree_pid_controller {
    float alpha_target   = 0.75f; // Target acceptance rate (75%)
    float kp             = 2.5f;  // Proportional gain
    float ki             = 0.4f;  // Integral gain
    float kd             = 0.8f;  // Derivative gain

    // Rolling exponential acceptance rate over window (~16 tokens)
    float alpha_rolling  = 0.75f;
    float integral_err   = 0.0f;
    float prev_err       = 0.0f;

    int32_t current_depth   = 3;     // Modulated depth in [1, 7]
    int32_t current_width   = 2;     // Tree width
    float   current_p_split = 0.15f; // Modulated split threshold in [0.05, 0.40]

    void update(int32_t n_accepted, int32_t n_drafted) {
        if (n_drafted <= 0) return;
        float current_rate = (float)n_accepted / (float)n_drafted;
        const float beta = 0.875f; // 16-token EWMA
        alpha_rolling = beta * alpha_rolling + (1.0f - beta) * current_rate;

        // Positive error indicates acceptance above target (exploit: deepen tree, lower p_split)
        // Negative error indicates acceptance below target (contract: shallow tree, higher p_split)
        float err = alpha_rolling - alpha_target;
        integral_err = std::clamp(integral_err + err, -3.0f, 3.0f);
        float derivative = err - prev_err;
        prev_err = err;

        float control_signal = kp * err + ki * integral_err + kd * derivative;

        // Base depth 3: modulate between [1, 7]
        int32_t target_depth = 3 + (int32_t)std::round(control_signal);
        current_depth = std::clamp(target_depth, 1, 7);

        // Modulate split threshold: contract width on low confidence, expand on high confidence
        float target_p_split = 0.15f - (control_signal * 0.05f);
        current_p_split = std::clamp(target_p_split, 0.05f, 0.40f);
    }

    void reset() {
        alpha_rolling   = alpha_target;
        integral_err    = 0.0f;
        prev_err        = 0.0f;
        current_depth   = 3;
        current_width   = 2;
        current_p_split = 0.15f;
    }
};

// Phase 10 / Technique #30: Pipelined Double-Buffered Speculative Execution Buffer
struct common_spec_pipe_buffer {
    common_spec_tree tree_buf[2];
    llama_tokens     primary_branch[2];
    int32_t          active_idx = 0;    // Currently verifying tree
    int32_t          staged_idx = 1;    // Pre-computed candidate tree
    bool             staged_valid = false;
    llama_token      staged_expected_root = -1;

    void swap() {
        std::swap(active_idx, staged_idx);
        staged_valid = false;
    }

    void invalidate() {
        staged_valid = false;
        staged_expected_root = -1;
    }

    common_spec_tree & get_active_tree() {
        return tree_buf[active_idx];
    }

    common_spec_tree & get_staged_tree() {
        return tree_buf[staged_idx];
    }
};

// ============================================================================
// Core Phase 7 Functional API
// ============================================================================

/**
 * Task 1: Tree-Causal Attention Mask Generation
 * Populates a 2D attention mask matrix of shape [n_kv, n_tree_tokens] such that:
 * - Each tree query token attends to all previous prompt/context tokens [0 .. n_prompt - 1]
 * - Each tree query token attends to the root token at index n_prompt
 * - Each tree query token attends ONLY to its ancestors and itself in [n_prompt + 1 .. n_prompt + n_tree]
 * - All other cells (sibling branches, future descendants) are set to mask_drop (-INFINITY)
 */
void common_spec_tree_generate_mask(
    const common_spec_tree & tree,
    float * mask_data,
    int64_t n_kv,
    int64_t n_prompt,
    float mask_keep = 0.0f,
    float mask_drop = -100000.0f);

void common_spec_tree_generate_mask_fp16(
    const common_spec_tree & tree,
    ggml_fp16_t * mask_data,
    int64_t n_kv,
    int64_t n_prompt,
    ggml_fp16_t mask_keep,
    ggml_fp16_t mask_drop);

/**
 * Task 2: Tree Drafting Logic (Drafter Side)
 * Expands candidate tree from the draft context level-by-level up to max_depth and max_width.
 * Employs entropy-aware branching: secondary branch only added if candidate prob >= p_split.
 */
common_spec_tree common_spec_tree_build(
    llama_context * ctx_dft,
    llama_token id_last,
    llama_pos pos0,
    int32_t max_depth,
    int32_t max_width,
    float p_split = 0.15f,
    float p_min   = 0.05f);

common_spec_tree common_spec_tree_build_from_candidates(
    const llama_tokens & draft_tokens,
    const std::vector<std::vector<llama_token_data>> & draft_q,
    llama_pos pos0,
    int32_t max_depth,
    int32_t max_width,
    float p_split = 0.15f);

/**
 * Task 3: Tree Verification & Acceptance (Target Side)
 * Walks tree branches against target model's output logits.
 * Identifies the longest accepted path and emits the novel target token.
 */
struct common_sampler;

common_spec_tree_result common_spec_tree_verify(
    const common_spec_tree & tree,
    llama_context * ctx_tgt,
    const std::vector<int32_t> & batch_indices, // index in target output logits for root + each tree node
    struct common_sampler * smpl = nullptr);

/**
 * Task 4: State Rollback & KV Cache Pruning
 * Prunes unselected/rejected branch cells from target and draft KV caches,
 * keeping the winning path aligned for the next generation turn.
 */
void common_spec_tree_prune_kv(
    llama_context * ctx_tgt,
    llama_context * ctx_dft,
    const common_spec_tree & tree,
    const common_spec_tree_result & result,
    llama_seq_id seq_id,
    llama_pos pos0);

/**
 * Visualizes the tree structure and highlighted accepted path in ANSI colors.
 */
void common_spec_tree_print(
    const llama_vocab * vocab,
    const common_spec_tree & tree,
    const common_spec_tree_result * result = nullptr);

inline void common_spec_tree_apply_to_context(llama_context * ctx, const common_spec_tree & tree, int64_t n_prompt) {
    if (tree.empty()) {
        llama_clear_spec_tree(ctx);
        return;
    }
    std::vector<int32_t> parents(tree.nodes.size());
    for (size_t i = 0; i < tree.nodes.size(); ++i) {
        parents[i] = tree.nodes[i].parent;
    }
    llama_set_spec_tree(ctx, parents.data(), (int64_t)parents.size(), n_prompt);
}
