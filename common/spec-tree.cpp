#include "spec-tree.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <queue>
#include <functional>

// ============================================================================
// Phase 7: EAGLE3 / P-EAGLE Speculative Tree Decoding Implementation
// ============================================================================

void common_spec_tree_generate_mask(
    const common_spec_tree & tree,
    float * mask_data,
    int64_t n_kv,
    int64_t n_prompt,
    float mask_keep,
    float mask_drop)
{
    const int64_t n_tree = (int64_t)tree.size();
    const int64_t total_batch = n_tree + 1; // 1 root token + n_tree tokens

    for (int64_t q = 0; q < total_batch; ++q) {
        float * row = mask_data + q * n_kv;

        // Step 1: Prompt tokens [0 .. n_prompt - 1] are visible to all tokens in the batch
        for (int64_t j = 0; j < n_prompt; ++j) {
            row[j] = mask_keep;
        }

        // Step 2: Root token (index n_prompt in the KV cache)
        row[n_prompt] = mask_keep;

        // Step 3: Speculative tree tokens [n_prompt + 1 .. n_prompt + n_tree]
        if (q == 0) {
            // Root token cannot attend forward to any speculative tree tokens
            for (int64_t k = 0; k < n_tree; ++k) {
                row[n_prompt + 1 + k] = mask_drop;
            }
        } else {
            // Query is tree node i = q - 1
            const int32_t node_i = (int32_t)(q - 1);

            for (int64_t k = 0; k < n_tree; ++k) {
                const int32_t node_k = (int32_t)k;

                // Node i attends to node k iff k is an ancestor of i (or k == i)
                if (tree.is_ancestor(node_k, node_i)) {
                    row[n_prompt + 1 + k] = mask_keep;
                } else {
                    row[n_prompt + 1 + k] = mask_drop;
                }
            }
        }

        // Step 4: Unallocated future positions in KV cache receive mask_drop
        for (int64_t j = n_prompt + total_batch; j < n_kv; ++j) {
            row[j] = mask_drop;
        }
    }
}

void common_spec_tree_generate_mask_fp16(
    const common_spec_tree & tree,
    ggml_fp16_t * mask_data,
    int64_t n_kv,
    int64_t n_prompt,
    ggml_fp16_t mask_keep,
    ggml_fp16_t mask_drop)
{
    const int64_t n_tree = (int64_t)tree.size();
    const int64_t total_batch = n_tree + 1;

    for (int64_t q = 0; q < total_batch; ++q) {
        ggml_fp16_t * row = mask_data + q * n_kv;

        for (int64_t j = 0; j < n_prompt; ++j) {
            row[j] = mask_keep;
        }

        row[n_prompt] = mask_keep;

        if (q == 0) {
            for (int64_t k = 0; k < n_tree; ++k) {
                row[n_prompt + 1 + k] = mask_drop;
            }
        } else {
            const int32_t node_i = (int32_t)(q - 1);

            for (int64_t k = 0; k < n_tree; ++k) {
                const int32_t node_k = (int32_t)k;
                if (tree.is_ancestor(node_k, node_i)) {
                    row[n_prompt + 1 + k] = mask_keep;
                } else {
                    row[n_prompt + 1 + k] = mask_drop;
                }
            }
        }

        for (int64_t j = n_prompt + total_batch; j < n_kv; ++j) {
            row[j] = mask_drop;
        }
    }
}

common_spec_tree common_spec_tree_build(
    llama_context * ctx_dft,
    llama_token id_last,
    llama_pos pos0,
    int32_t max_depth,
    int32_t max_width,
    float p_split,
    float p_min)
{
    common_spec_tree tree;
    tree.max_depth = max_depth;
    tree.max_width = max_width;
    tree.p_split   = p_split;
    tree.p_min     = p_min;

    if (ctx_dft == nullptr || max_depth <= 0) {
        return tree;
    }

    const llama_model * model_dft = llama_get_model(ctx_dft);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab_dft);

    // Structure for sorting candidate logits
    struct candidate {
        llama_token id;
        float prob;
    };

    auto extract_top_candidates = [&](int32_t batch_idx, int32_t k_limit) -> std::vector<candidate> {
        const float * logits = llama_get_logits_ith(ctx_dft, batch_idx);
        if (!logits) return {};

        // Find max logit for numerical stability
        float max_l = logits[0];
        for (int32_t v = 1; v < n_vocab; ++v) {
            if (logits[v] > max_l) max_l = logits[v];
        }

        // Softmax over top candidates
        std::vector<candidate> cands;
        cands.reserve(32);

        // First pass: select top 32 by logit
        for (int32_t v = 0; v < n_vocab; ++v) {
            if (logits[v] - max_l > -10.0f) { // filter out negligible candidates
                cands.push_back({ v, logits[v] });
            }
        }

        std::sort(cands.begin(), cands.end(), [](const candidate & a, const candidate & b) {
            return a.prob > b.prob;
        });

        if ((int32_t)cands.size() > 32) {
            cands.resize(32);
        }

        float sum_exp = 0.0f;
        for (auto & c : cands) {
            c.prob = std::exp(c.prob - max_l);
            sum_exp += c.prob;
        }

        if (sum_exp > 0.0f) {
            for (auto & c : cands) {
                c.prob /= sum_exp;
            }
        }

        if ((int32_t)cands.size() > k_limit) {
            cands.resize(k_limit);
        }

        return cands;
    };

    common_batch batch(ctx_dft);

    // Level 0: Direct children of root token
    batch.clear();
    batch.add(id_last, pos0, 0, true);
    if (llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        return tree;
    }

    auto top_cands = extract_top_candidates(0, max_width);
    if (top_cands.empty() || top_cands[0].prob < p_min) {
        return tree;
    }

    // Always admit the primary candidate
    int32_t idx0 = (int32_t)tree.nodes.size();
    tree.nodes.push_back({ idx0, top_cands[0].id, -1, 0, pos0 + 1, top_cands[0].prob, {} });

    // Entropy-directed branching: secondary branch admitted if prob >= p_split
    if (max_width >= 2 && top_cands.size() >= 2 && top_cands[1].prob >= p_split) {
        int32_t idx1 = (int32_t)tree.nodes.size();
        tree.nodes.push_back({ idx1, top_cands[1].id, -1, 0, pos0 + 1, top_cands[1].prob, {} });
    }

    // Levels 1 .. max_depth - 1
    for (int32_t d = 1; d < max_depth; ++d) {
        std::vector<int32_t> parent_indices;
        for (size_t i = 0; i < tree.nodes.size(); ++i) {
            if (tree.nodes[i].depth == d - 1) {
                parent_indices.push_back((int32_t)i);
            }
        }

        if (parent_indices.empty()) {
            break;
        }

        // Batch evaluate all parents in a single pass on drafter
        batch.clear();
        for (int32_t p : parent_indices) {
            batch.add(tree.nodes[p].token, tree.nodes[p].pos, 0, true);
        }

        if (llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            break;
        }

        for (size_t b = 0; b < parent_indices.size(); ++b) {
            int32_t p = parent_indices[b];
            auto cands = extract_top_candidates((int32_t)b, max_width);

            if (cands.empty() || cands[0].prob < p_min) {
                continue;
            }

            int32_t child0 = (int32_t)tree.nodes.size();
            tree.nodes[p].children.push_back(child0);
            tree.nodes.push_back({ child0, cands[0].id, p, d, pos0 + 1 + d, cands[0].prob, {} });

            if (max_width >= 2 && cands.size() >= 2 && cands[1].prob >= p_split) {
                int32_t child1 = (int32_t)tree.nodes.size();
                tree.nodes[p].children.push_back(child1);
                tree.nodes.push_back({ child1, cands[1].id, p, d, pos0 + 1 + d, cands[1].prob, {} });
            }
        }
    }

    return tree;
}

common_spec_tree common_spec_tree_build_from_candidates(
    const llama_tokens & draft_tokens,
    const std::vector<std::vector<llama_token_data>> & draft_q,
    llama_pos pos0,
    int32_t max_depth,
    int32_t max_width,
    float p_split)
{
    common_spec_tree tree;
    tree.max_depth = max_depth;
    tree.max_width = max_width;
    tree.p_split   = p_split;

    if (draft_tokens.empty() || max_depth <= 0) {
        return tree;
    }

    const int32_t effective_depth = std::min((int32_t)draft_tokens.size(), max_depth);

    // Level 0: Direct children of root
    int32_t idx0 = (int32_t)tree.nodes.size();
    float p0 = 1.0f;
    if (!draft_q.empty() && !draft_q[0].empty()) {
        p0 = draft_q[0][0].p;
    }
    tree.nodes.push_back({ idx0, draft_tokens[0], -1, 0, pos0 + 1, p0, {} });

    // Secondary branch if available and above split threshold
    if (max_width >= 2 && !draft_q.empty() && draft_q[0].size() >= 2) {
        const auto & c1 = draft_q[0][1];
        if (c1.p >= p_split && c1.id != draft_tokens[0]) {
            int32_t idx1 = (int32_t)tree.nodes.size();
            tree.nodes.push_back({ idx1, c1.id, -1, 0, pos0 + 1, c1.p, {} });
        }
    }

    // Expand subsequent depth levels
    for (int32_t d = 1; d < effective_depth; ++d) {
        std::vector<int32_t> parents;
        for (size_t i = 0; i < tree.nodes.size(); ++i) {
            if (tree.nodes[i].depth == d - 1) {
                parents.push_back((int32_t)i);
            }
        }

        if (parents.empty()) {
            break;
        }

        llama_token cand_primary = draft_tokens[d];
        float cand_primary_p = 1.0f;
        llama_token cand_secondary = -1;
        float cand_secondary_p = 0.0f;

        if ((size_t)d < draft_q.size() && !draft_q[d].empty()) {
            cand_primary_p = draft_q[d][0].p;
            if (draft_q[d].size() >= 2 && max_width >= 2) {
                cand_secondary = draft_q[d][1].id;
                cand_secondary_p = draft_q[d][1].p;
            }
        }

        for (int32_t p : parents) {
            int32_t child0 = (int32_t)tree.nodes.size();
            tree.nodes[p].children.push_back(child0);
            tree.nodes.push_back({ child0, cand_primary, p, d, pos0 + 1 + d, cand_primary_p, {} });

            if (cand_secondary >= 0 && cand_secondary_p >= p_split && cand_secondary != cand_primary) {
                int32_t child1 = (int32_t)tree.nodes.size();
                tree.nodes[p].children.push_back(child1);
                tree.nodes.push_back({ child1, cand_secondary, p, d, pos0 + 1 + d, cand_secondary_p, {} });
            }
        }
    }

    return tree;
}

common_spec_tree_result common_spec_tree_verify(
    const common_spec_tree & tree,
    llama_context * ctx_tgt,
    const std::vector<int32_t> & batch_indices,
    struct common_sampler * smpl)
{
    common_spec_tree_result result;
    if (tree.empty() || batch_indices.empty()) {
        return result;
    }

    const llama_model * model_tgt = llama_get_model(ctx_tgt);
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab_tgt);

    auto sample_at = [&](int32_t batch_idx) -> llama_token {
        if (smpl != nullptr) {
            return common_sampler_sample(smpl, ctx_tgt, batch_idx);
        }
        const float * logits = llama_get_logits_ith(ctx_tgt, batch_idx);
        if (!logits) return 0;
        int32_t best_id = 0;
        float max_l = logits[0];
        for (int32_t v = 1; v < n_vocab; ++v) {
            if (logits[v] > max_l) {
                max_l = logits[v];
                best_id = v;
            }
        }
        return best_id;
    };

    auto finalize_result = [&](common_spec_tree_result & res) -> common_spec_tree_result {
        if (smpl != nullptr) {
            for (llama_token tok : res.accepted_tokens) {
                common_sampler_accept(smpl, tok, true);
            }
            if (res.target_token >= 0) {
                common_sampler_accept(smpl, res.target_token, true);
            }
        }
        return res;
    };

    // 1. Root token prediction from batch index 0
    llama_token tgt_token = sample_at(batch_indices[0]);

    // 2. Search Level 0 nodes (direct children of root with parent == -1)
    int32_t curr_accepted_node = -1;
    for (const auto & node : tree.nodes) {
        if (node.parent == -1 && node.token == tgt_token) {
            curr_accepted_node = node.index;
            break;
        }
    }

    if (curr_accepted_node == -1) {
        // Zero draft tokens accepted; target token is novel token
        result.target_token = tgt_token;
        result.n_accepted = 0;
        result.total_tokens_emitted = 1;
        return finalize_result(result);
    }

    // Level 0 matched!
    result.accepted_tokens.push_back(tree.nodes[curr_accepted_node].token);
    result.accepted_node_indices.push_back(curr_accepted_node);

    // EOG termination check
    if (llama_vocab_is_eog(vocab_tgt, tree.nodes[curr_accepted_node].token)) {
        result.best_leaf_node = curr_accepted_node;
        result.n_accepted = (int32_t)result.accepted_tokens.size();
        result.total_tokens_emitted = result.n_accepted;
        return finalize_result(result);
    }

    // 3. Walk child branches
    while (!tree.nodes[curr_accepted_node].children.empty()) {
        int32_t batch_idx = batch_indices[curr_accepted_node + 1];
        llama_token next_tgt = sample_at(batch_idx);

        int32_t matched_child = -1;
        for (int32_t child_idx : tree.nodes[curr_accepted_node].children) {
            if (tree.nodes[child_idx].token == next_tgt) {
                matched_child = child_idx;
                break;
            }
        }

        if (matched_child == -1) {
            // Child rejected: stop at curr_accepted_node, next_tgt is the novel target token
            result.target_token = next_tgt;
            result.best_leaf_node = curr_accepted_node;
            result.n_accepted = (int32_t)result.accepted_tokens.size();
            result.total_tokens_emitted = result.n_accepted + 1;
            return finalize_result(result);
        }

        // Child accepted!
        curr_accepted_node = matched_child;
        result.accepted_tokens.push_back(tree.nodes[curr_accepted_node].token);
        result.accepted_node_indices.push_back(curr_accepted_node);

        if (llama_vocab_is_eog(vocab_tgt, tree.nodes[curr_accepted_node].token)) {
            result.best_leaf_node = curr_accepted_node;
            result.n_accepted = (int32_t)result.accepted_tokens.size();
            result.total_tokens_emitted = result.n_accepted;
            return finalize_result(result);
        }
    }

    // 4. Reached leaf of the tree: sample novel target token from this leaf's output logits
    int32_t leaf_batch_idx = batch_indices[curr_accepted_node + 1];
    result.target_token = sample_at(leaf_batch_idx);
    result.best_leaf_node = curr_accepted_node;
    result.n_accepted = (int32_t)result.accepted_tokens.size();
    result.total_tokens_emitted = result.n_accepted + 1;

    return finalize_result(result);
}

void common_spec_tree_prune_kv(
    llama_context * ctx_tgt,
    llama_context * ctx_dft,
    const common_spec_tree & tree,
    const common_spec_tree_result & result,
    llama_seq_id seq_id,
    llama_pos pos0)
{
    GGML_UNUSED(tree);
    if (ctx_tgt == nullptr) return;

    // The winning path contains result.n_accepted tokens at positions pos0+1 .. pos0+result.n_accepted.
    // Discard any speculative tokens at positions > pos0 + result.n_accepted.
    const llama_pos prune_pos = pos0 + result.n_accepted + 1;

    llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, prune_pos, -1);

    if (ctx_dft != nullptr) {
        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, prune_pos, -1);
    }
}

void common_spec_tree_print(
    const llama_vocab * vocab,
    const common_spec_tree & tree,
    const common_spec_tree_result * result)
{
    if (tree.empty()) {
        printf("  [Tree Speculation: Empty Tree]\n");
        return;
    }

    printf("  \033[1;36m[EAGLE3 Speculative Tree Topology (nodes=%zu, depth=%d, width=%d)]:\033[0m\n",
           tree.size(), tree.max_depth, tree.max_width);

    // Build map of accepted nodes for highlighting
    std::vector<bool> is_accepted(tree.size(), false);
    if (result != nullptr) {
        for (int32_t idx : result->accepted_node_indices) {
            if (idx >= 0 && idx < (int32_t)tree.size()) {
                is_accepted[idx] = true;
            }
        }
    }

    // Depth-first traversal to print tree lines
    std::function<void(int32_t, const std::string &, bool)> print_node =
        [&](int32_t node_idx, const std::string & prefix, bool is_last) {
            const auto & node = tree.nodes[node_idx];
            std::string piece = llama_vocab_get_text(vocab, node.token);

            // Clean printable display
            for (char & c : piece) {
                if (c == '\n') c = ' ';
            }

            const char * color = is_accepted[node_idx] ? "\033[1;32m" : "\033[0;37m";
            const char * mark  = is_accepted[node_idx] ? " \033[1;32m<-- ACCEPTED\033[0m" : "";

            printf("%s%s%s (%d) token %d ('%s') [p=%.2f]%s\033[0m\n",
                   color,
                   prefix.c_str(),
                   is_last ? "└─" : "├─",
                   node.index,
                   node.token,
                   piece.c_str(),
                   node.prob,
                   mark);

            std::string child_prefix = prefix + (is_last ? "    " : "│   ");
            for (size_t c = 0; c < node.children.size(); ++c) {
                bool child_is_last = (c == node.children.size() - 1);
                print_node(node.children[c], child_prefix, child_is_last);
            }
        };

    // Print all Level 0 root children
    std::vector<int32_t> root_children;
    for (size_t i = 0; i < tree.nodes.size(); ++i) {
        if (tree.nodes[i].parent == -1) {
            root_children.push_back((int32_t)i);
        }
    }

    for (size_t r = 0; r < root_children.size(); ++r) {
        bool is_last = (r == root_children.size() - 1);
        print_node(root_children[r], "    ", is_last);
    }

    if (result != nullptr) {
        printf("  \033[1;33mAccepted: %d draft tokens + 1 target token = %d tokens emitted (%.2f tokens/pass)\033[0m\n",
               result->n_accepted,
               result->total_tokens_emitted,
               (double)result->total_tokens_emitted);
    }
}
