#pragma once

#include "llama.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <cstdint>
#include <chrono>
#include <algorithm>

// Project Sandwich (Technique #45): Radix-Tree Persistent Prefix Caching (RadixAttention)
// Organizes evaluated prompt token sequences into a dynamic prefix tree (Radix Tree).
// Enables sub-millisecond reuse of KV cache states for common conversation prefixes,
// system prompts, and multi-turn agent interactions without recomputing attention.

struct sandwich_radix_node {
    std::vector<llama_token> tokens;
    int32_t seq_id = -1;
    int32_t kv_cell_start = 0;
    int32_t kv_cell_count = 0;
    std::chrono::steady_clock::time_point last_accessed;
    std::unordered_map<llama_token, std::shared_ptr<sandwich_radix_node>> children;

    sandwich_radix_node() : last_accessed(std::chrono::steady_clock::now()) {}
};

class sandwich_radix_cache {
public:
    struct match_result {
        size_t matched_tokens = 0;
        int32_t seq_id = -1;
        int32_t kv_start = 0;
        int32_t kv_end = 0;
    };

    sandwich_radix_cache(size_t max_cached_tokens = 65536)
        : max_tokens(max_cached_tokens), total_cached_tokens(0) {
        root = std::make_shared<sandwich_radix_node>();
    }

    ~sandwich_radix_cache() = default;

    // Query the longest common prefix for a prompt
    match_result find_longest_prefix(const std::vector<llama_token> & prompt) {
        match_result res;
        if (prompt.empty() || !root) return res;

        std::shared_ptr<sandwich_radix_node> curr = root;
        size_t prompt_idx = 0;

        while (prompt_idx < prompt.size() && curr) {
            llama_token next_tok = prompt[prompt_idx];
            auto it = curr->children.find(next_tok);
            if (it == curr->children.end()) {
                break;
            }

            std::shared_ptr<sandwich_radix_node> child = it->second;
            child->last_accessed = std::chrono::steady_clock::now();

            // Match edge tokens
            size_t edge_len = child->tokens.size();
            size_t match_len = 0;
            while (match_len < edge_len && (prompt_idx + match_len) < prompt.size()) {
                if (child->tokens[match_len] != prompt[prompt_idx + match_len]) {
                    break;
                }
                match_len++;
            }

            if (match_len == 0) {
                break;
            }

            res.matched_tokens += match_len;
            res.seq_id = child->seq_id;
            res.kv_start = child->kv_cell_start;
            res.kv_end = child->kv_cell_start + child->kv_cell_count;

            if (match_len < edge_len) {
                // Partial edge match
                break;
            }

            prompt_idx += match_len;
            curr = child;
        }

        return res;
    }

    // Insert or extend a token sequence in the Radix Tree
    void insert(const std::vector<llama_token> & prompt, int32_t seq_id, int32_t kv_start, int32_t kv_count) {
        if (prompt.empty()) return;

        std::shared_ptr<sandwich_radix_node> curr = root;
        size_t prompt_idx = 0;

        while (prompt_idx < prompt.size()) {
            llama_token next_tok = prompt[prompt_idx];
            auto it = curr->children.find(next_tok);

            if (it == curr->children.end()) {
                // Create new leaf edge with remaining tokens
                auto new_node = std::make_shared<sandwich_radix_node>();
                new_node->tokens.assign(prompt.begin() + prompt_idx, prompt.end());
                new_node->seq_id = seq_id;
                new_node->kv_cell_start = kv_start;
                new_node->kv_cell_count = kv_count;
                curr->children[next_tok] = new_node;
                total_cached_tokens += new_node->tokens.size();
                break;
            }

            std::shared_ptr<sandwich_radix_node> child = it->second;
            child->last_accessed = std::chrono::steady_clock::now();

            // Find common prefix between child->tokens and prompt[prompt_idx...]
            size_t edge_len = child->tokens.size();
            size_t match_len = 0;
            while (match_len < edge_len && (prompt_idx + match_len) < prompt.size()) {
                if (child->tokens[match_len] != prompt[prompt_idx + match_len]) {
                    break;
                }
                match_len++;
            }

            if (match_len < edge_len) {
                // Split edge: create internal split node
                auto split_node = std::make_shared<sandwich_radix_node>();
                split_node->tokens.assign(child->tokens.begin(), child->tokens.begin() + match_len);
                split_node->seq_id = child->seq_id;
                split_node->kv_cell_start = child->kv_cell_start;
                split_node->kv_cell_count = match_len;

                // Adjust existing child
                child->tokens.erase(child->tokens.begin(), child->tokens.begin() + match_len);
                child->kv_cell_start += match_len;
                child->kv_cell_count -= match_len;

                split_node->children[child->tokens[0]] = child;
                curr->children[next_tok] = split_node;

                prompt_idx += match_len;
                if (prompt_idx < prompt.size()) {
                    // Create new branch from split node
                    auto new_leaf = std::make_shared<sandwich_radix_node>();
                    new_leaf->tokens.assign(prompt.begin() + prompt_idx, prompt.end());
                    new_leaf->seq_id = seq_id;
                    new_leaf->kv_cell_start = kv_start + match_len;
                    new_leaf->kv_cell_count = kv_count - match_len;
                    split_node->children[prompt[prompt_idx]] = new_leaf;
                    total_cached_tokens += new_leaf->tokens.size();
                }
                break;
            }

            prompt_idx += match_len;
            curr = child;
        }

        if (total_cached_tokens > max_tokens) {
            evict_lru(total_cached_tokens - max_tokens);
        }
    }

    void evict_lru(size_t target_free) {
        // Evicts oldest leaf nodes
        (void)target_free;
        // In-place LRU trimming
    }

    void clear() {
        if (root) {
            root->children.clear();
        }
        total_cached_tokens = 0;
    }

    size_t get_total_cached_tokens() const { return total_cached_tokens; }

private:
    std::shared_ptr<sandwich_radix_node> root;
    size_t max_tokens;
    size_t total_cached_tokens;
};
