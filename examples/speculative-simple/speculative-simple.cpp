#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "spec-tree.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>
#include <utility>

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    const auto output_limits = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max = output_limits.total;
    params.n_outputs_max_per_seq = output_limits.per_seq;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;

    llama_context * ctx_tgt = NULL;

    // load the target model
    auto llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // load the draft model (if any) - this also creates the MTP draft context when MTP speculation is enabled
    common_speculative_init_result_ptr spec_init;

    {
        common_params params_dft = common_base_params_to_speculative(params);

        spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);

        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = spec_init->context();
    }

    llama_context * ctx_dft = params.speculative.draft.ctx_dft;

    // check if the context supports partial sequence removal
    const bool use_ckpt_tgt = common_context_can_seq_rm(ctx_tgt) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    const bool use_ckpt_dft = common_context_can_seq_rm(ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

    if (use_ckpt_tgt) {
        LOG_INF("speculative decoding will use checkpoints (context does not support partial sequence removal)\n");
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;

    llama_seq_id seq_id = 0;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));

    // init the speculator
    const auto & params_spec = params.speculative;

    struct common_speculative * spec = common_speculative_init(params.speculative, 1);

    if (spec == nullptr) {
        LOG_ERR("%s", "failed to initialize speculative decoding\n");
        return 1;
    }

    // eval the prompt on the target and feed it to the speculative implementation(s)
    {
        common_batch batch_prompt(ctx_tgt);
        for (size_t i = 0; i < inp.size() - 1; ++i) {
            batch_prompt.add(inp[i], i, seq_id, false);
        }

        llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_prompt.get());

        if (!common_speculative_process(spec, batch_prompt)) {
            LOG_ERR("%s", "failed to process speculative prompt\n");
            return 1;
        }
    }

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = inp.size() - 1;

    common_speculative_begin(spec, seq_id, prompt_tgt);

    common_batch batch_tgt(ctx_tgt);

    llama_tokens draft;

    common_prompt_checkpoint ckpt;

    const auto t_enc_end = ggml_time_us();

    common_spec_tree_stats stats_tree;

    const auto t_dec_start = ggml_time_us();

    while (true) {
        if (params.spec_tree && ctx_dft != nullptr) {
            // Phase 7: EAGLE3 / P-EAGLE Speculative Tree Decoding
            int n_draft_max = (int) llama_n_ctx(ctx_tgt) - n_past - 2;
            if (params.n_predict >= 0) {
                n_draft_max = std::min(n_draft_max, params.n_predict - n_predict - 1);
            }
            if (n_draft_max <= 0) {
                batch_tgt.clear();
                batch_tgt.add(id_last, n_past++, seq_id, true);
                llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get());
                id_last = common_sampler_sample(smpl.get(), ctx_tgt, 0);
                common_sampler_accept(smpl.get(), id_last, true);
                prompt_tgt.push_back(id_last);
                n_predict++;
                const std::string token_str = common_token_to_piece(ctx_tgt, id_last);
                LOG("%s", token_str.c_str());
                if ((params.n_predict >= 0 && n_predict > params.n_predict) || llama_vocab_is_eog(vocab, id_last)) {
                    break;
                }
                continue;
            }

            // 1. Build candidate tree on drafter / speculator
            common_spec_tree tree;
            llama_tokens spec_draft;
            std::vector<std::vector<llama_token_data>> draft_q;

            if (spec != nullptr) {
                common_speculative_get_draft_params(spec, seq_id) = {
                    /* .drafting   = */ true,
                    /* .n_max      = */ std::min(n_draft_max, params.spec_tree_depth),
                    /* .pos0       = */ n_past,
                    /* .id_last    = */ id_last,
                    /* .prompt     = */ &prompt_tgt,
                    /* .result     = */ &spec_draft,
                    /* .result_q   = */ &draft_q,
                };
                common_speculative_draft(spec);
                if (!spec_draft.empty()) {
                    tree = common_spec_tree_build_from_candidates(
                        spec_draft, draft_q, n_past, params.spec_tree_depth, params.spec_tree_width, params.spec_tree_p_split);
                }

                if (params.early_exit_thresh > 0.0f && !draft_q.empty() && !draft_q[0].empty() && draft_q[0][0].p >= params.early_exit_thresh) {
                    // Phase 8: Technique #28 Boundary Early-Exit directly from VRAM
                    id_last = spec_draft[0];
                    common_sampler_accept(smpl.get(), id_last, true);
                    prompt_tgt.push_back(id_last);
                    n_predict++;
                    n_past++;
                    stats_tree.n_emitted_total++;
                    const std::string token_str = common_token_to_piece(ctx_tgt, id_last);
                    if (params.use_color) {
                        LOG("\u001b[36m%s\u001b[0m", token_str.c_str());
                    } else {
                        LOG("%s", token_str.c_str());
                    }
                    if ((params.n_predict >= 0 && n_predict > params.n_predict) || llama_vocab_is_eog(vocab, id_last)) {
                        break;
                    }
                    continue;
                }
            }

            if (tree.empty() && ctx_dft != nullptr) {
                tree = common_spec_tree_build(
                    ctx_dft, id_last, n_past, params.spec_tree_depth, params.spec_tree_width, params.spec_tree_p_split);
            }

            if (ctx_dft != nullptr) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, n_past, -1);
            }

            if (tree.empty()) {
                batch_tgt.clear();
                batch_tgt.add(id_last, n_past++, seq_id, true);
                llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get());
                id_last = common_sampler_sample(smpl.get(), ctx_tgt, 0);
                common_sampler_accept(smpl.get(), id_last, true);
                prompt_tgt.push_back(id_last);
                n_predict++;
                stats_tree.n_forward_passes++;
                stats_tree.n_emitted_total++;
                const std::string token_str = common_token_to_piece(ctx_tgt, id_last);
                LOG("%s", token_str.c_str());
                if ((params.n_predict >= 0 && n_predict > params.n_predict) || llama_vocab_is_eog(vocab, id_last)) {
                    break;
                }
                continue;
            }

            // 2. Batched target forward pass over root + all tree nodes with custom tree mask
            batch_tgt.clear();
            std::vector<int32_t> batch_indices;
            batch_indices.push_back(batch_tgt.add(id_last, n_past, seq_id, true));

            for (size_t i = 0; i < tree.size(); ++i) {
                batch_indices.push_back(batch_tgt.add(tree.nodes[i].token, tree.nodes[i].pos, seq_id, true));
            }

            // Task 1: Generate custom 2D Tree-Causal Attention Mask and apply to target context
            common_spec_tree_apply_to_context(ctx_tgt, tree, n_past);

            llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get());

            llama_clear_spec_tree(ctx_tgt);

            if (spec != nullptr) {
                common_speculative_process(spec, batch_tgt);
            }

            // 3. Tree Verification & Acceptance
            auto tree_res = common_spec_tree_verify(tree, ctx_tgt, batch_indices, smpl.get());

            // Telemetry
            stats_tree.n_drafts_total   += tree.size();
            stats_tree.n_accepted_total += tree_res.n_accepted;
            stats_tree.n_forward_passes += 1;
            stats_tree.n_emitted_total  += tree_res.total_tokens_emitted;

            // 4. Prune KV cache for rejected branches
            common_spec_tree_prune_kv(ctx_tgt, ctx_dft, tree, tree_res, seq_id, n_past);

            if (spec != nullptr) {
                common_speculative_accept(spec, seq_id, tree_res.n_accepted);
            }

            // 5. Emit accepted draft tokens and novel target token
            std::vector<llama_token> ids = tree_res.accepted_tokens;
            if (tree_res.target_token >= 0) {
                ids.push_back(tree_res.target_token);
            }

            for (size_t i = 0; i < ids.size(); ++i) {
                prompt_tgt.push_back(id_last);
                id_last = ids[i];
                if (llama_vocab_is_eog(vocab, id_last)) {
                    has_eos = true;
                    break;
                }
                const std::string token_str = common_token_to_piece(ctx_tgt, id_last);
                if (params.use_color && i < tree_res.accepted_tokens.size()) {
                    LOG("\u001b[32m%s\u001b[0m", token_str.c_str());
                } else {
                    LOG("%s", token_str.c_str());
                }
            }

            n_past    += tree_res.n_accepted + 1;
            n_drafted += (int)tree.size();
            n_accept  += tree_res.n_accepted;
            n_predict += tree_res.total_tokens_emitted;

            if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
                break;
            }
            continue;
        }

        // generate or reuse draft tokens
        //
        // this is the most important part of the speculation. the more probable tokens that are provided here
        // the better the performance will be. in theory, this computation can be performed asynchronously and even
        // offloaded to a remote device. it doesn't even have to be based on an LLM. instead, it can provide tokens
        // from a cache or lookup tables.
        //
        if (draft.empty()) {
            ckpt.update_pos(
                    prompt_tgt.size(),
                    llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id));

            if (use_ckpt_dft) {
                ckpt.update_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            // determine the max draft that fits the remaining context and generation budget
            int n_draft_max = (int) llama_n_ctx(ctx_tgt) - n_past - 2;
            if (params.n_predict >= 0) {
                n_draft_max = std::min(n_draft_max, params.n_predict - n_predict - 1);
            }
            n_draft_max = std::max(n_draft_max, 0);

            // generate a new draft
            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting   = */ true,
                /* .n_max      = */ n_draft_max,
                /* .pos0       = */ n_past,
                /* .id_last    = */ id_last,
                /* .prompt     = */ &prompt_tgt,
                /* .result     = */ &draft, // output
            };
            common_speculative_draft(spec);

            // save a checkpoint of the target context before evaluating the draft
            // this allows us to restore the state if partial draft acceptance occurs
            if (!draft.empty()) {
                if (use_ckpt_tgt) {
                    ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }

            // reset the draft context to the checkpoint before verification
            if (ctx_dft) {
                if (use_ckpt_dft) {
                    GGML_ASSERT(ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                }

                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }
        } else {
            // we have a previous (partial) draft to reuse from checkpoint restoration
            if (use_ckpt_tgt) {
                GGML_ASSERT(!ckpt.empty());
            }
        }

        // always have a token to evaluate from before - id_last
        batch_tgt.clear();
        batch_tgt.add(id_last, n_past++, seq_id, true);

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        {
            for (size_t i = 0; i < draft.size(); ++i) {
                batch_tgt.add(draft[i], n_past + i, seq_id, true);
            }


            llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get());
        }

        // feed the batch to the speculative implementation(s) - this drives the draft model, MTP, Eagle3, etc.
        if (!common_speculative_process(spec, batch_tgt)) {
            LOG_ERR("%s", "failed to process speculative batch\n");
            break;
        }

        // only save the sampler sampler state if we use checkpoints
        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }

        // save the size of the draft being verified
        const size_t n_draft = draft.size();

        // sample from the full target batch and return the accepted tokens based on the target sampler
        //
        // for each token to be accepted, the sampler would have to sample that same token
        // in such cases, instead of decoding the sampled token as we normally do, we simply continue with the
        // available logits from the batch and sample the next token until we run out of logits or the sampler
        // disagrees with the draft
        //
        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);

        //LOG_DBG("ids: %s\n", string_from(ctx_tgt, ids).c_str());

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        // check for partial draft acceptance:
        // if the context doesn't support partial sequence removal, restore the checkpoint
        // and make the accepted tokens the new partial draft for the next iteration
        if (use_ckpt_tgt && ids.size() - 1 < n_draft) {
            LOG_DBG("partial acceptance: %zu < %zu, restoring checkpoint\n", ids.size() - 1, n_draft);

            draft = std::move(ids);

            {
                GGML_ASSERT(ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));

                llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, ckpt.pos_max + 1, -1);
            }

            if (ctx_dft) {
                GGML_ASSERT(ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));

                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }

            prompt_tgt.resize(ckpt.n_tokens);
            smpl = std::move(smpl_save);

            n_past = (int) prompt_tgt.size();

            continue;
        }

        common_speculative_accept(spec, seq_id, ids.size() - 1);

        // full acceptance: consume the draft and commit accepted tokens
        n_past    += ids.size() - 1;
        n_drafted += n_draft; // note: we ignore the discarded small drafts
        n_accept  += ids.size() - 1;
        n_predict += ids.size();

        // process the accepted tokens and update contexts
        //
        // this is the standard token post-processing that we normally do
        // in this case, we do it for a group of accepted tokens at once
        //
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);

            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        // clear the draft since it has been consumed
        draft.clear();

        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, n_past, -1);

            if (ctx_dft) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, n_past, -1);
            }
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", params_spec.draft.n_max);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", n_drafted > 0 ? 100.0f * n_accept / n_drafted : 0.0f);

    if (params.spec_tree) {
        LOG_INF("\n");
        LOG_INF("=================================================================\n");
        LOG_INF("🥪 Phase 7: EAGLE3 / P-EAGLE Speculative Tree Decoding Telemetry\n");
        LOG_INF("=================================================================\n");
        LOG_INF("  Tree Topology:              depth=%d, width=%d\n", params.spec_tree_depth, params.spec_tree_width);
        LOG_INF("  Target Forward Passes:      %" PRIu64 "\n", stats_tree.n_forward_passes);
        LOG_INF("  Draft Candidates Proposed:  %" PRIu64 "\n", stats_tree.n_drafts_total);
        LOG_INF("  Draft Candidates Accepted:  %" PRIu64 "\n", stats_tree.n_accepted_total);
        LOG_INF("  Total Tokens Emitted:       %" PRIu64 "\n", stats_tree.n_emitted_total);
        LOG_INF("  Mean Accepted Length / Pass:\033[1;32m %6.2f tokens/pass\033[0m\n", stats_tree.tokens_per_pass());
        LOG_INF("  Tree Draft Acceptance Rate: \033[1;36m %6.2f%%\033[0m\n", 100.0 * stats_tree.acceptance_rate());
        LOG_INF("=================================================================\n");
    }

    LOG_INF("\n");
    LOG_INF("draft:\n\n");
    common_speculative_print_stats(spec);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl.get());


    common_speculative_free(spec);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
