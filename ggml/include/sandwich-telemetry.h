#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t hot_hits;
    uint64_t cold_misses;
    uint64_t total_evals;
    double   hit_rate_pct;
} sandwich_telemetry_stats_t;

void sandwich_telemetry_record_hits(uint64_t hits);
void sandwich_telemetry_record_misses(uint64_t misses);
void sandwich_telemetry_record_expert(int32_t il, int32_t expert_id);

sandwich_telemetry_stats_t sandwich_telemetry_get_stats(void);
void sandwich_telemetry_reset(void);
void sandwich_telemetry_print(void);

// Online warm-up trace inspection
int64_t sandwich_telemetry_get_expert_freq(int32_t il, int32_t expert_id);
uint64_t sandwich_telemetry_get_total_warmup_tokens(void);
void sandwich_telemetry_increment_warmup_tokens(void);

#ifdef __cplusplus
}
#endif
