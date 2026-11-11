#pragma once

#include <string>
#include <cstdint>
#include <mutex>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <chrono>
#include <algorithm>

namespace sandwich {

struct TokenOdometerData {
    uint64_t lifetime_tokens = 0;
    uint64_t lifetime_prompt_tokens = 0;
    uint64_t lifetime_compute_ms = 0;
    uint64_t total_sessions = 0;
    uint64_t session_tokens = 0;
    uint64_t session_prompt_tokens = 0;
    uint64_t session_compute_ms = 0;
    double   cost_saved_usd = 0.0;
    std::string milestone_badge = "Rookie Pilot";
};

class TokenOdometer {
public:
    static TokenOdometer & instance() {
        static TokenOdometer inst;
        return inst;
    }

    void record_completion(uint64_t prompt_tokens, uint64_t gen_tokens, uint64_t compute_us) {
        std::lock_guard<std::mutex> lock(mtx_);
        session_tokens_ += gen_tokens;
        session_prompt_tokens_ += prompt_tokens;
        session_compute_ms_ += compute_us / 1000;

        lifetime_tokens_ += gen_tokens;
        lifetime_prompt_tokens_ += prompt_tokens;
        lifetime_compute_ms_ += compute_us / 1000;

        uint64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        if (session_tokens_ - last_saved_tokens_ >= 50 || (now_ms - last_saved_time_ms_) > 10000) {
            save_locked();
            last_saved_tokens_ = session_tokens_;
            last_saved_time_ms_ = now_ms;
        }
    }

    TokenOdometerData get_data() {
        std::lock_guard<std::mutex> lock(mtx_);
        TokenOdometerData d;
        d.lifetime_tokens = lifetime_tokens_;
        d.lifetime_prompt_tokens = lifetime_prompt_tokens_;
        d.lifetime_compute_ms = lifetime_compute_ms_;
        d.total_sessions = total_sessions_;
        d.session_tokens = session_tokens_;
        d.session_prompt_tokens = session_prompt_tokens_;
        d.session_compute_ms = session_compute_ms_;

        // Cost estimation: Blended frontier API savings ($0.80 / 1M prompt, $2.50 / 1M output)
        double prompt_cost = (static_cast<double>(d.lifetime_prompt_tokens) / 1000000.0) * 0.80;
        double gen_cost    = (static_cast<double>(d.lifetime_tokens) / 1000000.0) * 2.50;
        d.cost_saved_usd = prompt_cost + gen_cost;

        // Milestones
        if (d.lifetime_tokens >= 10000000) {
            d.milestone_badge = "Legendary Overlord (10M+ Tokens)";
        } else if (d.lifetime_tokens >= 5000000) {
            d.milestone_badge = "Grandmaster (5M+ Tokens)";
        } else if (d.lifetime_tokens >= 1000000) {
            d.milestone_badge = "Silicon Titan (1M+ Tokens)";
        } else if (d.lifetime_tokens >= 500000) {
            d.milestone_badge = "Platinum Commander (500K+ Tokens)";
        } else if (d.lifetime_tokens >= 100000) {
            d.milestone_badge = "Gold Pioneer (100K+ Tokens)";
        } else if (d.lifetime_tokens >= 25000) {
            d.milestone_badge = "Silver Voyager (25K+ Tokens)";
        } else if (d.lifetime_tokens >= 5000) {
            d.milestone_badge = "Bronze Operator (5K+ Tokens)";
        } else {
            d.milestone_badge = "Novice Explorer";
        }
        return d;
    }

    void flush() {
        std::lock_guard<std::mutex> lock(mtx_);
        save_locked();
    }

private:
    TokenOdometer() {
        resolve_path();
        load();
        total_sessions_++;
        save_locked();
    }

    ~TokenOdometer() {
        flush();
    }

    std::string config_path_;
    std::mutex mtx_;
    uint64_t lifetime_tokens_ = 0;
    uint64_t lifetime_prompt_tokens_ = 0;
    uint64_t lifetime_compute_ms_ = 0;
    uint64_t total_sessions_ = 0;

    uint64_t session_tokens_ = 0;
    uint64_t session_prompt_tokens_ = 0;
    uint64_t session_compute_ms_ = 0;

    uint64_t last_saved_tokens_ = 0;
    uint64_t last_saved_time_ms_ = 0;

    void resolve_path() {
        const char * home = std::getenv("HOME");
        std::string base = home ? home : "/tmp";
        std::filesystem::path dir = std::filesystem::path(base) / ".config" / "sandwich";
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        config_path_ = (dir / "token_odometer.json").string();
    }

    void load() {
        std::ifstream file(config_path_);
        if (!file.is_open()) return;

        std::string line;
        while (std::getline(file, line)) {
            if (line.find("\"lifetime_tokens\"") != std::string::npos) {
                parse_uint64(line, lifetime_tokens_);
            } else if (line.find("\"lifetime_prompt_tokens\"") != std::string::npos) {
                parse_uint64(line, lifetime_prompt_tokens_);
            } else if (line.find("\"lifetime_compute_ms\"") != std::string::npos) {
                parse_uint64(line, lifetime_compute_ms_);
            } else if (line.find("\"total_sessions\"") != std::string::npos) {
                parse_uint64(line, total_sessions_);
            }
        }
    }

    void parse_uint64(const std::string & line, uint64_t & val) {
        auto pos = line.find(':');
        if (pos == std::string::npos) return;
        std::string sub = line.substr(pos + 1);
        while (!sub.empty() && (sub.back() == ',' || sub.back() == ' ' || sub.back() == '\r' || sub.back() == '\n')) {
            sub.pop_back();
        }
        try {
            val = std::stoull(sub);
        } catch (...) {}
    }

    void save_locked() {
        std::ofstream file(config_path_, std::ios::trunc);
        if (!file.is_open()) return;

        file << "{\n";
        file << "  \"lifetime_tokens\": " << lifetime_tokens_ << ",\n";
        file << "  \"lifetime_prompt_tokens\": " << lifetime_prompt_tokens_ << ",\n";
        file << "  \"lifetime_compute_ms\": " << lifetime_compute_ms_ << ",\n";
        file << "  \"total_sessions\": " << total_sessions_ << "\n";
        file << "}\n";
    }
};

} // namespace sandwich
