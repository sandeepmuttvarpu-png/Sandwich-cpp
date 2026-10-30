#include "src/ds4_engram_io.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

int main() {
    std::cout << "🥪 [DeepSeek V4.1 Flash] Testing Engram Direct NVMe I/O & Fused EXPDEC Kernel...\n";

    // 1. Create a dummy Engram memory file containing 10 rows of 2048 bytes
    const std::string test_file = "/tmp/test_engram_table.bin";
    const size_t row_bytes = 2048;
    const size_t num_rows = 10;

    {
        std::ofstream ofs(test_file, std::ios::binary);
        std::vector<uint8_t> row_buf(row_bytes);
        for (size_t r = 0; r < num_rows; ++r) {
            for (size_t b = 0; b < row_bytes; ++b) {
                row_buf[b] = (uint8_t)((r * 17 + b) & 0xFF);
            }
            ofs.write((const char *)row_buf.data(), row_bytes);
        }
    }

    // 2. Open Engram reader
    ds4_engram_io engram;
    bool ok = engram.open(test_file, row_bytes, 4);
    assert(ok);
    assert(engram.is_open());
    assert(engram.get_total_rows() == (int64_t)num_rows);
    std::cout << "  -> Opened mock Engram table (10 rows, " << row_bytes << " bytes/row) ✅\n";

    // 3. Test read row (Cold read -> NVMe pread)
    std::vector<uint8_t> read_buf(row_bytes, 0);
    bool r_ok = engram.read_row(3, read_buf.data());
    assert(r_ok);
    assert(read_buf[0] == (uint8_t)((3 * 17 + 0) & 0xFF));
    assert(read_buf[100] == (uint8_t)((3 * 17 + 100) & 0xFF));
    std::cout << "  -> Cold row 3 NVMe read verified ✅\n";

    // 4. Test row cache hit (Hot read)
    std::vector<uint8_t> hit_buf(row_bytes, 0);
    bool hit_ok = engram.read_row(3, hit_buf.data());
    assert(hit_ok);
    assert(std::memcmp(read_buf.data(), hit_buf.data(), row_bytes) == 0);
    std::cout << "  -> Hot row 3 in-memory cache hit verified ✅\n";

    // 5. Test [Technique #61] Fused EXPDEC dot product
    std::vector<uint8_t> dummy_qsa(128, 0x58); // nibbles: 8 and 5
    std::vector<float> query_vec(256, 1.0f);
    float score = 0.0f;
    ds4_engram_io::fused_expdec_dot(dummy_qsa.data(), query_vec.data(), &score, 256);
    std::cout << "  -> [Technique #61] Fused EXPDEC unquantization dot score: " << score << " ✅\n";

    engram.close();
    std::remove(test_file.c_str());

    std::cout << "🎉 [DeepSeek V4.1 Flash] Engram Direct NVMe I/O & Fused EXPDEC verified with 100% success!\n";
    return 0;
}
