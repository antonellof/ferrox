// Driver for shim.h. Built once per quant kind, with -DFX_KERNEL set to
// that kind's entry point and the kind's generated .cu textually
// included ahead of this file (see run.sh).
//
// Compare bit-exactly, not approximately: with FP contraction disabled
// the host and the twin perform the identical sequence of fp32
// multiplies and adds, so anything short of equality is a real
// divergence. A NaN on one side and not the other is a divergence too;
// a NaN on both is agreement (a degenerate f16 scale decoded the same
// way twice), which is the same rule `gpu.rs::assert_close_relative`
// arrived at from a real RTX 3060 run.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static std::vector<unsigned char> readf(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        fprintf(stderr, "missing %s\n", p.c_str());
        exit(2);
    }
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(f)),
                                      std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s <base> <n_rows> <n_cols> <batch> <row_bytes>\n", argv[0]);
        return 2;
    }
    const std::string base = argv[1];
    const int n_rows = atoi(argv[2]), n_cols = atoi(argv[3]);
    const int batch = atoi(argv[4]), row_bytes = atoi(argv[5]);

    auto w = readf(base + ".w");
    auto xb = readf(base + ".x");
    auto wantb = readf(base + ".want");
    const float* x = (const float*)xb.data();
    const float* want = (const float*)wantb.data();
    std::vector<float> dst(size_t(batch) * n_rows, 0.0f);

    const unsigned threads = FX_THREADS;
    const unsigned gx = (batch + FX_BN - 1) / FX_BN;
    const unsigned gy = (n_rows + FX_BM - 1) / FX_BM;
    blockDim = {threads, 1, 1};
    gridDim = {gx, gy, 1};

    for (unsigned by = 0; by < gy; by++) {
        for (unsigned bx = 0; bx < gx; bx++) {
            Barrier bar(threads);
            g_bar = &bar;
            std::vector<std::thread> ts;
            for (unsigned t = 0; t < threads; t++) {
                ts.emplace_back([&, t]() {
                    blockIdx = {bx, by, 0};
                    threadIdx = {t, 0, 0};
                    FX_KERNEL(w.data(), x, dst.data(), n_rows, n_cols, batch, row_bytes);
                });
            }
            for (auto& th : ts) th.join();
        }
    }

    const size_t n = size_t(batch) * n_rows;
    if (wantb.size() != n * 4) {
        fprintf(stderr, "%s: expected-output length mismatch\n", base.c_str());
        return 2;
    }
    size_t nan_both = 0, mismatch = 0;
    for (size_t i = 0; i < n; i++) {
        const bool gn = std::isnan(dst[i]), wn = std::isnan(want[i]);
        if (gn || wn) {
            if (gn && wn) {
                nan_both++;
            } else {
                mismatch++;
                if (mismatch <= 3)
                    fprintf(stderr, "%s[%zu]: kernel=%g twin=%g\n", base.c_str(), i, dst[i],
                            want[i]);
            }
            continue;
        }
        if (dst[i] != want[i]) {
            mismatch++;
            if (mismatch <= 3)
                fprintf(stderr, "%s[%zu]: kernel=%g twin=%g\n", base.c_str(), i, dst[i], want[i]);
        }
    }
    printf("%-24s n=%6zu mismatches=%zu (nan-agreements=%zu)\n", base.c_str(), n, mismatch,
           nan_both);
    return mismatch == 0 ? 0 : 1;
}
