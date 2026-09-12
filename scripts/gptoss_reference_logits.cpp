// Dumps llama.cpp's own logits for an explicit token-id sequence.
//
// This is the *reference* half of ferrox's gpt-oss coverage test: the
// golden values checked into `crates/ferrox-models/tests/` are produced
// by this program running against a real llama.cpp build, not by
// re-reading a spec or by ferrox checking itself.
//
// Build (out-of-source llama.cpp build in $LL):
//   clang++ -std=c++17 -O2 scripts/gptoss_reference_logits.cpp \
//     -I$LLAMA_SRC/include -I$LLAMA_SRC/ggml/include \
//     -L$LL/bin -lllama -Wl,-rpath,$LL/bin -o /tmp/gptoss_ref
//
// Run:
//   /tmp/gptoss_ref model.gguf 3 7 11 19 23 5
//   /tmp/gptoss_ref --lora adapter.gguf:0.5 model.gguf 3 7 11 19 23 5
//
// `--lora FNAME[:SCALE]` (repeatable, before the model path) loads a
// LoRA adapter GGUF through `llama_adapter_lora_init` and applies it at
// SCALE (default 1.0) through `llama_set_adapters_lora`, which is what
// `llama-cli --lora-scaled` does. It is the reference half of ferrox's
// LoRA coverage test.
//
// Prints one float per line, full precision, for the LAST position.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::vector<std::string> lora_paths;
    std::vector<float>       lora_scales;
    int argi = 1;
    while (argi + 1 < argc && strcmp(argv[argi], "--lora") == 0) {
        std::string spec = argv[argi + 1];
        float scale = 1.0f;
        size_t colon = spec.rfind(':');
        if (colon != std::string::npos) {
            scale = (float) atof(spec.c_str() + colon + 1);
            spec.resize(colon);
        }
        lora_paths.push_back(spec);
        lora_scales.push_back(scale);
        argi += 2;
    }

    if (argc - argi < 2) {
        fprintf(stderr, "usage: %s [--lora adapter.gguf[:scale]]... model.gguf tok0 [tok1 ...]\n", argv[0]);
        return 2;
    }

    const char * model_path = argv[argi];
    std::vector<llama_token> toks;
    for (int i = argi + 1; i < argc; ++i) {
        toks.push_back((llama_token) atoi(argv[i]));
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    std::vector<llama_adapter_lora *> adapters;
    for (const auto & path : lora_paths) {
        llama_adapter_lora * a = llama_adapter_lora_init(model, path.c_str());
        if (!a) {
            fprintf(stderr, "failed to load lora adapter %s\n", path.c_str());
            return 1;
        }
        adapters.push_back(a);
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 128;
    cparams.n_batch   = 128;
    cparams.n_ubatch  = 128;
    cparams.no_perf   = true;
    // F32 KV. llama.cpp defaults to an F16 cache, which alone puts a
    // ~1e-4 floor under any comparison and would hide a real error of
    // that size in the graph under test.
    cparams.type_k    = GGML_TYPE_F32;
    cparams.type_v    = GGML_TYPE_F32;
    // ggml's CPU flash-attention kernel accumulates V in F16
    // (`VKQ16`), which is a second ~1e-4 floor. ferrox's CPU attention
    // is plain F32, so compare against the F32 reference path.
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    if (!adapters.empty()) {
        if (llama_set_adapters_lora(ctx, adapters.data(), adapters.size(), lora_scales.data()) != 0) {
            fprintf(stderr, "failed to set lora adapters\n");
            return 1;
        }
    }

    llama_batch batch = llama_batch_init((int32_t) toks.size(), 0, 1);
    batch.n_tokens = (int32_t) toks.size();
    for (size_t i = 0; i < toks.size(); ++i) {
        batch.token[i]     = toks[i];
        batch.pos[i]       = (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1;
    }

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const float * logits = llama_get_logits_ith(ctx, (int32_t) toks.size() - 1);
    for (int i = 0; i < n_vocab; ++i) {
        printf("%.9g\n", logits[i]);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    for (auto * a : adapters) {
        llama_adapter_lora_free(a);
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
