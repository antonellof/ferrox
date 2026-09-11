//! Verifies `ferrox-models::kimi_loader`/`kimi_decoder` against a real,
//! downloaded slice of Kimi K3's actual checkpoint -- not the synthetic
//! weights every other test in this crate uses. `#[ignore]`d since it
//! needs a real ~2.3GB safetensors shard on disk, the same convention
//! `ferrox-cuda`'s real-GPU test uses for hardware it can't assume is
//! present.
//!
//! To re-run this verification:
//! 1. Download the real shard (contains all of layer 0's tensors --
//!    the sole dense-leading-layer + KDA-attention layer, no MoE
//!    experts needed):
//!    ```text
//!    curl -L -o model-00001-of-000096.safetensors \
//!      https://huggingface.co/moonshotai/Kimi-K3/resolve/main/model-00001-of-000096.safetensors
//!    ```
//! 2. `export FERROX_TEST_KIMI_SHARD_DIR=/path/to/that/directory`
//! 3. `cargo test -p ferrox-models --test kimi_real_data -- --ignored --nocapture`
//!
//! This builds its own minimal `weight_map` (rather than requiring the
//! real ~megabyte-scale `model.safetensors.index.json` covering all 96
//! shards, which would make `ShardedSafetensors::open_index` try to
//! open shards this test never downloads) pointing every real layer-0
//! tensor name at that one shard file.

#[test]
#[ignore]
fn loads_and_runs_a_real_kimi_k3_layer_from_downloaded_bytes() {
    let dir = std::env::var("FERROX_TEST_KIMI_SHARD_DIR")
        .expect("set FERROX_TEST_KIMI_SHARD_DIR to the directory containing the downloaded shard");
    let shard_dir = std::path::PathBuf::from(dir);
    let shard_filename = "model-00001-of-000096.safetensors";
    assert!(
        shard_dir.join(shard_filename).exists(),
        "expected {shard_filename} inside {shard_dir:?} -- see this test's module doc comment"
    );

    // Real per-layer hyperparameters, confirmed against Kimi K3's real
    // config.json (hidden_size=7168, KDA num_heads=96/head_dim=128,
    // dense-layer intermediate_size=33792).
    const HIDDEN_DIM: usize = 7168;
    const KDA_NUM_HEADS: usize = 96;
    const KDA_HEAD_DIM: usize = 128;
    const DENSE_INTERMEDIATE: usize = 33792;
    const EPS: f32 = 1e-5;
    const SITU_BETA: f32 = 4.0;
    const SITU_LINEAR_BETA: f32 = 25.0;

    let prefix = "language_model.model.layers.0";
    let real_tensor_names = [
        "input_layernorm.weight",
        "mlp.down_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp_res_norm.weight",
        "mlp_res_proj.weight",
        "post_attention_layernorm.weight",
        "self_attention_res_norm.weight",
        "self_attention_res_proj.weight",
        "self_attn.A_log",
        "self_attn.b_proj.weight",
        "self_attn.dt_bias",
        "self_attn.f_a_proj.weight",
        "self_attn.f_b_proj.weight",
        "self_attn.g_proj.weight",
        "self_attn.k_conv1d.weight",
        "self_attn.k_proj.weight",
        "self_attn.o_norm.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_conv1d.weight",
        "self_attn.q_proj.weight",
        "self_attn.v_conv1d.weight",
        "self_attn.v_proj.weight",
    ];

    let mut index_json = String::from("{\"weight_map\":{");
    for (i, name) in real_tensor_names.iter().enumerate() {
        if i > 0 {
            index_json.push(',');
        }
        index_json.push_str(&format!("\"{prefix}.{name}\":\"{shard_filename}\""));
    }
    index_json.push_str("}}");
    let index_path = shard_dir.join("ferrox_test_index.safetensors.index.json");
    std::fs::write(&index_path, &index_json).expect("must write minimal index");

    let shard = ferrox_safetensors::ShardedSafetensors::open_index(&index_path)
        .expect("must open real shard via the minimal index");
    std::fs::remove_file(&index_path).ok();

    println!(
        "opened real shard, {} tensors indexed",
        real_tensor_names.len()
    );

    let kda_weights = ferrox_models::kimi_loader::load_kda_attn(
        &shard,
        prefix,
        KDA_NUM_HEADS,
        KDA_HEAD_DIM,
        HIDDEN_DIM,
    )
    .expect("must load real KDA attention weights");
    let dense_weights =
        ferrox_models::kimi_loader::load_dense_mlp(&shard, prefix, HIDDEN_DIM, DENSE_INTERMEDIATE)
            .expect("must load real dense MLP weights");
    let block_res = ferrox_models::kimi_loader::load_block_residual(&shard, prefix)
        .expect("must load real block-residual weights");
    let input_layernorm_weight = ferrox_models::kimi_loader::load_f32_vec(
        &shard,
        &format!("{prefix}.input_layernorm.weight"),
    )
    .expect("must load real input_layernorm weight");
    let post_attention_layernorm_weight = ferrox_models::kimi_loader::load_f32_vec(
        &shard,
        &format!("{prefix}.post_attention_layernorm.weight"),
    )
    .expect("must load real post_attention_layernorm weight");

    println!("loaded all real layer-0 weights, building KimiDecoder layer");

    let layer = ferrox_models::kimi_decoder::KimiDecoderLayerWeights {
        input_layernorm_weight,
        attn: ferrox_models::kimi_decoder::KimiLayerAttention::Kda(Box::new(kda_weights)),
        post_attention_layernorm_weight,
        ffn: ferrox_models::kimi_decoder::KimiLayerFfn::Dense(Box::new(dense_weights)),
        self_attention_res_norm_weight: block_res.self_attention_res_norm_weight,
        self_attention_res_proj_weight: block_res.self_attention_res_proj_weight,
        mlp_res_norm_weight: block_res.mlp_res_norm_weight,
        mlp_res_proj_weight: block_res.mlp_res_proj_weight,
    };

    // Embedding row and output head are deliberately synthetic (small,
    // random): embed_tokens.weight/lm_head.weight are themselves
    // multi-gigabyte real tensors unrelated to what this test validates
    // (real per-layer KDA/dense-FFN/block-residual compute against real
    // bytes), so they're out of scope here rather than adding
    // multi-gigabyte downloads that don't exercise anything new.
    let mut lcg_state: u64 = 0x243F6A8885A308D3;
    let mut next = || {
        lcg_state = lcg_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((lcg_state >> 33) as f32 / u32::MAX as f32) * 0.1 - 0.05
    };
    let embedding_row: Vec<f32> = (0..HIDDEN_DIM).map(|_| next()).collect();
    let output_vocab = 8;
    let output_head_data: Vec<f32> = (0..output_vocab * HIDDEN_DIM).map(|_| next()).collect();
    let final_norm_weight: Vec<f32> = vec![1.0; HIDDEN_DIM];
    let output_attn_res_norm_weight: Vec<f32> = vec![1.0; HIDDEN_DIM];
    let output_attn_res_proj_weight: Vec<f32> = (0..HIDDEN_DIM).map(|_| next()).collect();

    let weights = ferrox_models::kimi_decoder::KimiDecoderWeights {
        embedding: ferrox_core::tensor::Tensor::new(embedding_row, vec![1, HIDDEN_DIM]),
        layers: vec![layer],
        output_attn_res_norm_weight,
        output_attn_res_proj_weight,
        final_norm_weight,
        output_head: ferrox_core::weight_matrix::WeightMatrix::F32(
            ferrox_core::tensor::Tensor::new(output_head_data, vec![output_vocab, HIDDEN_DIM]),
        ),
    };

    let cfg = ferrox_models::kimi_decoder::KimiDecoderConfig {
        attn_res_block_size: 12, // real value; irrelevant with 1 layer (0 % 12 == 0 either way)
        rms_norm_eps: EPS,
        situ_beta: SITU_BETA,
        situ_linear_beta: SITU_LINEAR_BETA,
        moe: ferrox_models::latent_moe::KimiMoeConfig {
            n_experts_active: 16,
            moe_renormalize: true,
            routed_scaling_factor: 1.0,
            situ_beta: SITU_BETA,
            situ_linear_beta: SITU_LINEAR_BETA,
            rms_norm_eps: EPS,
        },
    };
    let mla_cfg = ferrox_models::config::MlaConfig {
        num_heads: 96,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 128,
        qk_rope_head_dim: 64,
        v_head_dim: 128,
        use_output_gate: true,
        rope: None,
    };
    let kda_cfg = ferrox_models::config::KdaConfig {
        num_heads: KDA_NUM_HEADS,
        head_dim: KDA_HEAD_DIM,
        short_conv_kernel_size: 4,
        gate_lower_bound: -5.0,
        use_full_rank_gate: true,
    };

    let mut state = ferrox_models::kimi_decoder::KimiDecodeState::new(&weights, &kda_cfg);
    let logits = ferrox_models::kimi_decoder::kimi_forward_token(
        &weights, &cfg, &mla_cfg, &kda_cfg, 0, &mut state,
    );

    println!(
        "forward pass produced {} logits: {:?}",
        logits.len(),
        logits
    );
    assert_eq!(logits.len(), output_vocab);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "real-weight forward pass must produce finite output, got {logits:?}"
    );
    assert!(
        logits.iter().any(|&v| v != 0.0),
        "real-weight forward pass produced an all-zero output, which would indicate a loading bug"
    );
    println!("PASS: real Kimi K3 layer-0 (KDA attention + dense FFN) loaded and ran successfully");
}

/// Verifies `load_kimi_layer`'s MoE-FFN path (KDA attention + latent
/// MoE with real MXFP4 routed experts) against a real downloaded shard
/// -- layer 1, confirmed by fetching `model.safetensors.index.json` to
/// contain exactly this layer's 5404 real tensors (896 experts x 3
/// projections x 2 (weight_packed/weight_scale) + attention + norms),
/// all real shapes cross-checked directly against a real shard header
/// fetched via HTTP range request and matching `KimiRealHparams::real()`
/// exactly (moe_hidden_dim=3584, moe_intermediate_dim=3072, n_experts=896).
///
/// Eagerly dequantizing all 896 real experts to owned `f32` (this
/// crate's established, disclosed simplification -- see
/// `kimi_loader`'s module doc comment) uses roughly 896 experts x
/// (3072*3584*3 elements x 4 bytes) =~ 117GB of RAM -- a real,
/// concretely quantified cost of the current eager-dequant design that
/// a synthetic small-dims test can't surface. This test needs a
/// correspondingly large-RAM rented instance; see this test's module
/// doc comment for how it was run.
///
/// To re-run this verification:
/// 1. Download the real shard (contains all of layer 1's 5404 real
///    tensors -- the first MoE layer, KDA attention):
///    ```text
///    curl -L -o model-00002-of-000096.safetensors \
///      https://huggingface.co/moonshotai/Kimi-K3/resolve/main/model-00002-of-000096.safetensors
///    ```
/// 2. `export FERROX_TEST_KIMI_MOE_SHARD_DIR=/path/to/that/directory`
/// 3. `cargo test -p ferrox-models --test kimi_real_data moe_layer -- --ignored --nocapture`
#[test]
#[ignore]
fn loads_and_runs_a_real_kimi_k3_moe_layer_from_downloaded_bytes() {
    let dir = std::env::var("FERROX_TEST_KIMI_MOE_SHARD_DIR").expect(
        "set FERROX_TEST_KIMI_MOE_SHARD_DIR to the directory containing the downloaded shard",
    );
    let shard_dir = std::path::PathBuf::from(dir);
    let shard_filename = "model-00002-of-000096.safetensors";
    assert!(
        shard_dir.join(shard_filename).exists(),
        "expected {shard_filename} inside {shard_dir:?} -- see this test's doc comment"
    );

    let layer_idx = 1;
    let prefix = format!("language_model.model.layers.{layer_idx}");
    let hp = ferrox_models::kimi_loader::KimiRealHparams::real();

    let non_expert_names = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attention_res_norm.weight",
        "self_attention_res_proj.weight",
        "mlp_res_norm.weight",
        "mlp_res_proj.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.q_conv1d.weight",
        "self_attn.k_conv1d.weight",
        "self_attn.v_conv1d.weight",
        "self_attn.A_log",
        "self_attn.f_a_proj.weight",
        "self_attn.f_b_proj.weight",
        "self_attn.dt_bias",
        "self_attn.b_proj.weight",
        "self_attn.g_proj.weight",
        "self_attn.o_norm.weight",
        "self_attn.o_proj.weight",
        "block_sparse_moe.gate.weight",
        "block_sparse_moe.gate.e_score_correction_bias",
        "block_sparse_moe.routed_expert_down_proj.weight",
        "block_sparse_moe.routed_expert_up_proj.weight",
        "block_sparse_moe.routed_expert_norm.weight",
        "block_sparse_moe.shared_experts.gate_proj.weight",
        "block_sparse_moe.shared_experts.down_proj.weight",
        "block_sparse_moe.shared_experts.up_proj.weight",
    ];

    let mut entries: Vec<String> = non_expert_names
        .iter()
        .map(|name| format!("\"{prefix}.{name}\":\"{shard_filename}\""))
        .collect();
    for e in 0..hp.n_experts {
        for w in ["w1", "w2", "w3"] {
            for suf in ["weight_packed", "weight_scale"] {
                entries.push(format!(
                    "\"{prefix}.block_sparse_moe.experts.{e}.{w}.{suf}\":\"{shard_filename}\""
                ));
            }
        }
    }
    let index_json = format!("{{\"weight_map\":{{{}}}}}", entries.join(","));
    let index_path = shard_dir.join("ferrox_test_moe_index.safetensors.index.json");
    std::fs::write(&index_path, &index_json).expect("must write minimal MoE-layer index");

    let shard = ferrox_safetensors::ShardedSafetensors::open_index(&index_path)
        .expect("must open real shard via the minimal index");
    std::fs::remove_file(&index_path).ok();

    println!(
        "opened real shard, {} tensors indexed (layer {layer_idx}, {} routed experts)",
        entries.len(),
        hp.n_experts
    );

    let layer = ferrox_models::kimi_loader::load_kimi_layer(
        &shard,
        &hp,
        ferrox_models::config::LayerAttentionKind::KimiKda,
        false,
        layer_idx,
    )
    .expect("must load a real MoE layer (KDA attention + latent MoE with 896 real MXFP4 experts)");

    println!(
        "loaded all real layer-1 weights (896 real MXFP4 experts), building KimiDecoder layer"
    );

    match &layer.ffn {
        ferrox_models::kimi_decoder::KimiLayerFfn::Moe(moe) => {
            assert_eq!(moe.experts.n_experts(), hp.n_experts);
        }
        ferrox_models::kimi_decoder::KimiLayerFfn::Dense(_) => panic!("expected a real MoE FFN"),
    }
    assert!(matches!(
        layer.attn,
        ferrox_models::kimi_decoder::KimiLayerAttention::Kda(_)
    ));

    // Embedding row and output head are deliberately synthetic, same
    // rationale as the layer-0 test above.
    let hidden_dim = hp.hidden_dim;
    let mut lcg_state: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        lcg_state = lcg_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((lcg_state >> 33) as f32 / u32::MAX as f32) * 0.1 - 0.05
    };
    let embedding_row: Vec<f32> = (0..hidden_dim).map(|_| next()).collect();
    let output_vocab = 8;
    let output_head_data: Vec<f32> = (0..output_vocab * hidden_dim).map(|_| next()).collect();
    let final_norm_weight: Vec<f32> = vec![1.0; hidden_dim];
    let output_attn_res_norm_weight: Vec<f32> = vec![1.0; hidden_dim];
    let output_attn_res_proj_weight: Vec<f32> = (0..hidden_dim).map(|_| next()).collect();

    let weights = ferrox_models::kimi_decoder::KimiDecoderWeights {
        embedding: ferrox_core::tensor::Tensor::new(embedding_row, vec![1, hidden_dim]),
        layers: vec![layer],
        output_attn_res_norm_weight,
        output_attn_res_proj_weight,
        final_norm_weight,
        output_head: ferrox_core::weight_matrix::WeightMatrix::F32(
            ferrox_core::tensor::Tensor::new(output_head_data, vec![output_vocab, hidden_dim]),
        ),
    };

    const EPS: f32 = 1e-5;
    const SITU_BETA: f32 = 4.0;
    const SITU_LINEAR_BETA: f32 = 25.0;
    let cfg = ferrox_models::kimi_decoder::KimiDecoderConfig {
        attn_res_block_size: 12,
        rms_norm_eps: EPS,
        situ_beta: SITU_BETA,
        situ_linear_beta: SITU_LINEAR_BETA,
        moe: ferrox_models::latent_moe::KimiMoeConfig {
            n_experts_active: 16,
            moe_renormalize: true,
            routed_scaling_factor: 1.0,
            situ_beta: SITU_BETA,
            situ_linear_beta: SITU_LINEAR_BETA,
            rms_norm_eps: EPS,
        },
    };
    let mla_cfg = ferrox_models::config::MlaConfig {
        num_heads: hp.mla_num_heads,
        q_lora_rank: hp.mla_q_lora_rank,
        kv_lora_rank: hp.mla_kv_lora_rank,
        qk_nope_head_dim: hp.mla_qk_nope_head_dim,
        qk_rope_head_dim: hp.mla_qk_rope_head_dim,
        v_head_dim: hp.mla_v_head_dim,
        use_output_gate: true,
        rope: None,
    };
    let kda_cfg = ferrox_models::config::KdaConfig {
        num_heads: hp.kda_num_heads,
        head_dim: hp.kda_head_dim,
        short_conv_kernel_size: 4,
        gate_lower_bound: -5.0,
        use_full_rank_gate: true,
    };

    let mut state = ferrox_models::kimi_decoder::KimiDecodeState::new(&weights, &kda_cfg);
    let logits = ferrox_models::kimi_decoder::kimi_forward_token(
        &weights, &cfg, &mla_cfg, &kda_cfg, 0, &mut state,
    );

    println!(
        "forward pass produced {} logits: {:?}",
        logits.len(),
        logits
    );
    assert_eq!(logits.len(), output_vocab);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "real-weight MoE forward pass must produce finite output, got {logits:?}"
    );
    assert!(
        logits.iter().any(|&v| v != 0.0),
        "real-weight MoE forward pass produced an all-zero output, which would indicate a loading bug"
    );
    println!(
        "PASS: real Kimi K3 layer-1 (KDA attention + latent MoE, 896 real MXFP4 experts) loaded and ran successfully"
    );
}

/// A lighter-weight companion to the full-layer test above: loads real
/// layer-1 KDA attention (small, real bytes) plus a handful of
/// individual real routed experts via `load_kimi_expert` directly
/// (bypassing `load_latent_moe`'s all-896 loop), from the same
/// downloaded shard. Exists because the full-layer test's eager
/// dequant of all 896 real experts to owned `f32` was observed to
/// OOM-kill the process on a real 62GB-RAM rented instance (twice,
/// reproducibly) -- consistent with, and now a real-hardware
/// confirmation of, the ~117GB estimate in this file's module doc
/// comment. This test isolates and confirms the two things that
/// estimate doesn't call into question: (1) `load_kda_attn` generalizes
/// correctly to a second real layer's real bytes, not just layer 0's;
/// (2) individual real MXFP4 experts -- not just layer 0's small
/// per-head tensors -- decode to finite values from genuine on-disk
/// bytes. A handful of real experts (not all 896) costs a realistic,
/// small amount of memory, which is exactly what a lazy/streaming
/// per-selected-expert loader (the real fix a genuinely cheap/
/// small-memory Kimi K3 engine needs, per the module doc comment above)
/// would load per token in practice.
#[test]
#[ignore]
fn loads_real_kda_attention_and_a_sample_of_real_mxfp4_experts_for_layer_1() {
    let dir = std::env::var("FERROX_TEST_KIMI_MOE_SHARD_DIR").expect(
        "set FERROX_TEST_KIMI_MOE_SHARD_DIR to the directory containing the downloaded shard",
    );
    let shard_dir = std::path::PathBuf::from(dir);
    let shard_filename = "model-00002-of-000096.safetensors";
    assert!(
        shard_dir.join(shard_filename).exists(),
        "expected {shard_filename} inside {shard_dir:?} -- see this test's doc comment"
    );

    let layer_idx = 1;
    let prefix = format!("language_model.model.layers.{layer_idx}");
    let hp = ferrox_models::kimi_loader::KimiRealHparams::real();
    let sample_expert_indices = [0usize, 1, 2, 500, 895];

    let attn_names = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.q_conv1d.weight",
        "self_attn.k_conv1d.weight",
        "self_attn.v_conv1d.weight",
        "self_attn.A_log",
        "self_attn.f_a_proj.weight",
        "self_attn.f_b_proj.weight",
        "self_attn.dt_bias",
        "self_attn.b_proj.weight",
        "self_attn.g_proj.weight",
        "self_attn.o_norm.weight",
        "self_attn.o_proj.weight",
    ];
    let mut entries: Vec<String> = attn_names
        .iter()
        .map(|name| format!("\"{prefix}.{name}\":\"{shard_filename}\""))
        .collect();
    for &e in &sample_expert_indices {
        for w in ["w1", "w2", "w3"] {
            for suf in ["weight_packed", "weight_scale"] {
                entries.push(format!(
                    "\"{prefix}.block_sparse_moe.experts.{e}.{w}.{suf}\":\"{shard_filename}\""
                ));
            }
        }
    }
    let index_json = format!("{{\"weight_map\":{{{}}}}}", entries.join(","));
    let index_path = shard_dir.join("ferrox_test_sample_index.safetensors.index.json");
    std::fs::write(&index_path, &index_json).expect("must write minimal sample index");

    let shard = ferrox_safetensors::ShardedSafetensors::open_index(&index_path)
        .expect("must open real shard via the minimal index");
    std::fs::remove_file(&index_path).ok();

    let kda_weights = ferrox_models::kimi_loader::load_kda_attn(
        &shard,
        &prefix,
        hp.kda_num_heads,
        hp.kda_head_dim,
        hp.hidden_dim,
    )
    .expect("must load real layer-1 KDA attention weights");
    println!(
        "PASS: real layer-1 KDA attention loaded ({} q_proj rows)",
        kda_weights.q_proj.rows()
    );

    for &e in &sample_expert_indices {
        let expert = ferrox_models::kimi_loader::load_kimi_expert(
            &shard,
            &format!("{prefix}.block_sparse_moe"),
            e,
            hp.moe_hidden_dim,
            hp.moe_intermediate_dim,
        )
        .unwrap_or_else(|err| panic!("must load real expert {e}: {err}"));
        let x = vec![0.01f32; hp.moe_hidden_dim];
        let out = expert.forward(&x, 4.0, 25.0);
        assert_eq!(out.len(), hp.moe_hidden_dim);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "real MXFP4 expert {e}'s forward output must be finite, got {out:?}"
        );
        println!("PASS: real expert {e} (real MXFP4 w1/w2/w3) loaded and ran, output finite");
    }

    println!(
        "PASS: real layer-1 KDA attention + {} individual real MXFP4 experts all loaded and ran from genuine downloaded bytes",
        sample_expert_indices.len()
    );
}

/// Verifies `ferrox-models::kimi_tokenizer::KimiTokenizer` against a
/// real downloaded `tiktoken.model` file, cross-validated against the
/// *real* `tiktoken` Python library (not a hand-derived guess): each
/// golden ID sequence below was produced by running the actual
/// `tiktoken.Encoding` with this exact real vocab file and the exact
/// real `pat_str` from `tokenization_kimi.py`, then asserting Rust
/// produces byte-for-byte the same token IDs.
///
/// To re-run this verification:
/// 1. `curl -L -o tiktoken.model https://huggingface.co/moonshotai/Kimi-K3/resolve/main/tiktoken.model`
/// 2. `export FERROX_TEST_KIMI_TOKENIZER_PATH=/path/to/tiktoken.model`
/// 3. `cargo test -p ferrox-models --test kimi_real_data tokenizer -- --ignored --nocapture`
#[test]
#[ignore]
fn kimi_tokenizer_matches_the_real_tiktoken_library_on_real_vocab() {
    let path = std::env::var("FERROX_TEST_KIMI_TOKENIZER_PATH")
        .expect("set FERROX_TEST_KIMI_TOKENIZER_PATH to the downloaded tiktoken.model file");
    let text = std::fs::read_to_string(&path).expect("must read tiktoken.model");
    let ranks = ferrox_models::kimi_tokenizer::parse_tiktoken_vocab(&text)
        .expect("must parse real tiktoken.model");
    assert_eq!(
        ranks.len(),
        163584,
        "real Kimi K3 tiktoken.model must have exactly 163584 base tokens (config.json: vocab_size 163840 - 256 reserved)"
    );

    let tok =
        ferrox_models::kimi_tokenizer::KimiTokenizer::new(ranks, std::collections::HashMap::new())
            .expect("split regex must compile");

    // (input, expected token ids from the real `tiktoken` Python library
    // against this exact real vocab file and pat_str -- see this
    // test's doc comment).
    let cases: &[(&str, &[u32])] = &[
        ("Hello, world!", &[19180, 11, 2695, 0]),
        (
            "  multiple   spaces\n\nand newlines",
            &[220, 6810, 256, 14803, 382, 516, 814, 11541],
        ),
        (
            "I'll do it, don't you think?",
            &[65447, 859, 483, 11, 4536, 398, 2704, 30],
        ),
        ("你好，世界！", &[33845, 378, 2243, 856]),
        ("π = 3.14159", &[14888, 327, 220, 18, 13, 18702, 6276]),
        (
            "The quick brown fox jumps over the lazy dog.",
            &[1008, 5072, 16331, 69275, 60062, 1312, 276, 29292, 7751, 13],
        ),
        ("こんにちは世界", &[16444, 29194, 6880, 43566, 8831, 2243]),
        ("", &[]),
        ("a", &[64]),
        (
            "12345678901234567890",
            &[6694, 12972, 16242, 16349, 18439, 22523, 2788],
        ),
    ];

    for (text, expected) in cases {
        let ids = tok.encode(text, ferrox_models::tokenizer::SpecialTokens::AsText);
        assert_eq!(ids, *expected, "encode({text:?}) mismatch");
        let back = tok.decode(&ids);
        assert_eq!(&back, text, "decode roundtrip mismatch for {text:?}");
        println!("PASS: {text:?} -> {ids:?}");
    }

    println!(
        "PASS: KimiTokenizer matches the real tiktoken library on {} real test strings",
        cases.len()
    );
}
