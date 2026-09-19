//! BERT GGUF → [`BertEncoder`].
//!
//! Tensor names and requiredness follow llama.cpp
//! `llama_model_bert::load_arch_tensors` (`src/models/bert.cpp`), and
//! hparam keys follow its `load_arch_hparams` plus the shared
//! `LLM_KV_*` set.
//!
//! # `bert.cpp` upstream is five architectures; this is one of them
//!
//! `nomic-bert`, `nomic-bert-moe`, `jina-bert-v2`, `jina-bert-v3` and
//! `neo-bert` all build their graph from the same file, each switching
//! on `model.arch` for RoPE, a gated FFN, expert layers or a second
//! attention norm. [`crate::bert_encoder`] implements only the plain
//! `bert` shape, so every one of those differences is checked for here
//! and **refused by name**: a checkpoint that carries `ffn_gate` or
//! `attn_q_norm` is not silently run without it.
//!
//! The last line of defence is
//! [`crate::loader::assert_every_tensor_consumed`], which the load ends
//! with: for `bge-small-en-v1.5-q8_0.gguf` this graph reads all 197
//! tensors, so any weight a variant adds and this module has no home
//! for stops the load instead of being ignored.

use ferrox_gguf::{ShardedGguf, TensorSource};

use crate::bert_encoder::{BertEncoder, BertFfn, BertHparams, BertLayer};
use crate::loader::{
    assert_every_tensor_consumed, load_f32_vec, load_f32_vec_optional, load_weight_matrix,
    LoadError,
};
use crate::pooling::PoolingType;

/// The architecture string this loader implements.
pub const BERT_ARCH: &str = "bert";

/// llama.cpp's `tokenizer_model == "bert"` defaults, applied when the
/// GGUF carries no explicit id (`llama-vocab.cpp`).
const DEFAULT_CLS_ID: u32 = 101;
const DEFAULT_SEP_ID: u32 = 102;

fn meta_u64(file: &impl TensorSource, key: &str) -> Result<u64, LoadError> {
    file.metadata_u64(key)
        .ok_or_else(|| LoadError::MissingHparam(key.to_string()))
}

fn refuse(what: &str) -> LoadError {
    LoadError::UnsupportedFeature(BERT_ARCH.to_string(), what.to_string())
}

/// Refuses if `name` exists, naming the upstream variant that carries it.
fn reject_tensor(file: &ShardedGguf, name: &str, why: &str) -> Result<(), LoadError> {
    if file.find_tensor(name).is_some() {
        return Err(refuse(&format!("checkpoint carries '{name}': {why}")));
    }
    Ok(())
}

/// The whole architecture policy of this loader, in one place so it can
/// be tested without a GGUF: `bert` and nothing else.
pub fn check_arch(arch: &str) -> Result<(), LoadError> {
    if ENCODER_ARCHS.iter().any(|(a, _)| *a == arch) {
        Ok(())
    } else {
        Err(LoadError::UnsupportedArchitecture(arch.to_string()))
    }
}

/// The architectures this loader builds, with the FFN each one runs.
///
/// Both share `bert.cpp`'s graph; what differs is two lines of it,
/// and both are read from the architecture because upstream reads
/// them that way: the rotation at `:126-133` and the FFN at
/// `:179-201`. A row here is a promise that every OTHER line of that
/// graph is the same, which is why `nomic-bert-moe` is not in it (its
/// `moe_every_n_layers` layers are a second FFN shape) and
/// `jina-bert-v2` is not either (a second attention norm).
pub const ENCODER_ARCHS: &[(&str, BertFfn)] = &[
    ("bert", BertFfn::GeluSeq),
    ("nomic-bert", BertFfn::SwigluPar),
];

/// Reads and checks `bert.*` hparams. Fails closed on anything the
/// graph in [`crate::bert_encoder`] does not implement.
pub fn read_bert_hparams(file: &impl TensorSource) -> Result<BertHparams, LoadError> {
    let arch = file
        .metadata_str("general.architecture")
        .ok_or_else(|| LoadError::MissingHparam("general.architecture".into()))?
        .to_string();
    check_arch(&arch)?;
    let p = |suffix: &str| format!("{arch}.{suffix}");

    let n_layer = meta_u64(file, &p("block_count"))? as usize;
    let n_embd = meta_u64(file, &p("embedding_length"))? as usize;
    let n_ff = meta_u64(file, &p("feed_forward_length"))? as usize;
    let n_head = meta_u64(file, &p("attention.head_count"))? as usize;
    let n_head_kv = file
        .metadata_u64(&p("attention.head_count_kv"))
        .unwrap_or(n_head as u64) as usize;
    let n_ctx_train = meta_u64(file, &p("context_length"))? as usize;

    // `LLM_KV_ATTENTION_LAYERNORM_EPS` is read with `get_key(..., true)`
    // upstream, i.e. required: there is no sane default for a norm this
    // small (this checkpoint's is 1e-12, a thousand times tighter than
    // any RMSNorm eps in the rest of this codebase).
    let layer_norm_eps = file
        .metadata_f32(&p("attention.layer_norm_epsilon"))
        .ok_or_else(|| LoadError::MissingHparam(p("attention.layer_norm_epsilon")))?;

    // `n_token_types` is required by upstream's own loader, which
    // throws "model needs to define token type count".
    let n_token_types = meta_u64(file, "tokenizer.ggml.token_type_count")? as usize;
    if n_token_types == 0 {
        return Err(refuse("tokenizer.ggml.token_type_count is 0"));
    }

    // An encoder is bidirectional by construction. If a checkpoint ever
    // says otherwise, this graph is the wrong one for it.
    if file.metadata_bool(&p("attention.causal")).unwrap_or(false) {
        return Err(refuse(
            "bert.attention.causal is true, but this graph applies no mask — \
             a causal BERT would need a decoder path",
        ));
    }

    if n_head == 0 || n_head_kv == 0 || !n_head.is_multiple_of(n_head_kv) {
        return Err(refuse(&format!(
            "head_count {n_head} is not a multiple of head_count_kv {n_head_kv}"
        )));
    }
    if !n_embd.is_multiple_of(n_head) {
        return Err(refuse(&format!(
            "embedding_length {n_embd} is not divisible by head_count {n_head}"
        )));
    }
    if file.metadata_u64(&p("expert_count")).unwrap_or(0) != 0
        || file.metadata_u64(&p("moe_every_n_layers")).unwrap_or(0) != 0
    {
        return Err(refuse(
            "expert layers (nomic-bert-moe's moe_every_n_layers) are not implemented",
        ));
    }

    // Upstream defaults `hparams.pooling_type` to NONE and reads the key
    // as optional, so an absent key means "return every row", not
    // "guess CLS".
    let pooling = PoolingType::from_gguf(file, &arch)
        .map_err(|e| refuse(&e.to_string()))?
        .unwrap_or(PoolingType::None);

    let cls_id = file
        .metadata_u64("tokenizer.ggml.bos_token_id")
        .unwrap_or(u64::from(DEFAULT_CLS_ID)) as u32;
    let sep_id = file
        .metadata_u64("tokenizer.ggml.seperator_token_id")
        .unwrap_or(u64::from(DEFAULT_SEP_ID)) as u32;

    // `bert.cpp:126-133` rotates for the architectures listed there and
    // adds no position table for them (`:90` is gated on `bert`); the
    // two facts are one field on `BertHparams`.
    let ffn = ENCODER_ARCHS
        .iter()
        .find(|(a, _)| *a == arch)
        .map(|(_, f)| *f)
        .expect("check_arch admitted this architecture");
    let rope_theta =
        (arch != BERT_ARCH).then(|| file.metadata_f32(&p("rope.freq_base")).unwrap_or(10_000.0));
    let head_dim = n_embd / n_head;
    let rope_dim = file
        .metadata_u64(&p("rope.dimension_count"))
        .map(|v| v as usize)
        .unwrap_or(head_dim);
    if rope_theta.is_some() && (rope_dim == 0 || rope_dim > head_dim || !rope_dim.is_multiple_of(2))
    {
        return Err(refuse(&format!(
            "rope.dimension_count is {rope_dim}, which is not an even width at or under \
             the {head_dim}-wide head"
        )));
    }

    Ok(BertHparams {
        arch,
        rope_theta,
        rope_dim,
        ffn,
        n_layer,
        n_embd,
        n_ff,
        n_head,
        n_head_kv,
        n_ctx_train,
        n_token_types,
        layer_norm_eps,
        pooling,
        cls_id,
        sep_id,
    })
}

/// Loads a `bert` GGUF into a runnable encoder.
pub fn load_bert_encoder_from_path(
    path: impl AsRef<std::path::Path>,
) -> Result<BertEncoder, LoadError> {
    load_bert_encoder(&ShardedGguf::open(path.as_ref())?)
}

/// Same, from an already-open file — so a caller that also needs the
/// tokenizer out of it ([`crate::embedding_model`]) mmaps it once.
pub fn load_bert_encoder(file: &ShardedGguf) -> Result<BertEncoder, LoadError> {
    let hp = read_bert_hparams(file)?;

    let tok_embd = load_weight_matrix(file, "token_embd.weight")?;
    // `bert.cpp:32` creates the table for every architecture on the
    // graph, but `:90` reads it only for `bert` -- and libllama's own
    // load log never names it for `nomic-bert`, measured. So a
    // rotating file carries none and a rotating file that DOES carry
    // one is refused rather than silently ignored.
    let pos_embd = match hp.rope_theta {
        None => Some(load_weight_matrix(file, "position_embd.weight")?),
        Some(_) => {
            if file.find_tensor("position_embd.weight").is_some() {
                return Err(refuse(
                    "a rotating encoder (bert.cpp:126-133) carries position_embd.weight, \
                     which its graph never reads; llama.cpp refuses the file as carrying \
                     an unread tensor and so does ferrox",
                ));
            }
            None
        }
    };
    if let Some(table) = &pos_embd {
        if table.rows() != hp.n_ctx_train {
            return Err(refuse(&format!(
                "position_embd.weight has {} rows but {}.context_length says {} — the \
                 learned position table and the advertised context disagree",
                table.rows(),
                hp.arch,
                hp.n_ctx_train
            )));
        }
    }
    if pos_embd.as_ref().is_some_and(|t| t.cols() != hp.n_embd) || tok_embd.cols() != hp.n_embd {
        return Err(refuse(&format!(
            "embedding tables are {} / {} wide but embedding_length is {}",
            tok_embd.cols(),
            pos_embd.as_ref().map_or(hp.n_embd, |t| t.cols()),
            hp.n_embd
        )));
    }

    // The WHOLE table, not row 0. Upstream views `type_embd` at offset
    // 0 and adds it everywhere, because `llama_batch` carries no
    // segment ids; ferrox's encoder takes them as a parameter, so a
    // cross-encoder pair can put its document half on row 1 the way the
    // checkpoint was trained. Loading row 0 alone is what made
    // `/v1/rerank` rank the relevant document last (see
    // `bert_encoder`'s module docs). The tensor is
    // `TENSOR_NOT_REQUIRED` upstream, so its absence is not an error —
    // it means no segment embedding is added at all, and a reranker
    // checkpoint in that state is refused by `EmbeddingModel`.
    let type_embd = match file.find_tensor("token_types.weight") {
        Some(_) => {
            let table = load_weight_matrix(file, "token_types.weight")?;
            if table.rows() != hp.n_token_types || table.cols() != hp.n_embd {
                return Err(refuse(&format!(
                    "token_types.weight is {}x{}, expected {}x{}",
                    table.rows(),
                    table.cols(),
                    hp.n_token_types,
                    hp.n_embd
                )));
            }
            Some((0..table.rows()).map(|r| table.dequant_row(r)).collect())
        }
        None => None,
    };

    let tok_norm_w = load_f32_vec(file, "token_embd_norm.weight")?;
    let tok_norm_b = load_f32_vec(file, "token_embd_norm.bias")?;

    let mut layers = Vec::with_capacity(hp.n_layer);
    for l in 0..hp.n_layer {
        let b = format!("blk.{l}");
        reject_tensor(
            file,
            &format!("{b}.attn_qkv.weight"),
            "a fused QKV projection; this graph reads separate attn_q/attn_k/attn_v",
        )?;
        reject_tensor(
            file,
            &format!("{b}.attn_q_norm.weight"),
            "per-projection QK normalization (jina-bert-v3 / neo-bert), not implemented",
        )?;
        reject_tensor(
            file,
            &format!("{b}.attn_k_norm.weight"),
            "per-projection QK normalization (jina-bert-v3 / neo-bert), not implemented",
        )?;
        reject_tensor(
            file,
            &format!("{b}.attn_norm_2.weight"),
            "jina-bert-v2's second attention norm, not implemented",
        )?;
        // The gate belongs to `BertFfn::SwigluPar` and to nothing
        // else: a `bert` file that carries one is a file this graph
        // would run as an ungated GELU while llama.cpp ran it gated,
        // which is the silent-wrong shape the refusal exists for.
        if hp.ffn == BertFfn::GeluSeq {
            reject_tensor(
                file,
                &format!("{b}.ffn_gate.weight"),
                "a gated FFN (jina-bert-v2 GEGLU); this architecture's graph runs a plain \
                 GELU MLP (bert.cpp:179-187)",
            )?;
        }
        reject_tensor(
            file,
            &format!("{b}.ffn_up_exps.weight"),
            "MoE expert tensors (nomic-bert-moe), not implemented",
        )?;

        layers.push(BertLayer {
            wq: load_weight_matrix(file, &format!("{b}.attn_q.weight"))?,
            bq: load_f32_vec_optional(file, &format!("{b}.attn_q.bias"))?,
            wk: load_weight_matrix(file, &format!("{b}.attn_k.weight"))?,
            bk: load_f32_vec_optional(file, &format!("{b}.attn_k.bias"))?,
            wv: load_weight_matrix(file, &format!("{b}.attn_v.weight"))?,
            bv: load_f32_vec_optional(file, &format!("{b}.attn_v.bias"))?,
            wo: load_weight_matrix(file, &format!("{b}.attn_output.weight"))?,
            bo: load_f32_vec_optional(file, &format!("{b}.attn_output.bias"))?,
            attn_out_norm_w: load_f32_vec(file, &format!("{b}.attn_output_norm.weight"))?,
            attn_out_norm_b: load_f32_vec(file, &format!("{b}.attn_output_norm.bias"))?,
            ffn_up: load_weight_matrix(file, &format!("{b}.ffn_up.weight"))?,
            ffn_up_b: load_f32_vec_optional(file, &format!("{b}.ffn_up.bias"))?,
            ffn_gate: match hp.ffn {
                BertFfn::GeluSeq => None,
                BertFfn::SwigluPar => {
                    Some(load_weight_matrix(file, &format!("{b}.ffn_gate.weight"))?)
                }
            },
            ffn_down: load_weight_matrix(file, &format!("{b}.ffn_down.weight"))?,
            ffn_down_b: load_f32_vec_optional(file, &format!("{b}.ffn_down.bias"))?,
            layer_out_norm_w: load_f32_vec(file, &format!("{b}.layer_output_norm.weight"))?,
            layer_out_norm_b: load_f32_vec(file, &format!("{b}.layer_output_norm.bias"))?,
        });
    }

    let kv_dim = hp.n_head_kv * hp.head_dim();
    for (l, layer) in layers.iter().enumerate() {
        for (name, m, rows) in [
            ("attn_q", &layer.wq, hp.n_embd),
            ("attn_k", &layer.wk, kv_dim),
            ("attn_v", &layer.wv, kv_dim),
            ("attn_output", &layer.wo, hp.n_embd),
            ("ffn_up", &layer.ffn_up, hp.n_ff),
            ("ffn_down", &layer.ffn_down, hp.n_embd),
        ] {
            if m.rows() != rows {
                return Err(refuse(&format!(
                    "blk.{l}.{name}.weight has {} output rows, expected {rows}",
                    m.rows()
                )));
            }
        }
    }

    assert_every_tensor_consumed(file)?;

    Ok(BertEncoder {
        hp,
        tok_embd,
        type_embd,
        pos_embd,
        tok_norm_w,
        tok_norm_b,
        layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only `bert`. The other eleven encoder rows in the catalog share
    /// `bert.cpp` upstream, and each of them differs from this graph in
    /// a way that would load clean and embed wrong.
    #[test]
    fn an_architecture_outside_the_table_is_refused_by_name() {
        for arch in [
            "nomic-bert-moe",
            "jina-bert-v2",
            "jina-bert-v3",
            "neo-bert",
            "modern-bert",
            "llama",
        ] {
            assert!(
                !ENCODER_ARCHS.iter().any(|(a, _)| *a == arch),
                "`{arch}` is in the table; the refusal below would be wrong"
            );
            let err = check_arch(arch).unwrap_err();
            assert!(
                matches!(&err, LoadError::UnsupportedArchitecture(a) if a == arch),
                "{arch} was not refused: {err}"
            );
        }
        assert!(check_arch(BERT_ARCH).is_ok());
    }
}
