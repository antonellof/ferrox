//! llama.cpp's per-tensor MIX, transcribed from `llama_tensor_get_type`
//! in b7650's `src/llama-quant.cpp:178`.
//!
//! `--type Q4_K_M` does not mean "every tensor Q4_K". It means a
//! specific set of promotions: `output.weight` to Q6_K, `attn_v` on
//! roughly a quarter of the layers to Q6_K, `ffn_down` on the same
//! layers to Q6_K, `attn_qkv` to Q5_K. Writing uniform Q4_K under that
//! name produces a file that loads, runs, reports `general.file_type =
//! Q4_K_M`, and is a different file from the one every other tool in
//! the ecosystem means by the name. That is the failure this module
//! exists to close: [`Recipe`] is why `--pure` stopped being mandatory
//! for the K-quant mixes.
//!
//! # One table, and why the rest is not in it
//!
//! [`PROMOTIONS`] is upstream's else-if chain, in upstream's order, as
//! data: one row per (role, ftype) arm, first match wins. It is the
//! chain, so it is a table.
//!
//! Three things upstream does are NOT chain arms and are therefore not
//! rows -- they are `if` statements that run AFTER the chain has picked
//! a type, and can override it:
//!
//! * the `LLM_TYPE_70B` bump (`llama-quant.cpp:305`), Q3_K/Q4_K -> Q5_K;
//! * `n_expert == 8` -> Q8_0 on `attn_v` and `attn_k`
//!   (`llama-quant.cpp:311`, `:319`);
//! * the output head's shape and Falcon checks.
//!
//! Encoding a post-condition as a chain row would put it in the wrong
//! place: on a Mixtral, `attn_v` matches the Q4_K_M row AND then gets
//! overridden to Q8_0, and a table that could only express "first match
//! wins" would have to choose one. So they are code, each carrying the
//! upstream line it came from, and [`tests`] asserts the ordering.
//!
//! # This is stateful, and the state is the point
//!
//! `i_attention_wv` and `i_ffn_down` are running counters over the
//! tensors that REACH each arm -- not layer indices parsed from the
//! name (except on MoE checkpoints, where upstream falls back to
//! parsing `blk.%d.` precisely because the counter is wrong there).
//! [`Recipe::tensor_type`] therefore takes `&mut self` and must be
//! called once per eligible tensor, in llama.cpp's order, for exactly
//! the tensors llama.cpp would call it for. Calling it for a tensor
//! llama.cpp skips, or in the wrong order, shifts every later layer's
//! decision by one.
//!
//! **That order is NOT the file's.** llama.cpp reaches these tensors
//! through `llama_model_loader::weights_map`, a `std::map` keyed by
//! name with a `weight_name_comparer` (`llama-model-loader.h:53`) that
//! sorts by the layer number parsed out of `blk.%d.` first and by name
//! second. A GGUF written by `convert_hf_to_gguf.py` is not in that
//! order, so walking `file.tensors` gives different answers -- measured
//! on Llama-3.2-1B, 12 of 113 tensors got the wrong type that way,
//! because the counter reached `blk.9.ffn_down` while llama.cpp was at
//! `blk.8`. [`Recipe::resolve_all`] exists so no caller has to know
//! this: it sorts, walks, and hands back a name -> type map.

use std::collections::BTreeMap;

use ferrox_gguf::{GgmlType, GgufFile};

use super::policy::Target;

/// The tensor roles `llama_tensor_get_type` branches on, in the order
/// its else-if chain tests them.
///
/// The order is load-bearing and is not the obvious one:
/// `attn_qkv.weight` is tested AFTER `ffn_down` and `attn_output`, and
/// `output.weight` before everything. [`Role::of`] preserves it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// `output.weight`, or `token_embd.weight` on a checkpoint with no
    /// separate output head (tied embeddings).
    Output,
    /// `token_embd.weight` when `output.weight` also exists.
    TokenEmbd,
    AttnV,
    AttnK,
    AttnQ,
    FfnDown,
    AttnOutput,
    AttnQkv,
    FfnGate,
    FfnUp,
    Other,
}

impl Role {
    /// Which arm of the chain a tensor name lands in.
    ///
    /// `has_output` is whether the file carries an `output.weight` at
    /// all: without one, `token_embd.weight` IS the output head and
    /// gets the output head's type, which is how a tied-embedding
    /// model's Q4_K_M file ends up with a Q6_K embedding table.
    pub fn of(name: &str, has_output: bool) -> Role {
        // `llama-quant.cpp:207`: the output arm's condition, then the
        // token-embd arm's at `:237`. Both are exact-name matches
        // against `LLM_TN(arch)(...)`, which for these two tensors is
        // the same spelling for every architecture.
        if name == "output.weight" {
            return Role::Output;
        }
        let is_token_embd = name == "token_embd.weight" || name == "per_layer_token_embd.weight";
        if is_token_embd {
            return if has_output {
                Role::TokenEmbd
            } else {
                Role::Output
            };
        }
        // From here the chain is substring matches, and the ORDER is
        // upstream's. `attn_v` before `attn_k` before `attn_q` before
        // `ffn_down` before `attn_output` before `attn_qkv` before
        // `ffn_gate` before `ffn_up`. Reordering `ffn_gate` above
        // `ffn_down` would be invisible (no name contains both) and
        // reordering `attn_qkv` above `attn_output` would not, which is
        // why this is one ordered list rather than a match on a parsed
        // suffix.
        for (needle, role) in [
            ("attn_v.weight", Role::AttnV),
            ("attn_k.weight", Role::AttnK),
            ("attn_q.weight", Role::AttnQ),
            ("ffn_down", Role::FfnDown),
            ("attn_output.weight", Role::AttnOutput),
            ("attn_qkv.weight", Role::AttnQkv),
            ("ffn_gate", Role::FfnGate),
            ("ffn_up", Role::FfnUp),
        ] {
            if name.contains(needle) {
                return role;
            }
        }
        Role::Other
    }
}

/// The layer condition a promotion is gated on.
///
/// These four are every condition the five mixes ferrox can write
/// actually use. A fifth would be a new variant and a new match arm,
/// which is the point: a condition spelled inline in one row's closure
/// would be a condition nothing else could see.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum When {
    /// No condition: the arm fires for every tensor of this role.
    Always,
    /// `use_more_bits(i, n)` (`llama-quant.cpp:185`).
    UseMoreBits,
    /// `i < n/8`.
    FirstEighth,
    /// `i < 4`, which only `Q4_K_S`'s `attn_v` arm uses.
    FirstFour,
}

impl When {
    fn holds(self, i: usize, n: usize) -> bool {
        match self {
            When::Always => true,
            // `i_layer < n_layers/8 || i_layer >= 7*n_layers/8 ||
            //  (i_layer - n_layers/8) % 3 == 2`, with C's integer
            // division. The third term is only evaluated when the first
            // is false, so `i - n/8` cannot go negative -- which is the
            // only reason a `usize` subtraction is safe here, and the
            // reason it is written with the guard rather than as a
            // three-way `||`.
            When::UseMoreBits => {
                if i < n / 8 {
                    return true;
                }
                if n != 0 && i >= 7 * n / 8 {
                    return true;
                }
                (i - n / 8) % 3 == 2
            }
            When::FirstEighth => i < n / 8,
            When::FirstFour => i < 4,
        }
    }
}

/// One arm of `llama_tensor_get_type`'s else-if chain.
struct Promotion {
    role: Role,
    ftype: Target,
    /// Whether this arm is guarded by upstream's inline
    /// `arch != LLM_ARCH_FALCON`. There is no `only_falcon` counterpart
    /// because the two Falcon-only arms these mixes reach both need a
    /// fraction (`i < n/16`) or a shape no other arm uses, so they are
    /// code in [`Recipe::tensor_type`] rather than rows.
    not_falcon: bool,
    when: When,
    to: GgmlType,
}

/// `llama_tensor_get_type`'s else-if chain for the five mixes ferrox
/// can write, in upstream's order. First matching row wins.
///
/// A row per arm, and the row's `to` is the type that arm assigns. An
/// arm that assigns the default type (so, no promotion) has no row: the
/// absence IS the "leave it alone" case, and `the_recipe_only_promotes_
/// to_types_ferrox_can_encode` is what stops a row from naming a type
/// with no encoder.
///
/// The `not_falcon` rows are transcribed and UNEVIDENCED: there is no
/// Falcon checkpoint locally, so the recipe test cannot compare them
/// against `llama-quantize`. Saying so is better than omitting the
/// guard and silently writing a non-llama.cpp file for the one
/// architecture that takes a different arm.
const PROMOTIONS: &[Promotion] = &[
    // --- the output head (`llama-quant.cpp:207`) ---------------------
    // `else if (new_type != GGML_TYPE_Q8_0) new_type = GGML_TYPE_Q6_K;`
    // -- every K-quant mix sends the output head to Q6_K. The Falcon
    // and awkward-shape arms above it both send it to Q8_0 and are
    // handled in `tensor_type` because they read the tensor's shape.
    Promotion {
        role: Role::Output,
        ftype: Target::Q4_K_S,
        not_falcon: true,
        when: When::Always,
        to: GgmlType::Q6K,
    },
    Promotion {
        role: Role::Output,
        ftype: Target::Q4_K_M,
        not_falcon: true,
        when: When::Always,
        to: GgmlType::Q6K,
    },
    Promotion {
        role: Role::Output,
        ftype: Target::Q5_K_S,
        not_falcon: true,
        when: When::Always,
        to: GgmlType::Q6K,
    },
    Promotion {
        role: Role::Output,
        ftype: Target::Q5_K_M,
        not_falcon: true,
        when: When::Always,
        to: GgmlType::Q6K,
    },
    // Q6_K's own output head is already Q6_K, so the arm assigns the
    // default and there is deliberately no row.

    // --- attn_v (`llama-quant.cpp:279`) ------------------------------
    // `(Q4_K_M || Q5_K_M) && use_more_bits(i_attention_wv, n_attention_wv)`
    Promotion {
        role: Role::AttnV,
        ftype: Target::Q4_K_M,
        not_falcon: false,
        when: When::UseMoreBits,
        to: GgmlType::Q6K,
    },
    Promotion {
        role: Role::AttnV,
        ftype: Target::Q5_K_M,
        not_falcon: false,
        when: When::UseMoreBits,
        to: GgmlType::Q6K,
    },
    // `Q4_K_S && qs.i_attention_wv < 4`
    Promotion {
        role: Role::AttnV,
        ftype: Target::Q4_K_S,
        not_falcon: false,
        when: When::FirstFour,
        to: GgmlType::Q5K,
    },
    // --- ffn_down (`llama-quant.cpp:336`) ----------------------------
    // Q4_K_M on Falcon: `i < n/16 ? Q6_K : use_more_bits ? Q5_K : Q4_K`.
    // The `i < n/16` half has no `When` of its own because no other arm
    // needs it; it is folded into `tensor_type`'s Falcon branch below.
    Promotion {
        role: Role::FfnDown,
        ftype: Target::Q4_K_M,
        not_falcon: true,
        when: When::UseMoreBits,
        to: GgmlType::Q6K,
    },
    Promotion {
        role: Role::FfnDown,
        ftype: Target::Q5_K_M,
        not_falcon: false,
        when: When::UseMoreBits,
        to: GgmlType::Q6K,
    },
    Promotion {
        role: Role::FfnDown,
        ftype: Target::Q4_K_S,
        not_falcon: true,
        when: When::FirstEighth,
        to: GgmlType::Q5K,
    },
    // --- attn_qkv (`llama-quant.cpp:407`) ----------------------------
    Promotion {
        role: Role::AttnQkv,
        ftype: Target::Q4_K_M,
        not_falcon: false,
        when: When::Always,
        to: GgmlType::Q5K,
    },
    Promotion {
        role: Role::AttnQkv,
        ftype: Target::Q5_K_M,
        not_falcon: false,
        when: When::Always,
        to: GgmlType::Q6K,
    },
];

/// The Falcon-only `ffn_down` arm for Q4_K_M, which needs `i < n/16`
/// and is the only place that fraction appears.
const FALCON_Q4KM_FFN_DOWN_SIXTEENTH: GgmlType = GgmlType::Q6K;

/// The bits of a checkpoint's header the mix reads.
///
/// A struct rather than five arguments, and destructured EXHAUSTIVELY
/// where it is built, so a field added here has to be filled in by
/// whoever adds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelShape {
    /// `<arch>.block_count`, which llama.cpp calls `n_layer` and uses
    /// as the denominator of every `ffn_down` fraction.
    pub n_layer: usize,
    /// `<arch>.expert_count`. `8` is a magic number upstream: Mixtral's
    /// `attn_v` and `attn_k` go to Q8_0.
    pub n_expert: usize,
    /// `general.architecture == "falcon"`. Falcon takes a different arm
    /// in four places.
    pub is_falcon: bool,
    /// llama.cpp's `LLM_TYPE_70B`, which is a per-architecture layer
    /// count, not a parameter count. See [`ModelShape::from_header`].
    pub is_70b: bool,
}

impl ModelShape {
    /// Reads the shape from a GGUF header the way llama.cpp's loader
    /// does.
    ///
    /// `is_70b` is the only interesting one. llama.cpp assigns
    /// `LLM_TYPE_70B` from the LAYER COUNT, per architecture
    /// (`llama-model.cpp:656`, `:751`, `:1037`, `:1563`), and the only
    /// thing in the whole quantizer that reads it is one line that
    /// bumps a Q4_K `attn_v` to Q5_K. So a 70B Llama quantized to
    /// Q4_K_M has Q5_K `attn_v` tensors on the layers `use_more_bits`
    /// did not already promote, and a recipe that ignored this would
    /// differ from `llama-quantize` on exactly one model size.
    pub fn from_header(file: &GgufFile) -> ModelShape {
        let arch = file.metadata_str("general.architecture").unwrap_or("");
        let key =
            |suffix: &str| file.metadata_u64(&format!("{arch}.{suffix}")).unwrap_or(0) as usize;
        let n_layer = key("block_count");
        let n_head = key("attention.head_count");
        let n_head_kv = key("attention.head_count_kv");
        ModelShape {
            n_layer,
            n_expert: key("expert_count"),
            is_falcon: arch == "falcon",
            is_70b: match arch {
                // `hparams.n_head() == hparams.n_head_kv() ? 65B : 70B`
                "llama" | "llama-embed" => n_layer == 80 && n_head != n_head_kv,
                "deci" | "qwen2" | "olmo" => n_layer == 80,
                _ => false,
            },
        }
    }
}

/// A checkpoint's mix, walked once in tensor order.
///
/// Build it with [`Recipe::new`], which needs the whole tensor list
/// because `n_attention_wv` is a count over ALL tensors and
/// `has_output` is a question about the file, not about the tensor in
/// hand.
pub struct Recipe {
    target: Target,
    shape: ModelShape,
    /// Counted over every tensor in the file, before any type is
    /// chosen (`llama-quant.cpp:709`).
    n_attention_wv: usize,
    has_output: bool,
    i_attention_wv: usize,
    i_ffn_down: usize,
    i_ffn_gate: usize,
    i_ffn_up: usize,
}

impl Recipe {
    /// `tensor_names` must be every tensor in the file, in file order.
    pub fn new(target: Target, shape: ModelShape, tensor_names: &[String]) -> Recipe {
        // `llama-quant.cpp:705`: n_attention_wv counts attn_v, attn_qkv
        // AND attn_kv_b, while the chain arm that consumes the counter
        // only fires on attn_v. So on a checkpoint with fused QKV the
        // denominator is larger than the number of tensors that ever
        // increment `i_attention_wv`, and `use_more_bits` sees a
        // stretched range. That is upstream's behaviour and it is
        // reproduced, not corrected.
        let n_attention_wv = tensor_names
            .iter()
            .filter(|n| {
                n.contains("attn_v.weight")
                    || n.contains("attn_qkv.weight")
                    || n.contains("attn_kv_b.weight")
            })
            .count();
        Recipe {
            target,
            shape,
            n_attention_wv,
            has_output: tensor_names.iter().any(|n| n == "output.weight"),
            i_attention_wv: 0,
            i_ffn_down: 0,
            i_ffn_gate: 0,
            i_ffn_up: 0,
        }
    }

    /// The type llama.cpp would give this tensor, and the counter
    /// bookkeeping that goes with it.
    ///
    /// MUST be called once per eligible tensor, in llama.cpp's order.
    /// Prefer [`Recipe::resolve_all`], which owns that order; this is
    /// public for the tests that step the counters one tensor at a time.
    pub fn tensor_type(&mut self, name: &str, shape: &[u64]) -> GgmlType {
        let role = Role::of(name, self.has_output);
        let default = self.target.ggml_type();

        let ty = match role {
            Role::Output => self.output_head_type(shape, default),
            Role::AttnV => {
                let t = self.chain(role, self.i_attention_wv, self.n_attention_wv, default);
                // `llama-quant.cpp:305`, AFTER the chain: the 70B bump.
                let t = if self.shape.is_70b && matches!(t, GgmlType::Q3K | GgmlType::Q4K) {
                    GgmlType::Q5K
                } else {
                    t
                };
                // `llama-quant.cpp:311`, after that: 8 experts -> Q8_0,
                // overriding everything the chain and the 70B bump
                // decided. Last write wins, which is why this is not a
                // table row.
                let t = if self.shape.n_expert == 8 {
                    GgmlType::Q8_0
                } else {
                    t
                };
                self.i_attention_wv += 1;
                t
            }
            Role::AttnK => {
                // `llama-quant.cpp:319`. No counter: upstream does not
                // increment anything on this arm.
                if self.shape.n_expert == 8 {
                    GgmlType::Q8_0
                } else {
                    default
                }
            }
            Role::FfnDown => {
                let (i, n) = self.ffn_layer(name, self.i_ffn_down);
                let t = if self.shape.is_falcon && self.target == Target::Q4_K_M {
                    // The one arm that needs `i < n/16`.
                    if i < n / 16 {
                        FALCON_Q4KM_FFN_DOWN_SIXTEENTH
                    } else if When::UseMoreBits.holds(i, n) {
                        GgmlType::Q5K
                    } else {
                        default
                    }
                } else {
                    self.chain(role, i, n, default)
                };
                self.i_ffn_down += 1;
                t
            }
            Role::AttnOutput => {
                // `llama-quant.cpp:390`: the only Q4_K arm here is the
                // 8-expert one, and it is inside `if (arch != FALCON)`.
                if !self.shape.is_falcon
                    && self.shape.n_expert == 8
                    && matches!(self.target, Target::Q4_K_S | Target::Q4_K_M)
                {
                    GgmlType::Q5K
                } else {
                    default
                }
            }
            Role::FfnGate => {
                let (i, n) = self.ffn_layer(name, self.i_ffn_gate);
                let t = self.chain(role, i, n, default);
                self.i_ffn_gate += 1;
                t
            }
            Role::FfnUp => {
                let (i, n) = self.ffn_layer(name, self.i_ffn_up);
                let t = self.chain(role, i, n, default);
                self.i_ffn_up += 1;
                t
            }
            Role::AttnQkv | Role::AttnQ | Role::TokenEmbd | Role::Other => {
                self.chain(role, 0, 0, default)
            }
        };

        // `llama-quant.cpp:436`: the chosen type is then checked
        // against the tensor's row length, and an awkward row falls
        // back to a DIFFERENT type. ferrox refuses that case in `plan`
        // instead of following the fallback, because it would need Q5_0
        // and Q5_1 encoders it does not have -- so this returns the
        // type llama.cpp would have picked and the caller stops.
        ty
    }

    /// `llama-quant.cpp:207`, the two arms above the K-quant one.
    fn output_head_type(&self, shape: &[u64], default: GgmlType) -> GgmlType {
        let nx = shape.first().copied().unwrap_or(0) as usize;
        let qk_k = default.block_layout().1;
        if self.shape.is_falcon || qk_k == 0 || !nx.is_multiple_of(qk_k) {
            return GgmlType::Q8_0;
        }
        // `else if (new_type != GGML_TYPE_Q8_0) new_type = Q6_K;`
        self.chain(Role::Output, 0, 0, default)
    }

    /// First matching row of [`PROMOTIONS`] for this role and mix, or
    /// the default type.
    fn chain(&self, role: Role, i: usize, n: usize, default: GgmlType) -> GgmlType {
        for p in PROMOTIONS {
            if p.role != role || p.ftype != self.target {
                continue;
            }
            if p.not_falcon && self.shape.is_falcon {
                continue;
            }
            if p.when.holds(i, n) {
                return p.to;
            }
            // Upstream's chain: an arm whose ftype matches but whose
            // condition fails falls THROUGH to the next `else if`, it
            // does not stop the chain. So this keeps scanning.
        }
        default
    }

    /// Every eligible tensor's type, keyed by name.
    ///
    /// This is the entry point the planner uses, and it exists because
    /// the ORDER the counters see is llama.cpp's and not the file's.
    /// Sorting here rather than at the call site means a caller cannot
    /// get it wrong by iterating the obvious thing -- which is exactly
    /// what happened: walking `file.tensors` gave 12 of Llama-3.2-1B's
    /// 113 tensors a different type from `llama-quantize`, and the file
    /// still loaded and answered.
    ///
    /// `eligible` decides which tensors llama.cpp would call
    /// `llama_tensor_get_type` for; it is `policy::allows_quantization`
    /// in production and is passed in so this module does not have to
    /// depend on the keep-list to be tested.
    pub fn resolve_all<F>(
        target: Target,
        shape: ModelShape,
        tensors: &[(String, Vec<u64>)],
        eligible: F,
    ) -> BTreeMap<String, GgmlType>
    where
        F: Fn(&str, &[u64]) -> bool,
    {
        let names: Vec<String> = tensors.iter().map(|(n, _)| n.clone()).collect();
        let mut recipe = Recipe::new(target, shape, &names);

        let mut order: Vec<usize> = (0..tensors.len()).collect();
        order.sort_by(|&a, &b| llama_cpp_weight_order(&tensors[a].0, &tensors[b].0));

        let mut out = BTreeMap::new();
        for i in order {
            let (name, shape) = &tensors[i];
            if eligible(name, shape) {
                out.insert(name.clone(), recipe.tensor_type(name, shape));
            }
        }
        out
    }

    /// `layer_info` (`llama-quant.cpp:189`): the layer index for an FFN
    /// tensor, and the denominator.
    ///
    /// On a dense model this is the running counter. On a MoE model
    /// (`n_expert > 1`) upstream parses `blk.%d.` out of the name
    /// instead, because Mixtral's expert FFN tensors are not in layer
    /// order in the file and the counter would name the wrong layer.
    /// A name that does not parse is a `throw` upstream; here it falls
    /// back to the counter, which is the same answer for every
    /// checkpoint whose names are well formed and does not abort a
    /// quantize over a tensor whose type it would not have changed.
    fn ffn_layer(&self, name: &str, counter: usize) -> (usize, usize) {
        let n = self.shape.n_layer;
        if self.shape.n_expert.max(1) > 1 {
            if let Some(rest) = name.strip_prefix("blk.") {
                if let Some((digits, _)) = rest.split_once('.') {
                    if let Ok(i) = digits.parse::<usize>() {
                        return (i, n);
                    }
                }
            }
        }
        (counter, n)
    }
}

/// llama.cpp's `weight_name_comparer` (`llama-model-loader.h:53`): the
/// order `llama_model_loader::weights_map` puts tensors in, which is
/// the order the quantizer walks them.
///
/// Layer number parsed out of `blk.%d.` first, then the whole name.
/// `sscanf` leaves the layer at its initialiser of `-1` when the prefix
/// does not match, so every non-`blk.` tensor sorts BEFORE layer 0, in
/// name order. It is a numeric sort, not a lexicographic one: `blk.10`
/// comes after `blk.9`, which plain string ordering gets backwards, and
/// getting it backwards is invisible until a model has ten layers.
fn llama_cpp_weight_order(a: &str, b: &str) -> std::cmp::Ordering {
    fn layer(name: &str) -> i64 {
        name.strip_prefix("blk.")
            .and_then(|rest| rest.split_once('.'))
            .and_then(|(digits, _)| digits.parse::<i64>().ok())
            .unwrap_or(-1)
    }
    layer(a).cmp(&layer(b)).then_with(|| a.cmp(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(n_layer: usize) -> ModelShape {
        ModelShape {
            n_layer,
            n_expert: 0,
            is_falcon: false,
            is_70b: false,
        }
    }

    fn names(n_layer: usize, fused_qkv: bool) -> Vec<String> {
        let mut v = vec!["token_embd.weight".to_string()];
        for i in 0..n_layer {
            if fused_qkv {
                v.push(format!("blk.{i}.attn_qkv.weight"));
            } else {
                v.push(format!("blk.{i}.attn_q.weight"));
                v.push(format!("blk.{i}.attn_k.weight"));
                v.push(format!("blk.{i}.attn_v.weight"));
            }
            v.push(format!("blk.{i}.attn_output.weight"));
            v.push(format!("blk.{i}.ffn_gate.weight"));
            v.push(format!("blk.{i}.ffn_up.weight"));
            v.push(format!("blk.{i}.ffn_down.weight"));
        }
        v.push("output.weight".to_string());
        v
    }

    /// Walks a synthetic tensor list the way the quantizer does and
    /// returns every tensor's chosen type.
    fn walk(target: Target, sh: ModelShape, names: &[String]) -> Vec<(String, GgmlType)> {
        let mut r = Recipe::new(target, sh, names);
        names
            .iter()
            .map(|n| (n.clone(), r.tensor_type(n, &[4096, 4096])))
            .collect()
    }

    /// `use_more_bits` is C's integer arithmetic, and getting it wrong
    /// is invisible on a small model and wrong on a large one. These
    /// are the values upstream's expression produces for n = 32.
    #[test]
    fn use_more_bits_is_llama_cpps_integer_arithmetic() {
        let n = 32;
        let got: Vec<usize> = (0..n).filter(|&i| When::UseMoreBits.holds(i, n)).collect();
        // i < 4, or i >= 28, or (i - 4) % 3 == 2 -> 6, 9, 12, ...
        assert_eq!(
            got,
            vec![0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 28, 29, 30, 31]
        );
        // n = 0 must not divide by zero or panic on the subtraction.
        assert!(!When::UseMoreBits.holds(0, 0));
    }

    /// The output head is the promotion everyone notices, because it
    /// is the one that makes a "Q4_K_M" file bigger than uniform Q4_K.
    #[test]
    fn every_k_quant_mix_sends_the_output_head_to_q6_k() {
        for t in [
            Target::Q4_K_S,
            Target::Q4_K_M,
            Target::Q5_K_S,
            Target::Q5_K_M,
        ] {
            let ns = names(4, false);
            let got = walk(t, shape(4), &ns);
            let out = got.iter().find(|(n, _)| n == "output.weight").unwrap();
            assert_eq!(out.1, GgmlType::Q6K, "{t:?}");
            // And token_embd is NOT promoted when a real output head
            // exists -- it takes the default.
            let emb = got.iter().find(|(n, _)| n == "token_embd.weight").unwrap();
            assert_eq!(emb.1, t.ggml_type(), "{t:?} token_embd");
        }
        // Q6_K's own head is already Q6_K.
        let ns = names(4, false);
        let got = walk(Target::Q6_K, shape(4), &ns);
        assert_eq!(
            got.iter().find(|(n, _)| n == "output.weight").unwrap().1,
            GgmlType::Q6K
        );
    }

    /// With tied embeddings there is no `output.weight`, and
    /// `token_embd.weight` inherits the output head's type. A recipe
    /// that missed this would leave a Q4_K embedding table where
    /// llama.cpp writes Q6_K -- on every modern small model, since
    /// tied embeddings are the norm there.
    #[test]
    fn a_tied_embedding_table_gets_the_output_heads_type() {
        let ns: Vec<String> = names(4, false)
            .into_iter()
            .filter(|n| n != "output.weight")
            .collect();
        let got = walk(Target::Q4_K_M, shape(4), &ns);
        assert_eq!(
            got.iter()
                .find(|(n, _)| n == "token_embd.weight")
                .unwrap()
                .1,
            GgmlType::Q6K
        );
    }

    /// The per-layer half of the mix: `attn_v` and `ffn_down` are
    /// promoted on the layers `use_more_bits` selects and left at the
    /// default elsewhere. A recipe that promoted all of them or none of
    /// them would still produce a loadable file of the wrong size.
    #[test]
    fn q4_k_m_promotes_attn_v_and_ffn_down_on_exactly_the_use_more_bits_layers() {
        let n = 16;
        let ns = names(n, false);
        let got = walk(Target::Q4_K_M, shape(n), &ns);
        for (role, want_hits) in [("attn_v", true), ("ffn_down", true)] {
            let hits: Vec<usize> = got
                .iter()
                .filter(|(name, _)| name.contains(role))
                .enumerate()
                .filter(|(_, (_, t))| *t == GgmlType::Q6K)
                .map(|(i, _)| i)
                .collect();
            let want: Vec<usize> = (0..n).filter(|&i| When::UseMoreBits.holds(i, n)).collect();
            assert_eq!(hits, want, "{role}");
            assert!(want_hits && !hits.is_empty());
        }
        // attn_q and attn_k are untouched by this mix.
        for (name, t) in &got {
            if name.contains("attn_q.weight") || name.contains("attn_k.weight") {
                assert_eq!(*t, GgmlType::Q4K, "{name}");
            }
        }
    }

    /// Q4_K_S promotes a fixed prefix, not a `use_more_bits` pattern:
    /// the first four `attn_v` and the first eighth of `ffn_down`. Two
    /// different conditions on two roles in one mix, which is why
    /// `When` is per row rather than per mix.
    #[test]
    fn q4_k_s_promotes_a_prefix_where_q4_k_m_promotes_a_pattern() {
        let n = 16;
        let ns = names(n, false);
        let got = walk(Target::Q4_K_S, shape(n), &ns);
        let v: Vec<GgmlType> = got
            .iter()
            .filter(|(name, _)| name.contains("attn_v"))
            .map(|(_, t)| *t)
            .collect();
        assert_eq!(&v[..4], &[GgmlType::Q5K; 4]);
        assert!(v[4..].iter().all(|t| *t == GgmlType::Q4K));
        let d: Vec<GgmlType> = got
            .iter()
            .filter(|(name, _)| name.contains("ffn_down"))
            .map(|(_, t)| *t)
            .collect();
        assert_eq!(&d[..2], &[GgmlType::Q5K; 2]); // n/8 == 2
        assert!(d[2..].iter().all(|t| *t == GgmlType::Q4K));
    }

    /// A fused-QKV checkpoint takes the `attn_qkv` arm, which is a flat
    /// promotion with no layer condition -- and, separately, its
    /// `attn_qkv` tensors are what `n_attention_wv` counted. Both
    /// halves are upstream's and they are not consistent with each
    /// other; reproducing that is the job.
    #[test]
    fn a_fused_qkv_checkpoint_promotes_every_layer_and_still_counts_them() {
        let n = 8;
        let ns = names(n, true);
        let got = walk(Target::Q4_K_M, shape(n), &ns);
        let q: Vec<GgmlType> = got
            .iter()
            .filter(|(name, _)| name.contains("attn_qkv"))
            .map(|(_, t)| *t)
            .collect();
        assert_eq!(q, vec![GgmlType::Q5K; n]);
        // Q5_K_M sends the same tensors to Q6_K.
        let got5 = walk(Target::Q5_K_M, shape(n), &ns);
        assert!(got5
            .iter()
            .filter(|(name, _)| name.contains("attn_qkv"))
            .all(|(_, t)| *t == GgmlType::Q6K));
        // And the counter saw them: `n_attention_wv` is 8, even though
        // no tensor ever increments `i_attention_wv` on this file.
        let r = Recipe::new(Target::Q4_K_M, shape(n), &ns);
        assert_eq!(r.n_attention_wv, n);
    }

    /// The three post-chain overrides, each of which must beat the row
    /// the chain picked. Encoding them as rows would have let the chain
    /// win instead, and on a Mixtral that is the difference between a
    /// Q8_0 and a Q6_K `attn_v`.
    #[test]
    fn the_post_chain_overrides_beat_the_chain_row_they_follow() {
        let n = 16;
        let ns = names(n, false);

        // 8 experts: attn_v and attn_k go to Q8_0 even on a layer where
        // `use_more_bits` had already promoted attn_v to Q6_K.
        let moe = ModelShape {
            n_expert: 8,
            ..shape(n)
        };
        let got = walk(Target::Q4_K_M, moe, &ns);
        assert!(got
            .iter()
            .filter(|(name, _)| name.contains("attn_v") || name.contains("attn_k"))
            .all(|(_, t)| *t == GgmlType::Q8_0));
        // attn_output picks up the 8-expert Q5_K arm.
        assert!(got
            .iter()
            .filter(|(name, _)| name.contains("attn_output"))
            .all(|(_, t)| *t == GgmlType::Q5K));

        // 70B: an attn_v the chain left at Q4_K becomes Q5_K, and one
        // the chain promoted to Q6_K stays Q6_K.
        let big = ModelShape {
            is_70b: true,
            ..shape(n)
        };
        let got = walk(Target::Q4_K_M, big, &ns);
        let v: Vec<GgmlType> = got
            .iter()
            .filter(|(name, _)| name.contains("attn_v"))
            .map(|(_, t)| *t)
            .collect();
        for (i, t) in v.iter().enumerate() {
            let want = if When::UseMoreBits.holds(i, n) {
                GgmlType::Q6K
            } else {
                GgmlType::Q5K
            };
            assert_eq!(*t, want, "layer {i}");
        }

        // Falcon: the output head goes to Q8_0, not Q6_K.
        let falcon = ModelShape {
            is_falcon: true,
            ..shape(n)
        };
        let got = walk(Target::Q4_K_M, falcon, &ns);
        assert_eq!(
            got.iter().find(|(n, _)| n == "output.weight").unwrap().1,
            GgmlType::Q8_0
        );
    }

    /// An output head whose row length is not a whole number of the
    /// default type's blocks goes to Q8_0, whose blocks are 32 wide.
    /// This is the arm that fires on a vocabulary that is not a
    /// multiple of 256, and it is above the Q6_K arm.
    #[test]
    fn an_output_head_with_an_awkward_row_length_goes_to_q8_0() {
        let ns = names(2, false);
        let mut r = Recipe::new(Target::Q4_K_M, shape(2), &ns);
        assert_eq!(
            r.tensor_type("output.weight", &[4096 + 32, 100]),
            GgmlType::Q8_0
        );
        let mut r = Recipe::new(Target::Q4_K_M, shape(2), &ns);
        assert_eq!(r.tensor_type("output.weight", &[4096, 100]), GgmlType::Q6K);
    }

    /// A gate that reads as coverage: every type the table can produce
    /// must be one this build can actually encode. Adding a row naming
    /// a type with no encoder would otherwise produce a plan that
    /// refuses only once a tensor reaches the encoder dispatch, halfway
    /// through writing a file.
    #[test]
    fn the_recipe_only_promotes_to_types_ferrox_can_encode() {
        let writable = [GgmlType::Q8_0, GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q6K];
        for p in PROMOTIONS {
            assert!(
                writable.contains(&p.to),
                "{:?}/{:?} promotes to {:?}, which has no ferrox encoder",
                p.role,
                p.ftype,
                p.to
            );
        }
        assert!(writable.contains(&FALCON_Q4KM_FFN_DOWN_SIXTEENTH));
        // And the post-chain overrides, which are code rather than rows.
        for t in [GgmlType::Q5K, GgmlType::Q8_0, GgmlType::Q6K] {
            assert!(writable.contains(&t));
        }
    }

    /// Every mix in the table is a mix `parse_target` admits, and the
    /// two mixes with NO rows are the two that genuinely have none.
    ///
    /// The second half is the one that matters: a target added to the
    /// CLI without rows here would silently get a uniform file under a
    /// mix's name, which is the exact failure this module exists to
    /// close. Naming the two exceptions explicitly means adding a
    /// seventh target turns this red rather than passing quietly.
    #[test]
    fn the_table_and_the_target_list_cover_each_other() {
        for p in PROMOTIONS {
            assert!(Target::ALL.contains(&p.ftype), "{:?}", p.ftype);
        }
        let no_rows: Vec<&'static str> = Target::ALL
            .iter()
            .filter(|t| !PROMOTIONS.iter().any(|p| p.ftype == **t))
            .map(|t| t.name())
            .collect();
        // Q8_0: `llama_tensor_get_type`'s output arm reaches Q8_0 and
        // assigns Q8_0, because the promotion is `else if (new_type !=
        // GGML_TYPE_Q8_0)`. Q6_K: same arm, same reason, and no
        // per-layer arm mentions Q6_K as an ftype at all. Both still
        // pass through `tensor_type`, which is where their shape and
        // 8-expert overrides live.
        assert_eq!(no_rows, vec!["Q8_0", "Q6_K"]);
    }

    /// `Role::of` preserves upstream's chain order. `attn_qkv.weight`
    /// contains neither `attn_v.weight` nor `attn_q.weight` as a
    /// substring, but `blk.0.ffn_gate_shexp.weight` DOES contain
    /// `ffn_gate`, and a role table sorted differently would put
    /// `ffn_down_shexp` in the wrong arm.
    #[test]
    fn tensor_names_land_in_llama_cpps_arms() {
        let cases: &[(&str, Role)] = &[
            ("output.weight", Role::Output),
            ("token_embd.weight", Role::TokenEmbd),
            ("blk.0.attn_v.weight", Role::AttnV),
            ("blk.0.attn_k.weight", Role::AttnK),
            ("blk.0.attn_q.weight", Role::AttnQ),
            ("blk.0.attn_qkv.weight", Role::AttnQkv),
            ("blk.0.attn_output.weight", Role::AttnOutput),
            ("blk.0.ffn_down.weight", Role::FfnDown),
            ("blk.0.ffn_down_exps.weight", Role::FfnDown),
            ("blk.0.ffn_down_shexp.weight", Role::FfnDown),
            ("blk.0.ffn_gate.weight", Role::FfnGate),
            ("blk.0.ffn_up.weight", Role::FfnUp),
            ("blk.0.attn_norm.weight", Role::Other),
        ];
        for (name, want) in cases {
            assert_eq!(Role::of(name, true), *want, "{name}");
        }
        // Without an output.weight in the file, token_embd IS the head.
        assert_eq!(Role::of("token_embd.weight", false), Role::Output);
    }

    /// llama.cpp walks tensors in `weight_name_comparer` order, and the
    /// counters that decide the mix advance in that order. A GGUF from
    /// `convert_hf_to_gguf.py` is NOT in it.
    ///
    /// The numeric layer sort is the half that bites: plain string
    /// ordering puts `blk.10` before `blk.2`, which is invisible on a
    /// model with fewer than ten layers and wrong on every real one.
    #[test]
    fn tensors_are_walked_in_llama_cpps_weights_map_order_not_the_files() {
        let mut names = vec![
            "blk.10.ffn_down.weight",
            "token_embd.weight",
            "blk.2.attn_v.weight",
            "output_norm.weight",
            "blk.2.attn_k.weight",
            "blk.1.ffn_down.weight",
        ];
        names.sort_by(|a, b| llama_cpp_weight_order(a, b));
        assert_eq!(
            names,
            vec![
                // Everything without a `blk.N.` prefix parses as layer
                // -1 and sorts first, in name order.
                "output_norm.weight",
                "token_embd.weight",
                "blk.1.ffn_down.weight",
                // Within a layer, by name.
                "blk.2.attn_k.weight",
                "blk.2.attn_v.weight",
                // And 10 after 2, which is the whole point.
                "blk.10.ffn_down.weight",
            ]
        );
    }

    /// `resolve_all` gives the same answer whatever order the tensors
    /// arrive in, which is what makes the planner's own loop order
    /// irrelevant. Before it existed the planner walked the file and
    /// twelve of Llama-3.2-1B's tensors came out with a different type
    /// from `llama-quantize`'s -- a file that loaded and answered.
    #[test]
    fn resolve_all_is_independent_of_the_order_it_is_handed_the_tensors() {
        let n = 16;
        let tensors: Vec<(String, Vec<u64>)> = names(n, false)
            .into_iter()
            .map(|s| (s, vec![4096u64, 4096]))
            .collect();
        let forward = Recipe::resolve_all(Target::Q4_K_M, shape(n), &tensors, |_, _| true);

        let mut reversed = tensors.clone();
        reversed.reverse();
        assert_eq!(
            forward,
            Recipe::resolve_all(Target::Q4_K_M, shape(n), &reversed, |_, _| true)
        );

        // A lexicographic shuffle is the one that actually moved
        // llama.cpp's answers, because it interleaves blk.1x with blk.1.
        let mut lexicographic = tensors.clone();
        lexicographic.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            forward,
            Recipe::resolve_all(Target::Q4_K_M, shape(n), &lexicographic, |_, _| true)
        );

        // And it is not a constant map: if it were, order-independence
        // would be trivial.
        let distinct: std::collections::BTreeSet<_> =
            forward.values().map(|t| format!("{t:?}")).collect();
        assert!(distinct.len() > 1, "{distinct:?}");
    }

    /// On a MoE checkpoint the layer index is parsed from the name, not
    /// taken from the counter, because Mixtral's expert tensors are not
    /// in layer order. Feeding them out of order must still give each
    /// its own layer's answer.
    #[test]
    fn a_moe_checkpoints_ffn_layer_comes_from_the_name_not_the_counter() {
        let n = 16;
        let mut ns = names(n, false);
        ns.retain(|s| !s.contains("ffn_down"));
        let moe = ModelShape {
            n_expert: 4,
            ..shape(n)
        };
        let mut r = Recipe::new(Target::Q4_K_M, moe, &ns);
        // Layer 5 first: `use_more_bits(5, 16)` is false, while the
        // counter would say layer 0, which is true. So a recipe reading
        // the counter here answers Q6_K and this answers Q4_K.
        assert_eq!(
            r.tensor_type("blk.5.ffn_down.weight", &[4096, 4096]),
            GgmlType::Q4K
        );
        assert_eq!(
            r.tensor_type("blk.0.ffn_down.weight", &[4096, 4096]),
            GgmlType::Q6K
        );
    }
}
