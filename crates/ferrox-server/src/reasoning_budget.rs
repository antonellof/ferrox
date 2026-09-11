//! llama.cpp's `reasoning_budget_tokens`, from the wire to the sampler.
//!
//! The machine itself is
//! [`ferrox_models::sampling::reasoning_budget::ReasoningBudget`]; this
//! module is everything between a request body and that machine:
//!
//! - the wire value and its range (`-1` unrestricted, `N >= 0` a budget;
//!   `server-schema.cpp:383-385` hard-limits it to `[-1, INT32_MAX]`,
//!   `arg.cpp:3611` refuses anything below `-1`);
//! - the server default, llama.cpp's `--reasoning-budget`, which a
//!   request's `-1` (or absence) falls back to
//!   (`server-common.cpp:1128-1132`);
//! - which token sequences a checkpoint's opener and closer are, and
//!   whether the prompt already opened the block -- resolved at the one
//!   seam that holds both the tokenizer and the rendered prompt, the
//!   same seam that resolves single-token stop strings.
//!
//! Three shapes ride on [`crate::generate::GenerationParams`] as ONE
//! field, [`ReasoningBudget`], so a decode loop cannot be handed a
//! budget it does not know how to honour: `Unrestricted` is no machine
//! (llama.cpp builds none either, `sampling.cpp:316`), `Requested` is a
//! number waiting for the tokenizer and is an ERROR at the sampler, and
//! `Armed` is a plan the sampler builds its machine from. A path that
//! forgot to resolve the request would stop, not think unbounded.

use std::sync::Arc;

use ferrox_models::sampling::reasoning_budget::ReasoningBudgetPlan;
use serde::Deserialize;

use crate::policy::parser::ReasoningFormat;

/// Env var `--reasoning-budget` lowers to; llama.cpp's own is
/// `LLAMA_ARG_THINK_BUDGET`.
pub(crate) const SERVER_DEFAULT_ENV: &str = "FERROX_REASONING_BUDGET";

/// The wire value of `reasoning_budget_tokens` / `thinking_budget_tokens`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum BudgetTokens {
    /// llama.cpp's `-1`: no machine, the thought runs until the model
    /// closes it or `max_tokens` ends everything.
    Unrestricted,
    /// `N >= 0` tokens of thought after the opener; `0` forces the
    /// closer the moment the block opens (`arg.cpp:3609`).
    Tokens(u32),
}

impl BudgetTokens {
    /// llama.cpp's range: `-1` and every non-negative `int32`.
    pub(crate) fn parse(value: i64) -> Result<Self, String> {
        match value {
            -1 => Ok(BudgetTokens::Unrestricted),
            n if (0..=i64::from(i32::MAX)).contains(&n) => Ok(BudgetTokens::Tokens(n as u32)),
            n => Err(format!(
                "reasoning_budget_tokens must be -1 (unrestricted) or a token count from 0 to \
                 {}, not {n}",
                i32::MAX
            )),
        }
    }

    /// The server's `--reasoning-budget`, or unrestricted when unset.
    pub(crate) fn server_default() -> Self {
        Self::from_env(std::env::var(SERVER_DEFAULT_ENV).ok().as_deref())
    }

    fn from_env(value: Option<&str>) -> Self {
        value
            .and_then(|v| v.trim().parse::<i64>().ok())
            .and_then(|v| Self::parse(v).ok())
            .unwrap_or(BudgetTokens::Unrestricted)
    }

    /// llama.cpp's fallback (`server-common.cpp:1128-1132`): a request
    /// that said nothing, or said `-1`, gets the server's flag.
    pub(crate) fn effective(request: Option<Self>) -> Self {
        match request {
            Some(BudgetTokens::Tokens(n)) => BudgetTokens::Tokens(n),
            Some(BudgetTokens::Unrestricted) | None => Self::server_default(),
        }
    }
}

impl<'de> Deserialize<'de> for BudgetTokens {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = i64::deserialize(deserializer)?;
        BudgetTokens::parse(value).map_err(serde::de::Error::custom)
    }
}

/// The budget as it rides on a generation, in the three states the
/// module doc describes.
#[derive(Debug, Clone)]
pub enum ReasoningBudget {
    Unrestricted,
    /// The caller's `N`, not yet tokenized against a checkpoint.
    Requested(u32),
    /// Resolved: the machine is built from this on the first draw.
    Armed(Arc<ReasoningBudgetPlan>),
}

impl ReasoningBudget {
    pub(crate) fn from_tokens(tokens: BudgetTokens) -> Self {
        match tokens {
            BudgetTokens::Unrestricted => ReasoningBudget::Unrestricted,
            BudgetTokens::Tokens(n) => ReasoningBudget::Requested(n),
        }
    }

    /// The number the caller asked for, for the response cache key:
    /// the plan is derived from it and the (already keyed) model and
    /// prompt, so this is the whole of what can differ.
    pub(crate) fn key(&self) -> Option<u32> {
        match self {
            ReasoningBudget::Unrestricted => None,
            ReasoningBudget::Requested(n) => Some(*n),
            ReasoningBudget::Armed(plan) => Some(plan.budget),
        }
    }

    /// Whether the sampler has to see the whole vocabulary for this
    /// request: forcing the closer is a mask over every logit, so a
    /// backend that folds `lm_head + argmax` on device cannot serve it.
    pub(crate) fn needs_vocab_logits(&self) -> bool {
        !matches!(self, ReasoningBudget::Unrestricted)
    }

    /// Why a budget cannot be honoured for `format`, or `None` when it
    /// can. ONE rule, asked at the route (a 501 before any prompt is
    /// rendered) and again at the tokenizer seam (an error rather than a
    /// silently unbounded thought, should a path skip the route).
    ///
    /// A checkpoint with no reasoning format has no thought to bound,
    /// which is llama.cpp's own answer (`server-common.cpp:1134`: no
    /// thinking tags, no sampler); the budget is vacuous there, not
    /// dropped. A channel grammar (harmony, ATEM) has a reasoning block
    /// but no closer a mask can force one token at a time here, so it
    /// is refused BY NAME rather than served unbounded.
    pub(crate) fn unsupported_for(format: Option<ReasoningFormat>) -> Option<String> {
        let format = format?;
        if format.continuation_markers().is_some() {
            return None;
        }
        Some(format!(
            "`reasoning_budget_tokens` is not implemented for the {} reasoning format: its \
             chain of thought is a channel grammar, not a marker pair whose closer the \
             sampler can force. Bound the whole completion with `max_tokens`, or send -1.",
            format.as_str()
        ))
    }

    /// Resolve a requested budget against the checkpoint and the prompt
    /// it will decode from. `encode` is the model's tokenizer with
    /// special-token parsing on, so `<think>` is the control token and
    /// not five pieces of punctuation -- llama.cpp tokenizes its tags
    /// the same way (`server-schema.cpp:391,404`: `parse_special =
    /// true`).
    pub(crate) fn armed(
        &self,
        format: Option<ReasoningFormat>,
        prompt: &str,
        encode: impl Fn(&str) -> Vec<usize>,
    ) -> Result<Self, String> {
        let ReasoningBudget::Requested(budget) = self else {
            return Ok(self.clone());
        };
        if let Some(why) = Self::unsupported_for(format) {
            return Err(why);
        }
        let Some(format) = format else {
            return Ok(ReasoningBudget::Unrestricted);
        };
        let Some((start, end)) = format.continuation_markers() else {
            // `unsupported_for` said this family has markers.
            return Err(format!(
                "{} has no continuation markers although unsupported_for admitted it",
                format.as_str()
            ));
        };
        let closer = encode(end);
        let mut end_seqs = vec![closer.clone()];
        if let Some(tool) = format.tool_marker() {
            // A tool call can open before the block closes
            // (`common/chat.cpp:1135,2142`): it ends the thought too.
            end_seqs.push(encode(tool));
        }
        // What the template already put at and after the opener, so a
        // prompt that opened the block starts the count -- llama.cpp's
        // `prefill_tokens` (`common/sampling.cpp:283-297,324-327`).
        let prefill = if format.prompt_opens_reasoning(prompt) {
            prompt
                .rfind(start)
                .map(|at| encode(&prompt[at..]))
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        Ok(ReasoningBudget::Armed(Arc::new(ReasoningBudgetPlan {
            start_seqs: vec![encode(start)],
            end_seqs,
            forced: closer,
            budget: *budget,
            prefill,
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// llama.cpp's range, both ends: `-1` is the one negative value
    /// (`arg.cpp:3611`), and the schema's hard limit is `INT32_MAX`
    /// (`server-schema.cpp:384`).
    #[test]
    fn the_wire_value_admits_exactly_llama_cpps_range() {
        assert_eq!(BudgetTokens::parse(-1), Ok(BudgetTokens::Unrestricted));
        assert_eq!(BudgetTokens::parse(0), Ok(BudgetTokens::Tokens(0)));
        assert_eq!(BudgetTokens::parse(2000), Ok(BudgetTokens::Tokens(2000)));
        assert_eq!(
            BudgetTokens::parse(i64::from(i32::MAX)),
            Ok(BudgetTokens::Tokens(i32::MAX as u32))
        );
        for bad in [-2, i64::from(i32::MAX) + 1, i64::MIN] {
            let err = BudgetTokens::parse(bad).expect_err("out of range");
            assert!(err.contains("reasoning_budget_tokens"), "{err}");
        }
        let wire: BudgetTokens = serde_json::from_value(serde_json::json!(40)).unwrap();
        assert_eq!(wire, BudgetTokens::Tokens(40));
        assert!(serde_json::from_value::<BudgetTokens>(serde_json::json!("40")).is_err());
        assert!(serde_json::from_value::<BudgetTokens>(serde_json::json!(-2)).is_err());
    }

    /// `server-common.cpp:1128-1132`: the request's own number wins;
    /// `-1` and absence both mean "whatever the server was started
    /// with". Read through `from_env` so the test does not depend on
    /// the process environment.
    #[test]
    fn a_request_saying_nothing_or_minus_one_takes_the_server_flag() {
        assert_eq!(BudgetTokens::from_env(None), BudgetTokens::Unrestricted);
        assert_eq!(
            BudgetTokens::from_env(Some("-1")),
            BudgetTokens::Unrestricted
        );
        assert_eq!(
            BudgetTokens::from_env(Some(" 512 ")),
            BudgetTokens::Tokens(512)
        );
        assert_eq!(
            BudgetTokens::from_env(Some("nonsense")),
            BudgetTokens::Unrestricted
        );
    }

    fn encode(text: &str) -> Vec<usize> {
        // A toy tokenizer: each marker is one control token, everything
        // else one token per byte.
        match text {
            "<think>" => vec![1000],
            "</think>" => vec![1001],
            "<｜DSML｜" => vec![1002],
            other if other.starts_with("<think>") => {
                let mut v = vec![1000];
                v.extend(other["<think>".len()..].bytes().map(|b| b as usize));
                v
            }
            other => other.bytes().map(|b| b as usize).collect(),
        }
    }

    /// The plan for a marker family: the opener arms, the closer both
    /// ends and is forced, and a prompt that opened the block feeds its
    /// tail as prefill.
    #[test]
    fn a_marker_family_resolves_to_its_tokenized_opener_and_closer() {
        let budget = ReasoningBudget::Requested(40);
        let armed = budget
            .armed(
                Some(ReasoningFormat::Think),
                "<|User|>hi<|Assistant|><think>\n",
                encode,
            )
            .expect("resolves");
        let ReasoningBudget::Armed(plan) = armed else {
            panic!("expected a plan");
        };
        assert_eq!(plan.start_seqs, vec![vec![1000]]);
        assert_eq!(plan.end_seqs, vec![vec![1001]]);
        assert_eq!(plan.forced, vec![1001]);
        assert_eq!(plan.budget, 40);
        assert_eq!(plan.prefill, vec![1000, b'\n' as usize], "opener + newline");

        let closed = ReasoningBudget::Requested(40)
            .armed(Some(ReasoningFormat::Think), "<|Assistant|>", encode)
            .expect("resolves");
        let ReasoningBudget::Armed(plan) = closed else {
            panic!("expected a plan");
        };
        assert!(plan.prefill.is_empty(), "the prompt did not open the block");
    }

    /// A family whose tool opener can arrive inside the thought lists
    /// it as a second end sequence (`common/chat.cpp:2142`).
    #[test]
    fn a_tool_marker_is_a_second_end_sequence() {
        let armed = ReasoningBudget::Requested(5)
            .armed(Some(ReasoningFormat::DeepSeekV32), "<think>", encode)
            .expect("resolves");
        let ReasoningBudget::Armed(plan) = armed else {
            panic!("expected a plan");
        };
        assert_eq!(plan.end_seqs, vec![vec![1001], vec![1002]]);
        assert_eq!(plan.forced, vec![1001], "only the closer is forced");
    }

    /// No reasoning format is llama.cpp's "no thinking tags, no
    /// sampler" (`server-common.cpp:1134`); a channel grammar is a
    /// refusal by name; and the two answers come from the same rule
    /// the route asks.
    #[test]
    fn no_format_is_vacuous_and_a_channel_grammar_is_refused() {
        let plain = ReasoningBudget::Requested(5)
            .armed(None, "hello", encode)
            .expect("vacuous");
        assert!(matches!(plain, ReasoningBudget::Unrestricted));
        assert_eq!(ReasoningBudget::unsupported_for(None), None);

        for channel in [ReasoningFormat::GptOss, ReasoningFormat::MuseGlimmer] {
            let why = ReasoningBudget::unsupported_for(Some(channel)).expect("refused");
            assert!(why.contains(channel.as_str()), "{why}");
            let err = ReasoningBudget::Requested(5)
                .armed(Some(channel), "x", encode)
                .expect_err("the seam refuses what the route refuses");
            assert_eq!(err, why);
        }
        assert_eq!(
            ReasoningBudget::unsupported_for(Some(ReasoningFormat::Think)),
            None
        );
    }

    /// Unrestricted and already-armed budgets pass through the seam
    /// unchanged, so resolving twice is harmless.
    #[test]
    fn resolving_is_idempotent() {
        let none = ReasoningBudget::Unrestricted
            .armed(Some(ReasoningFormat::Think), "<think>", encode)
            .unwrap();
        assert!(matches!(none, ReasoningBudget::Unrestricted));
        let once = ReasoningBudget::Requested(3)
            .armed(Some(ReasoningFormat::Think), "<think>", encode)
            .unwrap();
        let twice = once
            .armed(Some(ReasoningFormat::Think), "different", encode)
            .unwrap();
        let (ReasoningBudget::Armed(a), ReasoningBudget::Armed(b)) = (&once, &twice) else {
            panic!("both armed");
        };
        assert_eq!(a, b);
    }

    /// The cache key sees the number and nothing else about the plan.
    #[test]
    fn the_key_is_the_requested_number() {
        assert_eq!(ReasoningBudget::Unrestricted.key(), None);
        assert_eq!(ReasoningBudget::Requested(7).key(), Some(7));
        let armed = ReasoningBudget::Requested(7)
            .armed(Some(ReasoningFormat::Think), "<think>", encode)
            .unwrap();
        assert_eq!(armed.key(), Some(7));
        assert!(!ReasoningBudget::Unrestricted.needs_vocab_logits());
        assert!(armed.needs_vocab_logits());
        assert!(ReasoningBudget::Requested(0).needs_vocab_logits());
    }
}
