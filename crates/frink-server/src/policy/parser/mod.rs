//! Turning a token stream back into an agent-shaped response.
//!
//! A coding agent does not read text: it reads a `content` field, a
//! `reasoning_content` field, and a list of tool calls with JSON
//! arguments. A model emits one undifferentiated stream of tokens with
//! markers in it. These two modules are the translation, and both of
//! them have to work *incrementally*, because the markers straddle
//! token boundaries and SSE cannot retract what it has sent.
//!
//! - [`reasoning`] cuts the chain of thought off the answer.
//! - [`tool_call`] recognizes a call in whichever of a dozen wire
//!   formats the checkpoint was trained on, and streams its arguments
//!   as prefix-stable JSON fragments.
//!
//! They compose in that order: the reasoning parser runs first and
//! hands any tool-call text through verbatim, so the tool parser sees a
//! stream with the thinking already removed.

pub mod reasoning;
pub mod tool_call;

pub use reasoning::{ReasoningFormat, ReasoningParser};
pub use tool_call::{ToolCallEvent, ToolCallFormat, ToolCallParser};
