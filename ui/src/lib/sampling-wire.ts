// The sampling panel's values, as the fields they put on a request.
//
// One function rather than a spread written inline in the runtime, so
// the rule for each knob -- which ones are omitted when unset, and what
// "unset" means on the wire -- is stated once and can be pinned without
// a browser. `reasoning_budget_tokens` is the knob that made this worth
// extracting: its "unset" is not "omit the field" in spirit but llama.cpp's
// `-1`, and a panel that sent `0` for an empty box would be asking the
// model to close its thought the moment it opened it.

/** The per-request knobs the panel holds; `null` is "let the server decide". */
export type WireKnobs = {
  temperature: number;
  topP: number;
  /** `null` sends no `max_tokens`: the context is the only limit. */
  maxTokens: number | null;
  /**
   * llama.cpp's `reasoning_budget_tokens`: tokens of thinking allowed
   * before the server forces the closing tag so the answer still comes.
   * `null` sends nothing, which the server reads as its own default
   * (`--reasoning-budget`, itself `-1`: unrestricted). `0` is a real
   * value -- end the thought the moment it opens.
   */
  reasoningBudget: number | null;
};

/** The request fields these knobs contribute. */
export type WireFields = {
  temperature: number;
  top_p: number;
  max_tokens?: number;
  reasoning_budget_tokens?: number;
};

export function samplingToWire(knobs: WireKnobs): WireFields {
  return {
    temperature: knobs.temperature,
    top_p: knobs.topP,
    ...(knobs.maxTokens !== null ? { max_tokens: knobs.maxTokens } : {}),
    ...(knobs.reasoningBudget !== null
      ? { reasoning_budget_tokens: knobs.reasoningBudget }
      : {}),
  };
}

/**
 * What the budget box's text means. Empty is unrestricted; `-1` is
 * llama.cpp's own spelling of the same and is folded onto it rather
 * than sent, so the panel shows one idea one way; a negative below
 * that is not a budget and stays as the last valid value.
 */
export function parseReasoningBudget(
  text: string,
  previous: number | null,
): number | null {
  if (text.trim() === "") return null;
  const n = Math.round(Number(text));
  if (!Number.isFinite(n)) return previous;
  if (n === -1) return null;
  if (n < 0) return previous;
  return n;
}
