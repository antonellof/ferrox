#!/usr/bin/env python3
"""Regenerate the chat-template goldens in `crates/frink-models/tests/templates/`.

Every `*.jinja` in that directory is the verbatim `tokenizer.chat_template`
string of a real GGUF in `models/`, read straight out of the file's metadata.
This script renders each one through **real jinja2**, configured exactly the
way the two engines that actually render chat templates configure it, and
writes the bytes next to the template as `*.expected` / `*.system.expected`.
`crates/frink-models/tests/chat_template_real_gguf.rs` then asserts that
frink's minijinja evaluator produces those same bytes.

That is the point of the file: the goldens are produced by an independent
implementation, not by the code under test, so a regression in
`frink_models::chat_template` cannot quietly rewrite its own expectations.

## The reference configuration

HuggingFace `PreTrainedTokenizerBase._compile_jinja_template`:

    ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                  extensions=[jinja2.ext.loopcontrols])
    env.globals["raise_exception"], env.globals["strftime_now"]
    env.filters["tojson"] = json.dumps(..., ensure_ascii=False)

llama.cpp agrees on both flags -- `common/jinja/lexer.cpp:112-118` says
"default config for chat template: lstrip_blocks = true, trim_blocks = true".
Stock jinja2 defaults both to false, and so does minijinja; that mismatch is
the bug these goldens caught (TinyLlama).

## Deliberately not covered here

`tools`. jinja2's stock `tojson` sorts keys, transformers' replacement does
not, and llama.cpp's refuses `sort_keys=true` outright. frink sorts, because
`serde_json::Map` is a `BTreeMap` without the workspace-wide `preserve_order`
feature, so the author's key order is already gone by the time the filter
runs. Pinning tool goldens here would pin that disagreement as if it were
agreement. The tool-rendering path is covered by the hand-written unit tests
in `chat_template.rs` instead.

Usage:  python3 scripts/chat_template_goldens.py [--check]
"""

import json
import os
import sys

try:
    import jinja2
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment
except ImportError:  # pragma: no cover
    sys.exit("needs jinja2: pip install jinja2")

HERE = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(HERE, os.pardir, "crates", "frink-models", "tests", "templates")

# Kept byte-identical with `chat_template_real_gguf.rs`.
CONVERSATION = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "And of Italy?"},
]
SYSTEM_CONVERSATION = [
    {"role": "system", "content": "Answer with one word."},
    {"role": "user", "content": "What is the capital of France?"},
]
BOS = "<s>"
EOS = "</s>"
# Llama-3.x bakes today's date into its system header via `strftime_now`
# unless the caller defines `date_string`. Pin it so the golden is stable.
DATE_STRING = "26 Jul 2024"


def _raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                      separators=separators, sort_keys=sort_keys)


def _strftime_now(fmt):
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime(fmt)


def environment():
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
    )
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise_exception
    env.globals["strftime_now"] = _strftime_now
    return env


def render(src, messages):
    return environment().from_string(src).render(
        messages=messages,
        add_generation_prompt=True,
        bos_token=BOS,
        eos_token=EOS,
        tools=None,
        date_string=DATE_STRING,
    )


SCENARIOS = [("", CONVERSATION), (".system", SYSTEM_CONVERSATION)]


def main():
    check = "--check" in sys.argv
    names = sorted(n for n in os.listdir(TPL) if n.endswith(".jinja"))
    if not names:
        sys.exit("no templates in %s" % TPL)
    bad = []
    for name in names:
        stem = name[: -len(".jinja")]
        src = open(os.path.join(TPL, name), encoding="utf-8").read()
        for suffix, messages in SCENARIOS:
            out_path = os.path.join(TPL, stem + suffix + ".expected")
            raise_path = os.path.join(TPL, stem + suffix + ".raises")
            try:
                text = render(src, messages)
                target, other = out_path, raise_path
            except jinja2.exceptions.TemplateError as exc:
                # A template that refuses this conversation shape (Mistral
                # v0.2 rejects a system role) is itself a pinned behaviour:
                # frink must surface the refusal, never a guessed framing.
                text = str(exc)
                target, other = raise_path, out_path
            if check:
                if not os.path.exists(target) or open(target, encoding="utf-8").read() != text:
                    bad.append(target)
                continue
            if os.path.exists(other):
                os.remove(other)
            with open(target, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            print("%-40s %s (%d bytes)" % (
                stem + suffix, os.path.basename(target), len(text)))
    if bad:
        sys.exit("stale goldens:\n  " + "\n  ".join(bad))


if __name__ == "__main__":
    main()
