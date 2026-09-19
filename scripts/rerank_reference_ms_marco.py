#!/usr/bin/env python3
"""An independent NumPy reference for a BERT cross-encoder reranker.

Prints the relevance scores that `cross-encoder/ms-marco-MiniLM-L6-v2`
gives for the query/document pairs in
`crates/frink-models/tests/rerank_cross_encoder_ordering.rs`, so the
golden values that test asserts against are reproducible rather than
recorded from frink itself (which would only prove frink agrees with
frink).

It is a transcription of HuggingFace `BertForSequenceClassification`
straight from the safetensors -- no `transformers` model classes, no
torch -- because the whole point is to be a *second* implementation.
Only `numpy`, `safetensors` and `huggingface_hub` are needed.

    python3 scripts/rerank_reference_ms_marco.py

# The three variants it prints, and why they differ

    hf      pooler + segment ids 0/1        the reference the checkpoint
                                            was trained to produce
    frink  no pooler + segment ids 0/1     what frink computes today
    seg0    no pooler + all segments 0      what frink and llama.cpp
                                            both computed before #44

`frink` is missing the pooler because llama.cpp's converter DROPS
`bert.pooler.dense` -- `conversion/bert.py`, "we are only using BERT for
embeddings so we don't need the pooling layer" -- so the GGUF simply
does not carry it, and no engine can apply a tensor that is not in the
file. That flattens the score range (about +-0.2 instead of about +-11)
and it is why the test asserts on the ORDER and on the `frink` column,
not on the `hf` column.

`seg0` is the one that was wrong: it puts the RELEVANT document LAST in
three of these four rankings.

# Dumping the pooler, so Rust can reach the `hf` column too

    python3 scripts/rerank_reference_ms_marco.py --dump-pooler \
        models/ms-marco-MiniLM-L6-v2-pooler.bin

writes `bert.pooler.dense.{weight,bias}` as raw little-endian float32
behind an eight-byte magic and two u32 dimensions. frink cannot invent
that tensor, but it CAN run one that is present (issue #82, route 1),
and `crates/frink-models/tests/rerank_cross_encoder_ordering.rs`
splices this file into a head-only GGUF to prove it reproduces the `hf`
column above rather than merely running. The dump is the checkpoint's
own weights, not a re-derivation, so the Rust side is still checked
against THIS script's arithmetic and not against its own.
"""

import json
import math
import sys
import unicodedata

import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open

REPO = "cross-encoder/ms-marco-MiniLM-L6-v2"

CASES = [
    (
        "How many people live in Berlin?",
        [
            "Berlin is well known for its museums.",
            "Berlin had a population of 3,520,031 registered inhabitants in an "
            "area of 891.82 square kilometers.",
            "The capital of France is Paris.",
            "Elephants are the largest land animals.",
            "Berlin is the capital and largest city of Germany by both area and "
            "population.",
        ],
    ),
    (
        "What is the boiling point of water?",
        [
            "Water freezes at 0 degrees Celsius at sea level.",
            "At standard atmospheric pressure water boils at 100 degrees Celsius.",
            "The Pacific Ocean is the largest ocean on Earth.",
            "Coffee is usually brewed just below boiling.",
        ],
    ),
    (
        "Who wrote Romeo and Juliet?",
        [
            "Romeo and Juliet is a tragedy written by William Shakespeare early "
            "in his career.",
            "The play is set in Verona, Italy.",
            "Python is a programming language created by Guido van Rossum.",
            "Juliet is fourteen years old in the play.",
        ],
    ),
    (
        "How do I install Rust on macOS?",
        [
            "Run the rustup installer script from rustup.rs to install the Rust "
            "toolchain.",
            "Rust is a systems programming language focused on safety.",
            "macOS Ventura was released in 2022.",
            "Cargo is the Rust package manager.",
        ],
    ),
]


def load(repo):
    d = snapshot_download(repo_id=repo, allow_patterns=["*.json", "*.txt", "*.safetensors"])
    w = {}
    with safe_open(d + "/model.safetensors", framework="numpy") as f:
        for k in f.keys():
            w[k] = f.get_tensor(k).astype(np.float64)
    cfg = json.load(open(d + "/config.json"))
    vocab = [line.rstrip("\n") for line in open(d + "/vocab.txt", encoding="utf-8")]
    return w, cfg, {t: i for i, t in enumerate(vocab)}


W, CFG, TOK = load(REPO)
NL, NH, D = CFG["num_hidden_layers"], CFG["num_attention_heads"], CFG["hidden_size"]
EPS = CFG["layer_norm_eps"]
HD = D // NH


def wordpiece(text):
    """`BertTokenizer(do_lower_case=True)`: strip accents, split on
    punctuation and whitespace, then greedy longest-match wordpiece."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    words, cur = [], ""
    for ch in text:
        if ch.isspace():
            if cur:
                words.append(cur)
            cur = ""
        elif not ch.isalnum() and ch != "_":
            if cur:
                words.append(cur)
            cur = ""
            words.append(ch)
        else:
            cur += ch
    if cur:
        words.append(cur)

    ids = []
    for word in words:
        start, pieces, ok = 0, [], True
        while start < len(word):
            end, hit = len(word), None
            while start < end:
                sub = word[start:end]
                if start > 0:
                    sub = "##" + sub
                if sub in TOK:
                    hit = sub
                    break
                end -= 1
            if hit is None:
                ok = False
                break
            pieces.append(hit)
            start = end
        ids.extend(TOK[p] for p in pieces) if ok else ids.append(TOK["[UNK]"])
    return ids


def layer_norm(x, weight, bias):
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + EPS) * weight + bias


def gelu(x):
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def encoder(ids, seg):
    h = (
        W["bert.embeddings.word_embeddings.weight"][ids]
        + W["bert.embeddings.token_type_embeddings.weight"][seg]
        + W["bert.embeddings.position_embeddings.weight"][: len(ids)]
    )
    h = layer_norm(h, W["bert.embeddings.LayerNorm.weight"], W["bert.embeddings.LayerNorm.bias"])
    n = len(ids)
    for i in range(NL):
        p = f"bert.encoder.layer.{i}."
        q = h @ W[p + "attention.self.query.weight"].T + W[p + "attention.self.query.bias"]
        k = h @ W[p + "attention.self.key.weight"].T + W[p + "attention.self.key.bias"]
        v = h @ W[p + "attention.self.value.weight"].T + W[p + "attention.self.value.bias"]
        q, k, v = (t.reshape(n, NH, HD).transpose(1, 0, 2) for t in (q, k, v))
        s = q @ k.transpose(0, 2, 1) / np.sqrt(HD)
        e = np.exp(s - s.max(-1, keepdims=True))
        ctx = ((e / e.sum(-1, keepdims=True)) @ v).transpose(1, 0, 2).reshape(n, D)
        x = ctx @ W[p + "attention.output.dense.weight"].T + W[p + "attention.output.dense.bias"]
        x = layer_norm(
            x + h,
            W[p + "attention.output.LayerNorm.weight"],
            W[p + "attention.output.LayerNorm.bias"],
        )
        u = gelu(x @ W[p + "intermediate.dense.weight"].T + W[p + "intermediate.dense.bias"])
        d = u @ W[p + "output.dense.weight"].T + W[p + "output.dense.bias"]
        h = layer_norm(d + x, W[p + "output.LayerNorm.weight"], W[p + "output.LayerNorm.bias"])
    return h


def score(query, document, pooler, segment_ids):
    a, b = wordpiece(query), wordpiece(document)
    ids = [TOK["[CLS]"]] + a + [TOK["[SEP]"]] + b + [TOK["[SEP]"]]
    seg = [0] * (len(a) + 2) + [1] * (len(b) + 1) if segment_ids else [0] * len(ids)
    cls = encoder(ids, seg)[0]
    if pooler:
        cls = np.tanh(cls @ W["bert.pooler.dense.weight"].T + W["bert.pooler.dense.bias"])
    return float(cls @ W["classifier.weight"][0] + W["classifier.bias"][0])


VARIANTS = [
    ("hf    ", dict(pooler=True, segment_ids=True)),
    ("frink", dict(pooler=False, segment_ids=True)),
    ("seg0  ", dict(pooler=False, segment_ids=False)),
]

# The Rust reader checks this before anything else, so a truncated or
# unrelated file is a named failure rather than 590 KB of garbage
# floats. Bump the digits if the layout below ever changes.
POOLER_MAGIC = b"FXPOOL01"


def dump_pooler(path):
    """Write `bert.pooler.dense.{weight,bias}` for the Rust fixture.

    Layout: the magic, then the weight's `[out, in]` as two
    little-endian u32, then `out * in` float32 in C order (which is
    exactly ggml's row-major `[out][in]`, so the bytes go into a GGUF
    `cls.weight` verbatim), then `out` float32 of bias.
    """
    w = np.ascontiguousarray(W["bert.pooler.dense.weight"], dtype="<f4")
    b = np.ascontiguousarray(W["bert.pooler.dense.bias"], dtype="<f4")
    out, inp = w.shape
    if b.shape != (out,):
        raise SystemExit(f"pooler bias is {b.shape}, expected ({out},)")
    with open(path, "wb") as f:
        f.write(POOLER_MAGIC)
        f.write(np.array([out, inp], dtype="<u4").tobytes())
        f.write(w.tobytes())
        f.write(b.tobytes())
    print(f"wrote {path}: pooler.dense [{out}, {inp}] + bias [{out}]")


def main():
    if "--dump-pooler" in sys.argv:
        i = sys.argv.index("--dump-pooler")
        if i + 1 >= len(sys.argv):
            raise SystemExit("--dump-pooler needs a path")
        dump_pooler(sys.argv[i + 1])
        return 0
    for query, documents in CASES:
        print(f"\nQ: {query}")
        for label, kwargs in VARIANTS:
            scores = [score(query, d, **kwargs) for d in documents]
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            print(f"  {label}  order={order}  scores={[round(s, 6) for s in scores]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
