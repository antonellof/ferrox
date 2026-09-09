"""Independent numpy reference for the gemma fixture, run twice:
once with exact tanh-GELU and once with ggml's f16 lookup-table GELU.

Answers one question: is the ~4e-5 gap between ferrox and llama.cpp on
this fixture the GELU table (GGML_GELU_FP16, ggml/src/ggml-cpu/vec.h:46)
or something structural?
"""

import sys

import numpy as np

sys.path.insert(0, "/Users/d695663/Desktop/Dev/ferrox/.scratch/llama.cpp/gguf-py")
import gguf  # noqa: E402

PATH = sys.argv[1]
GOLDEN = sys.argv[2]
TOKS = [3, 7, 11, 19, 23, 5]

r = gguf.GGUFReader(PATH)
T = {t.name: np.array(t.data, dtype=np.float32) for t in r.tensors}
S = {t.name: list(reversed(t.shape.tolist())) for t in r.tensors}


def w(name):
    return T[name].reshape(S[name])


N_EMBD, N_HEAD, N_KV, HD, N_FF, N_LAYER = 24, 4, 2, 8, 40, 2
EPS = 1e-5
BASE = 10000.0


def rms(x, g):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * g


def gelu_exact(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def gelu_f16_table(x):
    # ggml: index the table by the f16 bits of x, table holds f16(gelu(f16(x))).
    xh = x.astype(np.float16).astype(np.float32)
    return gelu_exact(xh).astype(np.float16).astype(np.float32)


def rope_neox(v, pos):
    # v: [n_head, head_dim]
    out = v.copy()
    half = HD // 2
    for i in range(half):
        theta = pos * BASE ** (-2.0 * i / HD)
        c, s = np.cos(theta), np.sin(theta)
        a, b = v[:, i].copy(), v[:, i + half].copy()
        out[:, i] = a * c - b * s
        out[:, i + half] = a * s + b * c
    return out


def run(gelu):
    emb = w("token_embd.weight")
    x = emb[TOKS] * np.sqrt(N_EMBD)  # gemma.cpp:49
    n = len(TOKS)
    for il in range(N_LAYER):
        p = f"blk.{il}."
        h = rms(x, w(p + "attn_norm.weight"))
        q = (h @ w(p + "attn_q.weight").T).reshape(n, N_HEAD, HD)
        k = (h @ w(p + "attn_k.weight").T).reshape(n, N_KV, HD)
        v = (h @ w(p + "attn_v.weight").T).reshape(n, N_KV, HD)
        q = np.stack([rope_neox(q[t], t) for t in range(n)])
        k = np.stack([rope_neox(k[t], t) for t in range(n)])
        q = q / np.sqrt(HD)  # gemma.cpp:86, then kq_scale = 1.0 at :91
        att = np.zeros((n, N_HEAD, HD), dtype=np.float32)
        rep = N_HEAD // N_KV
        for hh in range(N_HEAD):
            kv = hh // rep
            sc = q[:, hh, :] @ k[:, kv, :].T
            mask = np.triu(np.ones((n, n), dtype=bool), 1)
            sc = np.where(mask, -np.inf, sc)
            sc = sc - sc.max(-1, keepdims=True)
            e = np.exp(sc)
            att[:, hh, :] = (e / e.sum(-1, keepdims=True)) @ v[:, kv, :]
        x = x + att.reshape(n, N_HEAD * HD) @ w(p + "attn_output.weight").T
        h2 = rms(x, w(p + "ffn_norm.weight"))
        g = h2 @ w(p + "ffn_gate.weight").T
        u = h2 @ w(p + "ffn_up.weight").T
        x = x + (gelu(g) * u) @ w(p + "ffn_down.weight").T
    x = rms(x, w("output_norm.weight"))
    return x[-1] @ emb.T  # tied lm_head, gemma.cpp:20


golden = np.array([float(v) for v in open(GOLDEN)], dtype=np.float32)
for name, fn in [("exact tanh-gelu", gelu_exact), ("ggml f16-table gelu", gelu_f16_table)]:
    got = run(fn)
    print(f"{name:22s} max|numpy - llama.cpp| = {np.abs(got - golden).max():.3e}")
