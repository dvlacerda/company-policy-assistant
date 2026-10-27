"""Offline tests of the policy assistant with a fake Gemini client (no network)."""
import json, sys, types
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import policy_core as pc


class FakeModels:
    def embed_content(self, model, contents, config):
        vecs = []
        for t in contents:
            v = np.zeros(pc.EMBED_DIM)
            for w in pc.tokens(t):
                v[hash(w) % pc.EMBED_DIM] += 1
            vecs.append(types.SimpleNamespace(values=list(v + 1e-6)))
        return types.SimpleNamespace(embeddings=vecs)

    def generate_content(self, model, contents, config):
        q = contents.split("Employee question:")[-1]
        listed = [l for l in contents.splitlines() if l.startswith("- ") and "): " in l]
        title = listed[0][2:].split(" (")[0] if listed else "NONE"
        usage = types.SimpleNamespace(prompt_token_count=len(contents) // 4, candidates_token_count=30,
                                      thoughts_token_count=0)
        return types.SimpleNamespace(text=json.dumps({"answer": f"fake answer to{q}", "policy": title}),
                                     usage_metadata=usage)


def fake_client():
    return types.SimpleNamespace(models=FakeModels())


def test_types_shim(monkeypatch):
    gt = types.ModuleType("google.genai.types")
    gt.EmbedContentConfig = lambda **k: k
    gt.GenerateContentConfig = lambda **k: k
    genai = types.ModuleType("google.genai"); genai.types = gt
    g = types.ModuleType("google"); g.genai = genai
    monkeypatch.setitem(sys.modules, "google", g)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", gt)
    P = pc.load_policies()
    tmp = Path("/tmp/claude-0/idx.json")
    pc.build_index(fake_client(), P, tmp)
    idx = pc.load_index(tmp)
    assert idx[1].shape == (98, pc.EMBED_DIM)
    a = pc.llm_rag("How many vacation days do I get?", P, idx, fake_client())
    assert a.policy in {p["title"] for p in P} and a.input_tokens > 0 and len(a.retrieved) == 3
    b = pc.llm_full("How many vacation days do I get?", P, fake_client())
    assert b.input_tokens > a.input_tokens * 3      # full database costs far more tokens


def test_rules():
    r = pc.RulesSearch(pc.load_policies())
    assert r.answer("How often must I change my password?").policy == "Password Policy"
    assert r.answer("What is the per diem for meals?").policy == "NONE"
