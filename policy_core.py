"""Company policy assistant - three ways to answer a policy question.

1. rules   : keyword search (BM25) over policy titles and texts; the answer is the
             text of the best-matching policy, or "no policy found" below a threshold.
2. llm     : an LLM WITHOUT a vector index - every policy is put in the prompt.
3. rag     : an LLM WITH a vector index - the question is embedded, the 3 closest
             policies are retrieved by cosine similarity and only they go in the prompt.

Both LLM methods use the same generation model and the same answering rules, so the
only difference is how the policies reach the model.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
POLICY_CSV = HERE / "data" / "company_policies.csv"
INDEX_JSON = HERE / "data" / "policy_index.json"

GEN_MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-flash-lite-latest"]
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
TOP_K = 3
NO_POLICY = "NONE"
# USD per 1M tokens (paid tier) - https://ai.google.dev/gemini-api/docs/pricing, checked 27 Oct 2026
PRICES = {"gemini-3.1-flash-lite": (0.25, 1.50), "gemini-3.5-flash-lite": (0.30, 2.50),
          "gemini-flash-lite-latest": (0.10, 0.40)}


@dataclass
class Answer:
    method: str
    answer: str
    policy: str                 # title of the policy relied on, or NONE
    seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_calls: int = 0
    model: str = ""
    retrieved: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


def load_policies(path=POLICY_CSV):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def canonical_policy(raw, policies):
    """Map the model's 'policy' field to exact policy titles.

    The models often echo the title with its category ("Vacation Policy (Human Resources)") or
    name two policies; this returns the known titles found in the text, joined by " | ",
    or NONE. Unknown text is returned unchanged so it shows up as a mismatch."""
    raw = (raw or "").strip()
    if not raw or raw.upper().startswith(NO_POLICY):
        return NO_POLICY
    found = []
    for t in sorted({p["title"] for p in policies}, key=len, reverse=True):
        i = raw.lower().find(t.lower())
        if i >= 0 and not any(t.lower() in f.lower() for _, f in found):
            found.append((i, t))
    return " | ".join(t for _, t in sorted(found)) if found else raw


def policy_line(p):
    return f"- {p['title']} ({p['category']}): {p['policy_text']}"


# ----------------------------------------------------------------------------- 1. rules
STOP = set("""a an the and or of to in on for with at by from is are be been was were am i my me we our you
your it its this that these those can may must should do does did how what when where which who whom why
will would could there their they them as if not no any all about into than then so up out over per
have has had get got also just only need needs want allowed allow ok okay please tell know happen
something anything someone company employee employees policy policies rule rules""".split())


def tokens(text):
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOP:
            continue
        for suf in ("ing", "ies", "es", "s", "ed"):      # tiny stemmer
            if len(w) > 4 and w.endswith(suf):
                w = w[: -len(suf)] + ("y" if suf == "ies" else "")
                break
        out.append(w)
    return out


class RulesSearch:
    """BM25 keyword search; titles are counted twice because they name the topic."""

    def __init__(self, policies, k1=1.5, b=0.75, threshold=3.0):
        self.p = policies
        self.docs = [tokens(p["title"] + " " + p["title"] + " " + p["policy_text"] + " " + p["category"])
                     for p in policies]
        self.avg = sum(map(len, self.docs)) / len(self.docs)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(d) for d in self.docs]
        self.k1, self.b, self.threshold = k1, b, threshold

    def scores(self, query):
        q = tokens(query)
        res = []
        for d, tf in zip(self.docs, self.tf):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
            res.append(s)
        return np.array(res)

    def answer(self, question):
        t0 = time.perf_counter()
        s = self.scores(question)
        i = int(s.argmax())
        secs = time.perf_counter() - t0
        if s[i] < self.threshold:
            return Answer("rules", "No matching policy was found for this question.", NO_POLICY, secs)
        p = self.p[i]
        return Answer("rules", p["policy_text"], p["title"], secs,
                      retrieved=[{"title": self.p[j]["title"], "score": round(float(s[j]), 2)}
                                 for j in np.argsort(-s)[:3]])


# ----------------------------------------------------------------------------- LLM helpers
ANSWER_RULES = """You are the company's policy assistant. Answer the employee's question using ONLY the
company policies listed below.
- If a policy answers the question, answer in one or two sentences and name that policy.
- Do not add facts, numbers, conditions or exceptions that are not written in the policy text.
  If the policy only partly answers the question, give what it says and state what it does not specify.
- If no listed policy covers the question, say that the policy database does not cover it and use
  policy "NONE".
Return JSON: {"answer": "...", "policy": "<exact policy title or NONE>"}"""


def _client(api_key):
    from google import genai
    return genai.Client(api_key=api_key)


def generate(client, prompt, models=None):
    """Call Gemini with retry + model fallback (lecture 6)."""
    from google.genai import types
    errors = []
    for model in models or GEN_MODELS:
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json",
                                                       temperature=0, max_output_tokens=400))
                data = json.loads(resp.text)
                u = resp.usage_metadata
                out_tokens = (u.candidates_token_count or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
                return data, model, u.prompt_token_count or 0, out_tokens
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{model}: {str(exc)[:120]}")
                if "404" in str(exc) or "not found" in str(exc).lower():
                    break
                time.sleep(2 * (attempt + 1))
    raise RuntimeError("Gemini failed: " + " | ".join(errors))


def llm_full(question, policies, client, models=None):
    t0 = time.perf_counter()
    prompt = f"{ANSWER_RULES}\n\nCompany policies ({len(policies)}):\n" + \
             "\n".join(policy_line(p) for p in policies) + f"\n\nEmployee question: {question}"
    data, model, tin, tout = generate(client, prompt, models)
    return Answer("llm", data.get("answer", ""), canonical_policy(data.get("policy"), policies),
                  time.perf_counter() - t0, tin, tout, 0, model)


# ----------------------------------------------------------------------------- 3. RAG
def embed(client, texts, task):
    from google.genai import types
    res = client.models.embed_content(
        model=EMBED_MODEL, contents=texts,
        config=types.EmbedContentConfig(task_type=task, output_dimensionality=EMBED_DIM))
    v = np.array([e.values for e in res.embeddings], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def build_index(client, policies, path=INDEX_JSON):
    texts = [f"{p['title']}: {p['policy_text']}" for p in policies]
    vecs = embed(client, texts, "RETRIEVAL_DOCUMENT")
    path.write_text(json.dumps({"model": EMBED_MODEL, "dim": EMBED_DIM,
                                "titles": [p["title"] for p in policies],
                                "vectors": vecs.round(6).tolist()}), encoding="utf-8")
    return path


def load_index(path=INDEX_JSON):
    d = json.loads(path.read_text(encoding="utf-8"))
    return d["titles"], np.array(d["vectors"], dtype=np.float32)


def llm_rag(question, policies, index, client, models=None, k=TOP_K):
    titles, vecs = index
    by_title = {p["title"]: p for p in policies}
    t0 = time.perf_counter()
    q = embed(client, [question], "RETRIEVAL_QUERY")[0]
    sims = vecs @ q
    top = np.argsort(-sims)[:k]
    chosen = [by_title[titles[i]] for i in top]
    prompt = f"{ANSWER_RULES}\n\nMost relevant company policies:\n" + \
             "\n".join(policy_line(p) for p in chosen) + f"\n\nEmployee question: {question}"
    data, model, tin, tout = generate(client, prompt, models)
    return Answer("rag", data.get("answer", ""), canonical_policy(data.get("policy"), policies),
                  time.perf_counter() - t0, tin, tout, 1, model,
                  retrieved=[{"title": titles[i], "score": round(float(sims[i]), 3)} for i in top])


def api_key_from_env():
    key = os.getenv("GEMINI_API_KEY")
    env = HERE / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    return key
