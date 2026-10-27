"""Company Policy Assistant - compares three ways of answering policy questions.

Tab 1: ask a question and see rules-based search, an LLM without a vector index and
       an LLM with a vector index side by side (answer, policy, time, tokens).
Tab 2: benchmark of 30 questions (answerable, partly answerable, unanswerable).
Tab 3: the policy database.
The Gemini key comes from Streamlit secrets / environment, never from the code.
"""
import os
import re

import pandas as pd
import streamlit as st

import policy_core as pc

st.set_page_config(page_title="Company Policy Assistant", page_icon="📘", layout="wide")


def secret(name):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


@st.cache_resource
def resources():
    policies = pc.load_policies()
    index = pc.load_index() if pc.INDEX_JSON.exists() else None
    return policies, pc.RulesSearch(policies), index


@st.cache_resource
def client():
    key = secret("GEMINI_API_KEY")
    return pc._client(key) if key else None


policies, rules, index = resources()
policy_text = {p["title"]: p["policy_text"] for p in policies}
all_numbers = set(re.findall(r"\d+(?:\.\d+)?", " ".join(policy_text.values())))
LABEL = {"rules": "Rules-based search", "llm": "LLM, no vector index", "rag": "LLM + vector index"}
EXPLAIN = {
    "rules": "Keyword (BM25) search over policy titles and texts; returns the best policy's text.",
    "llm": f"All {len(policies)} policies are put in the prompt; Gemini answers from them.",
    "rag": f"The question is embedded; the {pc.TOP_K} most similar policies go in the prompt.",
}


def support_check(ans, question):
    """Quick, visible checks of whether the answer is backed by the policy database."""
    notes = []
    if ans.policy.upper() == pc.NO_POLICY:
        return "No policy cited - the method says the database does not cover this."
    if ans.policy not in policy_text:
        return f"⚠️ Cites '{ans.policy}', which is not in the policy database."
    extra = set(re.findall(r"\d+(?:\.\d+)?", ans.answer)) - all_numbers - \
        set(re.findall(r"\d+(?:\.\d+)?", question))
    if extra:
        notes.append(f"⚠️ Mentions {', '.join(sorted(extra))}, which appears in no policy.")
    notes.append(f"Source text: “{policy_text[ans.policy]}”")
    return " ".join(notes)


st.title("Company Policy Assistant")
st.caption("Three ways to answer an employee's policy question, compared on the same 98-policy database.")
tab_ask, tab_bench, tab_db = st.tabs(["Compare the three methods", "Benchmark (30 questions)", "Policy database"])

with tab_ask:
    examples = ["Can I work from Spain for a few weeks?", "How long is maternity leave?",
                "How many unused vacation days can I carry over?", "Can I bring my dog to the office?",
                "Do I need approval to buy a $700 monitor?"]
    pick = st.selectbox("Example questions", ["—"] + examples)
    q = st.text_input("Your policy question", value="" if pick == "—" else pick,
                      placeholder="e.g. How many days a week can I work from home?")
    if st.button("Ask all three", type="primary") and q.strip():
        cl = client()
        cols = st.columns(3)
        for col, m in zip(cols, ("rules", "llm", "rag")):
            with col:
                st.subheader(LABEL[m])
                st.caption(EXPLAIN[m])
                try:
                    if m == "rules":
                        a = rules.answer(q)
                    elif cl is None:
                        st.error("GEMINI_API_KEY is not configured in the app secrets.")
                        continue
                    elif m == "llm":
                        a = pc.llm_full(q, policies, cl)
                    else:
                        if index is None:
                            st.error("Vector index missing - run build_index.py.")
                            continue
                        a = pc.llm_rag(q, policies, index, cl)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed: {exc}")
                    continue
                st.markdown(f"**Answer:** {a.answer}")
                st.markdown(f"**Relevant policy:** {a.policy}")
                c1, c2 = st.columns(2)
                c1.metric("Response time", f"{a.seconds:.2f} s")
                c2.metric("Tokens", f"{a.input_tokens + a.output_tokens:,}" if m != "rules" else "0")
                if m != "rules":
                    pin, pout = pc.PRICES.get(a.model, (0, 0))
                    cost = (a.input_tokens * pin + a.output_tokens * pout) / 1e6
                    st.caption(f"{a.input_tokens:,} in / {a.output_tokens:,} out · {a.model} · "
                               f"≈ ${cost:.5f} at paid-tier prices (free tier: $0)"
                               + (" · +1 embedding call" if m == "rag" else ""))
                st.info(support_check(a, q))
                if a.retrieved:
                    st.caption("Retrieved: " + "; ".join(f"{r['title']} ({r['score']})" for r in a.retrieved))

with tab_bench:
    path = pc.HERE / "data" / "benchmark_results.csv"
    if not path.exists():
        st.info("Benchmark results not generated yet (run benchmark.py).")
    else:
        df = pd.read_csv(path)
        df["unsupported"] = pd.to_numeric(df["unsupported"], errors="coerce")
        df["tokens"] = df.input_tokens + df.output_tokens
        g = df.groupby("method")
        summary = pd.DataFrame({
            "Correct policy (all 30)": g.policy_correct.mean().map("{:.0%}".format),
            "Correct on answerable (20)": df[df.type == "answerable"].groupby("method").policy_correct.mean().map("{:.0%}".format),
            "Abstained on unanswerable (6)": (1 - df[df.type == "unanswerable"].groupby("method").answered_unanswerable.mean()).map("{:.0%}".format),
            "Unsupported answers (reviewed)": g.unsupported.mean().map(lambda v: "—" if pd.isna(v) else f"{v:.0%}"),
            "Median response time (s)": g.seconds.median().round(2),
            "Mean response time (s)": g.seconds.mean().round(2),
            "Mean tokens per question": g.tokens.mean().round(0).astype(int),
        }).reindex(["rules", "llm", "rag"]).rename(index=LABEL)
        st.dataframe(summary, width="stretch")
        st.caption("'Unsupported' = the answer states something no policy says (e.g. an invented limit or "
                   "rate) or answers with an unrelated policy. Each of the 90 answers was reviewed against "
                   "the policy text (see review_note). Mean times include free-tier retries after 503/429 "
                   "errors, so the median is the typical time. Free-tier model: gemini-3.1-flash-lite.")
        st.dataframe(df[["id", "type", "question", "method", "returned_policy", "policy_correct",
                         "unsupported", "review_note", "answer", "seconds", "tokens"]], width="stretch",
                     hide_index=True)

with tab_db:
    st.dataframe(pd.DataFrame(policies), width="stretch", hide_index=True)
