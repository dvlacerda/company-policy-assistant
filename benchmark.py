"""Run the 30 benchmark questions through all three methods.

Usage: GEMINI_API_KEY=... python benchmark.py      (or: python benchmark.py --rescore)
Writes data/benchmark_results.csv (one row per question x method) and prints a summary.

Scoring
- policy_correct : the policy returned matches the expected one (NONE for unanswerable)
- unsupported    : the answer states something the policy database does not support.
                   This is judged by a reviewer afterwards (column left blank here) - an
                   automatic judge would be another LLM grading LLMs. Two automatic
                   signals help the review: answered_unanswerable (gave a policy-based
                   answer to a question no policy covers) and invented_number (a number
                   in the answer that does not appear in any policy text).
"""
import csv
import re
import time

import policy_core as pc


def numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


def is_correct(got, expected):
    """Correct when every policy returned is one of the expected ones (NONE for unanswerable)."""
    returned = [g.strip() for g in got.split("|") if g.strip()]
    return bool(returned) and all(g in expected for g in returned)


def rescore():
    """Re-apply canonical policy names and scoring to saved results (no API calls)."""
    policies = pc.load_policies()
    path = pc.HERE / "data" / "benchmark_results.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        if r["method"] != "rules":
            r["returned_policy"] = pc.canonical_policy(r["returned_policy"], policies)
        r["policy_correct"] = int(is_correct(r["returned_policy"], r["expected_policy"].split("|")))
        r["answered_unanswerable"] = int(r["type"] == "unanswerable" and r["returned_policy"] != "NONE")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary(rows)


def summary(rows):
    for m in ("rules", "llm", "rag"):
        r = [x for x in rows if x["method"] == m]
        print(m, "policy correct", sum(int(x["policy_correct"]) for x in r), "/", len(r),
              "| mean s", round(sum(float(x["seconds"]) for x in r) / len(r), 2),
              "| mean tokens", round(sum(int(x["input_tokens"] or 0) + int(x["output_tokens"] or 0) for x in r) / len(r)))


def main():
    key = pc.api_key_from_env()
    if not key:
        raise SystemExit("Set GEMINI_API_KEY")
    client = pc._client(key)
    policies = pc.load_policies()
    rules = pc.RulesSearch(policies)
    index = pc.load_index()
    all_numbers = set().union(*(numbers(p["policy_text"]) for p in policies))
    qs = list(csv.DictReader(open(pc.HERE / "data" / "benchmark_questions.csv", encoding="utf-8")))
    rows = []
    for q in qs:
        expected = q["expected_policy"].split("|")
        for name, fn in (("rules", lambda s: rules.answer(s)),
                         ("llm", lambda s: pc.llm_full(s, policies, client)),
                         ("rag", lambda s: pc.llm_rag(s, policies, index, client))):
            a = fn(q["question"])
            got = a.policy.strip()
            correct = is_correct(got, expected)
            rows.append({
                "id": q["id"], "type": q["type"], "question": q["question"], "method": name,
                "expected_policy": q["expected_policy"], "returned_policy": got,
                "policy_correct": int(correct), "answer": a.answer,
                "seconds": round(a.seconds, 3), "input_tokens": a.input_tokens,
                "output_tokens": a.output_tokens, "model": a.model,
                "answered_unanswerable": int(q["type"] == "unanswerable" and got.upper() != "NONE"),
                "invented_number": int(bool(numbers(a.answer) - all_numbers - numbers(q["question"]))),
                "unsupported": "",
            })
            print(f"{q['id']:>2} {name:5s} {got[:32]:32s} {'OK' if correct else '--'} {a.seconds:.2f}s")
            if name != "rules":
                time.sleep(4)        # stay inside free-tier requests per minute
    with open(pc.HERE / "data" / "benchmark_results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary(rows)


if __name__ == "__main__":
    import sys
    rescore() if "--rescore" in sys.argv else main()
