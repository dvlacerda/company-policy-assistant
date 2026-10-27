# Company Policy Assistant

Answers employees' policy questions from a database of 98 company policies and names the relevant
policy. Built for the *AI in Accounting and Business* course (SDU).

## Three methods compared

| Method | How it works |
|---|---|
| Rules-based search | BM25 keyword search over policy titles and texts; returns the best policy's text, or "no policy found" below a score threshold. No AI. |
| LLM without a vector index | All 98 policies are placed in the prompt; Gemini answers from them. |
| LLM with a vector index (RAG) | Policies are embedded once (`gemini-embedding-001`, 768 dims, `data/policy_index.json`); each question is embedded and only the 3 most similar policies go into the prompt. |

Both LLM methods use the same model and the same answering rules (answer only from the listed
policies, add no facts, say when the database does not cover a question), so the vector index is the
only difference. The website shows, for each method: the answer, the relevant policy, response time,
token use and a check of whether the answer is backed by the database. The *Benchmark* tab reports 30
test questions (20 answerable, 4 partly answerable, 6 unanswerable).

## Files

- `app.py` - Streamlit comparison website
- `policy_core.py` - the three methods
- `build_index.py` - builds the vector index (run once)
- `benchmark.py` - runs the 30 questions through all methods
- `slack_bot.py`, `slack_manifest.yaml` - Slack bot (LLM + vector index)
- `data/` - policies, vector index, benchmark questions and results

## Run

```bash
pip install -r requirements.txt
cp .env.example .env            # add GEMINI_API_KEY (never commit .env)
python build_index.py
python benchmark.py
streamlit run app.py
```

Deploy on Streamlit Community Cloud: New app -> this repository -> `app.py`, and add
`GEMINI_API_KEY` under *Advanced settings -> Secrets*.

## Slack bot

1. https://api.slack.com/apps -> Create New App -> From a manifest -> pick your workspace -> paste
   `slack_manifest.yaml`.
2. Basic Information -> App-Level Tokens -> generate a token with `connections:write` (xapp-...).
3. Install App -> install to workspace -> copy the Bot User OAuth Token (xoxb-...).
4. Put both tokens and `GEMINI_API_KEY` in `.env`, run `python slack_bot.py`.
5. In the test channel: `/invite @PolicyBot`, then ask `@PolicyBot can I work from abroad?`
