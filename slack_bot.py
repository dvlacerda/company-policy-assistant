"""Slack policy bot (Socket Mode - runs on your own computer, no hosting needed).

Mention the bot in a channel:  @PolicyBot can I work from abroad?
It answers with the LLM + vector index method and names the relevant policy.

Setup: see README.md. Needs in .env (never commit it):
  SLACK_BOT_TOKEN=xoxb-...   SLACK_APP_TOKEN=xapp-...   GEMINI_API_KEY=...
Run:   python slack_bot.py
"""
import os
import re

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

import policy_core as pc


def load_env():
    env = pc.HERE / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


load_env()
app = App(token=os.environ["SLACK_BOT_TOKEN"])
POLICIES = pc.load_policies()
INDEX = pc.load_index()
CLIENT = pc._client(os.environ["GEMINI_API_KEY"])


@app.event("app_mention")
def on_mention(event, say):
    question = re.sub(r"<@[^>]+>", "", event.get("text", "")).strip()
    thread = event.get("thread_ts") or event["ts"]
    if not question:
        say(text="Ask me a policy question, e.g. *How many vacation days do I get?*", thread_ts=thread)
        return
    try:
        a = pc.llm_rag(question, POLICIES, INDEX, CLIENT)
    except Exception as exc:  # noqa: BLE001
        say(text=f"Sorry, I could not reach the AI service ({exc}).", thread_ts=thread)
        return
    source = "_No policy in the database covers this question._" if a.policy.upper() == pc.NO_POLICY \
        else f"*Relevant policy:* {a.policy}"
    say(text=f"{a.answer}\n{source}", thread_ts=thread)


if __name__ == "__main__":
    print("Policy bot running - mention it in a channel. Ctrl+C to stop.")
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
