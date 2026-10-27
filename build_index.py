"""Embed every policy once and save the vector index (data/policy_index.json).

Usage: GEMINI_API_KEY=... python build_index.py
"""
import policy_core as pc

if __name__ == "__main__":
    key = pc.api_key_from_env()
    if not key:
        raise SystemExit("Set GEMINI_API_KEY")
    path = pc.build_index(pc._client(key), pc.load_policies())
    print("saved", path)
