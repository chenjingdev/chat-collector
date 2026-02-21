from pathlib import Path
from llm_chat_archive.collector import run_collect, CollectConfig
from llm_chat_archive.adapters.antigravity import _ModernAntigravityIdeAdapter

adapter = _ModernAntigravityIdeAdapter()
sessions = adapter.collect()
print(f"Total sessions collected from DBs: {len(sessions)}")

for s in sessions:
    if s.source_variant == "ide-chat" and s.display_turns:
        print(f"FOUND A CHAT: {s.display_turns[0].text[:100]}")
