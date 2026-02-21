from pathlib import Path
from llm_chat_archive.collector import run_collect, CollectConfig
from llm_chat_archive.adapters.antigravity import _ModernAntigravityIdeAdapter

adapter = _ModernAntigravityIdeAdapter()
sessions = adapter.collect()
success = 0
for s in sessions:
    for t in s.display_turns:
        if "Workspace Prompts History" in t.text:
            print("FOUND ONE:", t.text[:100])
            success += 1
print(f"Total successes: {success}")
