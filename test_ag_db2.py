from pathlib import Path
from llm_chat_archive.collector import run_collect, CollectConfig
from llm_chat_archive.adapters.antigravity import _ModernAntigravityIdeAdapter

adapter = _ModernAntigravityIdeAdapter()
sessions = adapter.collect()
for s in sessions:
    print(f"variant: {s.source_variant}, events: {len(s.raw_events)}")
    if s.raw_events:
        print(f"Sample Event: {s.raw_events[0]}")
