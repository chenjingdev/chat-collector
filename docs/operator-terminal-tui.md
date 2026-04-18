# Operator Terminal TUI

`llm-chat-archive tui`는 recorded run manifest와 normalized archive JSONL을
그대로 재사용해서 terminal 안에서 triage 하는 read-only 화면이다.

## 실행

기본 archive root가 맞다면 다음처럼 바로 연다.

```bash
uv run llm-chat-archive tui --archive-root /Users/chenjing/dev/chat-history
```

특정 화면에서 시작하고 싶다면 `--view`를 붙인다.

```bash
uv run llm-chat-archive tui --archive-root /Users/chenjing/dev/chat-history --view runs
uv run llm-chat-archive tui --archive-root /Users/chenjing/dev/chat-history --view sources
uv run llm-chat-archive tui --archive-root /Users/chenjing/dev/chat-history --view samples --source cursor_editor
```

CI나 headless shell에서 curses 대신 plain-text snapshot만 보고 싶다면:

```bash
uv run llm-chat-archive tui --archive-root /Users/chenjing/dev/chat-history --snapshot --view overview
```

## 화면 구성

- `overview`: latest run, archive digest, source health를 한 화면에 요약한다.
- `runs`: 최근 run 목록과 선택한 run의 source별 상태를 보여준다.
- `sources`: 선택한 source의 digest/stats/profile 핵심 수치를 보여준다.
- `samples`: 선택한 source의 sample 목록과 실제 conversation message를 보여준다.

## 기본 키

- `1`: `overview`
- `2`: `runs`
- `3`: `sources`
- `4`: `samples`
- `?`: help
- `Up/Down`, `j/k`: 현재 화면의 선택 항목 이동
- `Enter`: `runs`에서는 선택한 run의 첫 degraded source로 이동, `overview`/`sources`에서는 sample drill-down으로 이동
- `r`: disk에서 run/digest/stats/profile/sample 상태 새로고침
- `q`: 종료

## 운영 메모

- TUI는 destructive action을 하지 않는다.
- source of truth는 여전히 `<archive-root>/runs/*/manifest.json`과 source별
  `memory_chat_v1-*.jsonl`이다.
- `archive list`, `archive find`, `archive stats`, `archive profile`과 마찬가지로
  index가 없거나 stale하면 `<archive-root>/archive-index/`를 갱신해서 조회한다.
