# Memory Export Contract

`archive export-memory`는 normalized archive row를 downstream memory ingestion이
직접 소비할 수 있는 conversation-oriented JSONL contract로 내보낸다. 목적은
archive 내부 파일 구조, source별 output naming, superseded row 처리 규칙을
downstream이 몰라도 ingest를 시작할 수 있게 하는 것이다.

## CLI

기본 실행 예시는 다음과 같다.

```bash
uv run llm-chat-archive archive export-memory \
  --archive-root /Users/chenjing/dev/chat-history \
  --output-dir /absolute/path/to/memory-export \
  --run <run-id> \
  --execute
```

watermark 기반 incremental export도 지원한다.

```bash
uv run llm-chat-archive archive export-memory \
  --archive-root /Users/chenjing/dev/chat-history \
  --output-dir /absolute/path/to/memory-export \
  --after-collected-at 2026-03-19T08:30:00Z \
  --execute
```

두 범위 옵션의 의미는 다음과 같다.

- `--run <run-id>`: 해당 run manifest가 가리키는 source output path들만 읽는다.
  나중에 superseded된 row는 자동으로 제외된다.
- `--after-collected-at <ISO-8601>`: `collected_at`이 watermark보다 엄격히 큰
  active row만 내보낸다.

## Output Files

- `memory-records.jsonl`: memory ingestion record 한 줄당 conversation 하나
- `memory-export-manifest.json`: export contract, range filter, record count 요약

## Record Shape

각 JSONL row는 다음 필드를 가진다.

- `contract`: schema version과 stable id 전략을 설명하는 self-describing metadata
- `id`: stable conversation id
- `record_type`: 현재는 항상 `conversation`
- `source`
- `execution_context`
- `collected_at`
- `source_session_id`
- `source_artifact_path`
- `transcript_completeness`
- `limitations`
- `redaction`: `{status, marker_count}`
- `message_count`
- `transcript_text`: archive message 순서를 유지한 role-prefixed flattened text
- `messages`: simplified message array
- `source_provenance`: normalized archive `provenance` payload
- `export_provenance`: archive file path, row number, selected `run_id`

`messages[]` 안에는 다음 필드가 들어간다.

- `id`: stable message id
- `index`: 1-based archive message order
- `role`
- `timestamp`
- `source_message_id`
- `text`
- `image_count`
- `has_images`
- `redaction`

## Stable ID Rules

- conversation id는 `source`와 다음 우선순위로 계산한다.
  - `source_session_id`
  - `source_artifact_path`
  - 둘 다 없으면 `collected_at + message_fingerprint`
- message id는 `conversation_id`와 다음 우선순위로 계산한다.
  - `source_message_id`
  - 없으면 `message_fingerprint + same-fingerprint occurrence`

이 규칙 때문에 transcript가 동일한 conversation/message는 rerun 이후에도 같은
id를 유지하고, richer transcript로 실제 message content가 달라지면 바뀐 record만
새 id를 갖는다.

## Redaction Semantics

export는 archive에 이미 들어있는 redacted text를 그대로 유지한다. `redaction`
status는 export 단계에서 `[REDACTED]`, `[REDACTED_API_KEY]` marker를 스캔해서
계산한다.

- `none_detected`: marker가 없었다.
- `redacted`: marker가 하나 이상 있었다.

이 contract는 embedding 생성, vector DB 적재, cross-source semantic dedupe를
직접 수행하지 않는다.
