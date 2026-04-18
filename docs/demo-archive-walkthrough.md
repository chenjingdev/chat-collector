# Demo Archive Walkthrough

이 문서는 실제 개인 artifact 없이도 archive 운영 흐름을 끝까지 따라가볼 수
있도록, repo 안의 redacted demo bundle을 외부 임시 archive root로 import해서
검증하는 절차를 설명한다.

## 1. Demo bundle 위치와 범위

- bundle 경로: `examples/demo-archive-bundle/`
- 용도: operator demo 전용 portable export bundle
- 비포함 항목: raw source artifact, 비밀정보, 대용량 realistic dataset, run manifest
- 테스트 fixture와의 구분: automated fixture는 `tests/fixtures/`에 있고, 이
  bundle은 walkthrough 전용이다.

번들에 들어있는 normalized conversation은 다음 세 개다.

| Source | Session | Transcript completeness | 메모 |
| --- | --- | --- | --- |
| `codex_cli` | `demo-codex-complete` | `complete` | complete conversation |
| `cursor_editor` | `demo-cursor-partial` | `partial` | draft 누락 limitation 포함 |
| `gemini_code_assist_ide` | `demo-gemini-unsupported` | `unsupported` | metadata-only recovery 예시 |

## 2. 준비

repo 루트에서 의존성을 맞춘다.

```bash
uv sync
```

demo bundle 경로와 외부 임시 archive root를 잡는다.

```bash
bundle_dir="$(pwd)/examples/demo-archive-bundle"
demo_root="$(mktemp -d "${TMPDIR:-/tmp}/llm-chat-archive-demo.XXXXXX")"
printf 'demo_root=%s\n' "$demo_root"
```

`demo_root`는 repo 밖의 절대경로여야 한다. walkthrough가 끝나면 지워도 된다.

## 3. `archive import`

먼저 dry-run으로 import 결과를 확인한다.

```bash
uv run llm-chat-archive archive import \
  --archive-root "$demo_root" \
  --bundle-dir "$bundle_dir"
```

이 단계에서는 `conversation_count=3`, `message_count=6`,
`imported_count=3`이 보여야 한다.

실제 canonical archive output을 쓴다.

```bash
uv run llm-chat-archive archive import \
  --archive-root "$demo_root" \
  --bundle-dir "$bundle_dir" \
  --execute
```

import 뒤에는 source별 canonical output이 생긴다.

- `"$demo_root/codex_cli/memory_chat_v1-codex_cli.jsonl"`
- `"$demo_root/cursor_editor/memory_chat_v1-cursor_editor.jsonl"`
- `"$demo_root/gemini_code_assist_ide/memory_chat_v1-gemini_code_assist_ide.jsonl"`

## 4. `archive list`, `show`, `find`

전체 conversation summary를 본다.

```bash
uv run llm-chat-archive archive list --archive-root "$demo_root"
```

세 conversation이 `unsupported`, `partial`, `complete` 순으로 보인다.

partial conversation 본문을 확인한다.

```bash
uv run llm-chat-archive archive show \
  --archive-root "$demo_root" \
  --source cursor_editor \
  --session demo-cursor-partial
```

text query로 limitation이 있는 대화를 찾는다.

```bash
uv run llm-chat-archive archive find \
  --archive-root "$demo_root" \
  --text rollback
```

## 5. `archive stats`, `profile`, `digest`

coverage와 completeness 분포를 본다.

```bash
uv run llm-chat-archive archive stats --archive-root "$demo_root"
```

role 분포와 limitation 빈도를 본다.

```bash
uv run llm-chat-archive archive profile --archive-root "$demo_root"
```

operator digest를 본다.

```bash
uv run llm-chat-archive archive digest --archive-root "$demo_root"
```

이 demo는 의도적으로 `partial`과 `unsupported` conversation을 포함하므로
`digest.status`는 보통 `warning`이다. 또한 portable export bundle은
normalized archive row만 가져오므로 `latest_run`은 비어 있다.

## 6. `archive verify`

archive file-level 무결성을 확인한다.

```bash
uv run llm-chat-archive archive verify --archive-root "$demo_root"
```

이 walkthrough에서는 bad row나 orphan file은 없어야 한다. 다만
`partial`과 `unsupported` row 때문에 warning finding은 남는다.

## 7. 정리

임시 demo archive root를 삭제한다.

```bash
rm -rf "$demo_root"
```
