# Ship Acceptance

이 문서는 실제 operator 로컬 artifact를 기준으로 `llm-chat-archive`의
ship-acceptance 기준을 고정한다. 목적은 fixture/unit test와 별도로, clean
archive root에서 collect부터 validate, archive verify, archive digest,
archive export, archive export-memory까지 한 번에 재현 가능한 운영 절차를
남기는 것이다.

## 1. 고정 acceptance source profile

이번 티켓에서 ship-acceptance source set은 다음 일곱 개로 고정한다.

| Source | Local storage / format | Support level | 판정 |
| --- | --- | --- | --- |
| `claude` | `~/.claude/projects/*/*.jsonl`, `subagents/*.jsonl` | `complete` | release-blocking |
| `codex_cli` | `~/.codex/sessions/**/rollout-*.jsonl`, `~/.codex/archived_sessions/rollout-*.jsonl` | `complete` | release-blocking |
| `codex_app` | shared Codex rollout JSONL + local app sqlite / shell provenance | `complete` | release-blocking |
| `codex_ide_extension` | shared Codex rollout JSONL + VS Code/Cursor bridge provenance | `complete` | release-blocking |
| `antigravity_editor_view` | `~/.gemini/antigravity/conversations/*.pb` | `partial` | allowed degraded |
| `cursor` | `~/Library/Application Support/Cursor/logs/*/cli.log` plus shared workspace state | `partial` | allowed degraded |
| `cursor_editor` | `~/Library/Application Support/Cursor/User/workspaceStorage/*/state.vscdb` | `partial` | allowed degraded |

판정 기준은 support metadata와 실제 operator-local artifact discovery를 같이
반영한다.

- `release-blocking`: `complete` source이며 실제 operator 사용 artifact가 확인된
  source다. collect 결과가 `complete`가 아니거나, candidate artifact를 찾지
  못하거나, digest에서 `attention_required`로 남으면 release를 막는다.
- `allowed degraded`: 실제 operator가 사용하지만 collector support가 아직
  `partial`인 source다. `partial` 또는 `unsupported` 자체는 허용하되, hard
  failure, zero-artifact, zero-conversation은 release를 막는다.

현재 ship-acceptance profile에서 제외한 source는 다음과 같다.

- `claude_code_ide`: 현재 operator checkout 기준으로 IDE-attached Claude
  transcript 후보가 확인되지 않았다.
- `gemini`, `gemini_code_assist_ide`: 현재 operator checkout 기준으로 transcript
  bearing Gemini artifact가 확인되지 않았다.
- `windsurf_editor`: repo direction의 primary ship target set에 포함하지 않고,
  현재 operator release bar에서도 고정하지 않는다.

## 2. 실행 절차

ship-acceptance는 비어 있는 외부 archive root에서만 시작한다. 기존 수집 결과가
남아 있으면 run을 재사용하지 말고 새 root를 잡는다.

```bash
accept_root="/Users/chenjing/dev/chat-history/ship-acceptance-$(date +%Y%m%d-%H%M%S)"
uv run llm-chat-archive acceptance ship \
  --archive-root "$accept_root" \
  --snapshot-path "$(pwd)/examples/ship-acceptance-golden/ship-acceptance.json"
```

명령은 내부적으로 다음을 순서대로 수행한다.

1. 고정 source profile로 batch collect
2. `validate --run <run-id>`
3. `archive verify`
4. `archive digest`
5. `archive export`
6. `archive export-memory`

## 3. 산출물

명령이 성공적으로 끝나면 다음 산출물이 남는다.

- `<archive-root>/runs/<run-id>/manifest.json`
- `<archive-root>/acceptance/archive-export/conversations.jsonl`
- `<archive-root>/acceptance/archive-export/export-manifest.json`
- `<archive-root>/acceptance/memory-export/memory-records.jsonl`
- `<archive-root>/acceptance/memory-export/memory-export-manifest.json`
- 선택한 경우 repo-safe golden snapshot JSON

`archive export` bundle과 `archive export-memory` output은 모두 redaction 이후의
normalized row를 기준으로 만들어진다. raw source artifact는 repo 안에 복사하지
않는다.

## 4. Pass / Fail 기준

ship-acceptance는 다음 조건을 모두 만족해야 `pass`다.

- 모든 release-blocking source가 `complete`로 끝난다.
- 모든 고정 source가 candidate artifact를 찾고 archive row를 한 건 이상 쓴다.
- `validate.error_count == 0`
- `archive verify.error_count == 0`
- `archive verify.bad_row_count == 0`
- `archive verify.orphan_file_count == 0`
- `archive export`와 `archive export-memory` conversation count가 run
  `written_conversation_count`와 일치한다.
- `memory_export.record_count`가 run `written_conversation_count`와 일치한다.

다음 상태는 `allowed degraded` source에 한해 허용한다.

- run status가 `partial`
- run status가 `unsupported`
- digest가 warning이지만 원인이 allowed degraded source limitation에만 묶여 있는
  경우

다음 상태는 source 종류와 무관하게 release-blocking이다.

- collect hard failure
- candidate artifact 0
- written archive row 0
- validate error
- archive verify error / bad row / orphan

`archive digest`는 항상 실행하고 snapshot에도 남기지만, warning 그 자체를 바로
release-blocker로 해석하지는 않는다. 현재 operator 실데이터에는 complete source여도
low-signal conversation 때문에 `suspicious`가 잡히는 경우가 있으므로, digest는
주요 원인 분류와 회귀 관찰용 신호로 사용한다.

## 5. Golden Snapshot

repo에는 raw operator archive를 넣지 않고, golden snapshot만 고정한다.

- 위치: `examples/ship-acceptance-golden/ship-acceptance.json`
- 내용: 고정 source profile, run 요약, validate/verify/digest/export summary,
  source별 기대치와 실제 결과, blocking finding
- 목적: 이후 operator acceptance 재실행 시 구조와 판정 기준이 drift하지 않았는지
  비교하는 기준점
