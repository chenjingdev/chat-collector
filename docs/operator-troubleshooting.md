# Operator Troubleshooting Guide

이 문서는 `doctor`, `validate`, `archive verify`, `archive anomalies`,
`archive audit-identities`, `runs diff`, `runs trend`, `rerun`,
`archive rewrite`, `archive prune`, `tui` 출력만 보고 다음 조치를 결정하기
위한 운영자용 기준이다.

## 빠른 판단표

| 신호 | 먼저 볼 명령 | 핵심 필드 | 기본 조치 |
| --- | --- | --- | --- |
| source artifact root가 없거나 읽기 불가 | `uv run llm-chat-archive doctor <source>` | `status`, `status_reason`, `roots[].exists`, `roots[].readable` | root/권한을 먼저 고친 뒤 `rerun` |
| 최근 run에서 source가 `failed`, `partial`, `unsupported` | `uv run llm-chat-archive runs show --archive-root <root> <run-id>` | `sources[].status`, `sources[].support_level`, `partial`, `unsupported`, `failure_reason` | 원인이 일시적이면 `rerun`, 구조적이면 `inspect` 또는 `ignore` |
| scheduled runner가 바로 종료됨 | `uv run llm-chat-archive scheduled run` | `reason`, `lock.stale`, `lock.age_seconds`, `scheduled.stale_after_seconds` | active lock이면 대기, stale lock이면 기준 확인 후 `--force-unlock-stale` |
| manifest와 output 계약 위반이 의심됨 | `uv run llm-chat-archive validate --archive-root <root> --run <run-id>` | `status`, `findings[].code`, `sources[].validation_status` | `inspect` 후 필요하면 `verify` 또는 `rerun` |
| archive 파일 자체가 깨졌거나 orphan이 보임 | `uv run llm-chat-archive archive verify --archive-root <root>` | `status`, `bad_row_count`, `orphan_file_count`, `findings[].code` | `inspect`, 그다음 `rewrite` 또는 `prune` |
| 낮은 신호 대화가 반복됨 | `uv run llm-chat-archive archive anomalies --archive-root <root>` | `source_reasons`, `suspicious_conversations[].reasons` | 일시적이면 `rerun`, 기대된 저품질이면 `ignore`, 애매하면 `inspect` |
| session id / artifact path 충돌 | `uv run llm-chat-archive archive audit-identities --archive-root <root>` | `collision_groups[].reasons`, `preferred_conversation` | richer row가 명확하면 `rewrite`, 아니면 `inspect` |
| run 간 상태 변화 추세를 보고 싶음 | `uv run llm-chat-archive runs diff ...`, `uv run llm-chat-archive runs trend ...` | `important_transitions`, `degraded_to_complete_count`, `latest_status` | transient면 `rerun`, 반복되면 `ignore` 또는 root cause `inspect` |
| 기존 archive를 정리하고 싶음 | `uv run llm-chat-archive archive rewrite ...`, `uv run llm-chat-archive archive prune ...` | `write_mode`, `dropped_row_count`, `upgraded_row_count`, `deleted_run_count` | `rewrite`는 논리 정리, `prune`은 삭제 전용 |

## 공통 원칙

- `--archive-root`는 repository 밖의 절대 경로여야 한다. 기본값은 `/Users/chenjing/dev/chat-history`다.
- `doctor`는 source artifact를 본다. `validate`와 `archive ...` 계열은 이미 써진 normalized archive를 본다.
- `support_level`은 collector의 구현 성숙도다. `status`는 이번 run의 실제 결과다.
- `partial`과 `unsupported`는 곧바로 장애를 의미하지 않는다. 해당 source의 `support_level`과 run trend를 같이 봐야 한다.
- 이 문서에서 `inspect`는 보통 `runs show`, `archive list`, `archive find`, `archive show`로 실제 run/source/session row를 확인하는 것을 뜻한다.
- `rerun`은 source artifact를 다시 읽어서 새 run을 만든다.
- `scheduled run`은 `collect --all` 또는 latest-based `rerun`을 non-interactive preset으로 실행하고, 겹침 방지를 위해 `<archive-root>/runs/.scheduled.lock`을 사용한다.
- `rewrite`는 이미 archive에 있는 row를 재정렬/압축한다.
- `prune`은 old run manifest와 보조 디렉터리를 지운다. 데이터 품질 복구 수단이 아니다.

## 0. Scheduled Lock 충돌과 stale lock 판단

외부 스케줄러에서는 `scheduled run`만 반복 호출한다.

```bash
uv run llm-chat-archive scheduled run
```

해석 기준:

- `reason = "scheduled_lock_held"`이면 다른 collect/rerun이 아직 같은 archive root에서 실행 중이다.
- `reason = "scheduled_lock_stale"`이면 lock age가 `scheduled.stale_after_seconds`를 넘었다.
- `lock.owner_pid`, `lock.owner_hostname`, `lock.acquired_at`, `lock.age_seconds`로 기존 lock의 주체와 경과 시간을 판단한다.
- active lock은 절대 강제로 깨지지 않는다. `--force-unlock-stale`는 stale lock에서만 의미가 있다.
- 성공한 scheduled run manifest에는 top-level `scheduled` 필드가 추가되어 수동 run과 구분된다.

Redacted example:

```json
{
  "status": "skipped",
  "reason": "scheduled_lock_stale",
  "scheduled": {
    "mode": "collect",
    "stale_after_seconds": 21600
  },
  "lock": {
    "owner_pid": 4242,
    "owner_hostname": "scheduler-host",
    "acquired_at": "2026-03-20T00:00:00Z",
    "age_seconds": 28800,
    "stale": true
  },
  "force_unlock_stale_available": true
}
```

다음 조치:

- `ignore`: 하지 않는다. active lock이면 그냥 다음 주기까지 기다린다.
- `inspect`: `lock.age_seconds`와 실제 운영 윈도우를 비교해 stale 판단이 맞는지 확인한다.
- `scheduled run --force-unlock-stale`: stale라고 확인된 경우에만 다시 실행한다.

## 1. Artifact Root가 없거나 읽기 불가한 경우

먼저 `doctor`로 source별 입력 root 상태를 확인한다.

```bash
uv run llm-chat-archive doctor claude --input-root /absolute/source-root
uv run llm-chat-archive doctor --all --profile default
```

해석 기준:

- `status = "missing"` 그리고 `status_reason = "no readable input roots"`이면 경로가 없거나 읽기 권한이 없다.
- `roots[].exists = false`면 경로 지정이 틀렸을 가능성이 크다.
- `roots[].exists = true`, `roots[].readable = false`면 권한 또는 디렉터리 접근 문제가 더 유력하다.
- `status = "partial-ready"`와 `candidate_artifact_count = 0`이면 root는 읽히지만 collector가 읽을 후보 artifact를 못 찾은 상태다.
- `support_level = "complete"`인데도 `candidate_artifact_count = 0`이 계속되면 운영 문제를 먼저 의심한다.

Redacted example:

```json
{
  "source": "claude",
  "support_level": "complete",
  "status": "missing",
  "status_reason": "no readable input roots",
  "candidate_artifact_count": 0,
  "roots": [
    {
      "path": "/absolute/source-root",
      "kind": "missing",
      "exists": false,
      "readable": false
    }
  ]
}
```

다음 조치:

- `rerun`: 바로 하지 않는다. root/권한을 먼저 고친 뒤 수행한다.
- `inspect`: root가 읽히는데 `candidate_artifact_count = 0`이면 source별 실제 저장 위치를 확인한다.
- `ignore`: 해당 source를 이번 운영 범위에서 의도적으로 쓰지 않는 경우에만 고려한다.

## 2. Source가 `unsupported` 또는 `partial`로 반복 수집되는 경우

한 번의 run만 보지 말고 최근 run과 추세를 같이 본다.

```bash
uv run llm-chat-archive runs latest --archive-root <root>
uv run llm-chat-archive runs show --archive-root <root> <run-id>
uv run llm-chat-archive runs diff --archive-root <root> --from <older-run> --to <newer-run>
uv run llm-chat-archive runs trend --archive-root <root> --source <source>
```

해석 기준:

- `support_level = "partial"` 또는 `support_level = "scaffold"`인 source에서 `status = "partial"` 또는 `status = "unsupported"`가 반복되면 구현 한계일 수 있다.
- `support_level = "complete"`인데 `status = "partial"`, `status = "unsupported"`, `status = "failed"`가 반복되면 운영 이슈일 가능성이 높다.
- `runs diff`의 `important_transitions`에 `failed_to_complete`, `partial_to_complete`, `unsupported_to_complete`가 보이면 rerun이나 환경 수정이 실제로 효과를 냈다는 뜻이다.
- `runs trend`에서 `degraded_to_complete_count = 0`이고 `latest_status`가 계속 `partial` 또는 `unsupported`면 blind rerun보다 root cause 확인이 우선이다.
- `runs trend`의 `timeline[].manifest.failure_reason`까지 같으면 반복 실패 패턴이다.

Redacted example:

```json
{
  "important_transitions": [
    {
      "source": "codex_cli",
      "label": "failed_to_complete",
      "category": "improved"
    }
  ],
  "sources": [
    {
      "source": "codex_cli",
      "status": {
        "from": "failed",
        "to": "complete",
        "changed": true
      }
    }
  ]
}
```

`rerun` 자체의 대상 선정은 `rerun.selection_reason`과 `rerun.matched_sources`로 확인한다.

```bash
uv run llm-chat-archive rerun --run <run-id> --reason degraded
uv run llm-chat-archive rerun --run <run-id> --reason failed_or_degraded --source <extra-source>
```

`degraded`는 `partial`과 `unsupported`를 같이 잡는다. `failed_or_degraded`는 가장 넓은 재시도 범위다.

Redacted example:

```json
{
  "selected_sources": [
    "complete_source",
    "failed_source",
    "partial_source"
  ],
  "rerun": {
    "origin_run_id": "20260319T080000Z",
    "selection_reason": "failed_or_degraded",
    "matched_sources": [
      "failed_source",
      "partial_source",
      "unsupported_source"
    ],
    "manual_include_sources": [
      "complete_source"
    ],
    "manual_exclude_sources": [
      "unsupported_source"
    ]
  }
}
```

다음 조치:

- `rerun`: artifact root를 고쳤거나, source 앱이 새 artifact를 쓴 뒤 다시 수집할 가치가 있을 때.
- `inspect`: 특정 source 하나만 이상하거나 `failure_reason`이 반복될 때.
- `ignore`: `support_level`이 낮고, trend상 기대된 degraded 상태가 안정적으로 반복될 때.

## 3. `validate` 결과 해석

`validate`는 특정 run manifest와 그 run이 가리키는 source output이 계약을 지키는지 본다.

```bash
uv run llm-chat-archive validate --archive-root <root> --run <run-id>
```

해석 기준:

- top-level `status`는 `success`, `warning`, `error` 중 하나다.
- exit code `1`은 `error_count > 0`일 때만 나온다. warning만 있으면 exit code는 `0`이다.
- `sources[].validation_status`는 source 단위 요약이다.
- `findings[].code`가 실제 판단의 중심이다.

자주 보는 finding code:

- `manifest_loaded`: 정상 시작 신호다.
- `degraded_support_level`, `degraded_source_status`, `incomplete_transcript`: degraded source를 반영한 warning일 수 있다.
- `missing_file`, `malformed_row`, `missing_required_field`, `invalid_enum`, `count_mismatch`: archive 또는 manifest가 실제 계약을 어겼다.
- `external_archive_only_violation`: archive root가 repo 안쪽으로 기록되었거나 잘못된 경로가 섞였다.

Redacted example:

```json
{
  "status": "warning",
  "summary": {
    "warning_count": 3,
    "error_count": 0
  },
  "sources": [
    {
      "source": "cursor_editor",
      "validation_status": "warning",
      "support_level": "partial",
      "status": "partial"
    }
  ],
  "findings": [
    {
      "code": "degraded_support_level",
      "level": "warning"
    },
    {
      "code": "degraded_source_status",
      "level": "warning"
    },
    {
      "code": "incomplete_transcript",
      "level": "warning"
    }
  ]
}
```

다음 조치:

- `ignore`: warning이 기대된 degraded source와 일치할 때.
- `inspect`: warning이지만 실제 운영 기대와 다를 때.
- `verify`: file-level 이상이 의심되면 바로 이어서 archive 전체를 확인한다.
- `rerun`: manifest는 정상인데 source output이 빠졌거나 source run 자체가 망가졌을 때.

## 4. `archive verify`, `archive anomalies`, `archive audit-identities` 해석

### 4.1 `archive verify`

`archive verify`는 archive root 아래의 source JSONL 파일을 직접 스캔한다.
반대로 `archive list`, `archive find`, `archive stats`, `archive profile`은
필요하면 `<archive-root>/archive-index/conversations.sqlite3`를 refresh해서
조회한다. verify 결과가 이상하면 index를 의심하기보다 raw JSONL부터 본다.

```bash
uv run llm-chat-archive archive verify --archive-root <root>
uv run llm-chat-archive archive verify --archive-root <root> --source <source>
```

해석 기준:

- `bad_row_count > 0`이면 malformed row나 contract mismatch가 실제 파일 안에 있다.
- `orphan_file_count > 0`이면 어떤 output file이 어떤 run manifest에도 연결되지 않는다.
- `files[].manifest_linked = false`와 `files[].orphan = true`는 stale output 후보를 뜻한다.
- `invalid_contract`와 `malformed_row`는 데이터 무결성 문제다.
- `orphan_output_file` 하나만 있으면 무조건 고장이라고 단정하지 말고, rewrite/prune이 안 끝난 상태인지 같이 본다.

Redacted example:

```json
{
  "status": "error",
  "bad_row_count": 2,
  "orphan_file_count": 1,
  "findings": [
    {
      "code": "invalid_contract",
      "level": "error"
    },
    {
      "code": "orphan_output_file",
      "level": "warning"
    },
    {
      "code": "malformed_row",
      "level": "error"
    }
  ]
}
```

다음 조치:

- `inspect`: `path`와 `row_number`가 찍힌 finding부터 본다.
- `rewrite`: append-only file들이 겹치고 canonical file로 정리할 가치가 있을 때.
- `prune`: orphan file이 stale auxiliary output이고 더 이상 필요 없을 때.
- `ignore`: `incomplete_transcript` warning만 있고 source support와 일치할 때.

### 4.2 `archive anomalies`

`archive anomalies`는 low-signal conversation과 source 단위 aggregate 이상치를 찾는다.

```bash
uv run llm-chat-archive archive anomalies --archive-root <root>
uv run llm-chat-archive archive anomalies --archive-root <root> --source <source>
```

해석 기준:

- `source_reasons`는 source 전체가 의심스럽다는 뜻이다.
- `suspicious_conversations[].reasons`는 row 단위 근거다.
- 기본 threshold는 `low_message_count = 1`, `limitations_count = 2`,
  `unsupported_count = 2`, `unsupported_ratio = 0.5`다.
- `low_message_count` 하나만으로는 짧은 정상 대화일 수도 있다.
- `unsupported_transcript`와 `high_unsupported_ratio`가 complete source에서 반복되면 운영 문제일 가능성이 커진다.

Redacted example:

```json
{
  "sources": {
    "cursor_editor": {
      "source_reasons": [
        {
          "code": "high_unsupported_ratio"
        }
      ],
      "suspicious_conversations": [
        {
          "source_session_id": "cursor-session-1",
          "reasons": [
            {
              "code": "low_message_count"
            },
            {
              "code": "excessive_limitations"
            },
            {
              "code": "unsupported_transcript"
            }
          ]
        }
      ]
    }
  }
}
```

다음 조치:

- `inspect`: suspicious session이 실제로 저품질인지 row를 확인한다.
- `rerun`: source artifact가 더 완전해졌을 가능성이 있을 때.
- `ignore`: 짧은 대화나 알려진 partial source라서 임계값 warning이 설명될 때.

### 4.3 `archive audit-identities`

`archive audit-identities`는 같은 논리 대화가 여러 identity shape로 저장된 후보를 찾는다.

```bash
uv run llm-chat-archive archive audit-identities --archive-root <root>
uv run llm-chat-archive archive audit-identities --archive-root <root> --source <source>
```

해석 기준:

- `collision_groups[].preferred_conversation`은 현재 archive 안에서 가장 풍부한 row다.
- `richer_transcript_available`가 있으면 archive 내부에 이미 더 나은 row가 있다.
- `conflicting_source_session_ids`나 `message_variants`가 있으면 충돌군이 정말 같은 대화인지 먼저 확인해야 한다.
- `mixed_identity_shapes`는 collector가 session-only와 session+artifact를 섞어 기록했다는 뜻이다.

Redacted example:

```json
{
  "sources": {
    "codex_cli": {
      "collision_groups": [
        {
          "source_session_ids": [
            "alpha",
            "beta"
          ],
          "reasons": [
            {
              "code": "duplicate_source_session_id"
            },
            {
              "code": "message_variants"
            },
            {
              "code": "richer_transcript_available"
            }
          ],
          "preferred_conversation": {
            "source_session_id": "beta",
            "transcript_completeness": "complete",
            "message_count": 3,
            "preferred": true
          }
        }
      ]
    }
  }
}
```

다음 조치:

- `rewrite`: `preferred_conversation`이 명확하고 archive를 canonical file로 정리하려는 경우.
- `inspect`: `conflicting_source_session_ids` 또는 `message_variants`가 있으면 먼저 실제 충돌인지 확인한다.
- `ignore`: collision이 없거나, source가 정상적으로 단일 canonical file만 유지할 때.

## 5. `rerun`, `rewrite`, `prune` 중 무엇을 써야 하는가

| 조치 | 언제 쓰는가 | 무엇이 바뀌는가 | 쓰면 안 되는 경우 |
| --- | --- | --- | --- |
| `rerun` | source artifact가 바뀌었거나, 직전 run의 `failed`/`partial`/`unsupported`를 다시 수집하고 싶을 때 | 새 run manifest와 새 source output이 생성된다 | 이미 archive 안에 있는 중복 row만 정리하고 싶은 경우 |
| `rewrite` | append-only output을 canonical file로 압축하고, richer row를 선택해 정리할 때 | 기존 archive JSONL을 재작성하거나 staging root에 새 canonical output을 만든다 | source artifact 자체가 없어서 다시 수집해야 하는 경우 |
| `prune` | old run manifest와 `archive-index`, `rewrite-staging`, `rewrite-backups`, `exports` 같은 보조 디렉터리를 지울 때 | run manifest/보조 파일이 삭제된다 | malformed row, missing row, contract drift를 고치려는 경우 |

추천 순서:

1. `inspect`
2. `archive verify`
3. `archive rewrite`를 dry-run으로 확인
4. 필요하면 `archive rewrite --execute`
5. 다시 `archive verify`
6. 마지막으로 `archive prune` dry-run과 `--execute`

`rewrite` 출력에서 볼 것:

- `write_mode = "dry_run"`, `"in_place"`, `"staging"`
- `changed_source_count`
- `dropped_row_count`
- `upgraded_row_count`
- `output_path`

`prune` 출력에서 볼 것:

- `write_mode = "dry_run"` 또는 `"prune"`
- `deleted_run_count`
- `deleted_auxiliary_directory_count`
- `latest_kept_run_id`
- `reclaimed_bytes`

## 6. Redaction/Validation 설정 때문에 기대와 다른 결과가 나온 경우

가장 먼저 해당 run의 `effective_config`를 확인한다.

```bash
uv run llm-chat-archive runs show --archive-root <root> <run-id>
```

즉시 실행한 `collect --all` 또는 `rerun` 출력에서도 같은 필드를 볼 수 있다.

확인할 필드:

- `effective_config.execution_policy.redaction`
- `redaction_event_count`
- `effective_config.execution_policy.validation`
- `validation.mode`
- `validation.status`
- `validation.error_count`

해석 기준:

- `redaction = "on"`이고 `redaction_event_count > 0`이면 secret-like 문자열이 `[REDACTED]` 형태로 바뀐 것이다. 이 자체는 정상 동작이다.
- row 수와 message 수는 정상인데 민감한 값만 치환되었다면 기본 조치는 `ignore`다.
- 민감하지 않은 본문까지 사라졌다고 느껴지면 `inspect`로 실제 row를 확인한 뒤 rerun 여부를 판단한다.
- `validation = "off"`면 `collect`/`rerun` 출력에 validation 요약이 없는 것이 정상이다. 필요하면 `validate`를 수동 실행한다.
- `validation = "report"`면 validation에 error가 있어도 command exit code가 source failure 때문에만 바뀔 수 있다. JSON의 `validation.status`를 반드시 같이 본다.
- `validation = "strict"`면 source 수집 자체가 끝났더라도 validation error 때문에 command가 non-zero로 끝날 수 있다.

다음 조치:

- `ignore`: redaction placeholder만 기대대로 생긴 경우.
- `inspect`: 설정은 맞는데 결과가 과하게 달라졌다고 느껴질 때.
- `rerun`: 설정을 의도적으로 바꾼 뒤 새 output을 다시 만들 때.

## 7. 최소 운영 루틴

문제가 생겼을 때는 아래 순서로 보면 대부분 다음 조치가 정리된다.

1. `doctor`로 source root와 artifact 가시성을 확인한다.
2. `runs latest` 또는 `runs show`로 해당 run의 source 상태를 확인한다.
3. `validate`로 manifest/output 계약 위반 여부를 본다.
4. `archive verify`로 file-level 무결성을 확인한다.
5. `archive anomalies`와 `archive audit-identities`로 내용 품질과 충돌을 확인한다.
6. source artifact를 다시 읽어야 하면 `rerun`, archive 내부만 정리하면 `rewrite`, 오래된 부가 산출물만 지우려면 `prune`, 기대된 degraded 상태면 `ignore`한다.
