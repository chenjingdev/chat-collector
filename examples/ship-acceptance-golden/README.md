# Ship Acceptance Golden

이 디렉터리는 실제 operator ship-acceptance run에서 추출한 repo-safe golden
snapshot을 보관한다.

- `ship-acceptance.json`: fixed acceptance profile, run summary, validate /
  verify / digest / export summary, source-level pass/fail 판단만 포함한다.
- raw normalized archive row나 source artifact는 넣지 않는다.
- redacted portable bundle과 memory export는 외부 archive root의
  `acceptance/` 아래에 남는다.

실행 절차와 판정 기준은 [docs/ship-acceptance.md](../../docs/ship-acceptance.md)를
따른다.
