---
tracker:
  kind: linear
  project_slug: "chat-collector-13bbc90bddca"
  active_states:
    - Todo
    - In Progress
  terminal_states:
    - Done
    - Canceled
polling:
  interval_ms: 5000
workspace:
  root: /Users/chenjing/code/workspaces/chat-collector
agent:
  max_concurrent_agents: 1
---

You are working on a Linear issue `{{ issue.identifier }}` in the `chat-collector` repository.

Issue context:
Identifier: {{ issue.identifier }}
Title: {{ issue.title }}
Current status: {{ issue.state }}
URL: {{ issue.url }}

Description:
{% if issue.description %}
{{ issue.description }}
{% else %}
No description provided.
{% endif %}

## Operating Model

- The default operating model is `direct-edit + serial execution`.
- The per-issue workspace under `/Users/chenjing/code/workspaces/chat-collector` is scratch only.
- Treat `/Users/chenjing/dev/chat-collector` as the only canonical checkout for product code, tests, and docs during the run.
- If `/Users/chenjing/dev/chat-collector/.git` is missing, record one concise blocker note and stop instead of inventing recovery work.
- Read repo-root `SPEC.md` and the current issue body together before planning or editing.
- Use `SPEC.md` for product scope and release rules, and use the current issue body for the immediate scope boundary and acceptance input.
- Do not treat old workspaces, old attachments, or prior recovery artifacts as canonical source unless the current issue explicitly says so.

## Status Map

- `Backlog`
  - out of scope for active execution
  - do not modify
- `Todo`
  - move to `In Progress` immediately, then execute
- `In Progress`
  - continue implementation
- `Done`
  - terminal state
  - stop
- `Canceled`
  - terminal state
  - stop
- Any other state
  - leave unchanged unless the issue explicitly requires something else

## Execution Rules

1. Do not ask a human to perform follow-up actions unless blocked by missing required auth, permissions, or secrets.
2. Work directly in `/Users/chenjing/dev/chat-collector`; use the scratch workspace only for temporary notes or artifacts.
3. Do not touch paths outside the scratch workspace except the canonical checkout above and `/Users/chenjing/dev/chat-history` when the issue explicitly requires real collection output or validation artifacts.
4. Keep the current ticket as the primary scope boundary. Do not silently expand into adjacent cleanup, publication, or infrastructure work.
5. Keep Linear comments sparse. Use at most one concise progress comment if a comment is needed at all.
6. Create follow-up issues only when a clearly independent next slice is required and the current ticket cannot reasonably absorb it.
7. Prefer zero new tickets. If a follow-up is necessary, create only the next runnable queue item.
8. Use `blockedBy` only when there is real execution order.
9. Do not create speculative backlog tickets during implementation work.
10. Validate the exact behavior you changed with focused commands or tests before moving the issue to `Done`.
11. For real collection runs, write archives under `/Users/chenjing/dev/chat-history`, never inside the repository.
12. Final reporting should include completed actions, validation, and blockers only.

## Release-Hardening Ticket Reference Rule

Release-hardening tickets should include this sentence near the top of the description:

`Before planning or editing, read repo-root SPEC.md, WORKFLOW.md, and this issue body together. Treat them as product scope, execution rules, and immediate acceptance input.`

Apply that sentence to the active release-hardening queue and keep it on any replacement follow-up ticket that continues the same chain.
