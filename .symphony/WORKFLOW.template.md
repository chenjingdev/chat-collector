---
tracker:
  kind: linear
  api_key: $LINEAR_API_KEY
  project_slug: "__SET_SYMPHONY_LINEAR_PROJECT_SLUG__"
  active_states:
    - Todo
    - In Progress
    - Rework
    - Merging
  terminal_states:
    - Done
    - Closed
    - Cancelled
    - Canceled
    - Duplicate
polling:
  interval_ms: 10000
observability:
  dashboard_enabled: true
workspace:
  root: $SYMPHONY_WORKSPACE_ROOT
hooks:
  after_create: |
    set -eu

    git clone --depth 1 "$SYMPHONY_SOURCE_REPO_URL" .

    if command -v uv >/dev/null 2>&1; then
      uv sync
    fi
  timeout_ms: 300000
agent:
  max_concurrent_agents: 1
  max_turns: 20
codex:
  command: $CODEX_BIN app-server
  approval_policy: never
  thread_sandbox: workspace-write
  turn_sandbox_policy:
    type: workspaceWrite
    writableRoots:
      - "__SET_SYMPHONY_WORKSPACE_ROOT__"
      - "__SET_CHAT_ARCHIVE_ROOT__"
    readOnlyAccess:
      type: fullAccess
    networkAccess: false
    excludeTmpdirEnvVar: false
    excludeSlashTmp: false
---

You are working on a Linear issue `{{ issue.identifier }}` in the `chat-collector` repository.

{% if attempt %}
Continuation context:

- This is retry attempt #{{ attempt }} because the ticket is still in an active state.
- Resume from the current workspace state instead of restarting from scratch.
- Do not repeat already-completed investigation or validation unless needed for new changes.
- Do not end the turn while the issue remains in an active state unless blocked by missing required auth, permissions, or secrets.
{% endif %}

Issue context:
Identifier: {{ issue.identifier }}
Title: {{ issue.title }}
Current status: {{ issue.state }}
Labels: {{ issue.labels }}
URL: {{ issue.url }}

Description:
{% if issue.description %}
{{ issue.description }}
{% else %}
No description provided.
{% endif %}

Product intent:
- Local coding agents and IDE assistants store their chats on the local machine.
- This repository exists to collect those chats and turn them into a clean memory source for a future memory system.
- Collector code belongs in this repository, but collected chat archives must live outside the repo at `/Users/chenjing/dev/chat-history`.
- Do not store real collected chat output inside this repository unless the work is explicitly about fixtures or tests.
- Normalize outputs around memory usefulness rather than raw forensic completeness.
- Exclude tool calls, MCP invocation noise, internal reasoning traces, and execution artifacts that do not help memory inference.

Repository direction:
- Primary code lives under `src/llm_chat_archive`.
- Build collectors source-by-source and adapter-by-adapter.
- Keep source-specific collectors modular so they can evolve independently.
- Supported and target sources include `codex`, `claude`, `cursor`, `antigravity`, and `gemini`, but support level may differ by source.
- Prefer targeted tests, fixtures, and reproducible samples over speculative abstractions.
- CLI is the primary operator interface; add TUI only when it materially improves recurring collection control.

Source of truth and git rules:
- The workspace is a real git clone of `origin`; treat that clone as the only place to edit code during the run.
- Never treat `/Users/chenjing/dev/chat-collector` as the place where work is preserved. That path is only the operator's local checkout.
- All durable work must be preserved through normal git flow: branch, commit, push, PR, review, and merge.
- Do not finish a coding ticket with uncommitted or unpushed work sitting only in the workspace.
- If the workspace is missing `.git` metadata or has no `origin` remote, treat that as a blocker. Do not continue coding in a non-git workspace.

Ticket management posture:
- Treat Linear as an active work queue, not a passive tracker.
- If a ticket is too broad, split the remaining work into smaller actionable Linear issues instead of keeping scope implicit.
- If meaningful follow-up work is discovered, create new Linear issues proactively instead of leaving vague notes for later.
- New tickets should include a concrete title, problem statement, scope, and acceptance criteria.
- For unattended overnight execution, create the next immediately actionable follow-up ticket in `Todo` so Symphony can keep pulling work.
- Use `Backlog` only for lower-confidence ideas, optional future improvements, or work that should not be pulled immediately.
- If a follow-up ticket blocks the current ticket, link it with `blockedBy` and explain the dependency.
- Prefer one ticket per source, parser, cleanup pass, CLI control, or TUI milestone when those concerns can be delivered independently.

Default posture:
1. This is an unattended orchestration session. Do not ask a human to perform follow-up actions unless blocked by missing required auth, permissions, or secrets.
2. Start every task by determining the current Linear status and following the matching flow below.
3. Use one persistent Linear comment headed `## Codex Workpad` as the source of truth for plan, progress, validation, and handoff.
4. Reproduce first. Record concrete evidence of current behavior before editing code.
5. Run `pull` before code edits so the branch starts from the latest `origin/main`.
6. Work only in the provided git workspace clone.
7. Final response must report completed actions and blockers only.

Related skills:
- `linear`: interact with Linear.
- `commit`: produce clean logical commits.
- `push`: publish the current branch.
- `pull`: sync with latest `origin/main`.
- `land`: when the ticket reaches `Merging`, explicitly follow `.codex/skills/land/SKILL.md` and keep going until the PR is merged.

Status map:
- `Backlog` -> out of scope for active execution. Do not modify.
- `Todo` -> immediately move to `In Progress`, create or refresh the workpad, then execute.
- `In Progress` -> implementation actively underway.
- `Human Review` -> PR is attached and validated; wait for human review decision.
- `Rework` -> reviewer requested changes; treat as a fresh execution pass.
- `Merging` -> approved by human; run the `land` flow until merged.
- `Done` -> terminal state; no further action.

Execution flow:
1. Read the current issue state and route using the status map above.
2. Find or create exactly one `## Codex Workpad` comment and keep it updated in place.
3. Put a compact environment stamp at the top of the workpad as `<hostname>:<abs-workdir>@<short-sha>`.
4. Maintain these workpad sections:
   - `Plan`
   - `Acceptance Criteria`
   - `Validation`
   - `Notes`
   - `Confusions` only when needed
5. Before implementing, capture:
   - current repo state (`branch`, `git status`, `HEAD`)
   - reproduction evidence
   - result of syncing with latest `origin/main`
6. Implement against the workpad checklist and update it after each meaningful milestone.
7. Validate the exact behavior you changed. If the issue description includes testing instructions, copy them into the workpad and treat them as required.
8. For real collection runs and validation output, write archives under `/Users/chenjing/dev/chat-history`, not inside this repo.

PR and handoff flow:
1. Before any handoff, ensure the workspace has committed changes on a branch derived from `origin/main`.
2. Push the branch to `origin`.
3. Create or update a GitHub PR for that branch.
4. Attach the PR URL to the Linear issue.
5. Ensure the PR has label `symphony`.
6. Before moving to `Human Review`:
   - required validation is green
   - acceptance criteria are explicitly checked off in the workpad
   - the branch is pushed
   - the PR exists and is linked on the issue
   - all known review feedback has been addressed or explicitly pushed back on
7. Only then move the issue to `Human Review`.
8. When the issue enters `Merging`, run the `land` skill flow. Do not call `gh pr merge` directly outside that flow.
9. After merge is complete, move the issue to `Done`.

Rework flow:
- Re-read the issue and review feedback in full.
- Update the workpad plan to reflect the new approach.
- Continue on the PR branch only if it is still open and reusable.
- If the branch PR is already closed or merged, create a fresh branch from `origin/main` and restart execution from reproduction.

Guardrails:
- Do not leave durable work only in the workspace without commit/push/PR.
- Do not mark an implementation ticket `Done` without a merged PR unless the ticket is explicitly non-code and the workpad explains why.
- Do not expand scope silently; create follow-up tickets instead.
- Do not touch paths outside the workspace except the external archive target `/Users/chenjing/dev/chat-history` when the task explicitly requires real collection output.
- If blocked by missing non-GitHub auth or tools, write a concise blocker brief in the workpad and move the ticket according to the workflow.
- If the current workspace was created by an older rsync-based setup and lacks git history, stop active implementation, record the blocker, and wait for a fresh git-backed workspace.

Expected outputs:
- Source discovery notes: storage paths, formats, caveats, and support level.
- Collector implementation per source.
- Clean normalized chat output focused on memory-relevant content, written to `/Users/chenjing/dev/chat-history` for real collection runs.
- Tests or fixtures that prove parsing and filtering behavior.
- Operator-facing CLI controls, with TUI added only when it materially improves recurring collection tasks.
- When appropriate, newly created Linear tickets that capture discovered follow-up work, with the immediate next work placed in `Todo`.
