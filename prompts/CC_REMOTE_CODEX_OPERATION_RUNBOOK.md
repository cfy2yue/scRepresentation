# CC Remote Codex Operation Runbook

Updated: 2026-07-01

Tracked concise runbook for CC/Cursor when it needs to coordinate remote Codex
on `cyx-server-cfy`.

## First Principle

CC is the local coordinator. Remote Codex is the server executor. Do not start
remote Codex from an unsynced local state. Before remote work, compare:

```powershell
git -C E:\cc_workspace\<repo> fetch origin --prune
git -C E:\cc_workspace\<repo> status -sb
git -C E:\cc_workspace\<repo> rev-list --left-right --count HEAD...origin/main
ssh cyx-server-cfy "cd /data/cyx/1030/<repo> && git fetch origin --prune && git status -sb && git rev-list --left-right --count HEAD...origin/main"
```

If local, GitHub, and server differ, sync first or report the divergence.

## Default Multi-Project Pattern

- Main CC does intake, sync checks, CC-side `ccusage` checks, synthesis, commits,
  pushes, remote session launch, and monitoring. The `ccusage` threshold is
  local-machine CC/Claude Code usage above USD 90 in the last 24 hours; it does
  not apply to remote Codex goal sessions unless the user explicitly says so.
- Use one CC subagent per project for audit and initial exploration. Keep each
  subagent scoped to one repo; subagents do not push or start remote jobs unless
  explicitly delegated.
- Use one remote Codex goal session per remote task, normally one `tmux` session
  per project/goal. Do not run unrelated projects in the same Codex session.
- CC preserves `goal.md` as the durable north-star objective and updates dated
  handoff/strategy docs such as `docs/CC_AUDIT_AND_HANDOFF_<date>_<slug>.md`.
  Remote Codex receives a thin pointer to those version-controlled docs.
- The same pattern should scale to more projects and more remote servers by
  updating the local project registry first.

## Local Command Paths

Use this in already-open terminals if `git` or `codex` is not found:

```powershell
$env:Path = 'C:\Users\lenovo\AppData\Roaming\npm;C:\Users\lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd;' + $env:Path

where.exe codex
codex --version
where.exe git
git --version
```

Expected local Codex path/version:

- `C:\Users\lenovo\AppData\Roaming\npm\codex.cmd`
- `codex-cli 0.142.4`

## SSH Status Check

```powershell
ssh cyx-server-cfy "cd /data/cyx/1030 && pwd && git -C scLatent status -sb && git -C CellClip status -sb && git -C stock status -sb && tmux ls 2>/dev/null || true && codex --version"
```

## Prompt File Pattern

Use a prompt file rather than fragile shell quoting:

```bash
cat > /tmp/codex_handoff_prompt.txt <<'PROMPT'
Project: scLatent
Server path: /data/cyx/1030/scLatent

Goal:
One concrete outcome with success criteria.

Files to read first:
- goal.md
- docs/START_HERE.md
- docs/CC_CODEX_COOPERATION_PROTOCOL.md

Permissions:
- No secrets.
- No experiments/GPU/API/backtests unless explicitly requested.
- Preserve datasets, runs, reports, logs, checkpoints, caches, and archives.

Output:
- Exact changed files or exact status paths.
- Blockers and next action.
PROMPT
```

## Read-Only Smoke

Use this for remote Codex audit without edits:

```bash
codex -a never exec \
  -C /data/cyx/1030/scLatent \
  -m gpt-5.4-mini \
  -s read-only \
  --output-last-message /tmp/codex_readonly_audit.md \
  - < /tmp/codex_handoff_prompt.txt
```

Current remote CLI detail: use `codex -a never exec ...`; `codex exec -a never`
failed in the dry run.

## Workspace-Write Task

Use this only after CC/user has decided the server-side doc/code task:

```bash
codex -a never exec \
  -C /data/cyx/1030/scLatent \
  -m gpt-5.5 \
  -s workspace-write \
  --output-last-message runs/<run>/codex_last_message.md \
  - < /tmp/codex_handoff_prompt.txt
```

Use `gpt-5.5` for hard implementation or research planning. Use `gpt-5.4-mini`
for cheap status/doc checks. Use `danger-full-access` only when the user
explicitly asks for fully unattended broad server control.

## Visible Interactive Tmux Goal Session (Default)

Use this by default for long-running work. It gives the user and CC a real
terminal session to watch, attach, steer, and resume. `--no-alt-screen` keeps
scrollback visible to `tmux capture-pane`.

```bash
cd /data/cyx/1030/scLatent
cat > /tmp/codex_goal_sclatent_example.txt <<'PROMPT'
/goal Read goal.md as the durable project objective and hard boundary. Execute
the implementation contract in docs/CC_AUDIT_AND_HANDOFF_20260701_example.md.
Do not rewrite the durable objective. Start by writing a brief plan to
runs/<run>/RUN_STATUS.md. Keep progress there, honor stop rules, and continue
until the DONE criteria are met or a real blocker is recorded.
PROMPT

cat > /tmp/launch_sclatent_goal_example.sh <<'LAUNCH'
#!/usr/bin/env bash
set -euo pipefail
PROJECT=/data/cyx/1030/scLatent
PROMPT_FILE="$1"
cd "$PROJECT"
codex features enable goals >/dev/null 2>&1 || true
PROMPT="$(cat "$PROMPT_FILE")"
exec codex -C "$PROJECT" -m gpt-5.5 -s workspace-write -a never --no-alt-screen "$PROMPT"
LAUNCH
chmod +x /tmp/launch_sclatent_goal_example.sh

SESSION=codex_sclatent_goal_$(date +%Y%m%d_%H%M)
tmux new-session -d -s "$SESSION" \
  "bash /tmp/launch_sclatent_goal_example.sh /tmp/codex_goal_sclatent_example.txt"
tmux attach -t "$SESSION"
```

Detach: `Ctrl-b d`.

Reattach:

```bash
tmux attach -t codex_sclatent_goal_YYYYMMDD_HHMM
tmux capture-pane -p -S -200 -t codex_sclatent_goal_YYYYMMDD_HHMM
```

## Resume

```bash
codex resume --last -C /data/cyx/1030/scLatent -m gpt-5.5 -s workspace-write -a never --no-alt-screen
```

Noninteractive:

```bash
codex -a never exec resume --last \
  -C /data/cyx/1030/scLatent \
  -m gpt-5.5 \
  -s workspace-write \
  "Continue the previous goal. First summarize current status and exact paths."
```

## Goal Mode

If goal support is needed:

```bash
codex features enable goals
```

For complex work, start with `/plan` and shape a measurable goal before
implementation. A good handoff goal includes success criteria, permissions,
stop rules, files to read first, files not to touch, and expected output paths.

Use `codex exec` only for smoke checks, short bounded tasks, or when the user
explicitly accepts an invisible one-shot run. Manual launch by the user is
preferred over invisible long-running `exec` when observability matters.

## Monitoring

For long work, remote Codex should use detached `tmux`, `nohup`, or a scheduler
and maintain `runs/<run>/RUN_STATUS.md`. CC should monitor sparingly and report
exact server paths, not copied log dumps.
