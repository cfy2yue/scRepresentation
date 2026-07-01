# CC Remote Codex Operation Runbook

Updated: 2026-07-01

Tracked concise runbook for CC/Cursor when it needs to coordinate remote Codex
on `cyx-server-proxy-cfy`.

## First Principle

CC is the local coordinator. Remote Codex is the server executor. Do not start
remote Codex from an unsynced local state. Before remote work, compare:

```powershell
git -C E:\cc_workspace\<repo> fetch origin --prune
git -C E:\cc_workspace\<repo> status -sb
git -C E:\cc_workspace\<repo> rev-list --left-right --count HEAD...origin/main
ssh cyx-server-proxy-cfy "cd /data/cyx/1030/<repo> && git fetch origin --prune && git status -sb && git rev-list --left-right --count HEAD...origin/main"
```

If local, GitHub, and server differ, sync first or report the divergence.

## Default Multi-Project Pattern

- Main CC does intake, sync checks, `ccusage` checks, synthesis, commits, pushes,
  remote session launch, and monitoring.
- Use one CC subagent per project for audit and initial exploration. Keep each
  subagent scoped to one repo; subagents do not push or start remote jobs unless
  explicitly delegated.
- Use one remote Codex goal session per remote task, normally one `tmux` session
  per project/goal. Do not run unrelated projects in the same Codex session.
- CC updates `goal.md` and `docs/CC_AUDIT_AND_HANDOFF_<date>.md`; remote Codex
  receives a thin pointer to those version-controlled docs.
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
ssh cyx-server-proxy-cfy "cd /data/cyx/1030 && pwd && git -C scLatent status -sb && git -C CellClip status -sb && git -C stock status -sb && tmux ls 2>/dev/null || true && codex --version"
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

## Interactive Tmux Session

```bash
cd /data/cyx/1030/scLatent
tmux new -s codex_sclatent_goal_$(date +%Y%m%d)
codex -C /data/cyx/1030/scLatent -m gpt-5.5 -s workspace-write -a never --no-alt-screen "$(cat /tmp/codex_handoff_prompt.txt)"
```

Detach: `Ctrl-b d`.

Reattach:

```bash
tmux attach -t codex_sclatent_goal_YYYYMMDD
```

## Resume

```bash
codex resume --last -C /data/cyx/1030/scLatent -m gpt-5.5 -s workspace-write -a never
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

## Monitoring

For long work, remote Codex should use detached `tmux`, `nohup`, or a scheduler
and maintain `runs/<run>/RUN_STATUS.md`. CC should monitor sparingly and report
exact server paths, not copied log dumps.
