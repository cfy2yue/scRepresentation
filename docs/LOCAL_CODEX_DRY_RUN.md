# Local Codex Dry Run

Updated: 2026-07-01

This file records the intended local-Codex dry-run workflow for the scLatent
repo cloned from:

```text
https://github.com/cfy2yue/scRepresentation.git
```

The local folder should still be named `scLatent`, for example:

```powershell
git clone https://github.com/cfy2yue/scRepresentation.git scLatent
```

## Purpose

This is a workflow and documentation smoke test, not the main future local
workflow. The expected long-term local collaborator is CC/Cursor. Local Codex may
be used briefly to validate that startup docs, Git hygiene, SSH handoff, and
remote-process monitoring instructions are understandable.

## Local Codex May Do

- Read and summarize docs/source.
- Audit project direction, stale paths, and handoff clarity.
- Draft goals, plans, review reports, and small documentation patches.
- Prepare handoff notes for server Codex.
- Run tiny local static checks if dependencies are already present.

## Local Codex Must Not Do

- Run GPU jobs, model training, checkpoint evaluation, or large data reads on
  Windows.
- Pretend local Windows has `/data/cyx/1030/dataset`, server caches, reports, or
  checkpoints.
- Commit or print tokens, API keys, secrets, runs, reports, logs, checkpoints,
  venvs, or local archives.
- Edit the same code files as server Codex without an explicit ownership note.

## Remote Cooperation Dry Run

Use SSH only for lightweight status checks unless the user explicitly asks for
server execution:

```powershell
ssh cyx-server-proxy-cfy "cd /data/cyx/1030/scLatent && git status -sb && git remote -v"
```

When handing a server task to Codex, include:

- objective;
- files inspected locally;
- exact requested server action;
- boundaries;
- expected output paths;
- acceptance check.

## Remote Process Monitoring

Do not start a remote process for this dry run. For future real long jobs:

- use detached `tmux`, `nohup`, or a scheduler;
- write `runs/<run>/RUN_STATUS.md`;
- record command, start time, PID/session, log path, expected outputs, and stop
  rule;
- check logs sparingly;
- report exact server paths, not copied log dumps.

If local Codex/CC is only monitoring, it should not repeatedly poll. Ask server
Codex for a concise status update when needed.

## Known Dry-Run Notes

- The correct Codex CLI order is `codex -a never exec ...`, not
  `codex exec -a never ...`.
- For long prompts, write the prompt to `/tmp/codex_handoff_prompt.txt` on the
  server and run `codex -a never exec ... - < /tmp/codex_handoff_prompt.txt` to
  avoid shell word splitting.
- `failed to refresh available models: timeout waiting for child process to
  exit` may appear as a non-blocking remote Codex smoke warning if the requested
  task still completes. Record it and escalate only if it becomes frequent.

