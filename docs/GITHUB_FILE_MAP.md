# GitHub File Map

Updated: 2026-07-01

GitHub target: `https://github.com/cfy2yue/scRepresentation`

URL notes: URLs below use `main` as the intended publication branch. The local
`/data/cyx/1030/scLatent` directory is initialized as a Git worktree with origin
`https://github.com/cfy2yue/scRepresentation.git`. Remote `main` was initialized from
the server workspace on 2026-07-01.

| Local path | Git repo | GitHub URL | Purpose | Track in git? | Notes |
|---|---|---|---|---|---|
| `README.md` | `cfy2yue/scRepresentation` | `https://github.com/cfy2yue/scRepresentation/blob/main/README.md` | Project entrypoint | yes | Server path and shared roots. |
| `AGENTS.md` | `cfy2yue/scRepresentation` | `https://github.com/cfy2yue/scRepresentation/blob/main/AGENTS.md` | Agent protocol | yes | Local file is a symlink to root `../AGENTS.md`. |
| `goal.md` | `cfy2yue/scRepresentation` | `https://github.com/cfy2yue/scRepresentation/blob/main/goal.md` | Current actionable state | yes | Historical path references are provenance. |
| `docs/START_HERE.md` | `cfy2yue/scRepresentation` | `https://github.com/cfy2yue/scRepresentation/blob/main/docs/START_HERE.md` | Short onboarding | yes | First doc for CC/Codex orientation. |
| `docs/WORKSPACE_ORGANIZATION.md` | `cfy2yue/scRepresentation` | `https://github.com/cfy2yue/scRepresentation/blob/main/docs/WORKSPACE_ORGANIZATION.md` | Workspace boundary rules | yes | Current path authority. |
| `docs/CODEX_CC_COLLABORATION.md` | `cfy2yue/scRepresentation` | `https://github.com/cfy2yue/scRepresentation/blob/main/docs/CODEX_CC_COLLABORATION.md` | Codex/CC role split | yes | Keep concise. |
| `docs/GIT_AND_COLLABORATION.md` | `cfy2yue/scRepresentation` | `https://github.com/cfy2yue/scRepresentation/blob/main/docs/GIT_AND_COLLABORATION.md` | Git and ownership rules | yes | Created for CC/Codex handoff. |
| `docs/GITHUB_FILE_MAP.md` | `cfy2yue/scRepresentation` | `https://github.com/cfy2yue/scRepresentation/blob/main/docs/GITHUB_FILE_MAP.md` | File URL map | yes | This file. |
| `docs/LOCAL_CODEX_DRY_RUN.md` | `cfy2yue/scRepresentation` | `https://github.com/cfy2yue/scRepresentation/blob/main/docs/LOCAL_CODEX_DRY_RUN.md` | Local Codex dry-run notes | yes | Supports workflow testing before normal CC flow. |
| `docs/CC_CODEX_COOPERATION_PROTOCOL.md` | `cfy2yue/scRepresentation` | `https://github.com/cfy2yue/scRepresentation/blob/main/docs/CC_CODEX_COOPERATION_PROTOCOL.md` | CC/local and remote Codex operating protocol | yes | Read before parallel local/remote work. |
| `prompts/CC_LOCAL_REMOTE_SYSTEM_PROMPT.md` | `cfy2yue/scRepresentation` | `https://github.com/cfy2yue/scRepresentation/blob/main/prompts/CC_LOCAL_REMOTE_SYSTEM_PROMPT.md` | Canonical CC system prompt | yes | Root workspace copy may also exist. |
| `prompts/CC_REMOTE_CODEX_OPERATION_RUNBOOK.md` | `cfy2yue/scRepresentation` | `https://github.com/cfy2yue/scRepresentation/blob/main/prompts/CC_REMOTE_CODEX_OPERATION_RUNBOOK.md` | Remote Codex command runbook for CC | yes | SSH/tmux/model/approval/sandbox/goal patterns. |
| `docs/PROJECT_OVERVIEW.md` | `cfy2yue/scRepresentation` | `https://github.com/cfy2yue/scRepresentation/blob/main/docs/PROJECT_OVERVIEW.md` | Current project overview | yes | Current paths should point under `scLatent/`. |
| `docs/PROJECT_REVIEW.md` | `cfy2yue/scRepresentation` | `https://github.com/cfy2yue/scRepresentation/blob/main/docs/PROJECT_REVIEW.md` | Current review state | yes | Full pre-slim log is server-local only. |
| `docs/EXPERIMENT_INDEX.md` | `cfy2yue/scRepresentation` | `https://github.com/cfy2yue/scRepresentation/blob/main/docs/EXPERIMENT_INDEX.md` | Compact experiment index | yes | Long run details stay in `runs/`/reports/local archive. |
| `prompts/` | `cfy2yue/scRepresentation` | `https://github.com/cfy2yue/scRepresentation/tree/main/prompts` | Reusable prompts | yes | Small text only. |
| `CoupledFM/` | `cfy2yue/scRepresentation` | `https://github.com/cfy2yue/scRepresentation/tree/main/CoupledFM` | LatentFM/CoupledFM source | yes | Previous nested Git metadata backed up under `backup/`. |
| `scFMBench/` | `cfy2yue/scRepresentation` | `https://github.com/cfy2yue/scRepresentation/tree/main/scFMBench` | Benchmark source | yes | Previous nested Git metadata backed up under `backup/`. |
| `runs/`, `reports/`, `logs/` | none by default | not tracked | Provenance outputs | no | Preserve locally; do not add large outputs. |
| `docs/local_archive/` | none | not tracked | Full pre-slim Markdown history | no | Server-local provenance only. |
| `dataset`, `scFM_pretrained/`, `pretrainckpt/` | none by default | not tracked | Shared/large assets | no | Keep on server or package separately. |
