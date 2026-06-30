# Git And Collaboration

Updated: 2026-07-01

## Project Identity

- Server directory: `/data/cyx/1030/scLatent`
- GitHub target: `https://github.com/cfy2yue/scRepresentation`
- SSH entry: `ssh cyx-server-proxy-cfy`, then `cd /data/cyx/1030/scLatent`
- Shared data root: `/data/cyx/1030/dataset`
- Shared runtime root: `/data/cyx/1030/software`

Current audit note: `/data/cyx/1030/scLatent` is now a local Git worktree on
branch `main` with origin `https://github.com/cfy2yue/scRepresentation.git`. Remote
`main` was initialized from this server workspace on 2026-07-01. `CoupledFM/` and
`scFMBench/` are treated as ordinary scLatent project directories, not
submodules; their previous nested `.git` metadata was moved to
`backup/git_metadata_20260630_225838/`, which is ignored by Git.

Naming note: the server/project folder remains `scLatent`; the GitHub repo name
for collaboration is `scRepresentation`.

## Codex Alone

Codex works on the server project directory. It may run experiments, edit code,
write docs, update `goal.md`, and integrate run results when the user asks for
active project execution.

Long jobs, GPU audits, resource caps, `RUN_STATUS.md`, and project review updates
must follow `/data/cyx/1030/AGENTS.md`. Before any commit, record the key result
paths, split boundary, and current best/blocked state in the relevant project
docs.

## CC Alone

CC should clone `https://github.com/cfy2yue/scRepresentation` on Windows for reading,
direction audit, goal refinement, documentation cleanup, and code review.

If CC needs server-only information, use:

```bash
ssh cyx-server-proxy-cfy
cd /data/cyx/1030/scLatent
```

CC should not pretend Windows can run the GPU/data-heavy workflow. If CC needs
to edit server code or launch jobs, confirm Codex is paused for that file,
branch, or task.

## Codex Plus CC

CC is best used for direction checks, goal polishing, independent critique,
doc/plan drafting, and monitoring. Codex owns server execution, resource
scheduling, experiment launches, implementation changes, and result integration.

When both are active, use a handoff note or dated doc section to record who owns
which files, tasks, and branches. Avoid editing the same code file at the same
time. Markdown can be parallelized, but prefer appending dated sections over
rewriting long history.

## Git Hygiene

Track candidates:

- `README.md`, `goal.md`, `docs/*.md`, `prompts/*.md`
- small configs, launch wrappers, tests, and source code
- `CoupledFM/` and `scFMBench/` source files as ordinary monorepo content

Do not track by default:

- `dataset`, `runs/`, `reports/`, `logs/`, `pretrainckpt/`, `scFM_cache/`,
  `scFM_output/`, `scFM_pretrained/`, `scFM_third_party/`, `.venvs/`,
  `docs/local_archive/`
- checkpoints, raw data, large generated artifacts, secrets, tokens, API keys
- nested external repos under ignored mirrors such as `external_review/` and
  `scFM_third_party/`

Runtime note: `init-scdfm.sh` resolves the scDFM Conda environment through
`/data/cyx/1030/software/miniconda3`, which is intended to be a compatibility
link to the existing `/data/cyx/software/miniconda3` install unless a future
maintenance window performs a tested physical migration.
