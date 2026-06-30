# Prompt: Organize The Stock Project

You are working in `/data/cyx/1030/stock`, a fully independent stock/quant
project. Do not modify `/data/cyx/1030/scLatent` or `/data/cyx/1030/CellClip`
unless explicitly asked.

Goal: make the stock project understandable and maintainable. This is a
documentation and cleanup task first. Do not run expensive backtests, external
data downloads, or long agents unless I explicitly ask.

Read first:

```text
AGENTS.md
goal.md
README.md
PROJECT_BRIEF.md
MEMORY.md
docs/START_HERE.md
docs/PROJECT_REVIEW.md
docs/PROJECT_PROGRESS_AND_NEXT_PLAN.md
docs/WORKFLOW.md
docs/DATA_FLOW.md
docs/DECISIONS.md
docs/BUGS_AND_FIXES.md
```

Then produce/update:

1. A concise project entry document:
   - goal;
   - current workflow;
   - data sources and permission boundaries;
   - current best strategy/evaluation status;
   - how an agent should resume.
2. A directory map that separates:
   - source code;
   - configs;
   - docs;
   - memory/ledger files;
   - data;
   - reports/runs/logs;
   - caches/temp files.
3. A cleanup inventory:
   - keep;
   - archive;
   - safe delete;
   - unknown/needs user decision.
4. Remove only clearly reproducible noise such as pytest caches, empty temp
   files, or duplicate generated scratch files. Do not delete data, reports,
   ledgers, strategy cards, configs, or backtest outputs unless the inventory
   proves they are redundant.
5. Update `docs/PROJECT_REVIEW.md` with a dated organization checkpoint and
   `docs/START_HERE.md` if it is stale.

Final response should list changed files, deleted cache/noise files, unresolved
cleanup decisions, and the next concrete stock-project action.
