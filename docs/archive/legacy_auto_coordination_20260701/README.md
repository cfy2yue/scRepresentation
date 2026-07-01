# Legacy Auto-Coordination Archive

Status: historical only. Do not use these files as active execution
instructions.

These documents were written during the 2026-07-01 local/remote auto
coordination dry run. They may still contain useful audit evidence and past
decisions, but the active workflow has been simplified.

Current workflow:

1. Local audit updates the repository-root `local_goal.md`,
   `local_audit.md`, and `local_suggestion.md`.
2. The user manually asks remote Codex to pull the latest GitHub state.
3. Remote Codex reads those three local files plus `goal.md`, then runs the
   next goal.
4. If remote Codex is blocked, it outputs a structured local-audit request
   instead of starting hidden local/remote automation.

