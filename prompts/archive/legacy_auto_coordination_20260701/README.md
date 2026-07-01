# Legacy Prompt Archive

Status: historical only. Do not use these prompts as active workflow prompts.

The active local/remote loop is manual:

- local side: understand the project, audit the goal and implementation route,
  then update `local_goal.md`, `local_audit.md`, and `local_suggestion.md`;
- remote side: after the user says local audit is finished, pull GitHub, read
  those files plus `goal.md`, and execute the user-started goal;
- blocked remote side: output a structured local-audit request for the user to
  bring back to the local side.

