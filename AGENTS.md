# AGENTS.md

## Purpose
This repository is edited by Codex on Windows, but code execution and validation should prefer WSL/Linux-compatible commands and scripts.

## Environment
- The source-of-truth working copy is on the Windows filesystem.
- Codex runs as a Windows-native agent.
- Validation should prefer WSL-based execution through short wrapper scripts checked into the repo.
- The remote server is not the primary edit target. Do not assume direct remote access is available.

## Working style
- For anything beyond a tiny one-file change, start in plan mode.
- Before editing, identify the minimal set of files that need to change.
- Prefer small, reviewable patches over broad refactors.
- Preserve existing code style and local conventions unless the task explicitly asks for cleanup.

## Command policy
- Do not compose long PowerShell one-liners.
- Do not generate complex inline `powershell -Command ...` commands.
- Do not generate complex inline `wsl.exe ... bash -lc '...'` commands unless there is no script-based alternative.
- Prefer calling short repository scripts instead of building shell commands dynamically.
- If a needed script does not exist, add a small script first, then call that script.

## Validation policy
- Prefer these repository entrypoints for verification when available:
  - `scripts\\test_wsl.cmd`
  - `scripts\\lint_wsl.cmd`
  - `scripts\\smoke_eval_wsl.cmd`
- If validation cannot be run safely or cheaply, explain exactly what was not run and why.
- Do not run destructive commands.
- Do not install global dependencies unless explicitly requested.

## Editing policy
- Prefer direct file edits over shell-based text transformation.
- Avoid using long `sed`, `perl`, or `python -c` one-liners to rewrite files.
- When changing logic, update only the truly necessary call sites.
- Do not rename files, move directories, or rewrite unrelated code unless required by the task.

## Git policy
- Do not create commits unless explicitly asked.
- Do not push, pull, rebase, or force-push unless explicitly asked.
- You may inspect local diffs and repository status when needed.

## Scope control
- Stay tightly scoped to the user’s request.
- When touching evaluation code, avoid incidental changes to training or data-processing code unless they are required for correctness.
- When changing interfaces, check for direct downstream call sites and update only those that are necessary.

## Done criteria
A task is done only when all of the following are true:
1. The requested code change is implemented.
2. The affected files are internally consistent.
3. Relevant lightweight validation has been run, or the response clearly states why it was not run.
4. The final response lists the key files changed and summarizes the behavioral impact.

## Response style
- Be concise and concrete.
- Report assumptions when they matter.
- When blocked by environment limitations, propose the smallest practical next step.