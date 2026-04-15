# AGENTS.md

## Purpose
This repository may be edited by Codex or similar coding agents.
The goal of this file is to define stable repository-wide working rules so that changes remain minimal, reviewable, reproducible, and consistent with the project's research and engineering workflows.

This repository is a research-oriented codebase with both engineering and experimental concerns.
Agents must optimize not only for code correctness, but also for reproducibility, backward compatibility, and clear validation boundaries.

---

## Environment assumptions
- The primary development environment is WSL2.
- WSL2 is configured with mirrored networking.
- Network access from the development environment may depend on an HTTPS proxy exposed by the Windows host on a designated local port.
- The coding agent should assume that repository work, local execution, and validation happen inside WSL/Linux unless the task explicitly requires Windows-specific behavior.
- Do not assume direct remote server access is available.
- Do not assume unrestricted outbound network access; prefer repository-local work and existing project environments.

---

## General principles
- Stay tightly scoped to the user’s request.
- Prefer the smallest correct and reviewable change over broad refactors.
- Preserve existing architecture, module boundaries, naming conventions, CLI behavior, file layout, and coding style unless the task explicitly requires otherwise.
- Favor clarity, maintainability, and reproducibility over cleverness.
- Avoid incidental edits in unrelated files.
- Do not silently change scientific or evaluation semantics.

---

## Research-code priorities
- Treat reproducibility as a first-class requirement.
- Preserve existing experiment behavior unless the requested task explicitly changes it.
- Distinguish clearly between:
  - training behavior,
  - inference/basecalling behavior,
  - data preparation,
  - evaluation/metrics,
  - experimental utilities,
  - documentation.
- Do not mix these concerns without necessity.
- When changing logic that may affect scientific results, ensure the behavioral impact is explicitly stated.

---

## Planning
- For anything beyond a trivial one-file edit, begin by identifying:
  1. the user-visible goal,
  2. the minimal set of files that must change,
  3. whether the change affects training, inference, data processing, evaluation, CLI, config, or docs,
  4. the cheapest meaningful validation step.
- Prefer incremental patches over large multi-purpose rewrites.
- When requirements are ambiguous, choose the interpretation that minimizes irreversible or repository-wide change.

---

## Editing policy
- Prefer direct file edits over shell-based text rewriting.
- Avoid long one-liners for `sed`, `perl`, `python -c`, or shell-based bulk rewrites unless there is no cleaner alternative.
- Do not rename files, move directories, or reorganize modules unless required for correctness or explicitly requested.
- Reuse existing helpers, patterns, and abstractions before introducing new ones.
- Keep new code locally understandable; avoid introducing framework-like abstractions for a narrow task.
- When changing an interface, update only the necessary downstream call sites.
- When changing default values or config semantics, verify that the change is intentional and clearly visible.

---

## CLI and user-surface policy
- This repository may expose user-facing commands and options.
- When adding or modifying CLI behavior:
  - preserve backward compatibility where practical,
  - avoid breaking existing commands unless explicitly requested,
  - update argument parsing, help text, and relevant documentation together,
  - keep new options discoverable and clearly named.
- Do not silently repurpose an existing flag to mean something substantially different.
- New functionality should prefer additive changes over breaking changes.

---

## Model, training, and inference policy
- Treat model architecture changes as high-impact changes.
- Preserve existing model-loading and inference behavior unless the task explicitly requires modification.
- When adding new outputs, heads, losses, or training pathways:
  - avoid breaking existing checkpoints unless explicitly intended,
  - keep old code paths valid where practical,
  - make compatibility assumptions explicit,
  - ensure failure modes are understandable.
- Do not silently change tensor shapes, output conventions, decoding assumptions, or checkpoint expectations without updating the affected call sites and documentation.
- Separate experimental paths from stable/default paths whenever practical.

---

## Configuration policy
- Treat config files and command-line defaults as part of the public interface of the repository.
- Do not silently alter config meaning.
- When adding a new config field or option:
  - use names consistent with existing conventions,
  - provide a sensible default,
  - update nearby docs or examples when appropriate.
- Keep backward compatibility in mind for old configs, scripts, and checkpoints.

---

## Data and evaluation policy
- Do not silently change dataset assumptions, label semantics, metric definitions, or evaluation criteria.
- When touching evaluation code:
  - avoid incidental changes to training or preprocessing code unless required for correctness,
  - preserve metric names and meanings unless explicitly requested,
  - clearly state any change that could alter historical comparability.
- When touching preprocessing or label-generation logic:
  - consider downstream impact on training and evaluation,
  - avoid hidden format changes.

---

## Validation policy
- Run the lightest meaningful validation that matches the scope of the change.
- Prefer targeted validation before broad validation.
- Examples:
  - small CLI change -> help text / argument parsing / focused smoke test,
  - small logic fix -> targeted unit or functional check,
  - model-path change -> shape/path smoke test before any expensive run,
  - doc-only change -> no code execution required.
- For expensive training workflows, prefer smoke tests, dry runs, import checks, shape checks, or minimal command-level validation rather than full training unless explicitly requested.
- If validation cannot be run safely, cheaply, or because of environment limitations, clearly state:
  - what was not run,
  - why it was not run,
  - what the next best validation step would be.

---

## Self-validation boundary
- Local validation in WSL2 is intended only to verify that the basic code path is wired correctly.
- Treat local validation primarily as smoke testing, not as final experimental validation.
- Prefer checks such as:
  - importability,
  - CLI help and argument parsing,
  - config loading,
  - minimal command invocation,
  - shape/path consistency,
  - non-destructive dry-run or short-run execution where available.
- Do not block progress merely because full datasets, large checkpoint files, or remote-only resources are unavailable in WSL2.
- Missing local dataset files, missing large artifacts, or environment-specific paths are acceptable during local self-validation as long as the modified code is internally consistent and the basic execution path is verified.
- Do not add fake dataset paths, fake files, or hardcoded machine-specific workarounds just to satisfy local checks.
- Prefer validating that the program fails in an expected and interpretable way when required local data is absent.
- The production-like execution environment is the remote server reached via Git synchronization, so final heavy validation should be framed as a remote follow-up step rather than a requirement for local completion.
- For expensive workflows such as model training, full evaluation, or dataset-dependent experiments, prefer local smoke tests over full runs unless the task explicitly asks for end-to-end execution.
- When reporting validation, clearly distinguish:
  - what was verified locally in WSL2,
  - what was not verified because it depends on remote datasets, checkpoints, or server resources.

---

## Execution policy
- Prefer short, repeatable, repository-local commands.
- Prefer running commands inside WSL/Linux.
- Avoid complex inline Windows shell commands unless the task is specifically about Windows behavior.
- Avoid rebuilding long shell commands repeatedly; use small scripts or existing repository entrypoints when practical.
- Prefer project-local environments and repository-managed tooling over global system modifications.
- Be mindful that network-dependent commands may require the existing WSL2 proxy setup to function.

---

## Proxy and network awareness
- Assume the development environment may depend on a Windows-host HTTPS proxy reachable from WSL2.
- Do not modify proxy environment variables, shell init files, mirrored-network settings, Windows host networking, or system-wide network configuration unless explicitly requested.
- Do not assume network failures are code failures.
- Avoid adding instructions or scripts that hardcode machine-specific hostnames, ports, or secrets unless the task explicitly asks for local environment setup.
- When a command depends on network access, prefer solutions compatible with the existing WSL2 mirrored-network and host-proxy workflow.

---

## Safety and dependency policy
- Do not run destructive commands.
- Do not delete user data, overwrite local work, reset unrelated changes, or modify untracked files unless explicitly requested.
- Do not install global dependencies or alter machine-level configuration unless explicitly requested.
- Prefer project-local, reversible changes.
- Treat secrets, tokens, credentials, dataset paths, and private data as sensitive; do not expose them in logs, code, or summaries.

---

## Git policy
- Do not create commits unless explicitly asked.
- Do not push, pull, rebase, merge, or force-push unless explicitly asked.
- You may inspect repository status, tracked files, and diffs when needed.
- Keep changes easy to review in a normal Git diff.
- Do not mix unrelated cleanup into a feature or bugfix branch.

---

## Documentation policy
- When code changes affect user-facing behavior, developer workflow, CLI usage, configuration, model compatibility, data format, or validation steps, update nearby documentation where appropriate.
- Prefer small, local documentation updates over broad rewrites.
- Keep examples aligned with current repository structure and commands.
- For research-facing changes, document behavior precisely enough that future users can understand what changed and why.

---

## Scope control
- Do not turn a local task into a repository-wide redesign.
- Do not perform opportunistic cleanup unless it is required for correctness, compatibility, or readability of the requested change.
- When touching shared code paths, check obvious direct dependents, but avoid cascading refactors without clear need.
- Keep experimental additions isolated when possible.

---

## Completion criteria
A task is complete only when all of the following are true:
1. The requested change is implemented.
2. The modified files are internally consistent.
3. Appropriate lightweight validation has been run, or the lack of validation is clearly explained.
4. Any important behavioral impact on training, inference, evaluation, CLI, config, or compatibility is explicitly summarized.
5. The final response identifies the key files changed.

---

## Response style
- Be concise, concrete, and technically grounded.
- State important assumptions when they affect correctness.
- Report validation results accurately.
- Distinguish what was changed from what was only inspected.
- When blocked, propose the smallest practical next step instead of a broad redesign.