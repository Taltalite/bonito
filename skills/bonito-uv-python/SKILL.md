---
name: bonito-uv-python
description: Use when working in the Bonito project and any task may run Python, install Python dependencies, or invoke project scripts. This skill enforces the project's uv-managed virtual environment workflow: prefer the local .venv, run Python through uv-aware paths, and assume the repository itself is installed editable via `uv pip install -e .` unless evidence shows otherwise.
---

# Bonito UV Python

This project uses `uv venv` for its Python environment and keeps the repository installed editable with `uv pip install -e .`.

## Use This Skill When

- You are about to run `python`, `pytest`, or a Python module.
- You need project imports such as `import bonito` to resolve correctly.
- You need to install or refresh Python dependencies.
- You are debugging why a script behaves differently from the checked-out source.

## Core Rules

1. Prefer the repository local environment at `.venv`.
2. Assume editable install is expected. If imports fail or project code looks stale, refresh with `uv pip install -e .` inside the active `.venv`.
3. Do not assume system Python matches the project environment.
4. When giving commands, favor forms that clearly use the project environment.

## Preferred Command Patterns

From the repository root:

```bash
. .venv/bin/activate
python -m module_name
python path/to/script.py
```

For one-off commands when activation is inconvenient:

```bash
.venv/bin/python -m module_name
.venv/bin/python path/to/script.py
```

To refresh the editable install:

```bash
. .venv/bin/activate
uv pip install -e .
```

## Working Norms

- Before running Python, check whether `.venv` exists in the repo root.
- If `.venv` is missing and the task depends on Python execution, state that the expected uv-managed environment is absent.
- If a command uses `python`, `pip`, or `pytest`, prefer the `.venv` interpreter or activate `.venv` first.
- Prefer `uv pip` over bare `pip` for environment-managed installs in this project.
- When reporting reproduction steps, include environment-aware commands rather than generic `python` examples.

## Quick Checks

- Interpreter: `.venv/bin/python --version`
- Editable import path: `.venv/bin/python -c "import bonito; print(bonito.__file__)"`
- Refresh install: `. .venv/bin/activate && uv pip install -e .`

## Do Not

- Use `pip install -e .` as the default recommendation for this project.
- Use a global Python interpreter when project-local `.venv` is available.
- Assume import errors mean the source tree is broken before checking the editable install and interpreter path.