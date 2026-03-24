---
name: nanopore-data-version-guard
description: Add nanopore raw-signal data context and enforce version-aware I/O decisions. Use when handling Oxford Nanopore current signal files, especially when chemistry/platform versions (for example R10.4.1 vs R9.4) affect file formats, parsing dependencies, and reader APIs. Require explicit mapping between version and file suffix (.pod5 for R10-series, .fast5 for R9-series), require dependency/API validation before coding, and require ingestion checks that prove reads were loaded correctly.
---

# Nanopore Data Version Guard

Apply deterministic checks before implementing or modifying data I/O.

## Core Rules

1. Identify and record the nanopore data version assumption first.
2. Map version to raw-file format before selecting code path.
3. Select dependencies and APIs that match the file format.
4. Add runtime checks that fail fast on wrong suffix, missing files, empty reads, or invalid API usage.
5. If version is ambiguous, ask the user to confirm and stop implementation.

## Version-Format Mapping

Use this baseline mapping unless project docs explicitly override it:

- R10 series (including R10.4 and R10.4.1): expect `.pod5`.
- R9 series (including R9.4): expect `.fast5`.

Treat extension as necessary but not sufficient evidence. Confirm with metadata when available.

## Project-Specific Guardrail (bonito)

In this repository, the format dispatcher is currently pod5-oriented (`bonito/reader.py` includes `"pod5"` in `__formats__`, and `bonito/pod5.py` contains the active reader path).

Therefore:

- If incoming data is `.pod5`, continue with the existing pod5 path.
- If incoming data is `.fast5`, do not silently proceed through pod5 logic.
- Implement an explicit decision: either add/enable fast5 reader support, or convert/prepare data upstream, then document that decision in code comments and PR notes.

## Dependency and API Validation Checklist

Before writing logic, verify imports and APIs for the chosen format:

- `.pod5` path: verify `pod5` package import and required reader API availability.
- `.fast5` path: verify chosen fast5 library import and reader API availability (for example `ont_fast5_api` or a vetted `h5py`-based path).

During implementation:

- Guard imports with clear error messages.
- Validate API entry points used by code (method names, argument semantics, return types).
- Prefer small wrapper functions for file opening and read iteration so checks are centralized.

## Ingestion Integrity Checks (Required)

For every new or modified reader flow, implement checks for:

1. Input discovery: directory exists, files found, allowed suffixes only.
2. Format consistency: reject mixed `.pod5` and `.fast5` unless explicitly supported.
3. Read presence: non-zero read count.
4. Signal validity: non-empty signal arrays, numeric dtype, expected rank.
5. Optional ID filtering: report missing IDs ratio when selection is used.
6. Failure behavior: raise explicit, actionable errors (no silent fallback).

## Output Contract For Responses

When using this skill, respond with this structure:

```md
Data Version Assumption
- ...

Format Decision
- Expected suffix: ...
- Evidence: ...

Dependency + API Plan
- Package(s): ...
- API entry points: ...
- Validation steps: ...

Ingestion Checks
- [ ] Path/suffix check
- [ ] Format consistency check
- [ ] Non-empty reads check
- [ ] Signal validity check
- [ ] Filtered-ID completeness check (if applicable)

Uncertainty
- Ambiguity present? Yes/No
- If Yes: ask clarification and stop.
```

Read additional details in `references/version-format-api-matrix.md` only when needed.
