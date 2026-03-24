# Version / Format / API Matrix

## Baseline mapping

- R10.4 / R10.4.1 -> `.pod5`
- R9.4 -> `.fast5`

Use this mapping as a hard default unless user-provided run metadata or project policy overrides it.

## Evidence priority

1. Run metadata or dataset manifest (highest confidence)
2. Repository-level documented convention
3. File extension scan (lowest confidence, still required)

## API checks by format

### POD5 path

- Confirm `import pod5` succeeds.
- Confirm reader entry points used by code exist before main loop.
- Verify iteration returns reads and each read exposes signal data expected by downstream pipeline.

### FAST5 path

- Confirm selected fast5 package import succeeds.
- Confirm open/read APIs match actual file structure and multi-read vs single-read assumptions.
- Validate signal extraction path with at least one sample file before integrating into training/basecalling pipeline.

## Implementation rule

Never auto-switch between `.pod5` and `.fast5` readers without explicit logging and checks.
