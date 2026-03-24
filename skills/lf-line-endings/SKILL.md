---
name: lf-line-endings
description: Keep file edits in this Bonito repository on LF line endings for Ubuntu and WSL workflows. Use when creating, rewriting, or patching files in this project so Codex avoids introducing CRLF and preserves Linux-friendly newlines.
---

# LF Line Endings

Keep repository file edits on LF (`\n`) line endings.

## Rules

- Default to LF for every file you create or rewrite in this repository.
- Preserve LF when editing existing files.
- Avoid Windows-style CRLF (`\r\n`) unless the user explicitly asks for it.
- Treat accidental CRLF introduction as a regression and fix it before finishing.

## Preferred Editing Behavior

- Prefer editing methods that preserve repository-friendly line endings.
- Be especially careful when writing files from PowerShell or Windows-side tools, because they may emit CRLF by default.
- After substantial rewrites or generated content, verify line endings if there is any doubt.

## Verification

- If a file was rewritten by a Windows-side command, check whether it now contains `\r\n`.
- If CRLF was introduced, normalize the file back to LF before finishing.
- When relevant, mention that LF was preserved or restored.