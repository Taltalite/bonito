---
name: first-principles-debug
description: Reframe stalled debugging from first principles. Use when normal troubleshooting no longer works, root causes are unclear, or the team is relying on habits instead of necessities. Force a reset: state the physical essence in one sentence, list non-negotiable facts, separate conventions from required constraints, derive a minimal path from axioms, and pause to clarify user goals before writing code when intent or motivation is ambiguous.
---

# First-Principles Debug

Follow this workflow strictly when debugging is stuck.

## Workflow

1. Pause and align intent before coding.
2. If user goal, motivation, or decision criteria are unclear, ask concise clarifying questions and stop.
3. State the physical essence of the problem in one sentence.
4. List immutable facts that are true regardless of implementation choice.
5. List current plan assumptions and label each as either:
   - Habitual choice
   - Necessary constraint
6. Remove or challenge habitual choices that are not required by the facts.
7. Derive the shortest falsifiable path from the immutable facts.
8. Propose the smallest next experiment and expected observable result.
9. Only implement code after steps 1-8 are explicit.

## Output Format

Use this compact structure:

```md
Physical Essence (1 sentence)
- ...

Immutable Facts
- ...
- ...

Habitual vs Necessary
- [Habitual] ...
- [Necessary] ...

Minimal Path From Axioms
1. ...
2. ...
3. ...

Next Smallest Experiment
- Action: ...
- Expected result: ...
- What it would falsify: ...

Uncertainty Check
- Clear enough to proceed? Yes/No
- If No, ask clarifying question(s) and stop.
```

## Guardrails

- Do not assume the user fully knows what they want.
- Do not hide uncertainty; surface it early.
- Do not continue into deep implementation if objectives are still ambiguous.
- Prefer fewer assumptions, fewer steps, and directly testable claims.
