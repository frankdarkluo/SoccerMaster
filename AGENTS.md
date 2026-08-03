# AGENTS.md

## Project Priority Skills

This repository should prioritize these two skills whenever they match the task:

1. `karpathy-guidelines` (always on — trigger whenever the agent is invoked)
   - Any agent invocation triggers this skill, regardless of task.
   - Use for writing, reviewing, debugging, or refactoring code.
   - Leads when thinking through and adding/deleting/modifying code: explicit assumptions, simple solutions, surgical edits, verifiable success criteria.
   - Do not add speculative abstractions, unrelated refactors, or broad cleanup.
   - Pair with `ponytail` on any add/delete/modify (see "Code Changes: Ponytail First"): karpathy sets the approach, ponytail keeps the diff minimal.

2. `academic-research-suite`
   - Use for research planning, literature review, experiment planning, manuscript work, citation checks, paper review, and research-to-paper workflows.
   - Route broad or vague paper topics through Socratic research-question narrowing before outlining or drafting.
   - Verify claims, citations, statistics, policies, and current facts against primary or authoritative sources.

## Code Changes: Ponytail First

`karpathy-guidelines` leads the reasoning; `ponytail` applies specifically when adding/deleting/modifying code to keep the diff minimal. Run `ponytail` (full mode default) and its subtasks on any code change. Write no redundant code.

- `ponytail` — apply the ladder on ANY code change: YAGNI, reuse what's already here, stdlib/native before deps, one line before fifty, shortest working diff.
- `ponytail-review` — after a change, review the diff for over-engineering (what to delete).
- `ponytail-audit` — audit the whole repo for bloat when asked to clean up broadly.
- `ponytail-debt` — list every `ponytail:` shortcut comment left behind.
- Mark deliberate simplifications with a `ponytail:` comment naming the ceiling and upgrade path.

## Local Working Rules

- Keep implementation changes narrowly scoped to the user request.
- Prefer repository docs, code inspection, tests, and configs over assumptions.
- For non-trivial code changes, define the success criteria before editing and verify before claiming completion.
- Keep research outputs clear about what is evidence, inference, recommendation, or unverified.
- Commit only after a full plan is completed, or when I explicitly say it's OK to commit. Do not commit mid-plan or on your own initiative.
- When reporting information to me, be extremely concise and sacrifice grammar for the sake of concision.
