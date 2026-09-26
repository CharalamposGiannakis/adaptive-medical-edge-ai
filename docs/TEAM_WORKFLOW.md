# Team workflow

This document defines the work slices and their acceptance criteria. It is not
the live task board: GitHub Issues carry the current status, owner, discussion,
and pull request for each slice.

## Pull rules

Every issue title starts with exactly one status:

```text
[BLOCKED] -> [READY] -> [CLAIMED] -> [REVIEW] -> [DONE]
```

A `[DONE]` issue is also closed. All other states remain open. Codex or Claude Code can update
these prefixes directly, so the repository does not depend on custom labels or a
project board.

1. Claim the `[READY]` issue that you can work on.
2. Assign yourself and create a short branch named
   `T0xx-short-name`.
3. Open a draft pull request early so the files being changed are visible.
4. Work on only one implementation slice at a time.
5. Edit only the files named by the issue unless the reason is discussed first.
6. If blocked, record the exact reason, change the issue to `[BLOCKED]`, and
   take the next `[READY]` slice.
7. If no implementation issue is ready, review the oldest open pull request.
8. Every pull request needs one independent review before merge.
9. Nobody pushes directly to `main`; `main` must remain runnable.

Codex or Claude Code can create the branch, implement the slice, run its checks, prepare the
commit and pull request, and respond to review comments when it has repository
and GitHub permission. The human owner remains responsible for confirming that
the issue's intent and acceptance criteria are satisfied.

## Definition of done for every slice

- Its acceptance criteria pass.
- Relevant tests or a documented manual check pass.
- User-facing instructions match commands that actually exist.
- No unrelated files, caches, raw databases, or environment files are committed.
- The pull request states what changed, how it was checked, and what slice is
  unblocked next.
