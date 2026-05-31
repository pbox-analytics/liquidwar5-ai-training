# Agent Build → Review → Commit Workflow

_2026-05-30. Applies to agent (Claude) work in this repo and `liquidwar5-ai`._

## Why this exists

An agent committed a Dockerfile that did **not** build, with a commit message claiming
it was "verified." Nothing re-checked the claim before it landed. This workflow inserts an
**independent verification gate** so that can't happen: no claim of "works"/"verified"/"passing"
is trusted without re-run proof, and no commit lands until a separate reviewer confirms it.

## The rule

**The agent doing the work does NOT commit as a routine step.** Every change passes through:

```
prepare (uncommitted) → evidence block → independent review (re-verifies) → PASS → commit
```

A commit happens **only after** a reviewer returns PASS. On FAIL, fix and re-review.

## 1. Prepare

Make the change in the working tree. Leave it **uncommitted**. Do not `git commit` yet.

## 2. Evidence block

Write a short block stating, for each claim:
- **What changed** (files, summary).
- **The exact command** run to verify it.
- **The real output** proving it — pasted, not summarized. Examples of acceptable proof:
  - a build: the literal `BUILD_EXIT=0` (or `make` exit 0) **and** the artifact (`docker image inspect …` succeeds / `ls -l` the binary).
  - a headless game: the actual `result,<winner>,<ticks>,…` line.
  - a test: the runner's pass/fail summary line.
- If a step was skipped or is unverified, **say so explicitly.** "I believe" / "should work" is not evidence.

> Hard rule: never write "verified", "works", "passing", or "done" without the matching
> pasted output in the evidence block. If you don't have the output, you haven't verified it.

## 3. Independent review

A **separate reviewer** (a reviewer subagent, or the `/code-review` skill) receives the diff
+ the evidence block. Its job is adversarial:
- **Re-run the verification itself** — do not trust the evidence block; reproduce it.
- Review the diff for correctness, security, and that it does what's claimed.
- Return **PASS** (re-verified, diff sound) or **FAIL** (with specific reasons).

The reviewer must not be the same step that produced the change.

## 4. Commit (only on PASS)

After PASS, the worker may commit, with a message that reflects the **re-verified** state and
references the evidence. Show the review result + diff to the user alongside the commit.
Landing to `main` goes via PR so the review is a durable record.

## Roles (this project)

- **Worker** (main agent): prepares change + evidence block; commits **only after** PASS.
- **Reviewer** (subagent / `/code-review`): re-verifies and reviews; returns PASS/FAIL.
- **User**: sees the review + diff; authority to override. Strongly-irreversible or
  outward-facing steps (force-push, remote deletes, prod cluster changes) still get explicit
  user sign-off regardless of PASS.

## Notes

- Generated artifacts are a recurring trap here (stale `configure`, `base.h`, `.o`,
  `liquidwar.dat`). Verification must run from a **clean** state (fresh checkout / `make clean`
  / clean container) so a cached artifact can't fake a pass. See
  `liquidwar5-ai`'s build notes.
- Keep tool output honest: if the shell returns empty/garbled, re-run and confirm via files
  before reporting — don't infer success.
