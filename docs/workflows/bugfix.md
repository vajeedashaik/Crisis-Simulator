# Workflow — Fixing a Bug

## Purpose
Read this file every time you need to fix a bug.
Follow steps in order. Do not open files not mentioned in the log.

---

## Core Principle
**The log tells you where to look. Trust the log. Check one file.**
Do not scan the codebase. Do not refactor while fixing.

---

## Step 1 — Read the Error First
Before touching any code:
- [ ] Read `logs/errors.log` — find the exact error entry
- [ ] Identify: file, function, reason, fix hint from the log
- [ ] Read `docs/learnings.md` — has this failed before?
- [ ] If yes → apply the known fix, skip to Step 5

Error log format to look for:

[ERROR] [timestamp] [file:function] — reason — fix hint

---

## Step 2 — Isolate Before Fixing
State this out loud before touching code:
Error in: [file:function]
Reason: [from log]
Fix hint: [from log]
Only file I will open: [file]

- [ ] Open only the file named in the log
- [ ] Find only the function named in the log
- [ ] Do not open any other file unless the log explicitly points there

---

## Step 3 — Understand the Failure
Inside the identified function:
- [ ] Check what input is coming in
- [ ] Check where exactly it breaks (line from log)
- [ ] Check if it is a null/undefined issue
- [ ] Check if it is an auth/session issue
- [ ] Check if it is a Supabase RLS issue
- [ ] Reference `docs/debugging.md` → Supabase Specific Errors table

---

## Step 4 — Write the Fix
- [ ] Fix only the broken logic
- [ ] Do not rename variables or restructure the function
- [ ] Do not fix adjacent code that looks messy
- [ ] Add or improve error handling inline if it was missing
- [ ] Confirm fix hint from log is addressed

---

## Step 5 — Write or Update the Test
- [ ] Find the existing test file for this feature
- [ ] If the failing case was not covered → add a test for it now
- [ ] The test must reproduce the exact failure condition
- [ ] Run only the test for this file first:
```bash
npm test -- tests/unit/[feature-name].test.ts 2>&1 | tee logs/test.log
```
- [ ] Confirm it passes before running full suite

---

## Step 6 — Run Full Test Suite
```bash
bash scripts/test.sh
```
- [ ] All tests pass
- [ ] No previously passing test is now failing
- [ ] If a new failure appears → treat it as a new bug, do not chain fixes

---

## Step 7 — Update Docs
- [ ] Add one-liner to `docs/progress.md` under the feature
- [ ] If this was a repeated failure or needed a workaround → add to `docs/learnings.md`
- [ ] Append one line to `session/phase-log.md`
- [ ] Update `session/summary.md`

---

## Step 8 — Git Commit Message
Provide one-liner commit message in this format:

fix([scope]): [what was broken and what fixed it]
Examples:
fix(auth): handle null session before calling fetchUser
fix(dashboard): return empty array instead of null on no results
fix(db): add missing RLS policy for users table select

**Do NOT push to GitHub. Hand the message to the user.**

---

## Step 9 — Confirm with User
- [ ] Show exactly what was changed and in which file
- [ ] Show test results
- [ ] Show commit message
- [ ] Ask: "Confirmed fixed — ready to continue?"
- [ ] Do NOT proceed until user confirms

---

## Bugfix Decision Tree

Error occurs
│
▼
Read logs/errors.log
│
├── Entry found → go to Step 2
│
└── No entry → add logging first, reproduce error, then fix
│
▼
Check docs/learnings.md
│
├── Known issue → apply fix directly
│
└── New issue → isolate → fix → document

---

## Common Bug Patterns

| Symptom | Likely Cause | Where to Look |
|---------|-------------|---------------|
| Function returns null unexpectedly | Missing null check on input | The function's first few lines |
| Supabase returns empty data | RLS policy blocking query | Supabase Dashboard → Policies |
| Auth token errors | Session not refreshed | auth handler file |
| Test passes locally, fails in CI | Env var missing in CI | .env setup + CI config |
| Infinite re-render in React | useEffect dependency missing | The specific component file |
| Type error at runtime | TypeScript type not enforced at boundary | Input validation of the function |

---

## What NOT to Do
- Do not open the entire codebase to find the bug
- Do not fix multiple bugs in one session without separate commits
- Do not refactor while fixing — fix first, refactor separately
- Do not skip updating the test file
- Do not push to GitHub
- Do not chain fixes — one bug, one fix, one commit