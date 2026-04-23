# Workflow — Building a New Feature

## Purpose
Read this file every time you start building a new feature.
Follow steps in order. Do not skip any step.

---

## Step 1 — Understand Before Coding
- [ ] Read the current phase prompt from `prompts/phase-X.md`
- [ ] Read `docs/progress.md` to confirm what is already done
- [ ] Read `docs/architecture.md` to understand existing structure
- [ ] Identify the exact files you will create or modify
- [ ] List all edge cases before writing any code

---

## Step 2 — Plan the Feature
State this out loud before coding:

Feature: [name]
Files to create: [list]
Files to modify: [list]
DB changes needed: yes/no
Edge cases: [list]
Tests needed: [list]

---

## Step 3 — DB Changes First (if needed)
- [ ] Write migration file before any application code
- [ ] Enable RLS on every new table
- [ ] Write RLS policies in the migration file
- [ ] Run `supabase db push` to apply
- [ ] Confirm in dashboard or via `supabase migration list`
- [ ] Reference `docs/deployment.md` for exact commands

---

## Step 4 — Write the Code
- [ ] Create or modify only the files identified in Step 2
- [ ] Add error handling to every function — follow `docs/debugging.md`
- [ ] No function should silently fail
- [ ] Handle all edge cases identified in Step 1
- [ ] No hardcoded values — use env vars or constants file
- [ ] No unused imports or dead code

---

## Step 5 — Write Tests Immediately
- [ ] Create test file at `tests/unit/` or `tests/integration/`
- [ ] Follow naming convention from `docs/testing.md`
- [ ] Cover all edge cases from Step 1
- [ ] Cover happy path + at least 2 failure paths per function
- [ ] Reference `.claude/skills/testing.md` for reusable patterns

---

## Step 6 — Run Tests
```bash
# Run only this feature's tests first
npm test -- tests/unit/[feature-name].test.ts 2>&1 | tee logs/test.log

# If passing, run full suite
bash scripts/test.sh
```
- [ ] All tests pass before moving forward
- [ ] If a test fails — read `logs/test.log`, fix the specific function only

---

## Step 7 — Self Review Checklist
Before declaring feature done:
- [ ] Error handling in place for every function
- [ ] No console.log left in code (only console.error with format)
- [ ] All edge cases handled
- [ ] Tests written and passing
- [ ] No new files created outside the plan in Step 2
- [ ] DB migration applied and confirmed
- [ ] RLS policies applied if new table was created

---

## Step 8 — Update Docs
- [ ] Add one-liner to `docs/progress.md`
- [ ] Append one line to `session/phase-log.md`
- [ ] Update `session/summary.md` with what was done
- [ ] If any failure or workaround occurred → add to `docs/learnings.md`
- [ ] If architecture changed → update `docs/architecture.md`

---

## Step 9 — Git Commit Message
Provide one-liner commit message in this format:

feat([scope]): [what was done in plain english]
Examples:
feat(auth): add email login with session handling
feat(dashboard): add user profile fetch with error handling
feat(db): add users table migration with RLS policies

**Do NOT push to GitHub. Hand the message to the user.**

---

## Step 10 — Confirm with User
- [ ] Show user what was built
- [ ] Show test results
- [ ] Show commit message
- [ ] Ask: "Ready to move to next feature or phase?"
- [ ] Do NOT proceed until user confirms

---

## What NOT to Do
- Do not write tests after all features are done
- Do not modify files outside the plan without flagging it
- Do not proceed to next feature if current tests are failing
- Do not push to GitHub
- Do not skip the docs update in Step 8
- Do not start next phase without user confirmation