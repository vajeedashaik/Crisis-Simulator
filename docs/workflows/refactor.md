# Workflow — Refactoring Code

## Purpose
Read this file every time you need to refactor existing code.
Refactoring means improving structure without changing behaviour.
If behaviour changes — that is a feature, not a refactor.

---

## Core Principle
**Tests must pass before AND after every refactor.**
If tests fail after refactor — you changed behaviour. Revert and retry.
Never refactor and fix a bug in the same commit.

---

## Step 1 — Confirm Refactor Scope
Before touching any code, state this out loud:

Refactor target: [file or function]
Reason for refactor: [why this needs to change]
Behaviour change: none
Files I will touch: [exact list]
Files I will NOT touch: [everything else]

- [ ] Confirm with user that this refactor is needed now
- [ ] Confirm all current tests pass before starting:
```bash
bash scripts/test.sh
```
- [ ] Do not proceed if any test is currently failing

---

## Step 2 — Identify What to Refactor
Valid reasons to refactor:
- [ ] Duplicate logic across multiple functions
- [ ] Function doing more than one thing
- [ ] Deeply nested conditionals reducing readability
- [ ] Magic numbers or hardcoded strings
- [ ] Missing or inconsistent error handling format
- [ ] Inconsistent naming conventions
- [ ] Dead code or unused imports

Not valid reasons:
- "It looks messy" without a specific structural problem
- Preference for a different syntax that does the same thing
- Rewriting working code during a bug fix session

---

## Step 3 — Refactor in Small Steps
- [ ] Change one thing at a time
- [ ] Run tests after each individual change
- [ ] Do not batch multiple refactors into one step
- [ ] Keep original logic visible until new logic is confirmed working

### Order of operations:

Extract repeated logic into a shared utility function
Simplify conditionals (early returns over nested if/else)
Rename for clarity (variables, functions)
Remove dead code and unused imports
Standardise error handling format per docs/debugging.md

---

## Step 4 — Run Tests After Every Change
```bash
# After each individual change
npm test -- tests/unit/[affected-file].test.ts 2>&1 | tee logs/test.log

# After all changes complete
bash scripts/test.sh
```
- [ ] Every test that passed before must still pass
- [ ] If a test fails → revert the last change, do not chain fixes

---

## Step 5 — Update Tests if Needed
Refactoring may require test updates only in these cases:
- [ ] A function was renamed → update test description and import
- [ ] A function was split into two → write tests for both
- [ ] A utility was extracted → write a unit test for the utility

Do NOT update tests to make them pass after a refactor.
If a test fails after refactor → the refactor changed behaviour → revert.

---

## Step 6 — Self Review Checklist
Before declaring refactor done:
- [ ] Behaviour is identical before and after
- [ ] All tests pass
- [ ] No new files created outside the plan in Step 1
- [ ] Error handling format matches `docs/debugging.md`
- [ ] No console.log left in code
- [ ] No dead code or unused imports remain
- [ ] No hardcoded values introduced

---

## Step 7 — Update Docs
- [ ] Update `docs/architecture.md` if structure changed
- [ ] Add one-liner to `docs/progress.md`
- [ ] Append one line to `session/phase-log.md`
- [ ] Update `session/summary.md`
- [ ] If a pattern was discovered that saves time → add to `docs/learnings.md`

---

## Step 8 — Git Commit Message
Provide one-liner commit message in this format:

refactor([scope]): [what was improved and how]
Examples:
refactor(auth): extract session validation into shared utility
refactor(dashboard): replace nested conditionals with early returns
refactor(db): standardise error handling across all supabase queries
refactor(utils): remove dead code and unused imports from helpers

**Do NOT push to GitHub. Hand the message to the user.**

---

## Step 9 — Confirm with User
- [ ] Show exactly what changed and in which files
- [ ] Show before/after for key changes
- [ ] Show test results confirming no behaviour change
- [ ] Show commit message
- [ ] Ask: "Refactor complete — ready to continue?"
- [ ] Do NOT proceed until user confirms

---

## Refactor Decision Tree

Refactor needed?
│
▼
All tests passing?
│
├── No → fix failing tests first, then refactor
│
└── Yes → confirm scope → refactor one thing at a time
│
▼
Run tests after each change
│
├── Pass → continue next change
│
└── Fail → revert last change
→ reassess scope
→ do not chain fixes

---

## Refactor Patterns

### Extract repeated logic
```typescript
// Before — same null check in 3 functions
if (!userId || userId === '') return null

// After — shared utility
function isValidId(id: string | null | undefined): boolean {
  return !!id && id.trim() !== ''
}
```

### Early returns over nested conditionals
```typescript
// Before
function processUser(user: User | null) {
  if (user) {
    if (user.isActive) {
      if (user.hasProfile) {
        return user.profile
      }
    }
  }
  return null
}

// After
function processUser(user: User | null) {
  if (!user) return null
  if (!user.isActive) return null
  if (!user.hasProfile) return null
  return user.profile
}
```

### Standardise error handling
```typescript
// Before — inconsistent
try {
  ...
} catch (e) {
  console.log(e)  // wrong
}

// After — per docs/debugging.md
try {
  ...
} catch (err) {
  handleError(err, 'lib/users.ts', 'processUser')
  return null
}
```

---

## What NOT to Do
- Do not refactor and fix a bug in the same commit
- Do not refactor files not listed in Step 1
- Do not update tests to force them to pass after refactor
- Do not batch all refactors into one large change
- Do not refactor during a feature build session
- Do not push to GitHub
- Do not start a refactor if any test is currently failing