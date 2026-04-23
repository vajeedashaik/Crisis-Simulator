# Claude.md — Global Instructions

## Identity
You are a focused, phase-gated AI engineer on this project.
Read only what is needed. Act only on the current phase prompt.

---

## On Every Session Start
1. Read `docs/index.md` to orient yourself
2. Read `session/phase-log.md` to know current progress
3. Read `session/context.md` for any carry-over from last session
4. Load the current phase prompt from `prompts/phase-X.md`
5. Do NOT read all docs at once — load on demand only

---

## Mandatory Rules (Non-Negotiable)

### Phase Gating
- Work only on the current phase prompt
- Do not proceed to next phase without explicit user confirmation
- Confirm all tests pass before marking a phase done

### Error Handling
- Follow `docs/debugging.md` for all error handling patterns
- Every error must log: file name, function, reason, fix hint
- When debugging: check the specific file first, nothing else

### Testing
- Follow `docs/testing.md` for all test patterns
- Update test files after every feature, not at project end
- Provide CLI commands to run tests with every implementation

### Supabase
- Follow `.claude/skills/supabase.md` for all DB operations
- Always provide CLI command first, dashboard steps if CLI not possible

### Code Quality
- Handle all edge cases inline
- No infinite loops, no redundant API calls
- Fail fast, log clearly

---

## On Every Session End
1. Give a one-liner git commit message for every change made
2. Do NOT push to GitHub
3. Update `session/summary.md` with what was done
4. Append one line to `session/phase-log.md`
5. Update `docs/progress.md` with completed items
6. If something failed or a workaround was used → add to `docs/learnings.md`

---

## Reference Map (Load on Demand)

| Need | File |
|------|------|
| What does each file do | `docs/index.md` |
| Requirements + phases | `docs/prd.md` |
| Current progress | `docs/progress.md` |
| Past failures + fixes | `docs/learnings.md` |
| System architecture | `docs/architecture.md` |
| Error handling patterns | `docs/debugging.md` |
| Test strategy + commands | `docs/testing.md` |
| Supabase operations | `.claude/skills/supabase.md` |
| Deploy steps | `docs/deployment.md` |
| Feature workflow | `docs/workflows/feature.md` |
| Bug fix workflow | `docs/workflows/bugfix.md` |
| Refactor workflow | `docs/workflows/refactor.md` |

---

## Never Do
- Do not read all `.md` files at session start
- Do not push to GitHub
- Do not skip writing tests for a feature
- Do not proceed to next phase without user confirmation
- Do not guess Supabase schema — always confirm before migrating