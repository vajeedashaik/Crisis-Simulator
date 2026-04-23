# Index — What Every File Does & When to Read It

## Purpose
This is your orientation file. Read this once at session start.
It tells you what exists, what it does, and when to load it.

---

## Core Files

| File | Purpose | When to Read |
|------|---------|--------------|
| `claude.md` | Global rules, session start/end checklist | Every session start |
| `docs/index.md` | This file — master map | Every session start |
| `session/phase-log.md` | One-liner per phase, current status | Every session start |
| `session/context.md` | Carry-over notes from last session | Every session start |
| `session/summary.md` | What was done in last session | Only if context is unclear |

---

## Project State Files

| File | Purpose | When to Read |
|------|---------|--------------|
| `docs/prd.md` | Full PRD, all 8 phases, requirements | Phase start or on demand |
| `docs/progress.md` | One-liner per feature, pass/fail status | When checking what's done |
| `docs/learnings.md` | Past failures, workarounds, gotchas | When hitting a repeated error |
| `docs/architecture.md` | Stack, data models, folder structure | When building or modifying structure |

---

## How-To Reference Files

| File | Purpose | When to Read |
|------|---------|--------------|
| `docs/debugging.md` | Error format, log structure, debug steps | When an error occurs |
| `docs/testing.md` | Test strategy, CLI commands, log format | When writing or running tests |
| `docs/deployment.md` | Supabase CLI + dashboard deploy steps | When deploying any change |
| `.claude/skills/supabase.md` | Supabase patterns, queries, migrations | When writing DB code |
| `.claude/skills/debugging.md` | Reusable debug snippets by error type | When stuck on a specific error |
| `.claude/skills/testing.md` | Reusable test patterns by feature type | When writing tests |

---

## Workflow Files

| File | Purpose | When to Read |
|------|---------|--------------|
| `docs/workflows/feature.md` | Step-by-step for building a new feature | Every new feature |
| `docs/workflows/bugfix.md` | Step-by-step for fixing a bug | Every bug fix |
| `docs/workflows/refactor.md` | Step-by-step for refactoring code | Every refactor |

---

## Phase Prompts

| File | Purpose | When to Read |
|------|---------|--------------|
| `prompts/phase-1.md` through `phase-8.md` | Individual phase instructions | Only the current phase |

---

## Scripts & Logs

| File | Purpose | When to Read |
|------|---------|--------------|
| `scripts/test.sh` | Run all tests with logging | When running tests |
| `scripts/dev.sh` | Start dev environment | When starting dev |
| `scripts/deploy.sh` | Deploy to Supabase + environment | When deploying |
| `logs/errors.log` | Runtime error output | When debugging a failure |
| `logs/test.log` | Test run output | When a test fails |

---

## Rules for This File
- Do not add implementation details here
- Do not add code here
- Only update this file if a new reference file is added to the project