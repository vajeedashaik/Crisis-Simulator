# Phase Log — Full History

## Purpose
Read this file at every session start to know overall progress.
One line per phase or significant milestone. Never overwrite — only append.
This is the only file that keeps permanent history across all sessions.

---

## Format
[YYYY-MM-DD] [Phase X] [status] — [one liner of what happened]

## Status Tags
STARTED     — phase work has begun
PARTIAL     — some features done, phase not complete
COMPLETE    — all features done, all tests passing
BLOCKED     — cannot proceed, reason in line
ROLLED BACK — changes reverted, reason in line

---

## Log
[YYYY-MM-DD] [Phase 1] STARTED — project scaffolding begun

---

## Rules for This File
- Never delete or overwrite any line
- Append only — one line per session or milestone
- Keep each line under 15 words after the date and phase tag
- This file is the single source of truth for project history