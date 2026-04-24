# Context — Carry Over for Next Session

## Purpose
Read this file at every session start after index.md and phase-log.md.
Contains only what Claude needs to resume without re-reading everything.
Overwrite when context changes. Keep it minimal and current.

---

## Current Phase
Phase: 7
Prompt file: prompts/phase-7.md
Status: complete

---

## Currently Working On
Feature: HuggingFace Spaces deployment package
File(s): app.py, Dockerfile, README.md, client.py, requirements.txt
Status: done — all endpoints tested locally

---

## Open Questions

None.

---

## Known Blockers

None. openenv not on PyPI — omitted from requirements.txt, included in Dockerfile per spec.

---

## Last Commit Message
feat(deployment): implement phase 7 HuggingFace Spaces deployment package

---

## Do Not Forget

All 7 phases complete. Push to HuggingFace Spaces on user confirmation.

---

## Rules for This File
- Keep this file under 30 lines always
- Overwrite at end of every session
- Only include what is immediately needed to resume
- Do not include explanations — only facts and state
