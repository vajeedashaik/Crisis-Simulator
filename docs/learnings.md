# Learnings — Failures, Workarounds & Time Savers

## Purpose
Read this file when hitting a repeated error or unexpected behaviour.
Each bullet is a one-liner under 15 words.
No explanations. Only things that save time in future sessions.

---

## Rules for This File
- One line per learning, under 15 words
- Add immediately when a failure or workaround is found
- Never delete a line — they compound over time
- Group by category as list grows

---

## Supabase
- [ ] Add learnings here as they are discovered

## Auth
- [ ] Add learnings here as they are discovered

## Testing
- [ ] Add learnings here as they are discovered

## Next.js
- [ ] Add learnings here as they are discovered

## Deployment
- [ ] Add learnings here as they are discovered

## General
- [ ] Add learnings here as they are discovered

---

## Example Format (remove when real entries exist)
```
## Supabase
- RLS blocks all queries by default — always add select policy first
- supabase db push fails silently if migration has syntax error
- Foreign key inserts require parent row to exist first

## Auth
- JWT expires after 1hr — always call refreshSession before DB ops
- getSession returns null on hard refresh — use onAuthStateChange instead

## Testing
- Jest mock must be reset between tests or state bleeds across cases
- Supabase client must be mocked at module level not inside test

## Next.js
- useEffect runs twice in strict mode — guard with cleanup function
- API routes do not have access to cookies without explicit parsing
```