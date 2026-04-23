# Debugging — Error Handling, Logging & Debug Strategy

## Purpose
Read this file when an error occurs or when implementing error
handling for any feature. Do not guess — follow this exactly.

---

## Core Principle
**Check the specific file where the error occurred. Nothing else.**
Do not scan the entire codebase. Logs must tell you:
- WHERE it broke (file + function)
- WHY it broke (reason)
- HOW to fix it (hint)

---

## Standard Error Log Format

Every error logged must follow this exact structure:

[ERROR] [timestamp] [file:function] — reason — fix hint

**Example:**
[ERROR] [2025-01-15T10:32:00Z] [lib/supabase.ts:fetchUser] — user_id is null — check auth session before calling fetchUser

---

## Error Handler Template

### TypeScript / Next.js
```typescript
function handleError(error: unknown, file: string, fn: string): void {
  const timestamp = new Date().toISOString()
  const reason = error instanceof Error ? error.message : String(error)
  const hint = getFixHint(reason)

  console.error(`[ERROR] [${timestamp}] [${file}:${fn}] — ${reason} — ${hint}`)

  // Write to logs/errors.log in dev
  if (process.env.NODE_ENV === 'development') {
    appendToLog('logs/errors.log', `[ERROR] [${timestamp}] [${file}:${fn}] — ${reason} — ${hint}`)
  }
}

function getFixHint(reason: string): string {
  if (reason.includes('null')) return 'check for null before using this value'
  if (reason.includes('undefined')) return 'confirm the value exists before accessing'
  if (reason.includes('network')) return 'check network connection and API endpoint'
  if (reason.includes('permission')) return 'verify Supabase RLS policies'
  if (reason.includes('timeout')) return 'increase timeout or check slow query'
  return 'check function inputs and dependencies'
}
```

### Usage in any function
```typescript
// file: lib/supabase.ts
async function fetchUser(userId: string) {
  try {
    if (!userId) throw new Error('user_id is null')
    const { data, error } = await supabase.from('users').select('*').eq('id', userId)
    if (error) throw error
    return data
  } catch (err) {
    handleError(err, 'lib/supabase.ts', 'fetchUser')
    return null
  }
}
```

---

## Logging Levels

| Level | When to Use |
|-------|------------|
| `[ERROR]` | Something broke, feature cannot continue |
| `[WARN]` | Something unexpected but recoverable |
| `[INFO]` | Key state changes, successful operations |
| `[DEBUG]` | Verbose, dev-only, remove before production |

---

## Debug Steps — When Something Breaks

1. Read `logs/errors.log` — find the exact `[file:function]`
2. Open only that file
3. Check the function mentioned in the log
4. Verify inputs to that function
5. Check for null/undefined before the failure point
6. Fix inline, do not refactor adjacent code
7. Re-run the specific test for that function only

**Do not open other files unless the log explicitly points to them.**

---

## Supabase Specific Errors

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `permission denied` | RLS policy blocking query | Check policy in Supabase dashboard → Auth → Policies |
| `relation does not exist` | Migration not applied | Run `supabase db push` |
| `violates foreign key` | Referenced row missing | Insert parent record first |
| `JWT expired` | Auth token stale | Refresh session with `supabase.auth.refreshSession()` |
| `null value in column` | Missing required field | Validate inputs before insert |

---

## Edge Case Checklist
Before shipping any function, verify:
- [ ] What happens if input is null or undefined?
- [ ] What happens if the DB returns empty array?
- [ ] What happens if the API call times out?
- [ ] What happens if the user is not authenticated?
- [ ] What happens if this runs twice simultaneously?

---

## What NOT to Do
- Do not use `console.log` for errors — use `console.error` with the format above
- Do not catch an error and do nothing with it
- Do not log sensitive data (passwords, tokens, PII)
- Do not open unrelated files to debug an error
- Do not refactor while debugging — fix first, refactor later