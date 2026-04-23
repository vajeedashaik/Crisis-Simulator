# Skill — Debugging Patterns by Error Type

## Purpose
Read this file when stuck on a specific error type.
Contains reusable debug snippets organised by error category.
Always check logs/errors.log first before opening this file.

---

## Null & Undefined Errors

### Pattern
```typescript
// Guard at function entry — always first line
function myFunction(input: string | null) {
  if (!input) {
    handleError(new Error('input is null'), 'file.ts', 'myFunction')
    return null
  }
  // safe to use input below
}
```

### Checklist
- [ ] Is the value null before it enters the function?
- [ ] Is the value undefined because an async call hasn't resolved?
- [ ] Is optional chaining `?.` missing somewhere?
- [ ] Is the DB returning null instead of empty array?

---

## Async & Promise Errors

### Pattern
```typescript
// Always await, always catch
async function fetchData() {
  try {
    const result = await someAsyncCall()
    if (!result) throw new Error('result is empty')
    return result
  } catch (err) {
    handleError(err, 'file.ts', 'fetchData')
    return null
  }
}
```

### Checklist
- [ ] Is `await` missing before an async call?
- [ ] Is a `.then()` chain missing a `.catch()`?
- [ ] Is a Promise being returned without awaiting it?
- [ ] Are two async calls racing without proper sequencing?

---

## Type Errors

### Pattern
```typescript
// Validate type at boundary before using
function processUser(user: unknown) {
  if (!user || typeof user !== 'object') {
    handleError(new Error('invalid user type'), 'file.ts', 'processUser')
    return null
  }
  const typedUser = user as User
  // safe to use typedUser below
}
```

### Checklist
- [ ] Is an `any` type hiding a real type mismatch?
- [ ] Is an API response being used without type validation?
- [ ] Is a number being used where a string is expected?

---

## React Render Errors

### Pattern
```typescript
// Guard before rendering
function UserCard({ user }: { user: User | null }) {
  if (!user) return null  // or return <Skeleton />
  return <div>{user.name}</div>
}
```

### Checklist
- [ ] Is a component rendering before data is loaded?
- [ ] Is `useEffect` missing a dependency causing stale state?
- [ ] Is state being mutated directly instead of via setter?
- [ ] Is `useEffect` running twice due to React strict mode?

---

## API Route Errors

### Pattern
```typescript
// Always validate request body
export async function POST(req: Request) {
  try {
    const body = await req.json()
    if (!body?.userId) {
      return Response.json(
        { error: 'userId is required' },
        { status: 400 }
      )
    }
    // proceed
  } catch (err) {
    handleError(err, 'app/api/route.ts', 'POST')
    return Response.json({ error: 'internal server error' }, { status: 500 })
  }
}
```

### Checklist
- [ ] Is the request body being parsed correctly?
- [ ] Is the auth session being checked before processing?
- [ ] Is a 400 returned for bad input vs 500 for server error?
- [ ] Is the API route method correct (GET vs POST)?

---

## Environment Variable Errors

### Pattern
```typescript
// Validate env vars at startup
const requiredEnvVars = [
  'NEXT_PUBLIC_SUPABASE_URL',
  'NEXT_PUBLIC_SUPABASE_ANON_KEY'
]

requiredEnvVars.forEach((key) => {
  if (!process.env[key]) {
    throw new Error(`Missing required env var: ${key}`)
  }
})
```

### Checklist
- [ ] Is `.env.local` present and not committed?
- [ ] Are `NEXT_PUBLIC_` prefixes correct for client side vars?
- [ ] Are env vars available in the deployment environment?
- [ ] Was the dev server restarted after adding new env vars?

---

## What NOT to Do
- Do not add console.log and forget to remove it
- Do not catch an error and return undefined silently
- Do not assume the error is in a different file than the log says
- Do not fix the symptom without understanding the root cause