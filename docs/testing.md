# Testing — Strategy, Patterns & CLI Commands

## Purpose
Read this file when writing tests or running them.
Follow this exactly for every feature implementation.

---

## Core Principle
**Write tests after every feature. Never at the end of the project.**
Every test failure must tell you:
- WHICH test failed (test name + file)
- WHY it failed (expected vs received)
- WHERE it failed (line number)

---

## Testing Stack
- **Unit + Integration:** Jest + ts-jest
- **API Routes:** Supertest
- **UI Components:** React Testing Library
- **DB (Supabase):** Mocked with jest.mock or Supabase local instance

---

## Folder Structure

tests/
├── unit/
│   ├── lib/
│   └── utils/
├── integration/
│   ├── api/
│   └── db/
├── components/
└── setup.ts

---

## Test File Naming Convention
[feature-name].test.ts        ← unit test
[feature-name].integration.test.ts  ← integration test
[component-name].test.tsx     ← component test

---

## Standard Test Template

```typescript
// tests/unit/lib/fetchUser.test.ts
import { fetchUser } from '@/lib/supabase'

describe('fetchUser', () => {
  it('returns user when valid userId is provided', async () => {
    const result = await fetchUser('valid-uuid')
    expect(result).not.toBeNull()
    expect(result).toHaveProperty('id')
  })

  it('returns null when userId is null', async () => {
    const result = await fetchUser(null as any)
    expect(result).toBeNull()
  })

  it('returns null when userId is empty string', async () => {
    const result = await fetchUser('')
    expect(result).toBeNull()
  })

  it('handles DB error gracefully', async () => {
    // mock supabase to throw
    jest.spyOn(supabase, 'from').mockImplementationOnce(() => {
      throw new Error('DB connection failed')
    })
    const result = await fetchUser('valid-uuid')
    expect(result).toBeNull()
  })
})
```

---

## Edge Cases to Test for Every Feature

| Scenario | What to Test |
|----------|-------------|
| Empty input | null, undefined, empty string |
| Auth state | unauthenticated user, expired token |
| DB response | empty array, null, malformed data |
| Network | timeout, connection failure |
| Duplicates | calling same function twice simultaneously |
| Boundary | max length strings, zero values, negative numbers |

---

## CLI Commands

### Run all tests
```bash
npm test 2>&1 | tee logs/test.log
```

### Run a specific test file
```bash
npm test -- tests/unit/lib/fetchUser.test.ts 2>&1 | tee logs/test.log
```

### Run tests in watch mode
```bash
npm test -- --watch
```

### Run tests with coverage
```bash
npm test -- --coverage 2>&1 | tee logs/test.log
```

### Run only failed tests
```bash
npm test -- --onlyFailures 2>&1 | tee logs/test.log
```

---

## Test Log Format
All test output pipes to `logs/test.log`.
When a test fails, the log will show:

FAIL tests/unit/lib/fetchUser.test.ts
● fetchUser › returns null when userId is null

expect(received).toBeNull()

Received: { id: 'abc', name: 'test' }

  14 |   it('returns null when userId is null', async () => {
  15 |     const result = await fetchUser(null as any)
> 16 |     expect(result).toBeNull()
     |                    ^
  17 |   })

at Object.<anonymous> (tests/unit/lib/fetchUser.test.ts:16:20)

---

## scripts/test.sh
```bash
#!/bin/bash
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting test run..." >> logs/test.log
npm test 2>&1 | tee -a logs/test.log
EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -ne 0 ]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] TESTS FAILED — see above for details" >> logs/test.log
else
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ALL TESTS PASSED" >> logs/test.log
fi
exit $EXIT_CODE
```

---

## Test Update Rules
- [ ] Write tests immediately after each feature is implemented
- [ ] Never delete existing tests — only add or update
- [ ] If a test is skipped, add a comment explaining why
- [ ] All tests must pass before marking a phase complete
- [ ] Run full test suite before every deployment

---

## Mocking Supabase
```typescript
// tests/setup.ts
jest.mock('@/lib/supabase', () => ({
  supabase: {
    from: jest.fn().mockReturnValue({
      select: jest.fn().mockReturnValue({
        eq: jest.fn().mockResolvedValue({ data: [], error: null })
      }),
      insert: jest.fn().mockResolvedValue({ data: null, error: null }),
      update: jest.fn().mockResolvedValue({ data: null, error: null }),
      delete: jest.fn().mockResolvedValue({ data: null, error: null })
    }),
    auth: {
      getSession: jest.fn().mockResolvedValue({ data: { session: null }, error: null }),
      refreshSession: jest.fn().mockResolvedValue({ data: null, error: null })
    }
  }
}))
```

---

## What NOT to Do
- Do not write tests after the entire project is done
- Do not mock everything — integration tests must hit real logic
- Do not skip edge case tests to save time
- Do not ignore a failing test — fix it before moving on
- Do not write tests that always pass regardless of logic