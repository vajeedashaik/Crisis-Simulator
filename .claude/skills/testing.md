# Skill — Reusable Test Patterns by Feature Type

## Purpose
Read this file when writing tests for a specific feature type.
Contains reusable test templates organised by category.
Always follow naming conventions from docs/testing.md.

---

## Auth Feature Tests

```typescript
// tests/unit/auth/session.test.ts
describe('session handling', () => {
  it('returns null when no session exists', async () => {
    mockSupabase.auth.getSession.mockResolvedValueOnce({
      data: { session: null }, error: null
    })
    const result = await getSession()
    expect(result).toBeNull()
  })

  it('returns session when authenticated', async () => {
    mockSupabase.auth.getSession.mockResolvedValueOnce({
      data: { session: mockSession }, error: null
    })
    const result = await getSession()
    expect(result).toEqual(mockSession)
  })

  it('handles auth error gracefully', async () => {
    mockSupabase.auth.getSession.mockResolvedValueOnce({
      data: { session: null }, error: new Error('auth failed')
    })
    const result = await getSession()
    expect(result).toBeNull()
  })
})
```

---

## DB Query Tests

```typescript
// tests/unit/lib/users.test.ts
describe('fetchUser', () => {
  it('returns user for valid id', async () => {
    mockSupabase.from().select().eq.mockResolvedValueOnce({
      data: mockUser, error: null
    })
    const result = await fetchUser('valid-uuid')
    expect(result).toEqual(mockUser)
  })

  it('returns null for null id', async () => {
    const result = await fetchUser(null as any)
    expect(result).toBeNull()
  })

  it('returns null for empty string id', async () => {
    const result = await fetchUser('')
    expect(result).toBeNull()
  })

  it('returns null on DB error', async () => {
    mockSupabase.from().select().eq.mockResolvedValueOnce({
      data: null, error: new Error('DB error')
    })
    const result = await fetchUser('valid-uuid')
    expect(result).toBeNull()
  })

  it('returns null when DB returns null data', async () => {
    mockSupabase.from().select().eq.mockResolvedValueOnce({
      data: null, error: null
    })
    const result = await fetchUser('valid-uuid')
    expect(result).toBeNull()
  })
})
```

---

## API Route Tests

```typescript
// tests/integration/api/users.integration.test.ts
describe('POST /api/users', () => {
  it('returns 400 when userId is missing', async () => {
    const res = await fetch('/api/users', {
      method: 'POST',
      body: JSON.stringify({})
    })
    expect(res.status).toBe(400)
  })

  it('returns 401 when not authenticated', async () => {
    mockGetSession.mockResolvedValueOnce(null)
    const res = await fetch('/api/users', {
      method: 'POST',
      body: JSON.stringify({ userId: 'uuid' })
    })
    expect(res.status).toBe(401)
  })

  it('returns 200 with valid input and session', async () => {
    mockGetSession.mockResolvedValueOnce(mockSession)
    const res = await fetch('/api/users', {
      method: 'POST',
      body: JSON.stringify({ userId: 'uuid' })
    })
    expect(res.status).toBe(200)
  })

  it('returns 500 on unexpected server error', async () => {
    mockGetSession.mockRejectedValueOnce(new Error('unexpected'))
    const res = await fetch('/api/users', {
      method: 'POST',
      body: JSON.stringify({ userId: 'uuid' })
    })
    expect(res.status).toBe(500)
  })
})
```

---

## React Component Tests

```typescript
// tests/components/UserCard.test.tsx
import { render, screen } from '@testing-library/react'
import UserCard from '@/components/UserCard'

describe('UserCard', () => {
  it('renders user name when user exists', () => {
    render(<UserCard user={mockUser} />)
    expect(screen.getByText(mockUser.name)).toBeInTheDocument()
  })

  it('renders nothing when user is null', () => {
    const { container } = render(<UserCard user={null} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders loading state when isLoading is true', () => {
    render(<UserCard user={null} isLoading={true} />)
    expect(screen.getByTestId('skeleton')).toBeInTheDocument()
  })
})
```

---

## Utility Function Tests

```typescript
// tests/unit/utils/validation.test.ts
describe('isValidId', () => {
  it('returns true for valid uuid', () => {
    expect(isValidId('550e8400-e29b-41d4-a716-446655440000')).toBe(true)
  })

  it('returns false for null', () => {
    expect(isValidId(null)).toBe(false)
  })

  it('returns false for empty string', () => {
    expect(isValidId('')).toBe(false)
  })

  it('returns false for whitespace only', () => {
    expect(isValidId('   ')).toBe(false)
  })

  it('returns false for undefined', () => {
    expect(isValidId(undefined)).toBe(false)
  })
})
```

---

## Mock Templates

### Mock user
```typescript
export const mockUser = {
  id: '550e8400-e29b-41d4-a716-446655440000',
  email: 'test@example.com',
  created_at: '2025-01-01T00:00:00Z'
}
```

### Mock session
```typescript
export const mockSession = {
  access_token: 'mock-token',
  user: mockUser,
  expires_at: Date.now() + 3600
}
```

### Reset mocks between tests
```typescript
beforeEach(() => {
  jest.clearAllMocks()
})
```

---

## What NOT to Do
- Do not write tests that always pass regardless of logic
- Do not skip null and undefined test cases
- Do not forget to reset mocks between tests
- Do not test implementation details — test behaviour
- Do not mock everything — let utility functions run real logic