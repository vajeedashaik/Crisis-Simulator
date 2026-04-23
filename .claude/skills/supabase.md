# Skill — Supabase Patterns & Operations

## Purpose
Read this file when writing any Supabase code.
Contains reusable patterns, query templates and common gotchas.
Do not guess schema or policies — confirm first.

---

## Client Setup

### Client initialisation (Next.js)
```typescript
// lib/supabase.ts
import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

export const supabase = createClient(supabaseUrl, supabaseAnonKey)
```

### Server side client (API routes / server components)
```typescript
// lib/supabase-server.ts
import { createClient } from '@supabase/supabase-js'

export const supabaseAdmin = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)
```

---

## Auth Patterns

### Get current session
```typescript
const { data: { session }, error } = await supabase.auth.getSession()
if (!session) {
  handleError(new Error('no active session'), 'file.ts', 'functionName')
  return null
}
```

### Listen to auth state changes
```typescript
supabase.auth.onAuthStateChange((event, session) => {
  if (event === 'SIGNED_OUT') {
    // clear local state
  }
  if (event === 'TOKEN_REFRESHED') {
    // update session in state
  }
})
```

### Sign out
```typescript
const { error } = await supabase.auth.signOut()
if (error) handleError(error, 'file.ts', 'signOut')
```

---

## Query Patterns

### Select with error handling
```typescript
async function getRows(table: string, userId: string) {
  try {
    const { data, error } = await supabase
      .from(table)
      .select('*')
      .eq('user_id', userId)

    if (error) throw error
    return data ?? []
  } catch (err) {
    handleError(err, 'lib/supabase.ts', 'getRows')
    return []
  }
}
```

### Insert with error handling
```typescript
async function insertRow(table: string, payload: object) {
  try {
    const { data, error } = await supabase
      .from(table)
      .insert(payload)
      .select()
      .single()

    if (error) throw error
    return data
  } catch (err) {
    handleError(err, 'lib/supabase.ts', 'insertRow')
    return null
  }
}
```

### Update with error handling
```typescript
async function updateRow(table: string, id: string, payload: object) {
  try {
    const { data, error } = await supabase
      .from(table)
      .update(payload)
      .eq('id', id)
      .select()
      .single()

    if (error) throw error
    return data
  } catch (err) {
    handleError(err, 'lib/supabase.ts', 'updateRow')
    return null
  }
}
```

### Delete with error handling
```typescript
async function deleteRow(table: string, id: string) {
  try {
    const { error } = await supabase
      .from(table)
      .delete()
      .eq('id', id)

    if (error) throw error
    return true
  } catch (err) {
    handleError(err, 'lib/supabase.ts', 'deleteRow')
    return false
  }
}
```

---

## Migration Patterns

### Standard migration file structure
```sql
-- supabase/migrations/[timestamp]_create_users.sql

-- Create table
CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT NOT NULL UNIQUE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Users can view own record"
ON users FOR SELECT
USING (auth.uid() = id);

CREATE POLICY "Users can update own record"
ON users FOR UPDATE
USING (auth.uid() = id);
```

### Add column migration
```sql
ALTER TABLE users ADD COLUMN IF NOT EXISTS display_name TEXT;
```

### Add index migration
```sql
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
```

---

## RLS Policy Templates

### Full CRUD for owner
```sql
CREATE POLICY "owner select" ON table_name FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "owner insert" ON table_name FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "owner update" ON table_name FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "owner delete" ON table_name FOR DELETE USING (auth.uid() = user_id);
```

### Public read, owner write
```sql
CREATE POLICY "public read" ON table_name FOR SELECT USING (true);
CREATE POLICY "owner insert" ON table_name FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "owner update" ON table_name FOR UPDATE USING (auth.uid() = user_id);
```

### Service role bypass (admin operations)
```sql
-- Use supabaseAdmin client (service role) — bypasses RLS
-- Never expose service role key client side
```

---

## Realtime Patterns

### Subscribe to table changes
```typescript
const channel = supabase
  .channel('table-changes')
  .on('postgres_changes',
    { event: '*', schema: 'public', table: 'your_table' },
    (payload) => {
      console.info(`[INFO] realtime event: ${payload.eventType}`)
    }
  )
  .subscribe()

// Cleanup
channel.unsubscribe()
```

---

## Storage Patterns

### Upload file
```typescript
async function uploadFile(bucket: string, path: string, file: File) {
  try {
    const { data, error } = await supabase.storage
      .from(bucket)
      .upload(path, file, { upsert: true })

    if (error) throw error
    return data
  } catch (err) {
    handleError(err, 'lib/storage.ts', 'uploadFile')
    return null
  }
}
```

### Get public URL
```typescript
function getPublicUrl(bucket: string, path: string): string {
  const { data } = supabase.storage.from(bucket).getPublicUrl(path)
  return data.publicUrl
}
```

---

## Common Gotchas

| Gotcha | Rule |
|--------|------|
| RLS blocks everything by default | Always add policies after enabling RLS |
| `.single()` throws if no row found | Use `.maybeSingle()` if row may not exist |
| Service role bypasses RLS | Never use service role client on frontend |
| Auth session expires | Always check session before DB operations |
| Migration order matters | Never edit existing migration files |
| `.select()` after insert needed | Add `.select().single()` to get inserted row back |

---

## What NOT to Do
- Do not edit existing migration files — create new ones
- Do not disable RLS on any table
- Do not use service role key on client side
- Do not call DB without checking session first
- Do not use `.single()` when row might not exist