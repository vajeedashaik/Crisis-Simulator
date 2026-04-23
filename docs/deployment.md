# Deployment — Supabase CLI & Dashboard Instructions

## Purpose
Read this file when deploying any change to Supabase or your
environment. Always try CLI first. Use dashboard only if CLI
is not possible — dashboard steps will be clearly marked.

---

## Core Principle
**CLI first. Dashboard only when CLI cannot do it.**
Every deployment step must be logged and reversible.

---

## Prerequisites
```bash
# Install Supabase CLI
npm install -g supabase

# Login
supabase login

# Link to your project (run once per project)
supabase link --project-ref YOUR_PROJECT_REF

# Confirm link
supabase status
```

---

## Environment Variables
```bash
# .env.local (never commit this file)
NEXT_PUBLIC_SUPABASE_URL=your_project_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

**Where to find these:**
Dashboard → Project Settings → API → Project URL + Keys

---

## Database Migrations

### Create a new migration
```bash
supabase migration new migration_name
# Creates: supabase/migrations/[timestamp]_migration_name.sql
# Write your SQL inside this file, then push
```

### Push migration to remote
```bash
supabase db push
```

### Check migration status
```bash
supabase migration list
```

### Reset local DB (dev only)
```bash
supabase db reset
```

### Pull remote schema to local
```bash
supabase db pull
```

---

## Local Development

### Start local Supabase
```bash
supabase start
# Gives you local URL, anon key, service role key
```

### Stop local Supabase
```bash
supabase stop
```

### View local DB in browser
```bash
supabase studio
# Opens Supabase Studio at localhost:54323
```

---

## Edge Functions

### Create a new edge function
```bash
supabase functions new function-name
```

### Serve locally
```bash
supabase functions serve function-name --env-file .env.local
```

### Deploy edge function
```bash
supabase functions deploy function-name
```

### Set environment secret for edge function
```bash
supabase secrets set KEY=value
```

### List secrets
```bash
supabase secrets list
```

---

## Storage

### CLI — not fully supported for bucket creation
**→ Use Dashboard for bucket creation**

#### Dashboard Steps — Create Storage Bucket

Go to Supabase Dashboard → Storage
Click "New bucket"
Enter bucket name
Toggle public/private
Click "Create bucket"

### Upload file via CLI
```bash
supabase storage cp ./local-file.png ss:///bucket-name/path/file.png
```

---

## Row Level Security (RLS)

### CLI — enable RLS on a table
```sql
-- Inside a migration file
ALTER TABLE your_table ENABLE ROW LEVEL SECURITY;
```

### CLI — create a policy via migration
```sql
CREATE POLICY "Users can view own data"
ON your_table
FOR SELECT
USING (auth.uid() = user_id);
```

### Dashboard Steps — RLS Policy (if migration not possible)

Go to Supabase Dashboard → Authentication → Policies
Select your table
Click "New Policy"
Choose template or write custom
Click "Review" then "Save Policy"

---

## Auth Configuration

### CLI — not supported for OAuth provider setup
**→ Use Dashboard for OAuth setup**

#### Dashboard Steps — Enable OAuth Provider

Go to Dashboard → Authentication → Providers
Select provider (Google, GitHub, etc.)
Enter Client ID and Secret
Copy callback URL → paste into provider's OAuth app
Click "Save"

### Set auth email templates

Dashboard → Authentication → Email Templates
→ Edit confirm signup / reset password templates

---

## scripts/deploy.sh
```bash
#!/bin/bash
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting deployment..." >> logs/errors.log

# Run tests first — abort if they fail
bash scripts/test.sh
if [ $? -ne 0 ]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Deployment aborted — tests failed" >> logs/errors.log
  exit 1
fi

# Push DB migrations
echo "Pushing DB migrations..."
supabase db push
if [ $? -ne 0 ]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Migration failed — check supabase logs" >> logs/errors.log
  exit 1
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Deployment complete" >> logs/errors.log
```

---

## Rollback

### Rollback last migration
```bash
# Supabase does not have auto-rollback
# Write a reverse migration manually

supabase migration new rollback_migration_name
# Write reverse SQL (DROP TABLE, ALTER, etc.)
supabase db push
```

### Dashboard Steps — Rollback (if CLI fails)

Go to Dashboard → Database → Migrations
Identify the migration to reverse
Go to Dashboard → SQL Editor
Write and run reverse SQL manually

---

## Deployment Checklist
- [ ] All tests pass (`bash scripts/test.sh`)
- [ ] `.env.local` is not committed
- [ ] Migration file created for every schema change
- [ ] RLS enabled on every new table
- [ ] Edge functions tested locally before deploy
- [ ] `supabase db push` confirms no errors
- [ ] One-liner git commit message written

---

## What NOT to Do
- Do not edit schema directly in Dashboard without a migration file
- Do not push migrations without running tests first
- Do not commit `.env.local` or any file with keys
- Do not disable RLS on any table in production
- Do not deploy edge functions without local testing first