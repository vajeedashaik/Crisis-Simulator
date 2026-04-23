# Architecture — Stack, Structure & Data Models

## Purpose
Read this file when building or modifying any structural part
of the project. Update this file when architecture changes.
Do not guess the stack — confirm here first.

---

## Tech Stack
Frontend:     [e.g. Next.js 14, React, TypeScript]
Styling:      [e.g. Tailwind CSS, shadcn/ui]
Backend:      [e.g. Next.js API Routes / Edge Functions]
Database:     Supabase (PostgreSQL)
Auth:         Supabase Auth
Storage:      Supabase Storage
Deployment:   [e.g. Vercel / Railway]
Testing:      Jest, ts-jest, React Testing Library
Package Mgr:  [e.g. npm / pnpm]

---

## Folder Structure
project-root/
├── src/
│   ├── app/                  ← Next.js app router pages
│   │   ├── (auth)/           ← auth route group
│   │   ├── (dashboard)/      ← protected route group
│   │   └── api/              ← API routes
│   ├── components/           ← reusable UI components
│   │   ├── ui/               ← base components (shadcn)
│   │   └── [feature]/        ← feature specific components
│   ├── lib/                  ← shared utilities and clients
│   │   ├── supabase.ts       ← supabase client
│   │   ├── supabase-server.ts← server side supabase client
│   │   └── utils.ts          ← shared utility functions
│   ├── hooks/                ← custom React hooks
│   ├── types/                ← TypeScript type definitions
│   └── constants/            ← app wide constants
├── supabase/
│   └── migrations/           ← all DB migration files
├── tests/                    ← all test files
├── docs/                     ← project documentation
├── scripts/                  ← shell scripts
└── logs/                     ← runtime and test logs

---

## Database Schema

### Tables
[Update this section as tables are created]
users
id            UUID PRIMARY KEY DEFAULT gen_random_uuid()
email         TEXT NOT NULL UNIQUE
created_at    TIMESTAMPTZ DEFAULT NOW()
updated_at    TIMESTAMPTZ DEFAULT NOW()

### Relationships
[Document foreign keys and relationships here]
users.id ← referenced by [table].[column]

### RLS Summary
[Document which tables have RLS enabled and policy types]
Table         RLS     Policies

users         YES     owner CRUD

---

## Auth Flow

User lands on /login
Supabase Auth handles email/OAuth
On success → session stored in cookie
Protected routes check session via middleware
API routes validate session server side
On signout → session cleared, redirect to /login


---

## API Routes
[Document API routes as they are created]
POST  /api/auth/login     ← handle login
POST  /api/auth/logout    ← handle logout
GET   /api/user           ← get current user

---

## Environment Variables
NEXT_PUBLIC_SUPABASE_URL          ← supabase project URL
NEXT_PUBLIC_SUPABASE_ANON_KEY     ← supabase anon key
SUPABASE_SERVICE_ROLE_KEY         ← server only, never expose

---

## Key Architectural Decisions
[Document WHY decisions were made as project grows]
[YYYY-MM-DD] — Used app router over pages router for better
server component support
[YYYY-MM-DD] — Used Supabase RLS over API-level auth checks
for defence in depth

---

## Constraints
[Document hard technical constraints]

No direct DB access from client side components
Service role key only used in server side code
All DB changes must go through migration files
RLS must be enabled on every table


---

## Rules for This File
- Update when a new table is added
- Update when a new API route is created
- Update when a key architectural decision is made
- Keep schema section in sync with actual migrations
- Do not document implementation details — only structure