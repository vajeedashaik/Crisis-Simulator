# Progress — Feature & Phase Status

## Purpose
Read this file to know exactly what is done, in progress, or pending.
One liner per feature. Update after every feature implementation.
Do not read entire codebase to understand progress — read this file.

---

## Status Legend

✅ done        — implemented, tested, committed
🔄 in progress — currently being worked on
⏳ pending     — not started yet
❌ blocked     — cannot proceed, reason noted
🐛 bug         — implemented but has known failing test
📄 spec only   — design/spec written, implementation not started

---

## Phase 1 — Data Schema
✅ enums — ActionType, ServiceType, HazardType, SeverityLevel (4 enums)
✅ dataclasses — Zone, Person, Hazard, SensorReading, BuildingState, AgentAction, AgentObservation, RewardBreakdown (8 classes)
✅ tests — 61/61 passing in tests/unit/test_schema.py

## Phase 2 — RL Environment
✅ CrisisCoreEnv.__init__ — config loading with defaults
✅ CrisisCoreEnv.reset — grid zone layout, adjacency wiring, exits, people, hazards, sensors
✅ CrisisCoreEnv.step — all 4 action types, hazard spread, sensor refresh, reward, termination
✅ CrisisCoreEnv._shortest_path — BFS across zone connections
✅ CrisisCoreEnv._get_observation — partial observability, occupancy, SOS, exits
✅ CrisisCoreEnv._check_termination — all-evacuated or max-ticks
✅ CrisisCoreEnv.get_info — counts and hazard zones
✅ tests — 86/86 passing in tests/unit/test_environment.py

## Phase 3 — Reward Functions
✅ evacuation_speed_reward — +1.0/new evacuee with time bonus, capped at 5.0
✅ route_safety_reward — -3.0 zone hazard, -2.0 exit-path hazard
✅ dispatch_accuracy_reward — FIRE_BRIGADE/FIRE+SMOKE +2.0, EMS +1.0, POLICE penalized
✅ severity_accuracy_reward — ground-truth ratio → level, scored by distance
✅ format_compliance_reward — +0.2 all required fields present, -1.0 missing
✅ timeout_penalty — -5.0 done with unrescued people
✅ compute_reward — weighted sum into RewardBreakdown
✅ tests — 46/46 passing in tests/unit/test_rewards.py

## Phase 4 — Multi-Agent Hierarchy
✅ EvacuationAgent — observe (zone_occupancy/exits/hazards) + act (ROUTE_ZONE, safe default to nearest exit)
✅ DispatchAgent — observe (sos/severity/hazard_types) + act (DISPATCH_SERVICE, safe default FIRE_BRIGADE)
✅ CommsAgent — observe (routing_decision/occupancy) + act (BROADCAST_PA, safe default generic message)
✅ OrchestratorAgent — parallel evac+dispatch via ThreadPoolExecutor, feeds evac action to comms
✅ build_agent_prompt — system role paragraph + JSON obs + optional incident log
✅ tests — 31/31 passing in tests/unit/test_agents.py

## Phase 5 — Curriculum Learning
✅ CURRICULUM_LEVELS — 3 difficulty configs (1-floor/4-zone → 3-floor/6-zone)
✅ CurriculumManager — promotion_threshold + sliding window, auto-advance levels
✅ IncidentLog — deque-based log, mean_reward(), recent_actions() helpers
✅ run_curriculum_episode — single episode runner returning (actions, reward_sum, done)
✅ tests — 31/31 passing in tests/unit/test_curriculum.py

## Phase 6 — GRPO Training (Unsloth)
✅ train.py — GRPO training loop targeting Qwen2.5-3B-Instruct via Unsloth
✅ model config — LoRA r=16, q_proj+v_proj, max_seq_len=2048
✅ dataset builder — generates prompt/completion pairs from CrisisCoreEnv episodes
✅ reward fn wiring — compute_reward piped into GRPO reward signal
✅ curriculum integration — training uses CURRICULUM_LEVELS progression
✅ training plot — matplotlib loss/reward curve saved after run
⏳ tests — no automated tests (Colab/GPU dependency); manual validation only

## Phase 7 — HuggingFace Spaces Deployment
✅ app.py — FastAPI server wrapping CrisisCoreEnv (POST /reset, POST /step, GET /state)
✅ app.py — CORSMiddleware added (allow_origins=["*"]) so dashboard.html works from file:// or any origin
✅ OrchestratorAgent wired — rule-based fallback model_fn, no LLM needed for demo
✅ Dockerfile — HF Spaces compatible, port 7860, uvicorn entrypoint
✅ requirements.txt — all deps pinned (fastapi, uvicorn, pydantic)
✅ client.py — demo client showing full episode via HTTP calls
✅ README.md — project overview and API reference

## Phase 8 — Dashboard
✅ dashboard design spec — docs/superpowers/specs/2026-04-24-dashboard-design.md (156 lines, fully detailed)
✅ POST /trigger-crisis in app.py — injects random hazard into active episode
✅ dashboard.html — pure HTML/CSS/JS, no frameworks, dark cinematic theme (#0a0a0f)
✅ floor map panel — CSS grid zones, people dots, hazard-pulse animation, floor tabs, blocked-exit overlay
✅ agent decision log — scrolling feed, color-coded by reward sign, auto-scrolls to latest
✅ metrics panel — 2×2 cards: evacuated, tick, severity badge, episode reward
✅ responder payload panel — syntax-highlighted JSON (keys blue, strings green, numbers orange), slide-in on dispatch
✅ dashboard controls — start/pause/reset, speed slider (0.5×–3×), random/trained toggle, trigger crisis button
✅ tests — 3/3 passing in tests/unit/test_trigger_crisis.py

---

## Test Summary (2026-04-25)
258/258 passing across 6 test files
- test_schema.py: 61
- test_environment.py: 86
- test_rewards.py: 46
- test_agents.py: 31
- test_curriculum.py: 31
- test_trigger_crisis.py: 3

---

## What's Left to Build
Nothing — all 8 phases complete.

---

## Blocked Items
❌ train.py GPU execution — requires Colab or GPU machine with Unsloth installed; not runnable locally

---

## Rules for This File
- One line per feature, no paragraphs
- Update status after every feature, not at end of phase
- Never delete a line — only update its status
- If blocked, note the reason inline
