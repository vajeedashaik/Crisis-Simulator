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

## Phase 4 — [Phase Name]
⏳ [feature name] — [one line description]

## Phase 5 — [Phase Name]
⏳ [feature name] — [one line description]

## Phase 6 — [Phase Name]
⏳ [feature name] — [one line description]

## Phase 7 — [Phase Name]
⏳ [feature name] — [one line description]

## Phase 8 — [Phase Name]
⏳ [feature name] — [one line description]

---

## Blocked Items
❌ [feature name] — blocked by: [reason]

---

## Rules for This File
- One line per feature, no paragraphs
- Update status after every feature, not at end of phase
- Never delete a line — only update its status
- If blocked, note the reason inline