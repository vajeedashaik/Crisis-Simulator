# Phase 8 Dashboard Design Spec
Date: 2026-04-24

## Goal
Single self-contained `dashboard.html` — cinematic real-time visualization of CrisisCoreEnv. Connects to running FastAPI app. No external dependencies.

## Architecture

### Files changed
- `app.py` — add `POST /trigger-crisis` endpoint only
- `dashboard.html` — new file, all HTML/CSS/JS inline

### Approach: Hybrid (C)
- Server manages environment state only
- Dashboard manages: action generation, action log history, polling loop
- Only new server endpoint: `/trigger-crisis`

---

## API Surface

### Existing endpoints used
| Method | Path | Purpose |
|--------|------|---------|
| POST | /reset | Start new episode, clear JS state |
| POST | /step | Advance sim one tick, get reward |
| GET | /state | Poll BuildingState for rendering |

### New endpoint
```
POST /trigger-crisis
```
- Picks random zone not already in active hazard's `affected_zones`
- Injects new `Hazard` into `_env.state.hazards` with random `HazardType`
- Returns updated `BuildingState` snapshot
- Raises 400 if no episode active

---

## Data Flow

```
dashboard.html
  │
  ├─ POST /reset  → clears actionLog[], resets episodeReward, redraws map
  │
  ├─ setInterval (interval = 1000ms / speed)
  │    ├─ generateAction(agentMode)  → AgentAction JSON
  │    ├─ POST /step(action)         → {observation, reward_breakdown, done}
  │    │    └─ push to actionLog[]  → re-render decision log
  │    └─ GET /state                → re-render floor map + metrics
  │
  └─ "Trigger Crisis" btn → POST /trigger-crisis → GET /state → re-render
```

### Action generation (client-side)
- **Random mode**: pick random ActionType, random zone/exit/service from observation
- **Trained mode**: priority heuristic — route hazard zones first, dispatch if severity HIGH/CRITICAL, broadcast otherwise

---

## Layout

```
┌─────────────────────────────────────────────────────────┐
│ ● LIVE   CRISIS CORE DASHBOARD                          │
├────────────────────────────┬────────────────────────────┤
│                            │  AGENT DECISION LOG        │
│   FLOOR MAP                │  (scrolling feed)          │
│   (60% width)              ├────────────────────────────┤
│                            │  METRICS (2x2 cards)       │
│                            ├────────────────────────────┤
│                            │  RESPONDER PAYLOAD (JSON)  │
└────────────────────────────┴────────────────────────────┘
│  [Start] [Pause] [Reset]  Speed: ─●─  [Random|Trained] [Trigger Crisis] │
└─────────────────────────────────────────────────────────┘
```

---

## Panel Specs

### 1. Floor Map (left, 60%)
- Zones drawn as rectangles in CSS Grid
- Colors: green=safe, orange=hazard nearby, red=active hazard, gray=evacuated+empty, blue=exit
- People as `<div>` dots, `requestAnimationFrame` repositions on zone change
- Hazard zones: CSS `@keyframes pulse` red border
- Blocked exits: `×` overlay
- Floor tabs if `floors > 1`
- Zone label + occupancy badge in each cell

### 2. Agent Decision Log (right top)
- `div` with `overflow-y: auto`, JS auto-scrolls to bottom
- Each entry: `[tick] [SubAgent] action → reward`
- SubAgent inferred from action_type:
  - `route_zone` → Evacuation
  - `dispatch_service` → Dispatch
  - `broadcast_pa` / `update_severity` → Comms
- Color: green if reward > 0, red if reward < 0, gray if 0

### 3. Metrics (right middle, 2×2 grid)
| Card | Value source |
|------|-------------|
| People Evacuated / Total | count `is_evacuated=true` / total from `/state` |
| Current Tick / Max | `state.tick` / `state.max_ticks` |
| Severity Level | `observation.current_severity` from `/step` response |
| Episode Reward | running sum of `reward_breakdown.total` |

### 4. Responder Payload (right bottom)
- Shown when last action was `dispatch_service`
- JSON object: `{incident_type, location_coordinates, affected_persons, hazards_present, access_routes}`
- Syntax highlight: keys in `#7ec8e3`, strings in `#a8e6a3`, numbers in `#ffb347`
- Slide-in CSS animation on new dispatch

---

## Controls
| Control | Behavior |
|---------|---------|
| Start | Sets `isRunning=true`, begins interval loop |
| Pause | Sets `isRunning=false`, clears interval |
| Reset | POST /reset, clears all JS state, redraws |
| Speed slider | 0.5×–3× maps to interval `1000/speed` ms |
| Agent toggle | Switches `agentMode` between `random`/`trained` |
| Trigger Crisis | POST /trigger-crisis, GET /state, re-renders map |

---

## Visual Theme
- Background: `#0a0a0f`
- Panel bg: `#12121a`
- Panel border: `#1e1e2e`
- Text: `#e0e0e0`
- Accent green: `#00ff88`
- Accent red: `#ff3355`
- Accent orange: `#ff8800`
- Accent blue: `#0088ff`
- Font: system monospace stack
- LIVE indicator: pulsing red dot, CSS `@keyframes pulse`, visible only when `isRunning=true`

---

## Implementation Notes
- No frameworks, no CDN imports — fully offline-capable
- All JS in one `<script>` block, all CSS in one `<style>` block
- Zone layout derived from `BuildingState.zones` at runtime (not hardcoded)
- `requestAnimationFrame` loop runs always; dot positions lerp toward target zone center
- Polling and stepping share same `setInterval`; one tick per interval fire
- Episode auto-pauses when `state.episode_done = true`

---

## Out of Scope
- Real trained model inference (toggle uses heuristic only)
- Multi-episode history
- WebSocket (polling sufficient per spec)
