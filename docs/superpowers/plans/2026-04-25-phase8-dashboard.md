# Phase 8 Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `POST /trigger-crisis` to the FastAPI server and create a self-contained `dashboard.html` that visualizes CrisisCoreEnv in real time.

**Architecture:** A single HTML file contains all CSS and JS; it polls the running FastAPI server via `fetch`. The server adds one new endpoint to inject hazards mid-episode. No build step, no frameworks.

**Tech Stack:** FastAPI (existing), Python `uuid` (stdlib), Pure HTML/CSS/JS, pytest + `httpx` for endpoint test.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `app.py` | Add `POST /trigger-crisis` endpoint |
| Create | `tests/unit/test_trigger_crisis.py` | Endpoint test |
| Create | `dashboard.html` | Complete dashboard — all HTML/CSS/JS inline |

---

## Task 1: POST /trigger-crisis endpoint

**Files:**
- Modify: `app.py` (after the `/state` endpoint, before `/health`)
- Create: `tests/unit/test_trigger_crisis.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_trigger_crisis.py`:

```python
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_trigger_crisis_without_reset_returns_400():
    response = client.post("/trigger-crisis")
    assert response.status_code == 400
    assert "No active episode" in response.json()["detail"]


def test_trigger_crisis_after_reset_injects_one_hazard():
    client.post("/reset")
    state_before = client.get("/state").json()
    hazards_before = len(state_before["hazards"])

    response = client.post("/trigger-crisis")
    assert response.status_code == 200
    data = response.json()
    assert len(data["hazards"]) == hazards_before + 1


def test_trigger_crisis_returns_building_state_shape():
    client.post("/reset")
    response = client.post("/trigger-crisis")
    data = response.json()
    assert "zones" in data
    assert "people" in data
    assert "hazards" in data
    assert "tick" in data
```

- [ ] **Step 2: Run to verify tests fail**

```bash
python -m pytest tests/unit/test_trigger_crisis.py -v
```

Expected: 3 failures — `404 Not Found` (endpoint doesn't exist yet).

- [ ] **Step 3: Implement the endpoint**

In `app.py`, add this import at the top alongside the existing schema imports:

```python
import uuid
from schema import Hazard, HazardType
```

Then add this endpoint **after** the `/state` endpoint and **before** `/health`:

```python
@app.post("/trigger-crisis", summary="Inject a new hazard mid-episode")
async def trigger_crisis():
    """Pick a random zone not already affected by a hazard and inject a new one."""
    if _env is None or _env.state is None:
        raise HTTPException(status_code=400, detail="No active episode — call POST /reset first.")

    affected: set[str] = set()
    for h in _env.state.hazards.values():
        affected.update(h.affected_zones)

    available = [zid for zid in _env.state.zones if zid not in affected]
    if not available:
        raise HTTPException(status_code=400, detail="All zones already have active hazards.")

    zone_id = random.choice(available)
    hazard_type = random.choice(list(HazardType))
    new_hazard = Hazard(
        hazard_id=f"crisis_{uuid.uuid4().hex[:8]}",
        hazard_type=hazard_type,
        affected_zones=[zone_id],
        spread_rate=0.3,
        intensity=0.8,
    )
    _env.state.hazards[new_hazard.hazard_id] = new_hazard

    try:
        return _to_dict(_env.state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/unit/test_trigger_crisis.py -v
```

Expected output:
```
PASSED tests/unit/test_trigger_crisis.py::test_trigger_crisis_without_reset_returns_400
PASSED tests/unit/test_trigger_crisis.py::test_trigger_crisis_after_reset_injects_one_hazard
PASSED tests/unit/test_trigger_crisis.py::test_trigger_crisis_returns_building_state_shape
3 passed in ...
```

- [ ] **Step 5: Run full suite to catch regressions**

```bash
python -m pytest tests/ --tb=short -q
```

Expected: all 258 passed (255 original + 3 new).

- [ ] **Step 6: Commit**

```bash
git add app.py tests/unit/test_trigger_crisis.py
git commit -m "feat(api): add POST /trigger-crisis endpoint with 3 tests"
```

---

## Task 2: dashboard.html

**Files:**
- Create: `dashboard.html`

No automated tests — this is a browser UI. Verification is manual (Task 3).

- [ ] **Step 1: Create dashboard.html**

Create `dashboard.html` with the full content below. This is the complete file — do not split it across multiple edits:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Crisis Core Dashboard</title>
<style>
:root {
  --bg:     #0a0a0f;
  --panel:  #12121a;
  --border: #1e1e2e;
  --text:   #e0e0e0;
  --green:  #00ff88;
  --red:    #ff3355;
  --orange: #ff8800;
  --blue:   #0088ff;
  --gray:   #555566;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Courier New', Courier, monospace;
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* ── Header ─────────────────────────────────────────── */
#header {
  padding: 8px 16px;
  background: var(--panel);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
}
#live-dot {
  width: 10px; height: 10px;
  border-radius: 50%;
  background: var(--red);
  display: none;
}
#live-dot.active { display: block; animation: blink 1s infinite; }
#live-label { font-size: 11px; color: var(--red); display: none; letter-spacing: 1px; }
#header-title { font-size: 13px; letter-spacing: 3px; color: var(--text); }

/* ── Main layout ─────────────────────────────────────── */
#main {
  display: flex;
  flex: 1;
  gap: 8px;
  padding: 8px;
  overflow: hidden;
  min-height: 0;
}
#left  { width: 60%; display: flex; flex-direction: column; min-height: 0; }
#right { width: 40%; display: flex; flex-direction: column; gap: 8px; min-height: 0; }

/* ── Panels ──────────────────────────────────────────── */
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 10px 12px;
}
.panel-title {
  font-size: 10px;
  color: var(--gray);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-bottom: 8px;
}

/* ── Floor map ───────────────────────────────────────── */
#map-panel { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
#floor-tabs { display: inline-flex; gap: 4px; margin-left: 8px; }
.floor-tab {
  padding: 1px 8px;
  font-size: 10px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 3px;
  color: var(--gray);
  cursor: pointer;
  font-family: inherit;
}
.floor-tab.active { border-color: var(--green); color: var(--green); }
#zone-grid-wrap { flex: 1; overflow: auto; }
#zone-grid { display: grid; gap: 6px; padding: 4px; }

.zone-cell {
  background: #111118;
  border: 2px solid var(--border);
  border-radius: 4px;
  padding: 6px 8px;
  position: relative;
  min-height: 72px;
  transition: border-color 0.3s;
}
.zone-cell.safe         { border-color: var(--green); background: #0a120a; }
.zone-cell.hazard-near  { border-color: var(--orange); background: #140f00; }
.zone-cell.hazard-active{ border-color: var(--red); background: #140008; animation: hazard-pulse 0.8s infinite; }
.zone-cell.exit-zone    { border-color: var(--blue); }
.zone-cell.empty-evac   { border-color: var(--gray); background: #0e0e14; opacity: 0.55; }

@keyframes hazard-pulse {
  0%,100% { border-color: var(--red); box-shadow: 0 0 6px rgba(255,51,85,0.4); }
  50%      { border-color: #880022; box-shadow: none; }
}
.zone-id    { font-size: 10px; color: var(--gray); }
.zone-occ   { font-size: 13px; font-weight: bold; margin-top: 2px; }
.exit-badge {
  position: absolute; top: 4px; right: 5px;
  font-size: 9px; color: var(--blue); letter-spacing: 0.5px;
}
.blocked-x {
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 28px; color: var(--red);
  background: rgba(255,51,85,0.08);
  border-radius: 3px;
  pointer-events: none;
}
.dots-row { display: flex; flex-wrap: wrap; gap: 3px; margin-top: 4px; }
.person-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--green);
  display: inline-block;
}
.person-dot.sos { background: var(--red); animation: blink 0.5s infinite; }

/* ── Decision log ────────────────────────────────────── */
#log-panel { flex: 2; display: flex; flex-direction: column; overflow: hidden; }
#decision-log { flex: 1; overflow-y: auto; font-size: 11px; line-height: 1.7; }
.log-entry { padding: 2px 0; border-bottom: 1px solid #1a1a28; }
.log-entry.pos { color: var(--green); }
.log-entry.neg { color: var(--red); }
.log-entry.neu { color: var(--gray); }

/* ── Metrics ─────────────────────────────────────────── */
#metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.metric-card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 10px 12px;
}
.m-label { font-size: 9px; color: var(--gray); text-transform: uppercase; letter-spacing: 1px; }
.m-value { font-size: 18px; font-weight: bold; color: var(--green); margin-top: 3px; }
.sev-badge {
  display: inline-block;
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
  font-weight: bold;
}
.sev-LOW      { background: #003318; color: var(--green); }
.sev-MODERATE { background: #2a1a00; color: var(--orange); }
.sev-HIGH     { background: #1a0008; color: var(--red); }
.sev-CRITICAL { background: #1a0000; color: #ff0000; animation: blink 0.5s infinite; }

/* ── Responder payload ───────────────────────────────── */
#payload-panel {
  transition: opacity 0.3s, transform 0.3s;
  flex-shrink: 0;
}
#payload-json {
  font-size: 11px;
  line-height: 1.5;
  white-space: pre;
  overflow: auto;
  max-height: 110px;
}
.jk { color: #7ec8e3; }
.js { color: #a8e6a3; }
.jn { color: #ffb347; }

/* ── Controls ────────────────────────────────────────── */
#controls {
  padding: 8px 14px;
  background: var(--panel);
  border-top: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  flex-shrink: 0;
}
button {
  background: var(--bg);
  color: var(--text);
  border: 1px solid var(--border);
  padding: 5px 14px;
  border-radius: 3px;
  cursor: pointer;
  font-family: inherit;
  font-size: 12px;
  transition: background 0.15s;
}
button:hover { background: #1e1e2e; }
#btn-start   { border-color: var(--green); color: var(--green); }
#btn-pause   { border-color: var(--orange); color: var(--orange); }
#btn-trigger { border-color: var(--red); color: var(--red); }
.toggle-grp { display: flex; }
.toggle-grp button { border-radius: 0; border-right-width: 0; }
.toggle-grp button:first-child { border-radius: 3px 0 0 3px; }
.toggle-grp button:last-child  { border-radius: 0 3px 3px 0; border-right-width: 1px; }
.toggle-grp button.active { background: #1a2a1a; border-color: var(--green); color: var(--green); }
.ctrl-label { font-size: 11px; color: var(--gray); }
input[type=range] { accent-color: var(--green); width: 80px; }
#speed-label { font-size: 11px; color: var(--text); min-width: 28px; }

@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.25} }
</style>
</head>
<body>

<!-- Header -->
<div id="header">
  <div id="live-dot"></div>
  <span id="live-label">LIVE</span>
  <span id="header-title">CRISIS CORE DASHBOARD</span>
</div>

<!-- Main -->
<div id="main">
  <!-- Left: Floor map -->
  <div id="left">
    <div class="panel" id="map-panel">
      <div class="panel-title">
        Floor Map
        <span id="floor-tabs"></span>
      </div>
      <div id="zone-grid-wrap">
        <div id="zone-grid"></div>
      </div>
    </div>
  </div>

  <!-- Right: Log + Metrics + Payload -->
  <div id="right">
    <div class="panel" id="log-panel">
      <div class="panel-title">Agent Decision Log</div>
      <div id="decision-log"></div>
    </div>
    <div class="panel">
      <div class="panel-title">Metrics</div>
      <div id="metrics-grid">
        <div class="metric-card">
          <div class="m-label">Evacuated / Total</div>
          <div class="m-value" id="m-evac">— / —</div>
        </div>
        <div class="metric-card">
          <div class="m-label">Tick / Max</div>
          <div class="m-value" id="m-tick">— / —</div>
        </div>
        <div class="metric-card">
          <div class="m-label">Severity</div>
          <div class="m-value" id="m-sev">—</div>
        </div>
        <div class="metric-card">
          <div class="m-label">Episode Reward</div>
          <div class="m-value" id="m-reward">0.00</div>
        </div>
      </div>
    </div>
    <div class="panel" id="payload-panel" style="opacity:0.35">
      <div class="panel-title">Responder Payload</div>
      <div id="payload-json">No dispatch yet.</div>
    </div>
  </div>
</div>

<!-- Controls -->
<div id="controls">
  <button id="btn-start"   onclick="startSim()">▶ Start</button>
  <button id="btn-pause"   onclick="pauseSim()">⏸ Pause</button>
  <button id="btn-reset"   onclick="resetSim()">↺ Reset</button>
  <span class="ctrl-label">Speed:</span>
  <input  type="range" id="speed-slider" min="0.5" max="3" step="0.5" value="1"
          oninput="setSpeed(this.value)">
  <span id="speed-label">1×</span>
  <div class="toggle-grp">
    <button id="btn-random"  class="active" onclick="setMode('random')">Random</button>
    <button id="btn-trained"               onclick="setMode('trained')">Trained</button>
  </div>
  <button id="btn-trigger" onclick="triggerCrisis()">⚡ Trigger Crisis</button>
</div>

<script>
'use strict';
const API = 'http://localhost:8000';

// ── State ─────────────────────────────────────────────
let isRunning     = false;
let simTimer      = null;
let speed         = 1;
let agentMode     = 'random';
let episodeReward = 0;
let actionLog     = [];
let currentState  = null;   // BuildingState
let currentObs    = null;   // AgentObservation
let activeFloor   = 0;

// ── API helpers ───────────────────────────────────────
async function post(path, body) {
  const opts = { method: 'POST' };
  if (body) { opts.headers = {'Content-Type':'application/json'}; opts.body = JSON.stringify(body); }
  const r = await fetch(API + path, opts);
  if (!r.ok) { const t = await r.text(); throw new Error(t); }
  return r.json();
}
async function get(path) {
  const r = await fetch(API + path);
  if (!r.ok) { const t = await r.text(); throw new Error(t); }
  return r.json();
}

// ── Action generation ─────────────────────────────────
function makeAction(obs, state) {
  return agentMode === 'trained' ? trainedAction(obs, state) : randomAction(obs, state);
}

function randomAction(obs, state) {
  const types    = ['route_zone','dispatch_service','broadcast_pa','update_severity'];
  const type     = types[Math.floor(Math.random() * types.length)];
  const zones    = Object.keys(state ? state.zones : {});
  const exits    = (obs && obs.available_exits) ? obs.available_exits : [];
  const services = ['fire_brigade','ems','police'];
  const sevs     = ['LOW','MODERATE','HIGH','CRITICAL'];
  if (type === 'route_zone')
    return { action_type:'route_zone', zone_id: rand(zones), route_to_exit: rand(exits) || null };
  if (type === 'dispatch_service')
    return { action_type:'dispatch_service', service_type: rand(services) };
  if (type === 'broadcast_pa')
    return { action_type:'broadcast_pa', message:'Please evacuate via the nearest exit.' };
  return { action_type:'update_severity', severity: rand(sevs) };
}

function trainedAction(obs, state) {
  const hazZones = new Set();
  Object.values((state && state.hazards) || {}).forEach(h => h.affected_zones.forEach(z => hazZones.add(z)));
  const exits = (obs && obs.available_exits) ? obs.available_exits : [];
  const sev   = (obs && obs.current_severity) ? obs.current_severity : 'LOW';
  if (hazZones.size > 0)
    return { action_type:'route_zone', zone_id:[...hazZones][0], route_to_exit: exits[0] || null };
  if (sev === 'HIGH' || sev === 'CRITICAL')
    return { action_type:'dispatch_service', service_type:'fire_brigade' };
  return { action_type:'broadcast_pa', message:'Attention: evacuate calmly via the nearest exit.' };
}

function rand(arr) { return arr && arr.length ? arr[Math.floor(Math.random()*arr.length)] : null; }

// ── Simulation loop ───────────────────────────────────
function startSim() {
  if (isRunning) return;
  isRunning = true;
  document.getElementById('live-dot').classList.add('active');
  document.getElementById('live-label').style.display = 'inline';
  scheduleNext();
}
function pauseSim() {
  isRunning = false;
  clearTimeout(simTimer);
  document.getElementById('live-dot').classList.remove('active');
  document.getElementById('live-label').style.display = 'none';
}
function scheduleNext() {
  if (!isRunning) return;
  simTimer = setTimeout(tick, Math.round(1000 / speed));
}
async function tick() {
  if (!isRunning) return;
  try {
    const action = makeAction(currentObs, currentState);
    const result = await post('/step', action);
    currentObs    = result.observation;
    const reward  = result.reward_breakdown ? (result.reward_breakdown.total || 0) : 0;
    episodeReward += reward;

    // Log
    const sub = subagentName(action.action_type);
    actionLog.push({ tick: currentState ? currentState.tick : '?', sub, action, reward });
    renderLog();

    // Payload on dispatch
    if (action.action_type === 'dispatch_service') renderPayload(action, currentState);

    // Refresh state
    currentState = await get('/state');
    renderMap(currentState);
    renderMetrics(currentState, currentObs);

    if (result.done || (currentState && currentState.episode_done)) {
      pauseSim();
      logMsg('— Episode complete —', 'neu');
      return;
    }
  } catch (e) {
    logMsg('Error: ' + e.message, 'neg');
  }
  scheduleNext();
}

async function resetSim() {
  pauseSim();
  episodeReward = 0;
  actionLog     = [];
  document.getElementById('decision-log').innerHTML = '';
  document.getElementById('payload-json').textContent = 'No dispatch yet.';
  document.getElementById('payload-panel').style.opacity = '0.35';
  document.getElementById('m-evac').textContent   = '— / —';
  document.getElementById('m-tick').textContent   = '— / —';
  document.getElementById('m-sev').textContent    = '—';
  document.getElementById('m-reward').textContent = '0.00';
  try {
    currentObs   = await post('/reset');
    currentState = await get('/state');
    renderMap(currentState);
    renderMetrics(currentState, currentObs);
  } catch(e) { logMsg('Reset failed: ' + e.message, 'neg'); }
}

async function triggerCrisis() {
  try {
    currentState = await post('/trigger-crisis');
    renderMap(currentState);
    logMsg('⚡ Crisis injected!', 'neg');
  } catch(e) { logMsg('Trigger failed: ' + e.message, 'neg'); }
}

function setSpeed(v) {
  speed = parseFloat(v);
  document.getElementById('speed-label').textContent = v + '×';
}
function setMode(m) {
  agentMode = m;
  document.getElementById('btn-random').classList.toggle('active',  m === 'random');
  document.getElementById('btn-trained').classList.toggle('active', m === 'trained');
}

// ── Render: Floor Map ─────────────────────────────────
function renderMap(state) {
  if (!state) return;
  const allZones = Object.values(state.zones);
  const floors   = [...new Set(allZones.map(z => z.floor))].sort((a,b)=>a-b);

  // Floor tabs
  const tabsEl = document.getElementById('floor-tabs');
  tabsEl.innerHTML = floors.map(f =>
    `<button class="floor-tab${f===activeFloor?' active':''}" onclick="switchFloor(${f})">${f}</button>`
  ).join('');

  // Default to first floor
  if (!floors.includes(activeFloor)) activeFloor = floors[0];
  const fZones = allZones.filter(z => z.floor === activeFloor);

  const gridEl = document.getElementById('zone-grid');
  const cols   = Math.ceil(Math.sqrt(fZones.length)) || 1;
  gridEl.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;

  // Classify zones
  const hazardActive = new Set();
  const hazardNear   = new Set();
  Object.values(state.hazards || {}).forEach(h => h.affected_zones.forEach(z => hazardActive.add(z)));
  fZones.forEach(z => {
    if (!hazardActive.has(z.zone_id) && z.connected_zones.some(c => hazardActive.has(c)))
      hazardNear.add(z.zone_id);
  });

  const people  = Object.values(state.people || {});
  const blocked = new Set(state.blocked_exits || []);

  gridEl.innerHTML = fZones.map(z => {
    const zid        = z.zone_id;
    const present    = people.filter(p => p.current_zone === zid && !p.is_evacuated);
    const wereHere   = people.filter(p => p.current_zone === zid);
    const isHazard   = hazardActive.has(zid);
    const isNear     = !isHazard && hazardNear.has(zid);
    const isExit     = z.has_exit;
    const isBlocked  = isExit && z.exit_id && blocked.has(z.exit_id);
    const allGone    = wereHere.length > 0 && present.length === 0;

    let cls = 'zone-cell';
    if      (isHazard) cls += ' hazard-active';
    else if (isNear)   cls += ' hazard-near';
    else if (isExit)   cls += ' exit-zone';
    else               cls += ' safe';
    if (allGone)       cls += ' empty-evac';

    const dots = present.map(p =>
      `<div class="person-dot${p.has_sos?' sos':''}" title="${p.person_id}"></div>`
    ).join('');

    const exitBadge   = isExit ? `<div class="exit-badge">EXIT${z.exit_id?' '+z.exit_id:''}</div>` : '';
    const blockedOver = isBlocked ? `<div class="blocked-x">×</div>` : '';

    return `<div class="${cls}">
      ${exitBadge}
      <div class="zone-id">${zid}</div>
      <div class="zone-occ">${present.length} <span style="font-size:10px;color:var(--gray)">ppl</span></div>
      <div class="dots-row">${dots}</div>
      ${blockedOver}
    </div>`;
  }).join('');
}

function switchFloor(f) {
  activeFloor = f;
  if (currentState) renderMap(currentState);
}

// ── Render: Decision Log ──────────────────────────────
function subagentName(type) {
  if (type === 'route_zone')       return 'Evacuation';
  if (type === 'dispatch_service') return 'Dispatch';
  return 'Comms';
}
function renderLog() {
  const el = document.getElementById('decision-log');
  el.innerHTML = actionLog.slice(-120).map(e => {
    const cls    = e.reward > 0 ? 'pos' : e.reward < 0 ? 'neg' : 'neu';
    const detail = e.action.zone_id || e.action.service_type
                || (e.action.message ? e.action.message.slice(0,28)+'…' : '');
    const r      = (e.reward >= 0 ? '+' : '') + (e.reward || 0).toFixed(2);
    return `<div class="log-entry ${cls}">[${e.tick}] ${e.sub} → ${e.action.action_type}`
         + `${detail ? ' ('+detail+')' : ''}`
         + ` <span style="float:right">${r}</span></div>`;
  }).join('');
  el.scrollTop = el.scrollHeight;
}
function logMsg(msg, cls) {
  const el = document.getElementById('decision-log');
  el.innerHTML += `<div class="log-entry ${cls}">${msg}</div>`;
  el.scrollTop  = el.scrollHeight;
}

// ── Render: Metrics ───────────────────────────────────
function renderMetrics(state, obs) {
  if (!state) return;
  const people    = Object.values(state.people || {});
  const evacuated = people.filter(p => p.is_evacuated).length;
  document.getElementById('m-evac').textContent = `${evacuated} / ${people.length}`;
  document.getElementById('m-tick').textContent = `${state.tick} / ${state.max_ticks}`;

  const sev   = (obs && obs.current_severity) ? obs.current_severity : '—';
  const sevEl = document.getElementById('m-sev');
  sevEl.innerHTML = sev === '—'
    ? '<span style="color:var(--gray)">—</span>'
    : `<span class="sev-badge sev-${sev}">${sev}</span>`;

  document.getElementById('m-reward').textContent = episodeReward.toFixed(2);
}

// ── Render: Responder Payload ─────────────────────────
function renderPayload(action, state) {
  if (!state) return;
  const hazZones = new Set();
  Object.values(state.hazards || {}).forEach(h => h.affected_zones.forEach(z => hazZones.add(z)));
  const people      = Object.values(state.people || {});
  const affected    = people.filter(p => hazZones.has(p.current_zone) && !p.is_evacuated).length;
  const hazTypes    = [...new Set(Object.values(state.hazards||{}).map(h=>h.hazard_type))];
  const accessRoutes= Object.values(state.zones||{}).filter(z=>z.has_exit).map(z=>z.zone_id);

  const payload = {
    incident_type:       action.service_type || 'unknown',
    location_coordinates:[...hazZones],
    affected_persons:    affected,
    hazards_present:     hazTypes,
    access_routes:       accessRoutes,
  };

  document.getElementById('payload-json').innerHTML = highlight(JSON.stringify(payload, null, 2));
  const panel = document.getElementById('payload-panel');
  panel.style.opacity   = '1';
  panel.style.transform = 'translateX(6px)';
  requestAnimationFrame(() => { panel.style.transition = 'opacity 0.3s, transform 0.3s'; });
  setTimeout(() => { panel.style.transform = 'translateX(0)'; }, 50);
}

function highlight(json) {
  return json
    .replace(/("[\w_]+")\s*:/g,  '<span class="jk">$1</span>:')
    .replace(/:\s*("(?:[^"\\]|\\.)*")/g, ': <span class="js">$1</span>')
    .replace(/:\s*(\d+\.?\d*)/g,        ': <span class="jn">$1</span>');
}

// ── Boot ──────────────────────────────────────────────
(async () => {
  try {
    currentObs   = await post('/reset');
    currentState = await get('/state');
    renderMap(currentState);
    renderMetrics(currentState, currentObs);
    logMsg('Ready — press Start to begin.', 'neu');
  } catch(e) {
    logMsg('Cannot reach server at ' + API + ' — start FastAPI first, then refresh.', 'neg');
  }
})();
</script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add dashboard.html
git commit -m "feat(dashboard): add cinematic real-time dashboard HTML"
```

---

## Task 3: Manual Integration Test

**Files:** none (verification only)

- [ ] **Step 1: Start the FastAPI server**

```bash
uvicorn app:app --reload --port 8000
```

Expected: `Application startup complete.` in terminal.

- [ ] **Step 2: Open dashboard in browser**

Open `dashboard.html` directly in a browser (File → Open, or drag the file into a browser tab).

Expected on load:
- Floor map renders zones (green-bordered cells)
- People dots visible in some zones
- Metrics show `0 / N` evacuated, `0 / 20` tick
- Log shows "Ready — press Start to begin."

- [ ] **Step 3: Verify Start/Pause loop**

Click **Start**. Expected:
- LIVE dot pulses red
- Tick counter increments every ~1 second
- Log fills with entries colored green/red/gray
- Zone colors change as hazard spreads

Click **Pause**. Expected: tick stops.

- [ ] **Step 4: Verify Trigger Crisis**

While paused or running, click **⚡ Trigger Crisis**. Expected:
- Log shows "⚡ Crisis injected!"
- Map immediately shows a new red-pulsing zone

- [ ] **Step 5: Verify Dispatch payload**

Set Agent mode to **Trained** and click **Start**. Wait for a dispatch action. Expected:
- Responder Payload panel fades in (opacity 1)
- JSON shows `incident_type`, `location_coordinates`, `affected_persons`, `hazards_present`, `access_routes`
- Keys are blue (`#7ec8e3`), strings are green (`#a8e6a3`), numbers are orange (`#ffb347`)

- [ ] **Step 6: Verify Reset**

Click **↺ Reset**. Expected:
- Log clears
- Episode reward resets to 0.00
- Map redraws fresh episode state

- [ ] **Step 7: Verify Speed slider**

Set speed to **3×**, click Start. Expected: ticks fire ~3× per second.

- [ ] **Step 8: Run full test suite one final time**

```bash
python -m pytest tests/ --tb=short -q
```

Expected: `258 passed`.

- [ ] **Step 9: Final commit**

```bash
git add .
git commit -m "phase 8 complete: trigger-crisis endpoint + dashboard"
```
