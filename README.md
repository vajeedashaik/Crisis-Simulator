---
title: CrisisCore — Multi-Agent Crisis Response RL Environment
emoji: 🚨
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
tags:
  - multi-agent
  - reinforcement-learning
  - crisis-response
  - openenv
---

# CrisisCore — Multi-Agent Crisis Response RL Environment

## What Problem It Solves

Training emergency-response AI systems requires realistic, dynamic environments where agents must coordinate evacuation routing, service dispatch, and communications under time pressure with noisy sensor data. CrisisCore provides a lightweight, fully-observable/partially-observable building crisis environment that integrates directly with standard RL training loops via a REST API.

## Environment Description

### reset()
Returns an `AgentObservation` JSON object containing:
- `tick` — current simulation step (0 on reset)
- `sensor_readings` — per-zone smoke, motion, and sound levels (may include noise)
- `zone_occupancy` — count of non-evacuated persons per zone
- `known_hazard_zones` — zones with sensor readings above detection threshold
- `available_exits` — zones with open exits
- `sos_signals` — person IDs broadcasting distress
- `current_severity` — last severity label set by the agent (or null)

### step()
Accepts a single `AgentAction` JSON body with fields:

| Field | Type | Required for |
|---|---|---|
| `action_type` | `route_zone` \| `dispatch_service` \| `broadcast_pa` \| `update_severity` | always |
| `zone_id` | string | `route_zone` |
| `route_to_exit` | string | `route_zone` |
| `service_type` | `fire_brigade` \| `ems` \| `police` | `dispatch_service` |
| `message` | string | `broadcast_pa` |
| `severity` | `low` \| `medium` \| `high` \| `critical` | `update_severity` |

Returns `(observation, reward_breakdown, done, info)` as JSON.

### Agent Observation
Each tick the agent sees partial state: sensor readings (possibly noisy), zone occupancy counts, which exits are open, and which persons have sent SOS signals. The agent does not see exact person positions or full hazard intensity.

## Reward Model

| Signal | Description |
|---|---|
| `evacuation_speed` | Points per newly evacuated person, scaled by time remaining |
| `route_safety` | Penalty for routing persons through or toward hazard zones |
| `dispatch_accuracy` | Bonus for matching dispatched service to active hazard type |
| `severity_accuracy` | Bonus for labelling incident severity within one bracket of ground truth |
| `format_compliance` | Small bonus for well-formed action (all required fields present) |

A `timeout_penalty` is applied at episode end if persons remain inside when `max_ticks` is reached.

## How to Use Locally

```bash
pip install fastapi uvicorn pydantic requests
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Example Python Client

```python
import requests

BASE = "http://localhost:7860"

# Start episode
obs = requests.post(f"{BASE}/reset").json()

done = False
while not done:
    exits = obs["available_exits"]
    zones = [z for z, n in obs["zone_occupancy"].items() if n > 0]

    if zones and exits:
        action = {
            "action_type": "route_zone",
            "zone_id": zones[0],
            "route_to_exit": exits[0],
        }
    else:
        action = {"action_type": "broadcast_pa", "message": "Evacuate now"}

    result = requests.post(f"{BASE}/step", json=action).json()
    obs = result["observation"]
    print(f"tick={obs['tick']}  reward={result['reward_breakdown']['total']:+.3f}")
    done = result["done"]

print(requests.get(f"{BASE}/health").json())
```

### Full Client

Use `client.py` (included in this repo) for a batteries-included client with a 3-episode random demo:

```bash
python client.py
```

---

![Reward Curve](reward_curve.png)
