"""FastAPI wrapper for CrisisCoreEnv — HuggingFace Spaces deployment."""
from __future__ import annotations

import json
import random
import uuid
from dataclasses import asdict
from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agents import OrchestratorAgent
from curriculum import CURRICULUM_LEVELS
from environment import CrisisCoreEnv
from schema import (
    ActionType,
    AgentAction,
    Hazard,
    HazardType,
    ServiceType,
    SeverityLevel,
)

app = FastAPI(
    title="CrisisCore",
    description="Multi-Agent Crisis Response RL Environment",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Global state (single-episode server — suitable for demo / HF Space)
# ---------------------------------------------------------------------------

_env: CrisisCoreEnv = None  # type: ignore[assignment]
_orchestrator: OrchestratorAgent = None  # type: ignore[assignment]
_curriculum_level: int = 1


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _enum_default(obj):
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _to_dict(dataclass_obj) -> dict:
    """Convert any dataclass (possibly nested, with enum fields) to a plain dict."""
    raw = asdict(dataclass_obj)
    return json.loads(json.dumps(raw, default=_enum_default))


# ---------------------------------------------------------------------------
# Default model_fn for OrchestratorAgent (rule-based fallback, no LLM needed)
# ---------------------------------------------------------------------------

def _default_model_fn(prompt: str) -> str:
    if "evacuation" in prompt.lower():
        try:
            data = json.loads(prompt.split("USER:")[-1].strip())
            exits = data.get("available_exits", [])
            zones = [z for z, n in data.get("zone_occupancy", {}).items() if n > 0]
            zone = random.choice(zones) if zones else None
            exit_ = random.choice(exits) if exits else None
        except Exception:
            zone, exit_ = None, None
        return json.dumps({"action_type": "route_zone", "zone_id": zone, "route_to_exit": exit_})

    if "dispatch" in prompt.lower():
        return json.dumps({"action_type": "dispatch_service", "service_type": "fire_brigade"})

    if "comms" in prompt.lower() or "communications" in prompt.lower():
        return json.dumps({"action_type": "broadcast_pa", "message": "Attention: please evacuate calmly via the nearest exit."})

    return json.dumps({"action_type": "broadcast_pa", "message": "Please evacuate immediately."})


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup() -> None:
    global _env, _orchestrator, _curriculum_level
    _curriculum_level = 1
    config = CURRICULUM_LEVELS[_curriculum_level]
    _env = CrisisCoreEnv(config)
    _orchestrator = OrchestratorAgent()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ActionRequest(BaseModel):
    action_type: str
    zone_id: Optional[str] = None
    route_to_exit: Optional[str] = None
    service_type: Optional[str] = None
    message: Optional[str] = None
    severity: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", summary="Start a new episode")
async def reset():
    """Reset environment to initial state. Returns the first AgentObservation."""
    try:
        obs = _env.reset()
        return _to_dict(obs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/step", summary="Submit one action and advance the simulation")
async def step(body: ActionRequest):
    """
    Accept a single AgentAction as JSON, advance the environment one tick,
    and return (observation, reward_breakdown, done, info).
    """
    try:
        action = AgentAction(
            action_type=ActionType(body.action_type),
            zone_id=body.zone_id,
            route_to_exit=body.route_to_exit,
            service_type=ServiceType(body.service_type) if body.service_type else None,
            message=body.message,
            severity=SeverityLevel(body.severity) if body.severity else None,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid action field: {exc}")

    try:
        obs, reward, done, info = _env.step(action)
        return {
            "observation": _to_dict(obs),
            "reward_breakdown": _to_dict(reward),
            "done": done,
            "info": info,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/state", summary="Current BuildingState snapshot")
async def state():
    """Return the full BuildingState for dashboard polling."""
    if _env.state is None:
        raise HTTPException(status_code=400, detail="No active episode — call POST /reset first.")
    try:
        return _to_dict(_env.state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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


@app.get("/health", summary="Liveness check")
async def health():
    """Return server status, current tick, and curriculum level."""
    tick = _env.state.tick if (_env and _env.state) else 0
    return {"status": "ok", "tick": tick, "level": _curriculum_level}
