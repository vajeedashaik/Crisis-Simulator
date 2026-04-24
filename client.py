"""Python client for CrisisCoreEnv FastAPI server."""
from __future__ import annotations

import random
from typing import Any

import requests

_HF_SPACE_URL = "https://yourusername-crisiscore.hf.space"


class CrisisCoreClient:
    def __init__(self, base_url: str = _HF_SPACE_URL) -> None:
        self.base_url = base_url.rstrip("/")

    def reset(self) -> dict[str, Any]:
        resp = requests.post(f"{self.base_url}/reset", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def step(self, action_dict: dict[str, Any]) -> dict[str, Any]:
        resp = requests.post(f"{self.base_url}/step", json=action_dict, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_state(self) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}/state", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def health_check(self) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Demo: 3 episodes with random actions
# ---------------------------------------------------------------------------

_ACTION_TYPES = ["route_zone", "dispatch_service", "broadcast_pa", "update_severity"]
_SERVICE_TYPES = ["fire_brigade", "ems", "police"]
_SEVERITIES = ["low", "medium", "high", "critical"]


def _random_action(obs: dict[str, Any]) -> dict[str, Any]:
    atype = random.choice(_ACTION_TYPES)

    if atype == "route_zone":
        exits = obs.get("available_exits", [])
        zones = [z for z, n in obs.get("zone_occupancy", {}).items() if n > 0]
        if exits and zones:
            return {
                "action_type": "route_zone",
                "zone_id": random.choice(zones),
                "route_to_exit": random.choice(exits),
            }
        return {"action_type": "broadcast_pa", "message": "Evacuate via nearest exit."}

    if atype == "dispatch_service":
        return {"action_type": "dispatch_service", "service_type": random.choice(_SERVICE_TYPES)}

    if atype == "broadcast_pa":
        return {"action_type": "broadcast_pa", "message": "Attention: proceed calmly to the nearest exit."}

    return {"action_type": "update_severity", "severity": random.choice(_SEVERITIES)}


if __name__ == "__main__":
    client = CrisisCoreClient(base_url="http://localhost:7860")

    print("Health:", client.health_check())

    for episode in range(1, 4):
        obs = client.reset()
        done = False
        total_reward = 0.0
        ticks = 0

        while not done:
            action = _random_action(obs)
            result = client.step(action)
            obs = result["observation"]
            total_reward += result["reward_breakdown"]["total"]
            done = result["done"]
            ticks += 1

        evacuated = obs.get("zone_occupancy")
        print(
            f"Episode {episode}: {ticks} ticks | "
            f"total_reward={total_reward:+.3f} | "
            f"sos_remaining={len(obs.get('sos_signals', []))}"
        )
