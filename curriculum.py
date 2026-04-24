"""Curriculum management and post-incident learning log for CrisisCoreEnv."""
from __future__ import annotations

import random
from collections import deque
from typing import Callable

from schema import AgentAction, BuildingState, RewardBreakdown

CURRICULUM_LEVELS: dict[int, dict] = {
    1: {
        "num_floors": 1,
        "num_zones_per_floor": 4,
        "num_people": 5,
        "num_hazards": 1,
        "max_ticks": 20,
        "sensor_noise_rate": 0.0,
    },
    2: {
        "num_floors": 2,
        "num_zones_per_floor": 4,
        "num_people": 15,
        "num_hazards": 2,
        "max_ticks": 30,
        "sensor_noise_rate": 0.1,
    },
    3: {
        "num_floors": 3,
        "num_zones_per_floor": 6,
        "num_people": 30,
        "num_hazards": 3,
        "max_ticks": 40,
        "sensor_noise_rate": 0.15,
    },
}


class CurriculumManager:
    def __init__(self, promotion_threshold: float = 0.7, window: int = 20) -> None:
        self._level = 1
        self._threshold = promotion_threshold
        self._window = window
        self._scores: deque[float] = deque(maxlen=window)

    def record_episode(self, reward_total: float, max_possible: float) -> None:
        normalized = reward_total / max_possible if max_possible != 0 else 0.0
        self._scores.append(normalized)
        if self.should_promote() and self._level < 3:
            self._level += 1
            self._scores.clear()

    def get_config(self) -> dict:
        return CURRICULUM_LEVELS[self._level]

    def current_level(self) -> int:
        return self._level

    def should_promote(self) -> bool:
        if len(self._scores) < self._window:
            return False
        return (sum(self._scores) / len(self._scores)) > self._threshold


class IncidentLog:
    def __init__(self, max_entries: int = 10) -> None:
        self._log: deque[str] = deque(maxlen=max_entries)

    def record(self, state: BuildingState, action: AgentAction, reward: RewardBreakdown) -> None:
        tick = state.tick
        if reward.route_safety < -1.0:
            zone_id = action.zone_id or "unknown"
            self._log.append(f"Routed zone {zone_id} through hazard at tick {tick}")
        if reward.dispatch_accuracy < 0.0:
            self._log.append(f"Wrong service dispatched at tick {tick}")
        if reward.severity_accuracy < 0.0:
            self._log.append(f"Severity misclassified at tick {tick}")
        if reward.timeout_penalty < 0.0:
            unrescued = sum(1 for p in state.people.values() if not p.is_evacuated)
            self._log.append(f"Episode timed out with {unrescued} people unrescued")

    def get_log(self) -> list[str]:
        return list(self._log)

    def clear(self) -> None:
        self._log.clear()


def run_curriculum_episode(
    env,
    orchestrator,
    model_fn: Callable[[str], str],
    incident_log: IncidentLog,
) -> tuple[float, dict]:
    obs = env.reset()
    total_reward = 0.0
    info: dict = {}
    done = False

    while not done:
        actions = orchestrator.act(obs, env.state, model_fn)
        obs, reward, done, info = env.step(actions[0])
        total_reward += reward.total
        incident_log.record(env.state, actions[0], reward)

    return float(total_reward), info


# ---------------------------------------------------------------------------
# Simulation: 30 episodes, print every 10, verify promotion
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    from agents import OrchestratorAgent
    from environment import CrisisCoreEnv

    curriculum = CurriculumManager(promotion_threshold=0.7, window=10)
    incident_log = IncidentLog()

    def _make_env() -> CrisisCoreEnv:
        return CrisisCoreEnv(curriculum.get_config())

    def random_model_fn(prompt: str) -> str:
        choices = [
            {"action_type": "route_zone", "zone_id": "z1_f1", "route_to_exit": "exit_z1_f1"},
            {"action_type": "dispatch_service", "service_type": random.choice(["fire_brigade", "ems", "police"])},
            {"action_type": "broadcast_pa", "message": "Please evacuate immediately."},
        ]
        return json.dumps(random.choice(choices))

    MAX_POSSIBLE = 50.0
    rewards_window: list[float] = []

    for episode in range(1, 31):
        env = _make_env()
        orchestrator = OrchestratorAgent()
        total, _ = run_curriculum_episode(env, orchestrator, random_model_fn, incident_log)
        curriculum.record_episode(total, MAX_POSSIBLE)
        rewards_window.append(total)

        if episode % 10 == 0:
            mean_reward = sum(rewards_window[-10:]) / 10
            print(f"\n=== After episode {episode} ===")
            print(f"  Level     : {curriculum.current_level()}")
            print(f"  Mean reward (last 10): {mean_reward:.3f}")
            print(f"  Incident log ({len(incident_log.get_log())} entries):")
            for entry in incident_log.get_log():
                print(f"    - {entry}")

    # Verify promotion with artificially high rewards
    print("\n--- Promotion verification with high rewards ---")
    verifier = CurriculumManager(promotion_threshold=0.7, window=5)
    for _ in range(5):
        verifier.record_episode(reward_total=10.0, max_possible=10.0)
    assert verifier.current_level() == 2, "Expected promotion to level 2"
    for _ in range(5):
        verifier.record_episode(reward_total=10.0, max_possible=10.0)
    assert verifier.current_level() == 3, "Expected promotion to level 3"
    for _ in range(10):
        verifier.record_episode(reward_total=10.0, max_possible=10.0)
    assert verifier.current_level() == 3, "Should not exceed level 3"
    print("Promotion verification passed: levels 1 -> 2 -> 3 (capped at 3).")
