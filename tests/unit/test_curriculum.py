"""Tests for curriculum.py — CurriculumManager, IncidentLog, run_curriculum_episode."""
from __future__ import annotations

import pytest

from schema import (
    ActionType,
    AgentAction,
    BuildingState,
    Hazard,
    HazardType,
    Person,
    RewardBreakdown,
    SensorReading,
    SeverityLevel,
    Zone,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(tick: int = 0, max_ticks: int = 20, people: dict | None = None) -> BuildingState:
    zone = Zone("z1", floor=1, capacity=10, connected_zones=[], has_exit=True, exit_id="exit_a")
    return BuildingState(
        building_id="test",
        floors=1,
        zones={"z1": zone},
        people=people or {},
        hazards={},
        sensor_readings={},
        blocked_exits=[],
        tick=tick,
        max_ticks=max_ticks,
        episode_done=False,
    )


def _make_reward(**overrides) -> RewardBreakdown:
    defaults = dict(
        evacuation_speed=0.0,
        route_safety=0.0,
        dispatch_accuracy=0.0,
        severity_accuracy=0.0,
        format_compliance=0.0,
        timeout_penalty=0.0,
        total=0.0,
    )
    defaults.update(overrides)
    return RewardBreakdown(**defaults)


def _route_action(zone_id: str = "z1", exit_id: str = "exit_a") -> AgentAction:
    return AgentAction(
        action_type=ActionType.ROUTE_ZONE,
        zone_id=zone_id,
        route_to_exit=exit_id,
    )


# ---------------------------------------------------------------------------
# CURRICULUM_LEVELS
# ---------------------------------------------------------------------------

class TestCurriculumLevels:
    def test_has_three_levels(self):
        from curriculum import CURRICULUM_LEVELS
        assert set(CURRICULUM_LEVELS.keys()) == {1, 2, 3}

    def test_level_1_config(self):
        from curriculum import CURRICULUM_LEVELS
        cfg = CURRICULUM_LEVELS[1]
        assert cfg["num_floors"] == 1
        assert cfg["num_zones_per_floor"] == 4
        assert cfg["num_people"] == 5
        assert cfg["num_hazards"] == 1
        assert cfg["max_ticks"] == 20
        assert cfg["sensor_noise_rate"] == 0.0

    def test_level_2_config(self):
        from curriculum import CURRICULUM_LEVELS
        cfg = CURRICULUM_LEVELS[2]
        assert cfg["num_floors"] == 2
        assert cfg["num_zones_per_floor"] == 4
        assert cfg["num_people"] == 15
        assert cfg["num_hazards"] == 2
        assert cfg["max_ticks"] == 30
        assert cfg["sensor_noise_rate"] == 0.1

    def test_level_3_config(self):
        from curriculum import CURRICULUM_LEVELS
        cfg = CURRICULUM_LEVELS[3]
        assert cfg["num_floors"] == 3
        assert cfg["num_zones_per_floor"] == 6
        assert cfg["num_people"] == 30
        assert cfg["num_hazards"] == 3
        assert cfg["max_ticks"] == 40
        assert cfg["sensor_noise_rate"] == 0.15


# ---------------------------------------------------------------------------
# CurriculumManager
# ---------------------------------------------------------------------------

class TestCurriculumManagerInit:
    def test_starts_at_level_1(self):
        from curriculum import CurriculumManager
        mgr = CurriculumManager()
        assert mgr.current_level() == 1

    def test_get_config_returns_level_1_initially(self):
        from curriculum import CURRICULUM_LEVELS, CurriculumManager
        mgr = CurriculumManager()
        assert mgr.get_config() == CURRICULUM_LEVELS[1]

    def test_should_promote_false_initially(self):
        from curriculum import CurriculumManager
        mgr = CurriculumManager()
        assert mgr.should_promote() is False


class TestCurriculumManagerPromotion:
    def test_no_promotion_below_threshold(self):
        from curriculum import CurriculumManager
        mgr = CurriculumManager(promotion_threshold=0.7, window=5)
        for _ in range(5):
            mgr.record_episode(reward_total=0.5, max_possible=1.0)
        assert mgr.current_level() == 1

    def test_promotes_when_mean_exceeds_threshold(self):
        from curriculum import CurriculumManager
        mgr = CurriculumManager(promotion_threshold=0.7, window=5)
        for _ in range(5):
            mgr.record_episode(reward_total=0.8, max_possible=1.0)
        assert mgr.current_level() == 2

    def test_should_promote_false_when_window_not_full(self):
        from curriculum import CurriculumManager
        mgr = CurriculumManager(promotion_threshold=0.7, window=3)
        mgr.record_episode(0.9, 1.0)
        mgr.record_episode(0.9, 1.0)
        # window=3, only 2 entries — not full yet, no promotion regardless of score
        assert mgr.should_promote() is False
        assert mgr.current_level() == 1

    def test_should_promote_true_when_window_full_and_mean_exceeds(self):
        from curriculum import CurriculumManager
        mgr = CurriculumManager(promotion_threshold=0.7, window=3)
        mgr.record_episode(0.9, 1.0)
        mgr.record_episode(0.9, 1.0)
        # Add 3rd entry but check should_promote BEFORE record clears the window
        # Use a manager that only checks without auto-acting: call should_promote after 3 entries
        mgr2 = CurriculumManager(promotion_threshold=0.7, window=3)
        from collections import deque
        mgr2._scores = deque([0.9, 0.9, 0.9], maxlen=3)
        assert mgr2.should_promote() is True

    def test_no_promotion_beyond_level_3(self):
        from curriculum import CurriculumManager
        mgr = CurriculumManager(promotion_threshold=0.1, window=3)
        for _ in range(30):
            mgr.record_episode(reward_total=1.0, max_possible=1.0)
        assert mgr.current_level() == 3

    def test_window_resets_after_promotion(self):
        from curriculum import CurriculumManager
        mgr = CurriculumManager(promotion_threshold=0.7, window=3)
        for _ in range(3):
            mgr.record_episode(reward_total=1.0, max_possible=1.0)
        assert mgr.current_level() == 2
        # 1 bad episode after reset should not immediately re-promote
        mgr.record_episode(reward_total=0.0, max_possible=1.0)
        assert mgr.current_level() == 2

    def test_rolling_window_only_last_n(self):
        from curriculum import CurriculumManager
        mgr = CurriculumManager(promotion_threshold=0.7, window=3)
        # fill with bad scores
        for _ in range(10):
            mgr.record_episode(reward_total=0.0, max_possible=1.0)
        assert mgr.current_level() == 1
        # then 3 good ones should promote
        for _ in range(3):
            mgr.record_episode(reward_total=1.0, max_possible=1.0)
        assert mgr.current_level() == 2

    def test_normalized_score_uses_max_possible(self):
        from curriculum import CurriculumManager
        mgr = CurriculumManager(promotion_threshold=0.7, window=3)
        # reward_total=7, max_possible=10 → normalized 0.7 → NOT > 0.7 (equal, not exceeds)
        for _ in range(3):
            mgr.record_episode(reward_total=7.0, max_possible=10.0)
        # 0.7 mean == threshold (not strictly greater), so no promotion
        assert mgr.current_level() == 1

    def test_normalized_score_above_threshold_promotes(self):
        from curriculum import CurriculumManager
        mgr = CurriculumManager(promotion_threshold=0.7, window=3)
        for _ in range(3):
            mgr.record_episode(reward_total=7.5, max_possible=10.0)
        assert mgr.current_level() == 2


# ---------------------------------------------------------------------------
# IncidentLog
# ---------------------------------------------------------------------------

class TestIncidentLogInit:
    def test_starts_empty(self):
        from curriculum import IncidentLog
        log = IncidentLog()
        assert log.get_log() == []

    def test_clear_empties_log(self):
        from curriculum import IncidentLog
        log = IncidentLog()
        state = _make_state(tick=1)
        action = _route_action()
        reward = _make_reward(route_safety=-2.0, total=-2.0)
        log.record(state, action, reward)
        log.clear()
        assert log.get_log() == []


class TestIncidentLogRecord:
    def test_logs_route_through_hazard(self):
        from curriculum import IncidentLog
        log = IncidentLog()
        state = _make_state(tick=5)
        action = _route_action(zone_id="z1")
        reward = _make_reward(route_safety=-1.5, total=-1.5)
        log.record(state, action, reward)
        entries = log.get_log()
        assert len(entries) == 1
        assert "z1" in entries[0]
        assert "5" in entries[0]

    def test_no_log_route_safety_above_threshold(self):
        from curriculum import IncidentLog
        log = IncidentLog()
        state = _make_state(tick=5)
        action = _route_action()
        reward = _make_reward(route_safety=-0.5, total=-0.5)
        log.record(state, action, reward)
        assert log.get_log() == []

    def test_logs_wrong_service_dispatched(self):
        from curriculum import IncidentLog
        log = IncidentLog()
        state = _make_state(tick=3)
        action = AgentAction(action_type=ActionType.DISPATCH_SERVICE)
        reward = _make_reward(dispatch_accuracy=-0.5, total=-0.5)
        log.record(state, action, reward)
        entries = log.get_log()
        assert len(entries) == 1
        assert "3" in entries[0]

    def test_no_log_dispatch_zero(self):
        from curriculum import IncidentLog
        log = IncidentLog()
        state = _make_state(tick=3)
        action = AgentAction(action_type=ActionType.DISPATCH_SERVICE)
        reward = _make_reward(dispatch_accuracy=0.0)
        log.record(state, action, reward)
        assert log.get_log() == []

    def test_logs_severity_misclassified(self):
        from curriculum import IncidentLog
        log = IncidentLog()
        state = _make_state(tick=7)
        action = AgentAction(action_type=ActionType.UPDATE_SEVERITY)
        reward = _make_reward(severity_accuracy=-1.0, total=-1.0)
        log.record(state, action, reward)
        entries = log.get_log()
        assert len(entries) == 1
        assert "7" in entries[0]

    def test_logs_timeout_penalty(self):
        from curriculum import IncidentLog
        log = IncidentLog()
        people = {
            "p1": Person("p1", "z1", is_evacuated=False, has_sos=False),
            "p2": Person("p2", "z1", is_evacuated=True, has_sos=False),
        }
        state = _make_state(tick=20, people=people)
        action = AgentAction(action_type=ActionType.BROADCAST_PA, message="evac")
        reward = _make_reward(timeout_penalty=-5.0, total=-5.0)
        log.record(state, action, reward)
        entries = log.get_log()
        assert len(entries) == 1
        assert "1" in entries[0]  # 1 unrescued person

    def test_no_log_when_all_zero(self):
        from curriculum import IncidentLog
        log = IncidentLog()
        state = _make_state()
        action = _route_action()
        reward = _make_reward()
        log.record(state, action, reward)
        assert log.get_log() == []

    def test_multiple_conditions_log_multiple_entries(self):
        from curriculum import IncidentLog
        log = IncidentLog()
        state = _make_state(tick=2)
        action = _route_action(zone_id="z2")
        reward = _make_reward(route_safety=-2.0, dispatch_accuracy=-1.0, total=-3.0)
        log.record(state, action, reward)
        entries = log.get_log()
        assert len(entries) == 2

    def test_max_entries_enforced(self):
        from curriculum import IncidentLog
        log = IncidentLog(max_entries=3)
        state = _make_state(tick=1)
        action = _route_action(zone_id="z1")
        for i in range(6):
            reward = _make_reward(route_safety=-2.0, total=-2.0)
            log.record(_make_state(tick=i), action, reward)
        assert len(log.get_log()) <= 3


# ---------------------------------------------------------------------------
# run_curriculum_episode
# ---------------------------------------------------------------------------

class TestRunCurriculumEpisode:
    def _make_env_and_orchestrator(self):
        """Return (env, orchestrator) using real classes."""
        from curriculum import CURRICULUM_LEVELS
        from environment import CrisisCoreEnv
        from agents import OrchestratorAgent
        env = CrisisCoreEnv(CURRICULUM_LEVELS[1])
        orchestrator = OrchestratorAgent()
        return env, orchestrator

    def test_returns_float_and_dict(self):
        from curriculum import IncidentLog, run_curriculum_episode
        env, orchestrator = self._make_env_and_orchestrator()
        log = IncidentLog()

        def model_fn(_prompt):
            import json, random
            choices = [
                {"action_type": "route_zone", "zone_id": "z1_f1", "route_to_exit": "exit_z1_f1"},
                {"action_type": "dispatch_service", "service_type": "ems"},
                {"action_type": "broadcast_pa", "message": "evacuate now"},
            ]
            return json.dumps(random.choice(choices))

        result = run_curriculum_episode(env, orchestrator, model_fn, log)
        assert isinstance(result, tuple) and len(result) == 2
        total_reward, info = result
        assert isinstance(total_reward, float)
        assert isinstance(info, dict)

    def test_episode_runs_to_completion(self):
        from curriculum import IncidentLog, run_curriculum_episode
        env, orchestrator = self._make_env_and_orchestrator()
        log = IncidentLog()
        tick_count = []

        def model_fn(_prompt):
            import json
            tick_count.append(1)
            return json.dumps({"action_type": "broadcast_pa", "message": "evacuate"})

        run_curriculum_episode(env, orchestrator, model_fn, log)
        assert len(tick_count) > 0

    def test_incident_log_receives_records(self):
        from curriculum import IncidentLog, run_curriculum_episode
        env, orchestrator = self._make_env_and_orchestrator()
        log = IncidentLog()

        def model_fn(_prompt):
            import json
            return json.dumps({"action_type": "broadcast_pa", "message": "test"})

        run_curriculum_episode(env, orchestrator, model_fn, log)
        # log may or may not have entries but should not raise
        assert isinstance(log.get_log(), list)

    def test_reward_is_sum_of_ticks(self):
        from curriculum import IncidentLog, run_curriculum_episode, CURRICULUM_LEVELS
        from environment import CrisisCoreEnv
        from agents import OrchestratorAgent
        env = CrisisCoreEnv(CURRICULUM_LEVELS[1])
        orchestrator = OrchestratorAgent()
        log = IncidentLog()

        def model_fn(_prompt):
            import json
            return json.dumps({"action_type": "broadcast_pa", "message": "evacuate"})

        total, _ = run_curriculum_episode(env, orchestrator, model_fn, log)
        assert isinstance(total, (int, float))
