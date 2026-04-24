"""Unit tests for rewards.py — phase 3."""
from __future__ import annotations

import pytest

from rewards import (
    compute_reward,
    dispatch_accuracy_reward,
    evacuation_speed_reward,
    format_compliance_reward,
    route_safety_reward,
    severity_accuracy_reward,
    timeout_penalty,
)
from schema import (
    ActionType,
    AgentAction,
    BuildingState,
    Hazard,
    HazardType,
    Person,
    RewardBreakdown,
    SeverityLevel,
    ServiceType,
    Zone,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _state(
    *,
    tick: int = 5,
    max_ticks: int = 20,
    people: dict | None = None,
    hazards: dict | None = None,
    zones: dict | None = None,
) -> BuildingState:
    if zones is None:
        zones = {
            "Z1": Zone("Z1", 0, 10, ["Z2"], False),
            "Z2": Zone("Z2", 0, 10, ["Z1", "EXIT"], True, "EXIT"),
        }
    if people is None:
        people = {
            "P1": Person("P1", "Z1", False, False),
            "P2": Person("P2", "Z1", False, False),
            "P3": Person("P3", "Z2", False, False),
        }
    if hazards is None:
        hazards = {}
    return BuildingState(
        building_id="B1",
        floors=1,
        zones=zones,
        people=people,
        hazards=hazards,
        sensor_readings={},
        blocked_exits=[],
        tick=tick,
        max_ticks=max_ticks,
        episode_done=False,
    )


def _fire_in(zone_id: str) -> dict:
    return {"H1": Hazard("H1", HazardType.FIRE, [zone_id], 0.1, 0.8)}


# ── evacuation_speed_reward ───────────────────────────────────────────────────

class TestEvacuationSpeedReward:
    def test_no_new_evacuees_returns_zero(self):
        s = _state(people={"P1": Person("P1", "Z1", True, False)})
        assert evacuation_speed_reward(s, prev_evacuated=1) == 0.0

    def test_one_new_evacuee_base_score(self):
        s = _state(tick=0, max_ticks=20, people={"P1": Person("P1", "Z1", True, False)})
        result = evacuation_speed_reward(s, prev_evacuated=0)
        expected = min(1 * (1 + (20 - 0) / 20), 5.0)
        assert abs(result - expected) < 1e-9

    def test_time_bonus_decreases_later(self):
        people = {"P1": Person("P1", "Z1", True, False)}
        early = evacuation_speed_reward(_state(tick=0, max_ticks=20, people=people), 0)
        late = evacuation_speed_reward(_state(tick=18, max_ticks=20, people=people), 0)
        assert early > late

    def test_cap_at_five(self):
        people = {f"P{i}": Person(f"P{i}", "Z1", True, False) for i in range(10)}
        s = _state(tick=0, max_ticks=20, people=people)
        assert evacuation_speed_reward(s, prev_evacuated=0) == 5.0

    def test_negative_delta_returns_zero(self):
        s = _state(people={"P1": Person("P1", "Z1", False, False)})
        assert evacuation_speed_reward(s, prev_evacuated=5) == 0.0


# ── route_safety_reward ───────────────────────────────────────────────────────

class TestRouteSafetyReward:
    def test_non_route_action_returns_zero(self):
        s = _state()
        a = AgentAction(action_type=ActionType.DISPATCH_SERVICE, service_type=ServiceType.EMS)
        assert route_safety_reward(s, a) == 0.0

    def test_safe_route_returns_zero(self):
        s = _state(hazards=_fire_in("Z1"))
        a = AgentAction(action_type=ActionType.ROUTE_ZONE, zone_id="Z2", route_to_exit="EXIT")
        assert route_safety_reward(s, a) == 0.0

    def test_routed_zone_has_hazard_returns_minus_three(self):
        s = _state(hazards=_fire_in("Z1"))
        a = AgentAction(action_type=ActionType.ROUTE_ZONE, zone_id="Z1", route_to_exit="EXIT")
        assert route_safety_reward(s, a) == -3.0

    def test_route_to_exit_through_hazard_returns_minus_two(self):
        s = _state(hazards=_fire_in("CORRIDOR"))
        a = AgentAction(action_type=ActionType.ROUTE_ZONE, zone_id="Z2", route_to_exit="CORRIDOR")
        assert route_safety_reward(s, a) == -2.0

    def test_zone_hazard_takes_priority_over_exit_hazard(self):
        hazards = {
            "H1": Hazard("H1", HazardType.FIRE, ["Z1"], 0.1, 0.8),
            "H2": Hazard("H2", HazardType.SMOKE, ["EXIT"], 0.1, 0.5),
        }
        s = _state(hazards=hazards)
        a = AgentAction(action_type=ActionType.ROUTE_ZONE, zone_id="Z1", route_to_exit="EXIT")
        assert route_safety_reward(s, a) == -3.0

    def test_no_hazards_returns_zero(self):
        s = _state()
        a = AgentAction(action_type=ActionType.ROUTE_ZONE, zone_id="Z1", route_to_exit="EXIT")
        assert route_safety_reward(s, a) == 0.0


# ── dispatch_accuracy_reward ──────────────────────────────────────────────────

class TestDispatchAccuracyReward:
    def test_non_dispatch_returns_zero(self):
        s = _state(hazards=_fire_in("Z1"))
        a = AgentAction(action_type=ActionType.ROUTE_ZONE, zone_id="Z1", route_to_exit="EXIT")
        assert dispatch_accuracy_reward(s, a) == 0.0

    def test_fire_brigade_for_fire_returns_two(self):
        s = _state(hazards=_fire_in("Z1"))
        a = AgentAction(action_type=ActionType.DISPATCH_SERVICE, service_type=ServiceType.FIRE_BRIGADE)
        assert dispatch_accuracy_reward(s, a) == 2.0

    def test_fire_brigade_for_smoke_returns_two(self):
        hazards = {"H1": Hazard("H1", HazardType.SMOKE, ["Z1"], 0.1, 0.5)}
        s = _state(hazards=hazards)
        a = AgentAction(action_type=ActionType.DISPATCH_SERVICE, service_type=ServiceType.FIRE_BRIGADE)
        assert dispatch_accuracy_reward(s, a) == 2.0

    def test_ems_any_hazard_returns_one(self):
        s = _state(hazards=_fire_in("Z1"))
        a = AgentAction(action_type=ActionType.DISPATCH_SERVICE, service_type=ServiceType.EMS)
        assert dispatch_accuracy_reward(s, a) == 1.0

    def test_police_no_hazard_returns_minus_half(self):
        s = _state()
        a = AgentAction(action_type=ActionType.DISPATCH_SERVICE, service_type=ServiceType.POLICE)
        assert dispatch_accuracy_reward(s, a) == -0.5

    def test_fire_brigade_no_fire_returns_zero(self):
        hazards = {"H1": Hazard("H1", HazardType.STRUCTURAL, ["Z1"], 0.0, 0.3)}
        s = _state(hazards=hazards)
        a = AgentAction(action_type=ActionType.DISPATCH_SERVICE, service_type=ServiceType.FIRE_BRIGADE)
        assert dispatch_accuracy_reward(s, a) == 0.0

    def test_ems_no_hazard_returns_zero(self):
        s = _state()
        a = AgentAction(action_type=ActionType.DISPATCH_SERVICE, service_type=ServiceType.EMS)
        assert dispatch_accuracy_reward(s, a) == 0.0


# ── severity_accuracy_reward ──────────────────────────────────────────────────

class TestSeverityAccuracyReward:
    def _people_in_hazard(self, n_in: int, n_out: int) -> dict:
        people = {}
        for i in range(n_in):
            people[f"IN{i}"] = Person(f"IN{i}", "Z1", False, False)  # Z1 = hazard zone
        for i in range(n_out):
            people[f"OUT{i}"] = Person(f"OUT{i}", "Z2", False, False)
        return people

    def test_non_update_action_returns_zero(self):
        s = _state()
        a = AgentAction(action_type=ActionType.DISPATCH_SERVICE, service_type=ServiceType.EMS)
        assert severity_accuracy_reward(s, a) == 0.0

    def test_exact_match_returns_one_five(self):
        # 0/5 in hazard → LOW
        people = {"P1": Person("P1", "Z2", False, False)}
        s = _state(people=people, hazards=_fire_in("Z1"))
        a = AgentAction(action_type=ActionType.UPDATE_SEVERITY, severity=SeverityLevel.LOW)
        assert severity_accuracy_reward(s, a) == 1.5

    def test_one_off_returns_half(self):
        # all 5 in hazard (Z1) → CRITICAL; action says HIGH (1 off)
        people = {f"P{i}": Person(f"P{i}", "Z1", False, False) for i in range(5)}
        s = _state(people=people, hazards=_fire_in("Z1"))
        a = AgentAction(action_type=ActionType.UPDATE_SEVERITY, severity=SeverityLevel.HIGH)
        assert severity_accuracy_reward(s, a) == 0.5

    def test_two_off_returns_minus_one(self):
        # all 5 in hazard → CRITICAL; action says MEDIUM (2 off)
        people = {f"P{i}": Person(f"P{i}", "Z1", False, False) for i in range(5)}
        s = _state(people=people, hazards=_fire_in("Z1"))
        a = AgentAction(action_type=ActionType.UPDATE_SEVERITY, severity=SeverityLevel.MEDIUM)
        assert severity_accuracy_reward(s, a) == -1.0

    def test_no_people_returns_zero(self):
        s = _state(people={}, hazards=_fire_in("Z1"))
        a = AgentAction(action_type=ActionType.UPDATE_SEVERITY, severity=SeverityLevel.LOW)
        assert severity_accuracy_reward(s, a) == 0.0

    def test_evacuated_people_excluded_from_ratio(self):
        # 5 total, 4 evacuated (not in hazard count), 1 in hazard → 1/5 = 0.2 → MEDIUM
        people = {
            "P0": Person("P0", "Z1", False, False),
            **{f"P{i}": Person(f"P{i}", "Z1", True, False) for i in range(1, 5)},
        }
        s = _state(people=people, hazards=_fire_in("Z1"))
        a = AgentAction(action_type=ActionType.UPDATE_SEVERITY, severity=SeverityLevel.MEDIUM)
        assert severity_accuracy_reward(s, a) == 1.5

    def test_medium_threshold(self):
        # 2/10 in hazard = 0.2 → MEDIUM
        people = {
            **{f"IN{i}": Person(f"IN{i}", "Z1", False, False) for i in range(2)},
            **{f"OUT{i}": Person(f"OUT{i}", "Z2", False, False) for i in range(8)},
        }
        s = _state(people=people, hazards=_fire_in("Z1"))
        a = AgentAction(action_type=ActionType.UPDATE_SEVERITY, severity=SeverityLevel.MEDIUM)
        assert severity_accuracy_reward(s, a) == 1.5

    def test_high_threshold(self):
        # 4/10 = 0.4 → HIGH
        people = {
            **{f"IN{i}": Person(f"IN{i}", "Z1", False, False) for i in range(4)},
            **{f"OUT{i}": Person(f"OUT{i}", "Z2", False, False) for i in range(6)},
        }
        s = _state(people=people, hazards=_fire_in("Z1"))
        a = AgentAction(action_type=ActionType.UPDATE_SEVERITY, severity=SeverityLevel.HIGH)
        assert severity_accuracy_reward(s, a) == 1.5

    def test_none_severity_returns_minus_one(self):
        s = _state()
        a = AgentAction(action_type=ActionType.UPDATE_SEVERITY, severity=None)
        assert severity_accuracy_reward(s, a) == -1.0


# ── format_compliance_reward ──────────────────────────────────────────────────

class TestFormatComplianceReward:
    def test_route_zone_complete_fields(self):
        a = AgentAction(action_type=ActionType.ROUTE_ZONE, zone_id="Z1", route_to_exit="EXIT")
        assert format_compliance_reward(a) == 0.2

    def test_route_zone_missing_zone_id(self):
        a = AgentAction(action_type=ActionType.ROUTE_ZONE, route_to_exit="EXIT")
        assert format_compliance_reward(a) == -1.0

    def test_route_zone_missing_route_to_exit(self):
        a = AgentAction(action_type=ActionType.ROUTE_ZONE, zone_id="Z1")
        assert format_compliance_reward(a) == -1.0

    def test_dispatch_complete(self):
        a = AgentAction(action_type=ActionType.DISPATCH_SERVICE, service_type=ServiceType.EMS)
        assert format_compliance_reward(a) == 0.2

    def test_dispatch_missing_service_type(self):
        a = AgentAction(action_type=ActionType.DISPATCH_SERVICE)
        assert format_compliance_reward(a) == -1.0

    def test_broadcast_complete(self):
        a = AgentAction(action_type=ActionType.BROADCAST_PA, message="Evacuate now")
        assert format_compliance_reward(a) == 0.2

    def test_broadcast_missing_message(self):
        a = AgentAction(action_type=ActionType.BROADCAST_PA)
        assert format_compliance_reward(a) == -1.0

    def test_update_severity_complete(self):
        a = AgentAction(action_type=ActionType.UPDATE_SEVERITY, severity=SeverityLevel.HIGH)
        assert format_compliance_reward(a) == 0.2

    def test_update_severity_missing_severity(self):
        a = AgentAction(action_type=ActionType.UPDATE_SEVERITY)
        assert format_compliance_reward(a) == -1.0


# ── timeout_penalty ───────────────────────────────────────────────────────────

class TestTimeoutPenalty:
    def test_not_done_returns_zero(self):
        s = _state(people={"P1": Person("P1", "Z1", False, False)})
        assert timeout_penalty(s, done=False) == 0.0

    def test_done_all_evacuated_returns_zero(self):
        s = _state(people={"P1": Person("P1", "Z1", True, False)})
        assert timeout_penalty(s, done=True) == 0.0

    def test_done_not_all_evacuated_returns_minus_five(self):
        s = _state(people={"P1": Person("P1", "Z1", False, False)})
        assert timeout_penalty(s, done=True) == -5.0

    def test_done_mixed_evacuated_returns_minus_five(self):
        people = {
            "P1": Person("P1", "Z1", True, False),
            "P2": Person("P2", "Z1", False, False),
        }
        s = _state(people=people)
        assert timeout_penalty(s, done=True) == -5.0

    def test_empty_people_done_returns_zero(self):
        s = _state(people={})
        assert timeout_penalty(s, done=True) == 0.0


# ── compute_reward ────────────────────────────────────────────────────────────

class TestComputeReward:
    def test_returns_reward_breakdown(self):
        s = _state()
        a = AgentAction(action_type=ActionType.BROADCAST_PA, message="Evacuate")
        result = compute_reward(s, a, prev_evacuated=0, done=False)
        assert isinstance(result, RewardBreakdown)

    def test_total_is_weighted_sum(self):
        s = _state(hazards=_fire_in("Z1"))
        a = AgentAction(action_type=ActionType.DISPATCH_SERVICE, service_type=ServiceType.FIRE_BRIGADE)
        r = compute_reward(s, a, prev_evacuated=0, done=False)
        expected_total = (
            r.evacuation_speed * 1.0
            + r.route_safety * 1.0
            + r.dispatch_accuracy * 0.8
            + r.severity_accuracy * 0.6
            + r.format_compliance * 0.4
            + r.timeout_penalty * 1.0
        )
        assert abs(r.total - expected_total) < 1e-9

    def test_all_fields_populated(self):
        s = _state()
        a = AgentAction(action_type=ActionType.BROADCAST_PA, message="Stay calm")
        r = compute_reward(s, a, prev_evacuated=0, done=False)
        for field in ("evacuation_speed", "route_safety", "dispatch_accuracy",
                      "severity_accuracy", "format_compliance", "timeout_penalty", "total"):
            assert hasattr(r, field)

    def test_timeout_included_in_total(self):
        s = _state(people={"P1": Person("P1", "Z1", False, False)})
        a = AgentAction(action_type=ActionType.BROADCAST_PA, message="Help")
        r = compute_reward(s, a, prev_evacuated=0, done=True)
        assert r.timeout_penalty == -5.0
        assert r.total < 0

    def test_dispatch_every_tick_not_unbounded(self):
        a = AgentAction(action_type=ActionType.DISPATCH_SERVICE, service_type=ServiceType.FIRE_BRIGADE)
        rewards = []
        for tick in range(1, 10):
            people = {"P1": Person("P1", "Z1", False, False)}
            hazards = {"H1": Hazard("H1", HazardType.FIRE, ["Z1"], 0.1, 0.8)}
            s = BuildingState(
                building_id="B", floors=1,
                zones={"Z1": Zone("Z1", 0, 10, [], False)},
                people=people, hazards=hazards, sensor_readings={},
                blocked_exits=[], tick=tick, max_ticks=20, episode_done=False,
            )
            rewards.append(compute_reward(s, a, prev_evacuated=0, done=False).total)
        assert max(rewards) < 10.0, "Reward grew unboundedly with repeated dispatch"
