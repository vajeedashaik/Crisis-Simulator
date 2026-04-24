from __future__ import annotations

from schema import (
    ActionType,
    AgentAction,
    AgentObservation,
    BuildingState,
    HazardType,
    RewardBreakdown,
    SeverityLevel,
    ServiceType,
)

_SEVERITY_ORDER = [SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH, SeverityLevel.CRITICAL]
_SEVERITY_RANK = {s: i for i, s in enumerate(_SEVERITY_ORDER)}


def evacuation_speed_reward(state: BuildingState, prev_evacuated: int) -> float:
    current_evacuated = sum(1 for p in state.people.values() if p.is_evacuated)
    new_evacuated = current_evacuated - prev_evacuated
    if new_evacuated <= 0:
        return 0.0
    time_bonus = 1.0 + (state.max_ticks - state.tick) / state.max_ticks
    return min(new_evacuated * time_bonus, 5.0)


def route_safety_reward(state: BuildingState, action: AgentAction) -> float:
    if action.action_type != ActionType.ROUTE_ZONE:
        return 0.0

    hazard_zones: set[str] = set()
    for hazard in state.hazards.values():
        hazard_zones.update(hazard.affected_zones)

    if action.zone_id is not None and action.zone_id in hazard_zones:
        return -3.0

    if action.route_to_exit is not None and action.route_to_exit in hazard_zones:
        return -2.0

    return 0.0


def dispatch_accuracy_reward(state: BuildingState, action: AgentAction) -> float:
    if action.action_type != ActionType.DISPATCH_SERVICE:
        return 0.0

    hazard_types = {h.hazard_type for h in state.hazards.values()}

    if action.service_type == ServiceType.FIRE_BRIGADE:
        if HazardType.FIRE in hazard_types or HazardType.SMOKE in hazard_types:
            return 2.0
        return 0.0

    if action.service_type == ServiceType.EMS:
        if hazard_types:
            return 1.0
        return 0.0

    if action.service_type == ServiceType.POLICE:
        crowd_risk_types = {HazardType.FIRE, HazardType.SMOKE, HazardType.STRUCTURAL}
        # POLICE penalized only when no crowd-risk hazard present
        if not hazard_types.intersection(crowd_risk_types):
            return -0.5
        return 0.0

    return 0.0


def severity_accuracy_reward(state: BuildingState, action: AgentAction) -> float:
    if action.action_type != ActionType.UPDATE_SEVERITY:
        return 0.0

    total_people = len(state.people)
    if total_people == 0:
        return 0.0

    hazard_zones: set[str] = set()
    for hazard in state.hazards.values():
        hazard_zones.update(hazard.affected_zones)

    people_in_hazard = sum(
        1 for p in state.people.values()
        if not p.is_evacuated and p.current_zone in hazard_zones
    )
    ratio = people_in_hazard / total_people

    if ratio < 0.1:
        ground_truth = SeverityLevel.LOW
    elif ratio < 0.3:
        ground_truth = SeverityLevel.MEDIUM
    elif ratio < 0.6:
        ground_truth = SeverityLevel.HIGH
    else:
        ground_truth = SeverityLevel.CRITICAL

    if action.severity is None:
        return -1.0

    diff = abs(_SEVERITY_RANK[action.severity] - _SEVERITY_RANK[ground_truth])
    if diff == 0:
        return 1.5
    if diff == 1:
        return 0.5
    return -1.0


def format_compliance_reward(action: AgentAction) -> float:
    if action.action_type == ActionType.ROUTE_ZONE:
        if action.zone_id is not None and action.route_to_exit is not None:
            return 0.2
        return -1.0

    if action.action_type == ActionType.DISPATCH_SERVICE:
        if action.service_type is not None:
            return 0.2
        return -1.0

    if action.action_type == ActionType.BROADCAST_PA:
        if action.message is not None:
            return 0.2
        return -1.0

    if action.action_type == ActionType.UPDATE_SEVERITY:
        if action.severity is not None:
            return 0.2
        return -1.0

    return -1.0


def timeout_penalty(state: BuildingState, done: bool) -> float:
    if done and not all(p.is_evacuated for p in state.people.values()):
        return -5.0
    return 0.0


def compute_reward(
    state: BuildingState,
    action: AgentAction,
    prev_evacuated: int,
    done: bool,
) -> RewardBreakdown:
    evac = evacuation_speed_reward(state, prev_evacuated)
    route = route_safety_reward(state, action)
    dispatch = dispatch_accuracy_reward(state, action)
    severity = severity_accuracy_reward(state, action)
    fmt = format_compliance_reward(action)
    timeout = timeout_penalty(state, done)

    total = (
        evac * 1.0
        + route * 1.0
        + dispatch * 0.8
        + severity * 0.6
        + fmt * 0.4
        + timeout * 1.0
    )

    return RewardBreakdown(
        evacuation_speed=evac,
        route_safety=route,
        dispatch_accuracy=dispatch,
        severity_accuracy=severity,
        format_compliance=fmt,
        timeout_penalty=timeout,
        total=total,
    )


# ── Anti-hacking test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from schema import Hazard, HazardType, Person, SeverityLevel, Zone

    def _make_state(tick: int = 5, evacuated_ids: set[str] | None = None) -> BuildingState:
        evacuated_ids = evacuated_ids or set()
        zones = {
            "Z1": Zone("Z1", 0, 10, ["Z2"], False),
            "Z2": Zone("Z2", 0, 10, ["Z1", "EXIT"], True, "EXIT"),
        }
        people = {
            f"P{i}": Person(f"P{i}", "Z1", f"P{i}" in evacuated_ids, False)
            for i in range(5)
        }
        hazards = {
            "H1": Hazard("H1", HazardType.FIRE, ["Z1"], 0.1, 0.8),
        }
        return BuildingState(
            building_id="B1",
            floors=1,
            zones=zones,
            people=people,
            hazards=hazards,
            sensor_readings={},
            blocked_exits=[],
            tick=tick,
            max_ticks=20,
            episode_done=False,
        )

    # Scenario A: DISPATCH_SERVICE FIRE_BRIGADE every tick (anti-hacking check)
    print("=== Anti-hacking: DISPATCH_SERVICE every tick ===")
    state = _make_state()
    action_dispatch = AgentAction(
        action_type=ActionType.DISPATCH_SERVICE,
        service_type=ServiceType.FIRE_BRIGADE,
    )
    rewards_over_ticks = []
    for t in range(1, 6):
        s = _make_state(tick=t)
        r = compute_reward(s, action_dispatch, prev_evacuated=0, done=False)
        rewards_over_ticks.append(r.total)

    print(f"Rewards per tick (same dispatch action): {rewards_over_ticks}")
    assert all(
        abs(rewards_over_ticks[i] - rewards_over_ticks[0]) < 0.01
        for i in range(1, len(rewards_over_ticks))
    ), "FAIL: reward grows unboundedly with repeated DISPATCH_SERVICE"
    print("PASS: reward stays constant — no unbounded growth\n")

    # Three scenarios
    scenarios = [
        (
            "Good: ROUTE safe zone to exit",
            AgentAction(action_type=ActionType.ROUTE_ZONE, zone_id="Z2", route_to_exit="EXIT"),
            0,
            False,
        ),
        (
            "Bad: ROUTE through hazard zone",
            AgentAction(action_type=ActionType.ROUTE_ZONE, zone_id="Z1", route_to_exit="EXIT"),
            0,
            False,
        ),
        (
            "Timeout: episode done, people trapped",
            AgentAction(action_type=ActionType.DISPATCH_SERVICE, service_type=ServiceType.EMS),
            0,
            True,
        ),
    ]

    header = f"{'Scenario':<40} {'EvacSpd':>7} {'Route':>6} {'Dispatch':>8} {'Severity':>8} {'Format':>6} {'Timeout':>7} {'Total':>7}"
    print(header)
    print("-" * len(header))
    for label, action, prev_evac, done in scenarios:
        s = _make_state()
        r = compute_reward(s, action, prev_evacuated=prev_evac, done=done)
        print(
            f"{label:<40} {r.evacuation_speed:>7.2f} {r.route_safety:>6.2f} "
            f"{r.dispatch_accuracy:>8.2f} {r.severity_accuracy:>8.2f} "
            f"{r.format_compliance:>6.2f} {r.timeout_penalty:>7.2f} {r.total:>7.2f}"
        )
