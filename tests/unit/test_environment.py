"""Unit tests for environment.py — Phase 2 CrisisCoreEnv."""
from __future__ import annotations

import math
import pytest

from environment import CrisisCoreEnv
from schema import (
    ActionType,
    ServiceType,
    HazardType,
    SeverityLevel,
    AgentAction,
    AgentObservation,
    RewardBreakdown,
    BuildingState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def default_config():
    return {
        "num_floors": 1,
        "num_zones_per_floor": 4,
        "num_people": 10,
        "num_hazards": 1,
        "max_ticks": 30,
        "sensor_noise_rate": 0.0,
    }


@pytest.fixture()
def env(default_config):
    return CrisisCoreEnv(default_config)


@pytest.fixture()
def env_reset(env):
    env.reset()
    return env


# ---------------------------------------------------------------------------
# __init__ — config loading
# ---------------------------------------------------------------------------

class TestInit:
    def test_config_stored(self, default_config):
        e = CrisisCoreEnv(default_config)
        assert e.config is default_config

    def test_defaults_applied_when_keys_missing(self):
        e = CrisisCoreEnv({})
        assert e.num_floors == 1
        assert e.num_zones_per_floor == 4
        assert e.num_people == 10
        assert e.num_hazards == 1
        assert e.max_ticks == 30
        assert e.sensor_noise_rate == 0.0

    def test_state_is_none_before_reset(self, env):
        assert env.state is None

    def test_config_values_used(self):
        e = CrisisCoreEnv({"num_floors": 2, "num_zones_per_floor": 6, "num_people": 5,
                           "num_hazards": 2, "max_ticks": 50, "sensor_noise_rate": 0.2})
        assert e.num_floors == 2
        assert e.num_zones_per_floor == 6
        assert e.num_people == 5
        assert e.num_hazards == 2
        assert e.max_ticks == 50
        assert e.sensor_noise_rate == 0.2


# ---------------------------------------------------------------------------
# reset — BuildingState construction
# ---------------------------------------------------------------------------

class TestReset:
    def test_returns_agent_observation(self, env):
        obs = env.reset()
        assert isinstance(obs, AgentObservation)

    def test_state_initialised(self, env_reset):
        assert isinstance(env_reset.state, BuildingState)

    def test_tick_zero_after_reset(self, env_reset):
        assert env_reset.state.tick == 0

    def test_correct_zone_count(self, env_reset):
        # 1 floor × 4 zones/floor = 4 zones
        assert len(env_reset.state.zones) == 4

    def test_correct_zone_count_multifloor(self):
        e = CrisisCoreEnv({"num_floors": 2, "num_zones_per_floor": 4})
        e.reset()
        assert len(e.state.zones) == 8

    def test_zone_ids_include_floor_number(self, env_reset):
        for zone_id in env_reset.state.zones:
            assert zone_id.startswith("f0_")

    def test_two_exit_zones_per_floor(self, env_reset):
        exits = [z for z in env_reset.state.zones.values() if z.has_exit]
        assert len(exits) == 2

    def test_exit_zones_have_exit_id(self, env_reset):
        for zone in env_reset.state.zones.values():
            if zone.has_exit:
                assert zone.exit_id is not None

    def test_adjacent_zones_are_connected(self, env_reset):
        for zone in env_reset.state.zones.values():
            for neighbour in zone.connected_zones:
                assert zone.zone_id in env_reset.state.zones[neighbour].connected_zones

    def test_correct_people_count(self, env_reset):
        assert len(env_reset.state.people) == 10

    def test_people_start_not_evacuated(self, env_reset):
        for person in env_reset.state.people.values():
            assert not person.is_evacuated

    def test_people_placed_in_valid_zones(self, env_reset):
        valid = set(env_reset.state.zones.keys())
        for person in env_reset.state.people.values():
            assert person.current_zone in valid

    def test_correct_hazard_count(self, env_reset):
        assert len(env_reset.state.hazards) == 1

    def test_hazards_not_in_exit_zones(self, env_reset):
        exit_zones = {z for z, zone in env_reset.state.zones.items() if zone.has_exit}
        for hazard in env_reset.state.hazards.values():
            for z in hazard.affected_zones:
                assert z not in exit_zones

    def test_sensor_readings_generated(self, env_reset):
        assert len(env_reset.state.sensor_readings) == 4

    def test_sensor_reading_keys_match_zones(self, env_reset):
        assert set(env_reset.state.sensor_readings.keys()) == set(env_reset.state.zones.keys())

    def test_reset_clears_dispatched_services(self, env_reset):
        env_reset._dispatched_services.append((ServiceType.EMS, None))
        env_reset.reset()
        assert env_reset._dispatched_services == []

    def test_reset_clears_pa_messages(self, env_reset):
        env_reset._pa_messages.append("test")
        env_reset.reset()
        assert env_reset._pa_messages == []

    def test_reset_clears_severity(self, env_reset):
        env_reset._current_severity = SeverityLevel.HIGH
        env_reset.reset()
        assert env_reset._current_severity is None

    def test_observation_tick_is_zero(self, env):
        obs = env.reset()
        assert obs.tick == 0

    def test_observation_has_available_exits(self, env):
        obs = env.reset()
        assert len(obs.available_exits) == 2

    def test_observation_zone_occupancy_sums_to_num_people(self, env):
        obs = env.reset()
        assert sum(obs.zone_occupancy.values()) == 10


# ---------------------------------------------------------------------------
# step — action dispatching
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_without_reset_raises(self, env):
        action = AgentAction(action_type=ActionType.BROADCAST_PA, message="test")
        with pytest.raises(RuntimeError):
            env.step(action)

    def test_step_returns_four_tuple(self, env_reset):
        action = AgentAction(action_type=ActionType.BROADCAST_PA, message="evacuate")
        result = env_reset.step(action)
        assert len(result) == 4

    def test_step_returns_correct_types(self, env_reset):
        action = AgentAction(action_type=ActionType.BROADCAST_PA, message="evacuate")
        obs, reward, done, info = env_reset.step(action)
        assert isinstance(obs, AgentObservation)
        assert isinstance(reward, RewardBreakdown)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_tick_increments(self, env_reset):
        action = AgentAction(action_type=ActionType.BROADCAST_PA, message="evacuate")
        env_reset.step(action)
        assert env_reset.state.tick == 1

    def test_broadcast_pa_records_message(self, env_reset):
        action = AgentAction(action_type=ActionType.BROADCAST_PA, message="go now")
        env_reset.step(action)
        assert "go now" in env_reset._pa_messages

    def test_broadcast_pa_missing_message_format_penalty(self, env_reset):
        action = AgentAction(action_type=ActionType.BROADCAST_PA, message=None)
        _, reward, _, _ = env_reset.step(action)
        assert reward.format_compliance == 0.0

    def test_update_severity_sets_severity(self, env_reset):
        action = AgentAction(action_type=ActionType.UPDATE_SEVERITY, severity=SeverityLevel.HIGH)
        env_reset.step(action)
        assert env_reset._current_severity == SeverityLevel.HIGH

    def test_update_severity_observation_reflects_change(self, env_reset):
        action = AgentAction(action_type=ActionType.UPDATE_SEVERITY, severity=SeverityLevel.CRITICAL)
        obs, _, _, _ = env_reset.step(action)
        assert obs.current_severity == SeverityLevel.CRITICAL

    def test_update_severity_missing_severity_format_penalty(self, env_reset):
        action = AgentAction(action_type=ActionType.UPDATE_SEVERITY, severity=None)
        _, reward, _, _ = env_reset.step(action)
        assert reward.format_compliance == 0.0

    def test_dispatch_service_records_service(self, env_reset):
        action = AgentAction(
            action_type=ActionType.DISPATCH_SERVICE,
            service_type=ServiceType.FIRE_BRIGADE,
        )
        env_reset.step(action)
        services = [s for s, _ in env_reset._dispatched_services]
        assert ServiceType.FIRE_BRIGADE in services

    def test_dispatch_service_missing_service_format_penalty(self, env_reset):
        action = AgentAction(action_type=ActionType.DISPATCH_SERVICE, service_type=None)
        _, reward, _, _ = env_reset.step(action)
        assert reward.format_compliance == 0.0

    def test_valid_action_gets_format_compliance(self, env_reset):
        action = AgentAction(action_type=ActionType.BROADCAST_PA, message="ok")
        _, reward, _, _ = env_reset.step(action)
        assert reward.format_compliance == 0.1

    def test_reward_total_is_sum_of_components(self, env_reset):
        action = AgentAction(action_type=ActionType.BROADCAST_PA, message="evacuate")
        _, reward, _, _ = env_reset.step(action)
        expected = round(
            reward.evacuation_speed
            + reward.route_safety
            + reward.dispatch_accuracy
            + reward.severity_accuracy
            + reward.format_compliance
            + reward.timeout_penalty,
            3,
        )
        assert reward.total == pytest.approx(expected, abs=1e-6)

    def test_done_false_before_max_ticks(self, env_reset):
        action = AgentAction(action_type=ActionType.BROADCAST_PA, message="x")
        _, _, done, _ = env_reset.step(action)
        assert not done

    def test_done_true_at_max_ticks(self):
        e = CrisisCoreEnv({"max_ticks": 1, "num_people": 0, "num_hazards": 0})
        e.reset()
        action = AgentAction(action_type=ActionType.BROADCAST_PA, message="x")
        _, _, done, _ = e.step(action)
        assert done

    def test_info_keys_present(self, env_reset):
        action = AgentAction(action_type=ActionType.BROADCAST_PA, message="x")
        _, _, _, info = env_reset.step(action)
        for key in ("total_people", "evacuated_count", "remaining_count",
                    "active_hazards", "current_tick", "hazard_zones"):
            assert key in info


# ---------------------------------------------------------------------------
# ROUTE_ZONE action
# ---------------------------------------------------------------------------

class TestRouteZone:
    def _get_exit_and_nonexits(self, env: CrisisCoreEnv):
        exit_zones = [z for z, zone in env.state.zones.items() if zone.has_exit]
        non_exit_zones = [z for z, zone in env.state.zones.items() if not zone.has_exit]
        return exit_zones, non_exit_zones

    def test_route_zone_missing_zone_id_format_penalty(self, env_reset):
        action = AgentAction(
            action_type=ActionType.ROUTE_ZONE,
            zone_id=None,
            route_to_exit="f0_r0_c0",
        )
        _, reward, _, _ = env_reset.step(action)
        assert reward.format_compliance == 0.0

    def test_route_zone_missing_exit_format_penalty(self, env_reset):
        action = AgentAction(
            action_type=ActionType.ROUTE_ZONE,
            zone_id="f0_r0_c0",
            route_to_exit=None,
        )
        _, reward, _, _ = env_reset.step(action)
        assert reward.format_compliance == 0.0

    def test_route_zone_invalid_zone_ids_are_no_ops(self, env_reset):
        count_before = sum(1 for p in env_reset.state.people.values() if p.is_evacuated)
        action = AgentAction(
            action_type=ActionType.ROUTE_ZONE,
            zone_id="nonexistent",
            route_to_exit="also_nonexistent",
        )
        env_reset.step(action)
        count_after = sum(1 for p in env_reset.state.people.values() if p.is_evacuated)
        assert count_before == count_after

    def test_people_move_toward_exit(self):
        e = CrisisCoreEnv({"num_zones_per_floor": 4, "num_people": 5,
                            "num_hazards": 0, "sensor_noise_rate": 0.0})
        e.reset()
        # Place all people in a non-exit zone
        exit_zones = [z for z, zone in e.state.zones.items() if zone.has_exit]
        non_exit = [z for z in e.state.zones if z not in exit_zones]
        if not non_exit:
            pytest.skip("No non-exit zones for this config")
        source = non_exit[0]
        target = exit_zones[0]
        for person in e.state.people.values():
            person.current_zone = source
            person.is_evacuated = False

        action = AgentAction(
            action_type=ActionType.ROUTE_ZONE,
            zone_id=source,
            route_to_exit=target,
        )
        e.step(action)
        locations = {p.current_zone for p in e.state.people.values() if not p.is_evacuated}
        assert source not in locations or all(p.is_evacuated for p in e.state.people.values())

    def test_people_at_exit_are_evacuated(self):
        e = CrisisCoreEnv({"num_zones_per_floor": 4, "num_people": 3,
                            "num_hazards": 0, "sensor_noise_rate": 0.0})
        e.reset()
        exit_zones = [z for z, zone in e.state.zones.items() if zone.has_exit]
        target = exit_zones[0]
        # Find a zone adjacent to the exit
        adj = e.state.zones[target].connected_zones
        if not adj:
            pytest.skip("Exit has no adjacent zones")
        source = adj[0]
        for person in e.state.people.values():
            person.current_zone = source
            person.is_evacuated = False

        action = AgentAction(
            action_type=ActionType.ROUTE_ZONE,
            zone_id=source,
            route_to_exit=target,
        )
        e.step(action)
        assert all(p.is_evacuated for p in e.state.people.values())

    def test_done_when_all_evacuated(self):
        e = CrisisCoreEnv({"num_zones_per_floor": 4, "num_people": 3,
                            "num_hazards": 0, "sensor_noise_rate": 0.0, "max_ticks": 30})
        e.reset()
        exit_zones = [z for z, zone in e.state.zones.items() if zone.has_exit]
        target = exit_zones[0]
        adj = e.state.zones[target].connected_zones
        if not adj:
            pytest.skip("Exit has no adjacent zones")
        for person in e.state.people.values():
            person.current_zone = adj[0]
            person.is_evacuated = False

        action = AgentAction(
            action_type=ActionType.ROUTE_ZONE,
            zone_id=adj[0],
            route_to_exit=target,
        )
        _, _, done, _ = e.step(action)
        assert done


# ---------------------------------------------------------------------------
# _shortest_path
# ---------------------------------------------------------------------------

class TestShortestPath:
    def test_same_zone_returns_single_element(self, env_reset):
        zone = next(iter(env_reset.state.zones))
        path = env_reset._shortest_path(zone, zone)
        assert path == [zone]

    def test_adjacent_zones_path_length_two(self, env_reset):
        zone_a = "f0_r0_c0"
        neighbours = env_reset.state.zones[zone_a].connected_zones
        if not neighbours:
            pytest.skip("no neighbours")
        zone_b = neighbours[0]
        path = env_reset._shortest_path(zone_a, zone_b)
        assert len(path) == 2
        assert path[0] == zone_a
        assert path[-1] == zone_b

    def test_path_starts_at_from_zone(self, env_reset):
        zones = list(env_reset.state.zones.keys())
        path = env_reset._shortest_path(zones[0], zones[-1])
        if path:
            assert path[0] == zones[0]

    def test_path_ends_at_target(self, env_reset):
        zones = list(env_reset.state.zones.keys())
        path = env_reset._shortest_path(zones[0], zones[-1])
        if path:
            assert path[-1] == zones[-1]

    def test_invalid_from_zone_returns_empty(self, env_reset):
        zones = list(env_reset.state.zones.keys())
        assert env_reset._shortest_path("invalid", zones[0]) == []

    def test_invalid_to_zone_returns_empty(self, env_reset):
        zones = list(env_reset.state.zones.keys())
        assert env_reset._shortest_path(zones[0], "invalid") == []

    def test_path_is_contiguous(self, env_reset):
        zones = list(env_reset.state.zones.keys())
        path = env_reset._shortest_path(zones[0], zones[-1])
        for i in range(len(path) - 1):
            assert path[i + 1] in env_reset.state.zones[path[i]].connected_zones


# ---------------------------------------------------------------------------
# _check_termination
# ---------------------------------------------------------------------------

class TestCheckTermination:
    def test_not_done_at_start(self, env_reset):
        assert not env_reset._check_termination()

    def test_done_when_max_ticks_reached(self, env_reset):
        env_reset.state.tick = env_reset.max_ticks
        assert env_reset._check_termination()

    def test_done_when_all_evacuated(self, env_reset):
        for person in env_reset.state.people.values():
            person.is_evacuated = True
        assert env_reset._check_termination()

    def test_not_done_when_some_remain(self, env_reset):
        people = list(env_reset.state.people.values())
        for p in people[:-1]:
            p.is_evacuated = True
        assert not env_reset._check_termination()


# ---------------------------------------------------------------------------
# get_info
# ---------------------------------------------------------------------------

class TestGetInfo:
    def test_info_total_people(self, env_reset):
        assert env_reset.get_info()["total_people"] == 10

    def test_info_evacuated_count_starts_zero(self, env_reset):
        assert env_reset.get_info()["evacuated_count"] == 0

    def test_info_remaining_equals_total_minus_evacuated(self, env_reset):
        info = env_reset.get_info()
        assert info["remaining_count"] == info["total_people"] - info["evacuated_count"]

    def test_info_active_hazards(self, env_reset):
        assert env_reset.get_info()["active_hazards"] == 1

    def test_info_current_tick(self, env_reset):
        assert env_reset.get_info()["current_tick"] == 0

    def test_info_hazard_zones_is_list(self, env_reset):
        assert isinstance(env_reset.get_info()["hazard_zones"], list)

    def test_info_evacuated_count_updates(self, env_reset):
        for person in list(env_reset.state.people.values())[:3]:
            person.is_evacuated = True
        assert env_reset.get_info()["evacuated_count"] == 3


# ---------------------------------------------------------------------------
# _get_observation — partial observability
# ---------------------------------------------------------------------------

class TestGetObservation:
    def test_occupancy_sums_to_unevacuated_count(self, env_reset):
        obs = env_reset._get_observation()
        total = sum(obs.zone_occupancy.values())
        unevacuated = sum(1 for p in env_reset.state.people.values() if not p.is_evacuated)
        assert total == unevacuated

    def test_available_exits_not_in_blocked(self, env_reset):
        exit_zone = next(z for z, zone in env_reset.state.zones.items() if zone.has_exit)
        env_reset.state.blocked_exits.append(exit_zone)
        obs = env_reset._get_observation()
        assert exit_zone not in obs.available_exits

    def test_known_hazard_zones_subset_of_all_hazard_zones(self, env_reset):
        all_hazard = {z for h in env_reset.state.hazards.values() for z in h.affected_zones}
        obs = env_reset._get_observation()
        for z in obs.known_hazard_zones:
            assert z in all_hazard

    def test_sos_signals_only_unevacuated(self, env_reset):
        people = list(env_reset.state.people.values())
        people[0].has_sos = True
        people[1].has_sos = True
        people[1].is_evacuated = True
        obs = env_reset._get_observation()
        assert people[0].person_id in obs.sos_signals
        assert people[1].person_id not in obs.sos_signals

    def test_current_severity_none_by_default(self, env_reset):
        obs = env_reset._get_observation()
        assert obs.current_severity is None

    def test_sensor_readings_present_for_all_zones(self, env_reset):
        obs = env_reset._get_observation()
        assert set(obs.sensor_readings.keys()) == set(env_reset.state.zones.keys())


# ---------------------------------------------------------------------------
# Hazard spreading
# ---------------------------------------------------------------------------

class TestHazardSpreading:
    def test_hazard_does_not_shrink(self, env_reset):
        initial = len(env_reset.state.hazards["hazard_0"].affected_zones)
        env_reset._spread_hazards()
        assert len(env_reset.state.hazards["hazard_0"].affected_zones) >= initial

    def test_spread_at_rate_1_always_expands_if_space(self):
        e = CrisisCoreEnv({"num_zones_per_floor": 4, "num_hazards": 1,
                            "num_people": 0, "sensor_noise_rate": 0.0})
        e.reset()
        # Force spread_rate to 1.0
        hazard = next(iter(e.state.hazards.values()))
        hazard.spread_rate = 1.0
        zone = hazard.affected_zones[0]
        # Ensure the hazard zone has neighbours not yet affected
        neighbours = e.state.zones[zone].connected_zones
        non_affected = [n for n in neighbours if n not in hazard.affected_zones]
        if not non_affected:
            pytest.skip("No room to spread")
        before = len(hazard.affected_zones)
        e._spread_hazards()
        assert len(hazard.affected_zones) == before + 1

    def test_spread_at_rate_0_never_expands(self):
        e = CrisisCoreEnv({"num_zones_per_floor": 4, "num_hazards": 1,
                            "num_people": 0, "sensor_noise_rate": 0.0})
        e.reset()
        hazard = next(iter(e.state.hazards.values()))
        hazard.spread_rate = 0.0
        before = list(hazard.affected_zones)
        e._spread_hazards()
        assert hazard.affected_zones == before

    def test_hazard_spreads_max_one_zone_per_tick(self):
        e = CrisisCoreEnv({"num_zones_per_floor": 9, "num_hazards": 1,
                            "num_people": 0, "sensor_noise_rate": 0.0})
        e.reset()
        hazard = next(iter(e.state.hazards.values()))
        hazard.spread_rate = 1.0
        before = len(hazard.affected_zones)
        e._spread_hazards()
        assert len(hazard.affected_zones) <= before + 1


# ---------------------------------------------------------------------------
# Sensor noise
# ---------------------------------------------------------------------------

class TestSensorNoise:
    def test_no_noise_sensors_not_noisy(self):
        e = CrisisCoreEnv({"num_zones_per_floor": 4, "num_hazards": 0,
                            "num_people": 0, "sensor_noise_rate": 0.0})
        e.reset()
        for reading in e.state.sensor_readings.values():
            assert not reading.is_noisy

    def test_all_noise_sensors_all_noisy(self):
        e = CrisisCoreEnv({"num_zones_per_floor": 4, "num_hazards": 0,
                            "num_people": 0, "sensor_noise_rate": 1.0})
        e.reset()
        for reading in e.state.sensor_readings.values():
            assert reading.is_noisy

    def test_sensor_readings_cover_all_zones(self):
        e = CrisisCoreEnv({"num_zones_per_floor": 4, "num_hazards": 1,
                            "num_people": 5, "sensor_noise_rate": 0.5})
        e.reset()
        assert set(e.state.sensor_readings.keys()) == set(e.state.zones.keys())


# ---------------------------------------------------------------------------
# Reward scoring
# ---------------------------------------------------------------------------

class TestRewardScoring:
    def test_dispatch_fire_brigade_for_fire_scores_positive(self):
        e = CrisisCoreEnv({"num_hazards": 1, "num_people": 0, "sensor_noise_rate": 0.0})
        e.reset()
        hazard = next(iter(e.state.hazards.values()))
        hazard.hazard_type = HazardType.FIRE
        score = e._score_dispatch(ServiceType.FIRE_BRIGADE)
        assert score == 1.0

    def test_dispatch_wrong_service_scores_zero(self):
        e = CrisisCoreEnv({"num_hazards": 1, "num_people": 0, "sensor_noise_rate": 0.0})
        e.reset()
        hazard = next(iter(e.state.hazards.values()))
        hazard.hazard_type = HazardType.STRUCTURAL
        score = e._score_dispatch(ServiceType.FIRE_BRIGADE)
        assert score == 0.0

    def test_severity_correct_bracket_scores_positive(self):
        e = CrisisCoreEnv({"num_hazards": 1, "num_people": 0, "sensor_noise_rate": 0.0})
        e.reset()
        hazard = next(iter(e.state.hazards.values()))
        hazard.intensity = 0.9
        score = e._score_severity(SeverityLevel.CRITICAL)
        assert score == 0.5

    def test_severity_wrong_bracket_scores_zero(self):
        e = CrisisCoreEnv({"num_hazards": 1, "num_people": 0, "sensor_noise_rate": 0.0})
        e.reset()
        hazard = next(iter(e.state.hazards.values()))
        hazard.intensity = 0.9
        score = e._score_severity(SeverityLevel.LOW)
        assert score == 0.0

    def test_timeout_penalty_applied_when_people_remain(self):
        e = CrisisCoreEnv({"max_ticks": 1, "num_people": 5, "num_hazards": 0,
                            "sensor_noise_rate": 0.0})
        e.reset()
        action = AgentAction(action_type=ActionType.BROADCAST_PA, message="x")
        _, reward, done, _ = e.step(action)
        assert done
        remaining = sum(1 for p in e.state.people.values() if not p.is_evacuated)
        if remaining > 0:
            assert reward.timeout_penalty < 0.0

    def test_no_timeout_penalty_when_all_evacuated(self):
        e = CrisisCoreEnv({"max_ticks": 30, "num_people": 0, "num_hazards": 0,
                            "sensor_noise_rate": 0.0})
        e.reset()
        action = AgentAction(action_type=ActionType.BROADCAST_PA, message="x")
        _, reward, _, _ = e.step(action)
        assert reward.timeout_penalty == 0.0


# ---------------------------------------------------------------------------
# End-to-end smoke test: 5 episodes, zero crashes
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_five_episodes_no_crash(self):
        import random as _r
        config = {"num_floors": 1, "num_zones_per_floor": 4, "num_people": 10,
                  "num_hazards": 1, "max_ticks": 30, "sensor_noise_rate": 0.1}
        env = CrisisCoreEnv(config)

        for _ in range(5):
            obs = env.reset()
            done = False
            steps = 0
            while not done:
                atype = _r.choice(list(ActionType))
                if atype == ActionType.ROUTE_ZONE:
                    occupied = [z for z, n in obs.zone_occupancy.items() if n > 0]
                    if occupied and obs.available_exits:
                        action = AgentAction(
                            action_type=ActionType.ROUTE_ZONE,
                            zone_id=_r.choice(occupied),
                            route_to_exit=_r.choice(obs.available_exits),
                        )
                    else:
                        action = AgentAction(
                            action_type=ActionType.BROADCAST_PA, message="evacuate"
                        )
                elif atype == ActionType.DISPATCH_SERVICE:
                    action = AgentAction(
                        action_type=ActionType.DISPATCH_SERVICE,
                        service_type=_r.choice(list(ServiceType)),
                    )
                elif atype == ActionType.BROADCAST_PA:
                    action = AgentAction(
                        action_type=ActionType.BROADCAST_PA, message="go"
                    )
                else:
                    action = AgentAction(
                        action_type=ActionType.UPDATE_SEVERITY,
                        severity=_r.choice(list(SeverityLevel)),
                    )
                obs, _, done, _ = env.step(action)
                steps += 1
                assert steps <= config["max_ticks"] + 1
