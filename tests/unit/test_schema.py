"""Unit tests for schema.py — Phase 1 data structures."""
from __future__ import annotations

import pytest
from dataclasses import fields, asdict
from schema import (
    ActionType,
    ServiceType,
    HazardType,
    SeverityLevel,
    Zone,
    Person,
    Hazard,
    SensorReading,
    BuildingState,
    AgentAction,
    AgentObservation,
    RewardBreakdown,
)


# ---------------------------------------------------------------------------
# Enum membership tests
# ---------------------------------------------------------------------------

class TestActionType:
    def test_all_values_present(self):
        names = {e.name for e in ActionType}
        assert names == {"ROUTE_ZONE", "DISPATCH_SERVICE", "BROADCAST_PA", "UPDATE_SEVERITY"}

    def test_values_are_strings(self):
        for e in ActionType:
            assert isinstance(e.value, str)

    def test_lookup_by_value(self):
        assert ActionType("route_zone") is ActionType.ROUTE_ZONE

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ActionType("invalid_action")


class TestServiceType:
    def test_all_values_present(self):
        names = {e.name for e in ServiceType}
        assert names == {"FIRE_BRIGADE", "EMS", "POLICE"}

    def test_lookup_by_value(self):
        assert ServiceType("ems") is ServiceType.EMS

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ServiceType("ambulance")


class TestHazardType:
    def test_all_values_present(self):
        names = {e.name for e in HazardType}
        assert names == {"FIRE", "SMOKE", "STRUCTURAL"}

    def test_lookup_by_value(self):
        assert HazardType("structural") is HazardType.STRUCTURAL

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            HazardType("flood")


class TestSeverityLevel:
    def test_all_values_present(self):
        names = {e.name for e in SeverityLevel}
        assert names == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    def test_ordering_by_name(self):
        levels = [e.name for e in SeverityLevel]
        assert "LOW" in levels and "CRITICAL" in levels

    def test_lookup_by_value(self):
        assert SeverityLevel("critical") is SeverityLevel.CRITICAL

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            SeverityLevel("extreme")


# ---------------------------------------------------------------------------
# Zone dataclass
# ---------------------------------------------------------------------------

class TestZone:
    def _make(self, **kwargs):
        defaults = dict(
            zone_id="Z1",
            floor=1,
            capacity=20,
            connected_zones=["Z2", "Z3"],
            has_exit=True,
            exit_id="EXIT_A",
        )
        defaults.update(kwargs)
        return Zone(**defaults)

    def test_basic_construction(self):
        z = self._make()
        assert z.zone_id == "Z1"
        assert z.floor == 1
        assert z.capacity == 20
        assert z.connected_zones == ["Z2", "Z3"]
        assert z.has_exit is True
        assert z.exit_id == "EXIT_A"

    def test_exit_id_optional_defaults_none(self):
        z = Zone(zone_id="Z2", floor=2, capacity=10, connected_zones=[], has_exit=False)
        assert z.exit_id is None

    def test_no_exit_no_exit_id(self):
        z = self._make(has_exit=False, exit_id=None)
        assert z.has_exit is False
        assert z.exit_id is None

    def test_empty_connected_zones(self):
        z = self._make(connected_zones=[])
        assert z.connected_zones == []

    def test_capacity_zero(self):
        z = self._make(capacity=0)
        assert z.capacity == 0

    def test_floor_zero(self):
        z = self._make(floor=0)
        assert z.floor == 0

    def test_zone_id_empty_string(self):
        z = self._make(zone_id="")
        assert z.zone_id == ""

    def test_has_required_fields(self):
        field_names = {f.name for f in fields(Zone)}
        assert field_names == {"zone_id", "floor", "capacity", "connected_zones", "has_exit", "exit_id"}


# ---------------------------------------------------------------------------
# Person dataclass
# ---------------------------------------------------------------------------

class TestPerson:
    def _make(self, **kwargs):
        defaults = dict(person_id="P1", current_zone="Z1", is_evacuated=False, has_sos=False)
        defaults.update(kwargs)
        return Person(**defaults)

    def test_basic_construction(self):
        p = self._make()
        assert p.person_id == "P1"
        assert p.current_zone == "Z1"
        assert p.is_evacuated is False
        assert p.has_sos is False

    def test_evacuated_person(self):
        p = self._make(is_evacuated=True)
        assert p.is_evacuated is True

    def test_sos_active(self):
        p = self._make(has_sos=True)
        assert p.has_sos is True

    def test_has_required_fields(self):
        field_names = {f.name for f in fields(Person)}
        assert field_names == {"person_id", "current_zone", "is_evacuated", "has_sos"}


# ---------------------------------------------------------------------------
# Hazard dataclass
# ---------------------------------------------------------------------------

class TestHazard:
    def _make(self, **kwargs):
        defaults = dict(
            hazard_id="H1",
            hazard_type=HazardType.FIRE,
            affected_zones=["Z1", "Z2"],
            spread_rate=0.3,
            intensity=0.7,
        )
        defaults.update(kwargs)
        return Hazard(**defaults)

    def test_basic_construction(self):
        h = self._make()
        assert h.hazard_id == "H1"
        assert h.hazard_type is HazardType.FIRE
        assert h.affected_zones == ["Z1", "Z2"]
        assert h.spread_rate == 0.3
        assert h.intensity == 0.7

    def test_intensity_zero(self):
        h = self._make(intensity=0.0)
        assert h.intensity == 0.0

    def test_intensity_one(self):
        h = self._make(intensity=1.0)
        assert h.intensity == 1.0

    def test_spread_rate_zero(self):
        h = self._make(spread_rate=0.0)
        assert h.spread_rate == 0.0

    def test_empty_affected_zones(self):
        h = self._make(affected_zones=[])
        assert h.affected_zones == []

    def test_smoke_hazard_type(self):
        h = self._make(hazard_type=HazardType.SMOKE)
        assert h.hazard_type is HazardType.SMOKE

    def test_structural_hazard_type(self):
        h = self._make(hazard_type=HazardType.STRUCTURAL)
        assert h.hazard_type is HazardType.STRUCTURAL

    def test_has_required_fields(self):
        field_names = {f.name for f in fields(Hazard)}
        assert field_names == {"hazard_id", "hazard_type", "affected_zones", "spread_rate", "intensity"}


# ---------------------------------------------------------------------------
# SensorReading dataclass
# ---------------------------------------------------------------------------

class TestSensorReading:
    def _make(self, **kwargs):
        defaults = dict(zone_id="Z1", smoke_level=0.0, motion_detected=False, sound_level=0.0, is_noisy=False)
        defaults.update(kwargs)
        return SensorReading(**defaults)

    def test_basic_construction(self):
        s = self._make()
        assert s.zone_id == "Z1"
        assert s.smoke_level == 0.0
        assert s.motion_detected is False
        assert s.sound_level == 0.0
        assert s.is_noisy is False

    def test_high_smoke_motion(self):
        s = self._make(smoke_level=0.9, motion_detected=True, sound_level=0.8, is_noisy=True)
        assert s.smoke_level == 0.9
        assert s.motion_detected is True
        assert s.is_noisy is True

    def test_has_required_fields(self):
        field_names = {f.name for f in fields(SensorReading)}
        assert field_names == {"zone_id", "smoke_level", "motion_detected", "sound_level", "is_noisy"}


# ---------------------------------------------------------------------------
# BuildingState dataclass
# ---------------------------------------------------------------------------

class TestBuildingState:
    def _make(self, **kwargs):
        zone = Zone(zone_id="Z1", floor=1, capacity=10, connected_zones=[], has_exit=True, exit_id="E1")
        person = Person(person_id="P1", current_zone="Z1", is_evacuated=False, has_sos=False)
        hazard = Hazard(hazard_id="H1", hazard_type=HazardType.FIRE, affected_zones=["Z1"], spread_rate=0.1, intensity=0.5)
        sensor = SensorReading(zone_id="Z1", smoke_level=0.2, motion_detected=True, sound_level=0.1, is_noisy=False)
        defaults = dict(
            building_id="B1",
            floors=5,
            zones={"Z1": zone},
            people={"P1": person},
            hazards={"H1": hazard},
            sensor_readings={"Z1": sensor},
            blocked_exits=[],
            tick=0,
            max_ticks=100,
            episode_done=False,
        )
        defaults.update(kwargs)
        return BuildingState(**defaults)

    def test_basic_construction(self):
        b = self._make()
        assert b.building_id == "B1"
        assert b.floors == 5
        assert b.tick == 0
        assert b.max_ticks == 100
        assert b.episode_done is False

    def test_zones_dict(self):
        b = self._make()
        assert "Z1" in b.zones
        assert isinstance(b.zones["Z1"], Zone)

    def test_people_dict(self):
        b = self._make()
        assert "P1" in b.people
        assert isinstance(b.people["P1"], Person)

    def test_hazards_dict(self):
        b = self._make()
        assert "H1" in b.hazards
        assert isinstance(b.hazards["H1"], Hazard)

    def test_sensor_readings_dict(self):
        b = self._make()
        assert "Z1" in b.sensor_readings
        assert isinstance(b.sensor_readings["Z1"], SensorReading)

    def test_episode_done_flag(self):
        b = self._make(episode_done=True)
        assert b.episode_done is True

    def test_blocked_exits_list(self):
        b = self._make(blocked_exits=["E1", "E2"])
        assert b.blocked_exits == ["E1", "E2"]

    def test_empty_building(self):
        b = self._make(zones={}, people={}, hazards={}, sensor_readings={})
        assert b.zones == {}
        assert b.people == {}
        assert b.hazards == {}

    def test_has_required_fields(self):
        field_names = {f.name for f in fields(BuildingState)}
        assert field_names == {
            "building_id", "floors", "zones", "people", "hazards",
            "sensor_readings", "blocked_exits", "tick", "max_ticks", "episode_done",
        }


# ---------------------------------------------------------------------------
# AgentAction dataclass
# ---------------------------------------------------------------------------

class TestAgentAction:
    def test_route_zone_action(self):
        a = AgentAction(action_type=ActionType.ROUTE_ZONE, zone_id="Z1", route_to_exit="EXIT_A")
        assert a.action_type is ActionType.ROUTE_ZONE
        assert a.zone_id == "Z1"
        assert a.route_to_exit == "EXIT_A"
        assert a.service_type is None
        assert a.message is None
        assert a.severity is None

    def test_dispatch_service_action(self):
        a = AgentAction(action_type=ActionType.DISPATCH_SERVICE, zone_id="Z2", service_type=ServiceType.EMS)
        assert a.service_type is ServiceType.EMS

    def test_broadcast_pa_action(self):
        a = AgentAction(action_type=ActionType.BROADCAST_PA, message="Evacuate floor 3")
        assert a.message == "Evacuate floor 3"

    def test_update_severity_action(self):
        a = AgentAction(action_type=ActionType.UPDATE_SEVERITY, severity=SeverityLevel.CRITICAL)
        assert a.severity is SeverityLevel.CRITICAL

    def test_all_optional_fields_default_none(self):
        a = AgentAction(action_type=ActionType.BROADCAST_PA)
        assert a.zone_id is None
        assert a.route_to_exit is None
        assert a.service_type is None
        assert a.message is None
        assert a.severity is None

    def test_has_required_fields(self):
        field_names = {f.name for f in fields(AgentAction)}
        assert field_names == {
            "action_type", "zone_id", "route_to_exit", "service_type", "message", "severity"
        }


# ---------------------------------------------------------------------------
# AgentObservation dataclass
# ---------------------------------------------------------------------------

class TestAgentObservation:
    def _make(self, **kwargs):
        sensor = SensorReading(zone_id="Z1", smoke_level=0.1, motion_detected=False, sound_level=0.0, is_noisy=False)
        defaults = dict(
            tick=5,
            sensor_readings={"Z1": sensor},
            zone_occupancy={"Z1": 3},
            known_hazard_zones=["Z1"],
            available_exits=["EXIT_A"],
            sos_signals=["Z1"],
            current_severity=SeverityLevel.HIGH,
        )
        defaults.update(kwargs)
        return AgentObservation(**defaults)

    def test_basic_construction(self):
        obs = self._make()
        assert obs.tick == 5
        assert obs.zone_occupancy == {"Z1": 3}
        assert obs.known_hazard_zones == ["Z1"]
        assert obs.available_exits == ["EXIT_A"]
        assert obs.sos_signals == ["Z1"]
        assert obs.current_severity is SeverityLevel.HIGH

    def test_current_severity_optional_defaults_none(self):
        obs = AgentObservation(
            tick=0,
            sensor_readings={},
            zone_occupancy={},
            known_hazard_zones=[],
            available_exits=[],
            sos_signals=[],
        )
        assert obs.current_severity is None

    def test_empty_observations(self):
        obs = self._make(known_hazard_zones=[], sos_signals=[], available_exits=[])
        assert obs.known_hazard_zones == []
        assert obs.sos_signals == []
        assert obs.available_exits == []

    def test_tick_zero(self):
        obs = self._make(tick=0)
        assert obs.tick == 0

    def test_has_required_fields(self):
        field_names = {f.name for f in fields(AgentObservation)}
        assert field_names == {
            "tick", "sensor_readings", "zone_occupancy", "known_hazard_zones",
            "available_exits", "sos_signals", "current_severity",
        }


# ---------------------------------------------------------------------------
# RewardBreakdown dataclass
# ---------------------------------------------------------------------------

class TestRewardBreakdown:
    def _make(self, **kwargs):
        defaults = dict(
            evacuation_speed=0.8,
            route_safety=0.6,
            dispatch_accuracy=0.9,
            severity_accuracy=0.7,
            format_compliance=1.0,
            timeout_penalty=0.0,
            total=4.0,
        )
        defaults.update(kwargs)
        return RewardBreakdown(**defaults)

    def test_basic_construction(self):
        r = self._make()
        assert r.evacuation_speed == 0.8
        assert r.route_safety == 0.6
        assert r.dispatch_accuracy == 0.9
        assert r.severity_accuracy == 0.7
        assert r.format_compliance == 1.0
        assert r.timeout_penalty == 0.0
        assert r.total == 4.0

    def test_all_zeros(self):
        r = self._make(
            evacuation_speed=0.0,
            route_safety=0.0,
            dispatch_accuracy=0.0,
            severity_accuracy=0.0,
            format_compliance=0.0,
            timeout_penalty=0.0,
            total=0.0,
        )
        assert r.total == 0.0

    def test_negative_timeout_penalty(self):
        r = self._make(timeout_penalty=-1.0, total=-1.0)
        assert r.timeout_penalty == -1.0

    def test_has_required_fields(self):
        field_names = {f.name for f in fields(RewardBreakdown)}
        assert field_names == {
            "evacuation_speed", "route_safety", "dispatch_accuracy",
            "severity_accuracy", "format_compliance", "timeout_penalty", "total",
        }
