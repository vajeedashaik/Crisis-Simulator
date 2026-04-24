from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ActionType(Enum):
    ROUTE_ZONE = "route_zone"
    DISPATCH_SERVICE = "dispatch_service"
    BROADCAST_PA = "broadcast_pa"
    UPDATE_SEVERITY = "update_severity"


class ServiceType(Enum):
    FIRE_BRIGADE = "fire_brigade"
    EMS = "ems"
    POLICE = "police"


class HazardType(Enum):
    FIRE = "fire"
    SMOKE = "smoke"
    STRUCTURAL = "structural"


class SeverityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Zone:
    """Represents a physical zone in the building with connectivity and exit info."""

    zone_id: str
    floor: int
    capacity: int
    connected_zones: list[str]
    has_exit: bool
    exit_id: Optional[str] = None


@dataclass
class Person:
    """Tracks an individual's location, evacuation status, and distress signal."""

    person_id: str
    current_zone: str
    is_evacuated: bool
    has_sos: bool


@dataclass
class Hazard:
    """Describes an active hazard, its type, affected zones, and spread characteristics."""

    hazard_id: str
    hazard_type: HazardType
    affected_zones: list[str]
    spread_rate: float
    intensity: float  # 0.0–1.0


@dataclass
class SensorReading:
    """Raw sensor data from a single zone at a given tick."""

    zone_id: str
    smoke_level: float
    motion_detected: bool
    sound_level: float
    is_noisy: bool


@dataclass
class BuildingState:
    """Complete snapshot of the building environment at a single simulation tick."""

    building_id: str
    floors: int
    zones: dict[str, Zone]
    people: dict[str, Person]
    hazards: dict[str, Hazard]
    sensor_readings: dict[str, SensorReading]
    blocked_exits: list[str]
    tick: int
    max_ticks: int
    episode_done: bool


@dataclass
class AgentAction:
    """A structured action emitted by an agent to control the crisis response."""

    action_type: ActionType
    zone_id: Optional[str] = None
    route_to_exit: Optional[str] = None
    service_type: Optional[ServiceType] = None
    message: Optional[str] = None
    severity: Optional[SeverityLevel] = None


@dataclass
class AgentObservation:
    """Partial, agent-visible view of building state used as RL policy input."""

    tick: int
    sensor_readings: dict[str, SensorReading]
    zone_occupancy: dict[str, int]
    known_hazard_zones: list[str]
    available_exits: list[str]
    sos_signals: list[str]
    current_severity: Optional[SeverityLevel] = None


@dataclass
class RewardBreakdown:
    """Decomposed reward signal showing contribution from each evaluation criterion."""

    evacuation_speed: float
    route_safety: float
    dispatch_accuracy: float
    severity_accuracy: float
    format_compliance: float
    timeout_penalty: float
    total: float
