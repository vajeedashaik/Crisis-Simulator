Create a shared design document as a Python file called `schema.py` that defines all data structures for a multi-agent crisis response RL environment. Do not implement any logic yet — only define the dataclasses and enums.

Define the following:

1. An `ActionType` enum with values: ROUTE_ZONE, DISPATCH_SERVICE, BROADCAST_PA, UPDATE_SEVERITY
2. A `ServiceType` enum with values: FIRE_BRIGADE, EMS, POLICE
3. A `HazardType` enum with values: FIRE, SMOKE, STRUCTURAL
4. A `SeverityLevel` enum with values: LOW, MEDIUM, HIGH, CRITICAL
5. A `Zone` dataclass with fields: zone_id (str), floor (int), capacity (int), connected_zones (list of str), has_exit (bool), exit_id (optional str)
6. A `Person` dataclass with fields: person_id (str), current_zone (str), is_evacuated (bool), has_sos (bool)
7. A `Hazard` dataclass with fields: hazard_id (str), hazard_type (HazardType), affected_zones (list of str), spread_rate (float), intensity (float 0–1)
8. A `SensorReading` dataclass with fields: zone_id (str), smoke_level (float), motion_detected (bool), sound_level (float), is_noisy (bool)
9. A `BuildingState` dataclass with fields: building_id (str), floors (int), zones (dict of zone_id to Zone), people (dict of person_id to Person), hazards (dict of hazard_id to Hazard), sensor_readings (dict of zone_id to SensorReading), blocked_exits (list of str), tick (int), max_ticks (int), episode_done (bool)
10. An `AgentAction` dataclass with fields: action_type (ActionType), zone_id (optional str), route_to_exit (optional str), service_type (optional ServiceType), message (optional str), severity (optional SeverityLevel)
11. An `AgentObservation` dataclass with fields: tick (int), sensor_readings (dict), zone_occupancy (dict of zone_id to int), known_hazard_zones (list of str), available_exits (list of str), sos_signals (list of zone_id), current_severity (optional SeverityLevel)
12. A `RewardBreakdown` dataclass with fields: evacuation_speed (float), route_safety (float), dispatch_accuracy (float), severity_accuracy (float), format_compliance (float), timeout_penalty (float), total (float)

Add a docstring to each dataclass explaining its role in one sentence. Use Python dataclasses with type hints throughout. Add `from __future__ import annotations` at the top.