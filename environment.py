"""OpenEnv-compliant RL environment for multi-agent crisis response simulation."""
from __future__ import annotations

import math
import random
from collections import deque
from datetime import datetime, timezone
from typing import Optional

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


def _log_error(file: str, fn: str, reason: str, hint: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[ERROR] [{ts}] [{file}:{fn}] — {reason} — {hint}")


class CrisisCoreEnv:
    """Single-building crisis response environment."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.num_floors: int = config.get("num_floors", 1)
        self.num_zones_per_floor: int = config.get("num_zones_per_floor", 4)
        self.num_people: int = config.get("num_people", 10)
        self.num_hazards: int = config.get("num_hazards", 1)
        self.max_ticks: int = config.get("max_ticks", 30)
        self.sensor_noise_rate: float = config.get("sensor_noise_rate", 0.0)

        self.state: Optional[BuildingState] = None
        self._dispatched_services: list[tuple[ServiceType, Optional[str]]] = []
        self._pa_messages: list[str] = []
        self._current_severity: Optional[SeverityLevel] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> AgentObservation:
        self._dispatched_services = []
        self._pa_messages = []
        self._current_severity = None

        zones: dict[str, Zone] = {}
        cols = math.ceil(math.sqrt(self.num_zones_per_floor))
        rows = math.ceil(self.num_zones_per_floor / cols)

        for floor in range(self.num_floors):
            floor_zone_ids: list[str] = []
            grid: dict[tuple[int, int], str] = {}
            zone_index = 0

            for r in range(rows):
                for c in range(cols):
                    if zone_index >= self.num_zones_per_floor:
                        break
                    zone_id = f"f{floor}_r{r}_c{c}"
                    grid[(r, c)] = zone_id
                    floor_zone_ids.append(zone_id)
                    zones[zone_id] = Zone(
                        zone_id=zone_id,
                        floor=floor,
                        capacity=20,
                        connected_zones=[],
                        has_exit=False,
                        exit_id=None,
                    )
                    zone_index += 1

            # Wire up adjacency
            for (r, c), zone_id in grid.items():
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbour = grid.get((r + dr, c + dc))
                    if neighbour:
                        zones[zone_id].connected_zones.append(neighbour)

            # Two exits per floor: first and last zone in the grid
            for i, ez_id in enumerate([floor_zone_ids[0], floor_zone_ids[-1]]):
                zones[ez_id].has_exit = True
                zones[ez_id].exit_id = f"exit_f{floor}_{i + 1}"

        # Place people
        zone_ids = list(zones.keys())
        people: dict[str, Person] = {}
        for i in range(self.num_people):
            people[f"person_{i}"] = Person(
                person_id=f"person_{i}",
                current_zone=random.choice(zone_ids),
                is_evacuated=False,
                has_sos=False,
            )

        # Place hazards in non-exit zones
        non_exit = [z for z in zone_ids if not zones[z].has_exit] or zone_ids
        hazards: dict[str, Hazard] = {}
        for i in range(self.num_hazards):
            hazards[f"hazard_{i}"] = Hazard(
                hazard_id=f"hazard_{i}",
                hazard_type=random.choice(list(HazardType)),
                affected_zones=[random.choice(non_exit)],
                spread_rate=0.3,
                intensity=round(random.uniform(0.4, 1.0), 3),
            )

        self.state = BuildingState(
            building_id="building_0",
            floors=self.num_floors,
            zones=zones,
            people=people,
            hazards=hazards,
            sensor_readings={},
            blocked_exits=[],
            tick=0,
            max_ticks=self.max_ticks,
            episode_done=False,
        )

        self._refresh_sensors()
        return self._get_observation()

    def step(
        self, action: AgentAction
    ) -> tuple[AgentObservation, RewardBreakdown, bool, dict]:
        if self.state is None:
            _log_error(
                "environment.py", "step",
                "state is None",
                "call reset() before step()",
            )
            raise RuntimeError("Call reset() before step().")

        evacuated_before = sum(1 for p in self.state.people.values() if p.is_evacuated)
        hazard_zones_snapshot = {
            z for h in self.state.hazards.values() for z in h.affected_zones
        }

        format_ok = True
        route_safety = 0.0
        dispatch_score = 0.0
        severity_score = 0.0

        if action.action_type == ActionType.ROUTE_ZONE:
            if action.zone_id is None or action.route_to_exit is None:
                format_ok = False
            else:
                route_safety = self._apply_route_zone(
                    action.zone_id, action.route_to_exit, hazard_zones_snapshot
                )

        elif action.action_type == ActionType.DISPATCH_SERVICE:
            if action.service_type is None:
                format_ok = False
            else:
                self._dispatched_services.append((action.service_type, action.zone_id))
                dispatch_score = self._score_dispatch(action.service_type)

        elif action.action_type == ActionType.BROADCAST_PA:
            if action.message is None:
                format_ok = False
            else:
                self._pa_messages.append(action.message)

        elif action.action_type == ActionType.UPDATE_SEVERITY:
            if action.severity is None:
                format_ok = False
            else:
                self._current_severity = action.severity
                severity_score = self._score_severity(action.severity)

        self._spread_hazards()
        self._refresh_sensors()
        self.state.tick += 1

        done = self._check_termination()
        self.state.episode_done = done

        evacuated_after = sum(1 for p in self.state.people.values() if p.is_evacuated)
        newly_evacuated = evacuated_after - evacuated_before

        speed_reward = round(
            newly_evacuated * (1.0 - self.state.tick / self.max_ticks), 3
        )
        timeout_penalty = 0.0
        if done and evacuated_after < self.num_people:
            timeout_penalty = round(-(self.num_people - evacuated_after) * 1.0, 3)

        reward = RewardBreakdown(
            evacuation_speed=speed_reward,
            route_safety=round(route_safety, 3),
            dispatch_accuracy=round(dispatch_score, 3),
            severity_accuracy=round(severity_score, 3),
            format_compliance=0.1 if format_ok else 0.0,
            timeout_penalty=timeout_penalty,
            total=round(
                speed_reward
                + route_safety
                + dispatch_score
                + severity_score
                + (0.1 if format_ok else 0.0)
                + timeout_penalty,
                3,
            ),
        )

        return self._get_observation(), reward, done, self.get_info()

    def get_info(self) -> dict:
        evacuated = sum(1 for p in self.state.people.values() if p.is_evacuated)
        hazard_zones = list(
            {z for h in self.state.hazards.values() for z in h.affected_zones}
        )
        return {
            "total_people": self.num_people,
            "evacuated_count": evacuated,
            "remaining_count": self.num_people - evacuated,
            "active_hazards": len(self.state.hazards),
            "current_tick": self.state.tick,
            "hazard_zones": hazard_zones,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> AgentObservation:
        occupancy: dict[str, int] = {z: 0 for z in self.state.zones}
        for person in self.state.people.values():
            if not person.is_evacuated:
                occupancy[person.current_zone] = occupancy.get(person.current_zone, 0) + 1

        all_hazard_zones = {z for h in self.state.hazards.values() for z in h.affected_zones}
        known_hazard_zones = [
            z for z in all_hazard_zones
            if z in self.state.sensor_readings
            and (
                self.state.sensor_readings[z].smoke_level > 0.3
                or self.state.sensor_readings[z].sound_level > 0.5
            )
        ]

        available_exits = [
            z for z, zone in self.state.zones.items()
            if zone.has_exit and z not in self.state.blocked_exits
        ]

        sos_signals = [
            p.person_id for p in self.state.people.values()
            if p.has_sos and not p.is_evacuated
        ]

        return AgentObservation(
            tick=self.state.tick,
            sensor_readings=dict(self.state.sensor_readings),
            zone_occupancy=occupancy,
            known_hazard_zones=known_hazard_zones,
            available_exits=available_exits,
            sos_signals=sos_signals,
            current_severity=self._current_severity,
        )

    def _check_termination(self) -> bool:
        all_out = all(p.is_evacuated for p in self.state.people.values())
        return all_out or self.state.tick >= self.max_ticks

    def _shortest_path(self, from_zone: str, to_exit: str) -> list[str]:
        if from_zone not in self.state.zones or to_exit not in self.state.zones:
            return []
        if from_zone == to_exit:
            return [from_zone]

        queue: deque[list[str]] = deque([[from_zone]])
        visited = {from_zone}

        while queue:
            path = queue.popleft()
            for neighbour in self.state.zones[path[-1]].connected_zones:
                if neighbour == to_exit:
                    return path + [neighbour]
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(path + [neighbour])
        return []

    def _apply_route_zone(
        self, zone_id: str, target_exit: str, hazard_zones: set[str]
    ) -> float:
        if zone_id not in self.state.zones or target_exit not in self.state.zones:
            return 0.0

        movers = [
            p for p in self.state.people.values()
            if p.current_zone == zone_id and not p.is_evacuated
        ]
        penalty = 0.0

        for person in movers:
            path = self._shortest_path(person.current_zone, target_exit)
            if len(path) >= 2:
                next_zone = path[1]
                person.current_zone = next_zone
                if next_zone in hazard_zones:
                    penalty -= 0.5
                if self.state.zones[next_zone].has_exit:
                    person.is_evacuated = True
            elif len(path) == 1 and self.state.zones[path[0]].has_exit:
                person.is_evacuated = True

        return penalty

    def _spread_hazards(self) -> None:
        for hazard in self.state.hazards.values():
            candidates = [
                nb
                for z in hazard.affected_zones
                if z in self.state.zones
                for nb in self.state.zones[z].connected_zones
                if nb not in hazard.affected_zones
            ]
            if candidates and random.random() < hazard.spread_rate:
                hazard.affected_zones.append(random.choice(candidates))

    def _refresh_sensors(self) -> None:
        hazard_map: dict[str, tuple[HazardType, float]] = {}
        for hazard in self.state.hazards.values():
            for z in hazard.affected_zones:
                if z not in hazard_map or hazard.intensity > hazard_map[z][1]:
                    hazard_map[z] = (hazard.hazard_type, hazard.intensity)

        occupancy: dict[str, int] = {}
        for person in self.state.people.values():
            if not person.is_evacuated:
                occupancy[person.current_zone] = occupancy.get(person.current_zone, 0) + 1

        readings: dict[str, SensorReading] = {}
        for zone_id in self.state.zones:
            is_noisy = random.random() < self.sensor_noise_rate

            if not is_noisy:
                if zone_id in hazard_map:
                    h_type, intensity = hazard_map[zone_id]
                    smoke = intensity if h_type in (HazardType.FIRE, HazardType.SMOKE) else 0.1
                    sound = round(random.uniform(0.3, 1.0), 3)
                else:
                    smoke = 0.0
                    sound = round(random.uniform(0.0, 0.15), 3)
                motion = occupancy.get(zone_id, 0) > 0
            else:
                # Noisy sensor: randomised values to simulate unreliable reading
                smoke = round(random.random(), 3)
                sound = round(random.random(), 3)
                motion = random.choice([True, False])

            readings[zone_id] = SensorReading(
                zone_id=zone_id,
                smoke_level=round(smoke, 3),
                motion_detected=motion,
                sound_level=sound,
                is_noisy=is_noisy,
            )

        self.state.sensor_readings = readings

    def _score_dispatch(self, service: ServiceType) -> float:
        hazard_types = {h.hazard_type for h in self.state.hazards.values()}
        ideal = {
            HazardType.FIRE: ServiceType.FIRE_BRIGADE,
            HazardType.SMOKE: ServiceType.EMS,
            HazardType.STRUCTURAL: ServiceType.POLICE,
        }
        return 1.0 if any(ideal.get(ht) == service for ht in hazard_types) else 0.0

    def _score_severity(self, severity: SeverityLevel) -> float:
        if not self.state.hazards:
            return 0.0
        max_intensity = max(h.intensity for h in self.state.hazards.values())
        brackets = {
            SeverityLevel.LOW: (0.0, 0.3),
            SeverityLevel.MEDIUM: (0.3, 0.6),
            SeverityLevel.HIGH: (0.6, 0.8),
            SeverityLevel.CRITICAL: (0.8, 1.01),
        }
        lo, hi = brackets[severity]
        return 0.5 if lo <= max_intensity < hi else 0.0


# ---------------------------------------------------------------------------
# Quick smoke-test: 5 random episodes
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = {
        "num_floors": 1,
        "num_zones_per_floor": 4,
        "num_people": 10,
        "num_hazards": 1,
        "max_ticks": 30,
        "sensor_noise_rate": 0.1,
    }

    env = CrisisCoreEnv(config)

    for ep in range(5):
        obs = env.reset()
        done = False
        totals = {
            "evacuation_speed": 0.0,
            "route_safety": 0.0,
            "dispatch_accuracy": 0.0,
            "severity_accuracy": 0.0,
            "format_compliance": 0.0,
            "timeout_penalty": 0.0,
            "total": 0.0,
        }

        while not done:
            atype = random.choice(list(ActionType))

            if atype == ActionType.ROUTE_ZONE:
                occupied = [z for z, n in obs.zone_occupancy.items() if n > 0]
                if occupied and obs.available_exits:
                    action = AgentAction(
                        action_type=ActionType.ROUTE_ZONE,
                        zone_id=random.choice(occupied),
                        route_to_exit=random.choice(obs.available_exits),
                    )
                else:
                    action = AgentAction(
                        action_type=ActionType.BROADCAST_PA, message="Please evacuate"
                    )
            elif atype == ActionType.DISPATCH_SERVICE:
                action = AgentAction(
                    action_type=ActionType.DISPATCH_SERVICE,
                    service_type=random.choice(list(ServiceType)),
                )
            elif atype == ActionType.BROADCAST_PA:
                action = AgentAction(
                    action_type=ActionType.BROADCAST_PA, message="Evacuate now"
                )
            else:
                action = AgentAction(
                    action_type=ActionType.UPDATE_SEVERITY,
                    severity=random.choice(list(SeverityLevel)),
                )

            obs, reward, done, info = env.step(action)

            for k in totals:
                totals[k] += getattr(reward, k)

        print(f"\nEpisode {ep + 1}:")
        for k, v in totals.items():
            print(f"  {k:<20} {v:+.3f}")
        print(f"  evacuated            {info['evacuated_count']}/{info['total_people']}")

    print("\n[OK] 5 episodes complete — no crashes.")
