from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Callable

from schema import (
    ActionType, HazardType, ServiceType, SeverityLevel,
    AgentAction, AgentObservation, BuildingState,
)

logger = logging.getLogger(__name__)

_ROLE_DESCRIPTIONS = {
    "evacuation": (
        "You are the Evacuation Agent responsible for routing building occupants "
        "to safety. Given zone occupancy, known hazard zones, and available exits, "
        "decide which zone to route and which exit to use. Minimise exposure to hazards "
        "and maximise throughput through available exits."
    ),
    "dispatch": (
        "You are the Dispatch Agent responsible for coordinating emergency services. "
        "Given active SOS signals, current crisis severity, and hazard types present, "
        "decide which service to dispatch (FIRE_BRIGADE, EMS, or POLICE). Prioritise "
        "life-threatening situations and match service type to the hazard."
    ),
    "comms": (
        "You are the Communications Agent responsible for broadcasting public address "
        "messages to building occupants. Given the current routing decision and zone "
        "occupancy, compose a clear, calm, actionable PA message that directs people "
        "to the correct exits and away from hazards."
    ),
}


def build_agent_prompt(agent_role: str, observation: dict, last_incident_log: list[str]) -> str:
    system_desc = _ROLE_DESCRIPTIONS.get(
        agent_role.lower(),
        f"You are the {agent_role} agent in a crisis management system. "
        "Analyse the observation and return a structured action as JSON.",
    )
    obs_json = json.dumps(observation, indent=2, default=str)
    prompt = (
        f"SYSTEM: {system_desc}\n\n"
        f"USER: Current observation:\n{obs_json}"
    )
    if last_incident_log:
        log_str = "; ".join(last_incident_log)
        prompt += f"\n\nLast episode mistakes: {log_str}"
    return prompt


def _parse_action(raw: str, expected_type: ActionType, default: AgentAction) -> AgentAction:
    try:
        data = json.loads(raw)
        action_type = ActionType(data.get("action_type", expected_type.value))
        service_raw = data.get("service_type")
        severity_raw = data.get("severity")
        return AgentAction(
            action_type=action_type,
            zone_id=data.get("zone_id"),
            route_to_exit=data.get("route_to_exit"),
            service_type=ServiceType(service_raw) if service_raw else None,
            message=data.get("message"),
            severity=SeverityLevel(severity_raw) if severity_raw else None,
        )
    except Exception as exc:
        logger.warning(
            "agents._parse_action | parse failed | reason=%s | fix=returning safe default",
            exc,
        )
        return default


class EvacuationAgent:
    def observe(self, obs: AgentObservation) -> dict:
        return {
            "zone_occupancy": obs.zone_occupancy,
            "available_exits": obs.available_exits,
            "known_hazard_zones": obs.known_hazard_zones,
        }

    def act(self, narrowed_obs: dict, model_fn: Callable[[str], str]) -> AgentAction:
        exits = narrowed_obs.get("available_exits") or []
        default = AgentAction(
            action_type=ActionType.ROUTE_ZONE,
            route_to_exit=exits[0] if exits else None,
        )
        prompt = build_agent_prompt("evacuation", narrowed_obs, [])
        raw = model_fn(prompt)
        return _parse_action(raw, ActionType.ROUTE_ZONE, default)


class DispatchAgent:
    def observe(self, obs: AgentObservation, state: BuildingState) -> dict:
        active_hazard_types = [h.hazard_type for h in state.hazards.values()]
        return {
            "sos_signals": obs.sos_signals,
            "current_severity": obs.current_severity,
            "active_hazard_types": active_hazard_types,
        }

    def act(self, narrowed_obs: dict, model_fn: Callable[[str], str]) -> AgentAction:
        has_hazard = bool(narrowed_obs.get("active_hazard_types"))
        default = AgentAction(
            action_type=ActionType.DISPATCH_SERVICE,
            service_type=ServiceType.FIRE_BRIGADE if has_hazard else ServiceType.EMS,
        )
        prompt = build_agent_prompt("dispatch", narrowed_obs, [])
        raw = model_fn(prompt)
        return _parse_action(raw, ActionType.DISPATCH_SERVICE, default)


class CommsAgent:
    def observe(self, obs: AgentObservation, routing_decision: AgentAction) -> dict:
        return {
            "routing_decision": {
                "action_type": routing_decision.action_type.value,
                "zone_id": routing_decision.zone_id,
                "route_to_exit": routing_decision.route_to_exit,
            },
            "zone_occupancy": obs.zone_occupancy,
        }

    def act(self, narrowed_obs: dict, model_fn: Callable[[str], str]) -> AgentAction:
        default = AgentAction(
            action_type=ActionType.BROADCAST_PA,
            message="Attention: please evacuate calmly using the nearest available exit.",
        )
        prompt = build_agent_prompt("comms", narrowed_obs, [])
        raw = model_fn(prompt)
        return _parse_action(raw, ActionType.BROADCAST_PA, default)


class OrchestratorAgent:
    def __init__(self) -> None:
        self.evacuation_agent = EvacuationAgent()
        self.dispatch_agent = DispatchAgent()
        self.comms_agent = CommsAgent()

    def act(
        self,
        obs: AgentObservation,
        state: BuildingState,
        model_fn: Callable[[str], str],
    ) -> list[AgentAction]:
        evac_narrowed = self.evacuation_agent.observe(obs)
        dispatch_narrowed = self.dispatch_agent.observe(obs, state)

        with ThreadPoolExecutor(max_workers=3) as executor:
            evac_future = executor.submit(self.evacuation_agent.act, evac_narrowed, model_fn)
            dispatch_future = executor.submit(self.dispatch_agent.act, dispatch_narrowed, model_fn)

            evac_action = evac_future.result()
            comms_narrowed = self.comms_agent.observe(obs, evac_action)
            comms_future = executor.submit(self.comms_agent.act, comms_narrowed, model_fn)

            dispatch_action = dispatch_future.result()
            comms_action = comms_future.result()

        return [evac_action, dispatch_action, comms_action]


# ---------------------------------------------------------------------------
# Integration smoke-test: 3-tick run against mock environment
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import dataclasses
    from schema import (
        Zone, Person, Hazard, SensorReading,
        BuildingState, AgentObservation, SeverityLevel,
    )

    _HARDCODED = {
        "tick_0": json.dumps({"action_type": "route_zone", "zone_id": "z1", "route_to_exit": "exit_a"}),
        "tick_1": json.dumps({"action_type": "dispatch_service", "service_type": "fire_brigade"}),
        "tick_2": json.dumps({"action_type": "broadcast_pa", "message": "Please proceed to exit A."}),
    }

    _call_idx = [0]

    def dummy_model_fn(prompt: str) -> str:
        idx = _call_idx[0] % 3
        _call_idx[0] += 1
        if "evacuation" in prompt.lower():
            return json.dumps({"action_type": "route_zone", "zone_id": "z1", "route_to_exit": "exit_a"})
        if "dispatch" in prompt.lower():
            return json.dumps({"action_type": "dispatch_service", "service_type": "fire_brigade"})
        return json.dumps({"action_type": "broadcast_pa", "message": "Evacuate via exit A immediately."})

    zone = Zone("z1", floor=1, capacity=20, connected_zones=["z2"], has_exit=True, exit_id="exit_a")
    hazard = Hazard("h1", HazardType.FIRE, affected_zones=["z2"], spread_rate=0.2, intensity=0.6)
    person = Person("p1", current_zone="z1", is_evacuated=False, has_sos=False)
    sensor = SensorReading("z1", smoke_level=0.3, motion_detected=True, sound_level=0.4, is_noisy=False)

    mock_state = BuildingState(
        building_id="mock",
        floors=2,
        zones={"z1": zone},
        people={"p1": person},
        hazards={"h1": hazard},
        sensor_readings={"z1": sensor},
        blocked_exits=[],
        tick=0,
        max_ticks=3,
        episode_done=False,
    )

    mock_obs = AgentObservation(
        tick=0,
        sensor_readings={"z1": sensor},
        zone_occupancy={"z1": 5},
        known_hazard_zones=["z2"],
        available_exits=["exit_a"],
        sos_signals=[],
        current_severity=SeverityLevel.MEDIUM,
    )

    orchestrator = OrchestratorAgent()

    for tick in range(3):
        mock_obs.tick = tick
        mock_state.tick = tick
        actions = orchestrator.act(mock_obs, mock_state, dummy_model_fn)
        print(f"\n--- Tick {tick} ---")
        for action in actions:
            print(f"  {action.action_type.value}: zone={action.zone_id}, "
                  f"exit={action.route_to_exit}, service={action.service_type}, "
                  f"msg={action.message}")
