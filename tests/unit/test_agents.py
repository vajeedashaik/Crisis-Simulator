"""Tests for agents.py — Phase 4: multi-agent hierarchy."""
import json
import pytest
from unittest.mock import MagicMock

from schema import (
    ActionType, ServiceType, HazardType, SeverityLevel,
    Zone, Person, Hazard, SensorReading,
    BuildingState, AgentAction, AgentObservation,
)
from agents import (
    EvacuationAgent, DispatchAgent, CommsAgent,
    OrchestratorAgent, build_agent_prompt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def obs():
    return AgentObservation(
        tick=1,
        sensor_readings={
            "z1": SensorReading("z1", 0.1, True, 0.2, False),
            "z2": SensorReading("z2", 0.8, False, 0.5, True),
        },
        zone_occupancy={"z1": 3, "z2": 0, "z3": 5},
        known_hazard_zones=["z2"],
        available_exits=["exit_a", "exit_b"],
        sos_signals=["p1", "p2"],
        current_severity=SeverityLevel.HIGH,
    )


@pytest.fixture
def state():
    zone = Zone("z1", floor=1, capacity=10, connected_zones=["z2"], has_exit=True, exit_id="exit_a")
    hazard = Hazard("h1", HazardType.FIRE, affected_zones=["z2"], spread_rate=0.3, intensity=0.7)
    person = Person("p1", current_zone="z1", is_evacuated=False, has_sos=True)
    sensor = SensorReading("z1", smoke_level=0.1, motion_detected=True, sound_level=0.2, is_noisy=False)
    return BuildingState(
        building_id="b1",
        floors=3,
        zones={"z1": zone},
        people={"p1": person},
        hazards={"h1": hazard},
        sensor_readings={"z1": sensor},
        blocked_exits=[],
        tick=1,
        max_ticks=100,
        episode_done=False,
    )


# ---------------------------------------------------------------------------
# EvacuationAgent
# ---------------------------------------------------------------------------

class TestEvacuationAgentObserve:
    def test_returns_zone_occupancy(self, obs):
        agent = EvacuationAgent()
        narrowed = agent.observe(obs)
        assert narrowed["zone_occupancy"] == {"z1": 3, "z2": 0, "z3": 5}

    def test_returns_available_exits(self, obs):
        agent = EvacuationAgent()
        narrowed = agent.observe(obs)
        assert narrowed["available_exits"] == ["exit_a", "exit_b"]

    def test_returns_known_hazard_zones(self, obs):
        agent = EvacuationAgent()
        narrowed = agent.observe(obs)
        assert narrowed["known_hazard_zones"] == ["z2"]

    def test_excludes_other_fields(self, obs):
        agent = EvacuationAgent()
        narrowed = agent.observe(obs)
        assert set(narrowed.keys()) == {"zone_occupancy", "available_exits", "known_hazard_zones"}


class TestEvacuationAgentAct:
    def _make_model_fn(self, payload: dict):
        return lambda prompt: json.dumps(payload)

    def test_returns_route_zone_action(self, obs):
        agent = EvacuationAgent()
        narrowed = agent.observe(obs)
        model_fn = self._make_model_fn({
            "action_type": "route_zone",
            "zone_id": "z1",
            "route_to_exit": "exit_a",
        })
        action = agent.act(narrowed, model_fn)
        assert isinstance(action, AgentAction)
        assert action.action_type == ActionType.ROUTE_ZONE

    def test_model_fn_called_with_prompt(self, obs):
        agent = EvacuationAgent()
        narrowed = agent.observe(obs)
        calls = []
        def model_fn(prompt):
            calls.append(prompt)
            return json.dumps({"action_type": "route_zone"})
        agent.act(narrowed, model_fn)
        assert len(calls) == 1
        assert isinstance(calls[0], str)

    def test_safe_default_on_parse_failure(self, obs):
        agent = EvacuationAgent()
        narrowed = agent.observe(obs)
        model_fn = lambda prompt: "not valid json {{{"
        action = agent.act(narrowed, model_fn)
        assert isinstance(action, AgentAction)
        assert action.action_type == ActionType.ROUTE_ZONE

    def test_safe_default_has_exit_when_exits_available(self, obs):
        agent = EvacuationAgent()
        narrowed = agent.observe(obs)
        model_fn = lambda prompt: "bad json"
        action = agent.act(narrowed, model_fn)
        assert action.route_to_exit is not None


# ---------------------------------------------------------------------------
# DispatchAgent
# ---------------------------------------------------------------------------

class TestDispatchAgentObserve:
    def test_returns_sos_signals(self, obs, state):
        agent = DispatchAgent()
        narrowed = agent.observe(obs, state)
        assert narrowed["sos_signals"] == ["p1", "p2"]

    def test_returns_current_severity(self, obs, state):
        agent = DispatchAgent()
        narrowed = agent.observe(obs, state)
        assert narrowed["current_severity"] == SeverityLevel.HIGH

    def test_returns_active_hazard_types(self, obs, state):
        agent = DispatchAgent()
        narrowed = agent.observe(obs, state)
        assert HazardType.FIRE in narrowed["active_hazard_types"]

    def test_excludes_other_fields(self, obs, state):
        agent = DispatchAgent()
        narrowed = agent.observe(obs, state)
        assert set(narrowed.keys()) == {"sos_signals", "current_severity", "active_hazard_types"}


class TestDispatchAgentAct:
    def _make_model_fn(self, payload: dict):
        return lambda prompt: json.dumps(payload)

    def test_returns_dispatch_service_action(self, obs, state):
        agent = DispatchAgent()
        narrowed = agent.observe(obs, state)
        model_fn = self._make_model_fn({
            "action_type": "dispatch_service",
            "service_type": "fire_brigade",
        })
        action = agent.act(narrowed, model_fn)
        assert isinstance(action, AgentAction)
        assert action.action_type == ActionType.DISPATCH_SERVICE

    def test_safe_default_calls_fire_brigade_when_hazard_exists(self, obs, state):
        agent = DispatchAgent()
        narrowed = agent.observe(obs, state)
        model_fn = lambda prompt: "{{invalid}}"
        action = agent.act(narrowed, model_fn)
        assert action.action_type == ActionType.DISPATCH_SERVICE
        assert action.service_type == ServiceType.FIRE_BRIGADE

    def test_safe_default_no_hazard_still_returns_dispatch(self, obs, state):
        state.hazards = {}
        agent = DispatchAgent()
        narrowed = agent.observe(obs, state)
        model_fn = lambda prompt: "bad"
        action = agent.act(narrowed, model_fn)
        assert action.action_type == ActionType.DISPATCH_SERVICE


# ---------------------------------------------------------------------------
# CommsAgent
# ---------------------------------------------------------------------------

class TestCommsAgentObserve:
    def test_returns_routing_decision(self, obs):
        agent = CommsAgent()
        routing = AgentAction(ActionType.ROUTE_ZONE, zone_id="z1", route_to_exit="exit_a")
        narrowed = agent.observe(obs, routing)
        assert "routing_decision" in narrowed

    def test_returns_zone_occupancy(self, obs):
        agent = CommsAgent()
        routing = AgentAction(ActionType.ROUTE_ZONE, zone_id="z1", route_to_exit="exit_a")
        narrowed = agent.observe(obs, routing)
        assert narrowed["zone_occupancy"] == {"z1": 3, "z2": 0, "z3": 5}

    def test_excludes_other_fields(self, obs):
        agent = CommsAgent()
        routing = AgentAction(ActionType.ROUTE_ZONE)
        narrowed = agent.observe(obs, routing)
        assert set(narrowed.keys()) == {"routing_decision", "zone_occupancy"}


class TestCommsAgentAct:
    def _make_model_fn(self, payload: dict):
        return lambda prompt: json.dumps(payload)

    def test_returns_broadcast_pa_action(self, obs):
        agent = CommsAgent()
        routing = AgentAction(ActionType.ROUTE_ZONE)
        narrowed = agent.observe(obs, routing)
        model_fn = self._make_model_fn({
            "action_type": "broadcast_pa",
            "message": "Please evacuate via exit A.",
        })
        action = agent.act(narrowed, model_fn)
        assert isinstance(action, AgentAction)
        assert action.action_type == ActionType.BROADCAST_PA

    def test_parsed_action_has_message(self, obs):
        agent = CommsAgent()
        routing = AgentAction(ActionType.ROUTE_ZONE)
        narrowed = agent.observe(obs, routing)
        model_fn = self._make_model_fn({
            "action_type": "broadcast_pa",
            "message": "Evacuate now!",
        })
        action = agent.act(narrowed, model_fn)
        assert action.message == "Evacuate now!"

    def test_safe_default_is_generic_evacuation_message(self, obs):
        agent = CommsAgent()
        routing = AgentAction(ActionType.ROUTE_ZONE)
        narrowed = agent.observe(obs, routing)
        model_fn = lambda prompt: "not json"
        action = agent.act(narrowed, model_fn)
        assert action.action_type == ActionType.BROADCAST_PA
        assert isinstance(action.message, str)
        assert len(action.message) > 0


# ---------------------------------------------------------------------------
# OrchestratorAgent
# ---------------------------------------------------------------------------

class TestOrchestratorAgent:
    def _dummy_model_fn(self, prompt: str) -> str:
        if "evacuation" in prompt.lower():
            return json.dumps({"action_type": "route_zone", "zone_id": "z1", "route_to_exit": "exit_a"})
        if "dispatch" in prompt.lower():
            return json.dumps({"action_type": "dispatch_service", "service_type": "fire_brigade"})
        return json.dumps({"action_type": "broadcast_pa", "message": "Evacuate!"})

    def test_has_three_sub_agents(self):
        orch = OrchestratorAgent()
        assert isinstance(orch.evacuation_agent, EvacuationAgent)
        assert isinstance(orch.dispatch_agent, DispatchAgent)
        assert isinstance(orch.comms_agent, CommsAgent)

    def test_act_returns_list_of_three_actions(self, obs, state):
        orch = OrchestratorAgent()
        actions = orch.act(obs, state, self._dummy_model_fn)
        assert isinstance(actions, list)
        assert len(actions) == 3

    def test_actions_are_agent_action_instances(self, obs, state):
        orch = OrchestratorAgent()
        actions = orch.act(obs, state, self._dummy_model_fn)
        for action in actions:
            assert isinstance(action, AgentAction)

    def test_actions_cover_all_three_types(self, obs, state):
        orch = OrchestratorAgent()
        actions = orch.act(obs, state, self._dummy_model_fn)
        types = {a.action_type for a in actions}
        assert ActionType.ROUTE_ZONE in types
        assert ActionType.DISPATCH_SERVICE in types
        assert ActionType.BROADCAST_PA in types


# ---------------------------------------------------------------------------
# build_agent_prompt
# ---------------------------------------------------------------------------

class TestBuildAgentPrompt:
    def test_returns_string(self):
        prompt = build_agent_prompt("evacuation", {"zone_occupancy": {"z1": 3}}, [])
        assert isinstance(prompt, str)

    def test_contains_agent_role(self):
        prompt = build_agent_prompt("evacuation", {"zone_occupancy": {}}, [])
        assert "evacuation" in prompt.lower()

    def test_contains_observation_json(self):
        obs_dict = {"zone_occupancy": {"z1": 3}, "available_exits": ["exit_a"]}
        prompt = build_agent_prompt("dispatch", obs_dict, [])
        assert "z1" in prompt
        assert "exit_a" in prompt

    def test_no_incident_log_when_empty(self):
        prompt = build_agent_prompt("comms", {}, [])
        assert "Last episode mistakes" not in prompt

    def test_includes_incident_log_when_provided(self):
        log = ["failed to route zone z3", "PA message was empty"]
        prompt = build_agent_prompt("evacuation", {}, log)
        assert "Last episode mistakes" in prompt
        assert "failed to route zone z3" in prompt

    def test_observation_formatted_as_json(self):
        obs_dict = {"key": "value"}
        prompt = build_agent_prompt("dispatch", obs_dict, [])
        assert '"key"' in prompt or "key" in prompt
