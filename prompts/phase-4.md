Create a file called `agents.py` that implements the multi-agent hierarchy for CrisisCoreEnv. Import all types from schema.py.

The architecture is: one Orchestrator that receives full state and coordinates three sub-agents that act in parallel each tick.

1. `class EvacuationAgent`
   — `__init__(self)`: no state
   — `observe(self, obs: AgentObservation) -> dict`: extract zone_occupancy, available_exits, known_hazard_zones. Return a narrowed observation dict with only these fields.
   — `act(self, narrowed_obs: dict, model_fn: callable) -> AgentAction`: call model_fn with a prompt that includes the narrowed observation as JSON. Parse the returned JSON into an AgentAction of type ROUTE_ZONE. If parsing fails, return a safe default (route people toward nearest available exit).

2. `class DispatchAgent`
   — `observe(self, obs: AgentObservation, state: BuildingState) -> dict`: extract sos_signals, current_severity, active hazard types. Return narrowed dict.
   — `act(self, narrowed_obs: dict, model_fn: callable) -> AgentAction`: call model_fn. Parse response into AgentAction of type DISPATCH_SERVICE. Safe default: call FIRE_BRIGADE if any hazard exists.

3. `class CommsAgent`
   — `observe(self, obs: AgentObservation, routing_decision: AgentAction) -> dict`: extract current routing decision and zone occupancy. Return narrowed dict.
   — `act(self, narrowed_obs: dict, model_fn: callable) -> AgentAction`: call model_fn. Parse response into AgentAction of type BROADCAST_PA with a message string. Safe default: generic evacuation message.

4. `class OrchestratorAgent`
   — `__init__(self)`: instantiate all three sub-agents.
   — `act(self, obs: AgentObservation, state: BuildingState, model_fn: callable) -> list[AgentAction]`: call all three sub-agents in parallel using concurrent.futures.ThreadPoolExecutor. Collect their actions. Return list of all three AgentActions.

5. `def build_agent_prompt(agent_role: str, observation: dict, last_incident_log: list[str]) -> str`
   — build a system+user prompt string. System: describe the agent's specific role in one paragraph. User: inject observation as formatted JSON. If last_incident_log is non-empty, append "Last episode mistakes: {log}" at the end. Return the full prompt string.

At the bottom, add a test that instantiates OrchestratorAgent and runs it for 3 ticks against a mock environment using a dummy model_fn that returns a hardcoded valid JSON string. Print each sub-agent's action per tick.