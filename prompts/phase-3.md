Create a file called `rewards.py` that implements 5 independent reward functions for the CrisisCoreEnv environment. Import BuildingState, AgentAction, AgentObservation, and RewardBreakdown from schema.py.

Implement each reward as a standalone function, then combine them in a single `compute_reward` function.

1. `evacuation_speed_reward(state: BuildingState, prev_evacuated: int) -> float`
   — return +1.0 for each new person evacuated this tick. Apply a time bonus: multiply by (1 + (state.max_ticks - state.tick) / state.max_ticks) so early evacuations score higher. Cap at +5.0 per tick.

2. `route_safety_reward(state: BuildingState, action: AgentAction) -> float`
   — if action is ROUTE_ZONE and the route_to_exit path passes through a zone with an active hazard, return -2.0. If the routed zone itself has a hazard, return -3.0. Otherwise return 0.0. Use the hazard affected_zones to check.

3. `dispatch_accuracy_reward(state: BuildingState, action: AgentAction) -> float`
   — only applies when action is DISPATCH_SERVICE. Check if the service matches the dominant hazard type: FIRE/SMOKE → FIRE_BRIGADE scores +2.0, EMS for any hazard scores +1.0 (always useful), POLICE with no crowd-risk hazard scores -0.5. Return 0.0 for non-dispatch actions.

4. `severity_accuracy_reward(state: BuildingState, action: AgentAction) -> float`
   — only applies when action is UPDATE_SEVERITY. Compute ground truth severity: count total people in hazard zones, divide by total people. 0–0.1 = LOW, 0.1–0.3 = MEDIUM, 0.3–0.6 = HIGH, 0.6+ = CRITICAL. If action.severity matches ground truth return +1.5. One level off return +0.5. Two or more levels off return -1.0.

5. `format_compliance_reward(action: AgentAction) -> float`
   — return +0.2 if action has a valid action_type and all required fields for that type are non-None. Return -1.0 if required fields are missing. Required fields: ROUTE_ZONE needs zone_id and route_to_exit. DISPATCH_SERVICE needs service_type. BROADCAST_PA needs message. UPDATE_SEVERITY needs severity.

6. `timeout_penalty(state: BuildingState, done: bool) -> float`
   — if done is True and not all people are evacuated, return -5.0. Otherwise return 0.0.

7. `compute_reward(state: BuildingState, action: AgentAction, prev_evacuated: int, done: bool) -> RewardBreakdown`
   — call all 5 functions, populate a RewardBreakdown dataclass, set total = weighted sum: evacuation_speed * 1.0, route_safety * 1.0, dispatch_accuracy * 0.8, severity_accuracy * 0.6, format_compliance * 0.4, timeout_penalty * 1.0. Return the breakdown.

At the bottom, add an anti-hacking test: create a mock state and action that tries to exploit the reward (e.g. calling DISPATCH_SERVICE every tick). Confirm the reward doesn't grow unboundedly. Print a reward breakdown table for 3 different action scenarios.