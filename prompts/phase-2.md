Build a complete OpenEnv-compliant RL environment for a multi-agent crisis response simulation. Use the dataclasses defined in `schema.py` as your data structures.

The environment should be a Python class called `CrisisCoreEnv` in a file called `environment.py`. It must implement the following:

1. `__init__(self, config: dict)` — accept config keys: num_floors (default 1), num_zones_per_floor (default 4), num_people (default 10), num_hazards (default 1), max_ticks (default 30), sensor_noise_rate (default 0.0). Store config. Initialize empty state.

2. `reset(self) -> AgentObservation` — generate a fresh BuildingState. Create zones in a grid pattern per floor, connect adjacent zones, designate 2 exit zones per floor. Randomly place num_people people across zones. Randomly place num_hazards hazards in non-exit zones. Generate initial sensor readings. Reset tick to 0. Return the initial observation.

3. `step(self, action: AgentAction) -> tuple[AgentObservation, RewardBreakdown, bool, dict]` — apply the action to the state. For ROUTE_ZONE: move all people in that zone one step toward the specified exit along the shortest path. For DISPATCH_SERVICE: record which service was called and for what crisis. For BROADCAST_PA: record the message. For UPDATE_SEVERITY: update the current severity in state. Then advance hazards — each hazard spreads to 1 adjacent zone per tick with probability equal to spread_rate. Update sensor readings with noise based on sensor_noise_rate (flip is_noisy=True randomly). Increment tick. Check termination. Return (observation, reward_breakdown, done, info_dict).

4. `_get_observation(self) -> AgentObservation` — build and return AgentObservation from current state. Occupancy counts per zone. Only include hazard zones where a sensor has detected something (partial observability). List exits that are not in blocked_exits.

5. `_check_termination(self) -> bool` — return True if all people are evacuated OR tick >= max_ticks.

6. `_shortest_path(self, from_zone: str, to_exit: str) -> list[str]` — BFS across zone connections. Return list of zone_ids. Return empty list if no path.

7. `get_info(self) -> dict` — return dict with: total_people, evacuated_count, remaining_count, active_hazards, current_tick, hazard_zones.

Start simple: get a single-floor, 10-person, 1-hazard environment running correctly first. At the bottom of the file, add a `if __name__ == "__main__"` block that runs 5 random episodes (random valid actions each tick) and prints the final reward breakdown and evacuation count for each. Confirm zero crashes across all 5.