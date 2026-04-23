Create a file called `curriculum.py` that implements a 3-level curriculum and a post-incident learning log for CrisisCoreEnv.

1. `CURRICULUM_LEVELS` — a dict mapping level int (1, 2, 3) to a config dict:
   — Level 1: num_floors=1, num_zones_per_floor=4, num_people=5, num_hazards=1, max_ticks=20, sensor_noise_rate=0.0
   — Level 2: num_floors=2, num_zones_per_floor=4, num_people=15, num_hazards=2, max_ticks=30, sensor_noise_rate=0.1
   — Level 3: num_floors=3, num_zones_per_floor=6, num_people=30, num_hazards=3, max_ticks=40, sensor_noise_rate=0.15

2. `class CurriculumManager`
   — `__init__(self, promotion_threshold: float = 0.7, window: int = 20)`: start at level 1. Maintain a rolling window of last `window` episode rewards normalized 0–1.
   — `record_episode(self, reward_total: float, max_possible: float)`: append normalized score to window. If mean of window exceeds promotion_threshold and current level < 3, promote to next level and reset window.
   — `get_config(self) -> dict`: return the config dict for the current level.
   — `current_level(self) -> int`: return current level int.
   — `should_promote(self) -> bool`: return True if promotion condition is met.

3. `class IncidentLog`
   — `__init__(self, max_entries: int = 10)`: maintain a deque of the last max_entries log entries.
   — `record(self, state: BuildingState, action: AgentAction, reward: RewardBreakdown)`: if route_safety < -1.0, log "Routed zone {zone_id} through hazard at tick {tick}". If dispatch_accuracy < 0.0, log "Wrong service dispatched at tick {tick}". If severity_accuracy < 0.0, log "Severity misclassified at tick {tick}". If timeout_penalty < 0.0, log "Episode timed out with {n} people unrescued".
   — `get_log(self) -> list[str]`: return list of recent log strings.
   — `clear(self)`: clear the deque.

4. `def run_curriculum_episode(env: CrisisCoreEnv, orchestrator: OrchestratorAgent, model_fn: callable, incident_log: IncidentLog) -> tuple[float, dict]`
   — run one full episode: reset env, loop until done, call orchestrator.act each tick, apply first action from the list to env.step, record reward, append to incident_log. Return (total_reward, info_dict).

At the bottom, run a simulation: instantiate CurriculumManager and IncidentLog. Run 30 episodes with a random model_fn. Print the level, mean reward, and incident log after every 10 episodes. Confirm the manager promotes correctly when rewards are artificially high.