Create a deployment package for CrisisCoreEnv on HuggingFace Spaces using OpenEnv's FastAPI wrapper.

1. Create `app.py` — a FastAPI application that wraps CrisisCoreEnv:
   — On startup, instantiate CrisisCoreEnv with Level 1 config and OrchestratorAgent
   — POST /reset — call env.reset(), return observation as JSON
   — POST /step — accept AgentAction as JSON body, call env.step(), return (observation, reward_breakdown, done, info) as JSON
   — GET /state — return current BuildingState as JSON (for dashboard polling)
   — GET /health — return {"status": "ok", "tick": current_tick, "level": current_curriculum_level}
   — Handle all exceptions with proper HTTP error responses

2. Create `Dockerfile`:
   — FROM python:3.11-slim
   — COPY all .py files
   — RUN pip install openenv fastapi uvicorn pydantic
   — EXPOSE 7860
   — CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

3. Create `README.md` for the HuggingFace Space:
   — Title: CrisisCore — Multi-Agent Crisis Response RL Environment
   — Section 1 (2 sentences): what problem it solves
   — Section 2: environment description — what reset() produces, what step() accepts, what observation the agent sees
   — Section 3: reward model — list all 5 signals with one-line descriptions
   — Section 4: how to use it locally — pip install, uvicorn command, example Python client code
   — Embed reward_curve.png
   — Theme tags: multi-agent, reinforcement-learning, crisis-response, openenv

4. Create `client.py` — a simple Python client:
   — `class CrisisCoreClient` with methods: reset(), step(action_dict), get_state(), health_check()
   — Use httpx or requests. Accept base_url in __init__ defaulting to the HF Space URL.
   — Add a demo script at the bottom that connects to the Space, runs 3 episodes with random actions, and prints rewards.

5. Create `requirements.txt` listing all dependencies with pinned versions.

Test locally: run uvicorn app:app and confirm all endpoints respond correctly before pushing.