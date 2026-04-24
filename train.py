# Dependencies (install in Colab):
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install "trl>=0.8.0" transformers torch datasets matplotlib

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from __future__ import annotations

import dataclasses
import functools
import json
import warnings
from statistics import mean

import matplotlib.pyplot as plt
import torch
from datasets import Dataset

from agents import OrchestratorAgent
from curriculum import CURRICULUM_LEVELS, IncidentLog
from environment import CrisisCoreEnv
from rewards import compute_reward
from schema import (
    ActionType,
    AgentAction,
    AgentObservation,
    SeverityLevel,
    ServiceType,
)

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# ---------------------------------------------------------------------------
# Load model + tokenizer with Unsloth 4-bit quantization
# ---------------------------------------------------------------------------
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,
)

# ---------------------------------------------------------------------------
# Section 1: Prompt formatting
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = (
    "You are the {role} agent in a crisis response system. "
    "Analyse the building observation and return a single JSON action. "
    "Valid formats:\n"
    '  {{"action_type": "route_zone", "zone_id": "<id>", "route_to_exit": "<exit_id>"}}\n'
    '  {{"action_type": "dispatch_service", "service_type": "fire_brigade|ems|police"}}\n'
    '  {{"action_type": "broadcast_pa", "message": "<message>"}}\n'
    "Return ONLY the JSON object. No explanation."
)


def format_prompt(observation: dict, last_incident_log: list[str], agent_role: str) -> str:
    obs_json = json.dumps(observation, indent=2, default=str)
    system_msg = _SYSTEM_TEMPLATE.format(role=agent_role)
    user_content = f"Current observation:\n{obs_json}"
    if last_incident_log:
        user_content += "\n\nPrevious episode mistakes:\n" + "\n".join(
            f"- {e}" for e in last_incident_log
        )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Section 2: Rollout
# ---------------------------------------------------------------------------

def _obs_to_dict(obs: AgentObservation) -> dict:
    return json.loads(json.dumps(dataclasses.asdict(obs), default=str))


def rollout(
    env: CrisisCoreEnv,
    orchestrator: OrchestratorAgent,
    model,
    tokenizer,
    incident_log: IncidentLog,
) -> list[dict]:
    obs = env.reset()
    done = False
    records: list[dict] = []

    while not done:
        obs_dict = _obs_to_dict(obs)
        prompt = format_prompt(obs_dict, incident_log.get_log(), "orchestration")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        # Reuse the single generated completion for all orchestrator sub-agents
        _cached = completion

        def _model_fn(_prompt: str) -> str:
            return _cached

        actions = orchestrator.act(obs, env.state, _model_fn)
        obs, reward, done, _ = env.step(actions[0])
        incident_log.record(env.state, actions[0], reward)

        records.append({
            "prompt": prompt,
            "completion": completion,
            "reward": float(reward.total),
        })

    return records


# ---------------------------------------------------------------------------
# Section 3: GRPO reward function
# ---------------------------------------------------------------------------

def compute_rewards_for_grpo(
    prompts: list[str],
    completions: list[str],
    env: CrisisCoreEnv,
) -> list[float]:
    rewards: list[float] = []
    for completion in completions:
        try:
            data = json.loads(completion)
            action_type = ActionType(data.get("action_type", "route_zone"))
            service_raw = data.get("service_type")
            severity_raw = data.get("severity")
            action = AgentAction(
                action_type=action_type,
                zone_id=data.get("zone_id"),
                route_to_exit=data.get("route_to_exit"),
                service_type=ServiceType(service_raw) if service_raw else None,
                message=data.get("message"),
                severity=SeverityLevel(severity_raw) if severity_raw else None,
            )
            env.reset()
            rb = compute_reward(env.state, action, prev_evacuated=0, done=False)
            rewards.append(float(rb.total))
        except Exception:
            rewards.append(-2.0)
    return rewards


# ---------------------------------------------------------------------------
# Section 4: Training setup
# ---------------------------------------------------------------------------

# LoRA wrap
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=LORA_TARGET_MODULES,
    lora_alpha=LORA_R,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Level 1 env for dataset construction and reward function
level1_env = CrisisCoreEnv(CURRICULUM_LEVELS[1])

# Build training dataset from 200 Level 1 resets
_training_prompts: list[dict] = []
for _ in range(200):
    _obs = level1_env.reset()
    _prompt = format_prompt(_obs_to_dict(_obs), [], "orchestration")
    _training_prompts.append({"prompt": _prompt})

train_dataset = Dataset.from_list(_training_prompts)

# Bind env into the reward function so GRPOTrainer can call fn(prompts, completions)
_grpo_reward_fn = functools.partial(compute_rewards_for_grpo, env=level1_env)

from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    output_dir="./grpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    num_generations=8,
    max_new_tokens=256,
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[_grpo_reward_fn],
    args=training_args,
    train_dataset=train_dataset,
)

# ---------------------------------------------------------------------------
# Section 5: Training loop with baseline / trained comparison
# ---------------------------------------------------------------------------

NUM_EVAL_EPISODES = 20

def _eval_mean_reward(n: int) -> float:
    rewards: list[float] = []
    for _ in range(n):
        env = CrisisCoreEnv(CURRICULUM_LEVELS[1])
        inc_log = IncidentLog()
        orch = OrchestratorAgent()
        episode_records = rollout(env, orch, model, tokenizer, inc_log)
        rewards.append(mean(r["reward"] for r in episode_records) if episode_records else 0.0)
    return mean(rewards)


print("Running baseline evaluation...")
baseline_reward = _eval_mean_reward(NUM_EVAL_EPISODES)

print("Training...")
train_result = trainer.train()
step_rewards: list[float] = [
    log["reward_mean"] for log in trainer.state.log_history if "reward_mean" in log
]

print("Running post-training evaluation...")
trained_reward = _eval_mean_reward(NUM_EVAL_EPISODES)

print(
    f"Baseline: {baseline_reward:.2f} → Trained: {trained_reward:.2f} "
    f"| Improvement: {trained_reward - baseline_reward:+.2f}"
)

# ---------------------------------------------------------------------------
# Section 6: Reward curve plot
# ---------------------------------------------------------------------------

if step_rewards:
    plt.figure(figsize=(10, 5))
    plt.plot(step_rewards, linewidth=1.5)
    plt.xlabel("Training step")
    plt.ylabel("Mean episode reward")
    plt.title("GRPO reward curve — CrisisCoreEnv")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reward_curve.png", dpi=150)
    plt.close()
    print("Reward curve saved → reward_curve.png")
else:
    print("No step-level reward data logged — skipping reward_curve.png")

# ---------------------------------------------------------------------------
# Section 7: Save model
# ---------------------------------------------------------------------------

try:
    model.save_pretrained_merged("crisiscore_model", tokenizer, save_method="merged_16bit")
    print("Model saved → crisiscore_model/")
except Exception as exc:
    warnings.warn(
        f"Merged 16-bit save failed: {exc}. "
        "Try model.save_pretrained('crisiscore_model') for LoRA-only save.",
        RuntimeWarning,
        stacklevel=2,
    )
