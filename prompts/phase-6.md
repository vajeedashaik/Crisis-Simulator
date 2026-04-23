Create a Colab-compatible training script called `train.py` that implements GRPO training using TRL and Unsloth for CrisisCoreEnv.

Install dependencies at the top as comments: unsloth, trl>=0.8.0, transformers, torch, datasets.

1. `MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"` — use this as the base model. Load with Unsloth's FastLanguageModel with 4-bit quantization, max_seq_length=2048.

2. `def format_prompt(observation: dict, last_incident_log: list[str], agent_role: str) -> str`
   — build a chat-formatted prompt using the model's tokenizer.apply_chat_template. System message: describe the agent role and expected JSON output format. User message: inject observation JSON and incident log. Return the formatted string.

3. `def rollout(env: CrisisCoreEnv, orchestrator: OrchestratorAgent, model, tokenizer, incident_log: IncidentLog) -> list[dict]`
   — run one full episode. Each tick: format the orchestrator prompt, generate model output (max 256 tokens, temperature 0.7, do_sample=True), parse the output into AgentAction, call env.step, collect reward. Return list of dicts with keys: prompt, completion, reward (float scalar — use reward_breakdown.total).

4. `def compute_rewards_for_grpo(prompts: list[str], completions: list[str], env: CrisisCoreEnv) -> list[float]`
   — this is the reward function signature GRPOTrainer expects. For each (prompt, completion) pair, parse the completion as an AgentAction JSON, compute reward using compute_reward from rewards.py, return list of floats. Handle JSON parse failures by returning -2.0.

5. Training setup:
   — Wrap model with Unsloth's get_peft_model (LoRA r=16, target_modules=["q_proj","v_proj"])
   — Initialize GRPOTrainer with: model, tokenizer, reward_funcs=[compute_rewards_for_grpo], num_train_epochs=1, per_device_train_batch_size=4, num_generations=8, max_new_tokens=256
   — Use a Dataset built from 200 reset() observations from Level 1 env as training prompts

6. Training loop:
   — Before training: run 20 rollout episodes, record mean reward as `baseline_reward`
   — Train for 1 epoch
   — After training: run 20 rollout episodes, record mean reward as `trained_reward`
   — Print: f"Baseline: {baseline_reward:.2f} → Trained: {trained_reward:.2f} | Improvement: {trained_reward - baseline_reward:+.2f}"

7. Reward curve: use matplotlib to plot episode reward over training steps. Save as `reward_curve.png`.

8. Save model: use model.save_pretrained_merged("crisiscore_model", tokenizer, save_method="merged_16bit"). Print a warning if merged save fails.

Add clear section comments throughout so the notebook reads cleanly top to bottom.