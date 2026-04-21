#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import dataclasses
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

try:
    import wandb
except Exception as e:
    raise ImportError("Please install wandb first: pip install wandb") from e

# ============================================================
# Environment import
# ============================================================
try:
    from tdrp.envs.env import TruckMultiDroneCleanEnv, EnvConfig, RewardConfig, UncertaintyConfig
except Exception as e:
    raise ImportError(
        "Cannot import TDRP environment. Install the project with `pip install -e .` "
        "or run from the repository root."
    ) from e


# ============================================================
# Config
# ============================================================
@dataclass
class TrainConfig:
    save_dir: str = "checkpoints_masked_ppo"

    project: str = "tadpop-online"
    entity: str | None = None
    run_name: str = "masked_ppo_single_truck_single_drone"
    wandb_mode: str = "online"  # online / offline / disabled

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_deterministic: bool = True

    num_envs: int = 8
    rollout_steps: int = 256
    total_updates: int = 1500

    gamma: float = 0.995
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    update_epochs: int = 8
    num_minibatches: int = 8
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03

    hidden_dim: int = 512
    dropout: float = 0.0

    log_every_updates: int = 1
    eval_every_updates: int = 25
    eval_episodes: int = 10
    save_every_updates: int = 50

    fixed_scale: str | None = None
    road_network_type: str = "euclidean"
    dynamic_task_ratio: float = 0.30
    dynamic_generate_online: bool = True

    heter_task_enabled: bool = True
    urgent_task_ratio: float = 0.25
    critical_task_ratio: float = 0.10

    enable_wind: bool = True
    wind_intensity: float = 1.0
    enable_internal_failures: bool = False
    battery_intensity: float = 1.0
    payload_intensity: float = 1.0

    time_coef: float = 5.0
    task_done_bonus: float = 14.0
    early_finish_coef: float = 2.0
    deadline_miss_coef: float = 7.0


# ============================================================
# Utils
# ============================================================
def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


OBS_EXCLUDE_KEYS = {"action_mask"}


def preprocess_obs_array(key: str, arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32, copy=False)
    if key == "scale_id":
        x = x / 4.0
    elif key in ("truck_stop", "truck_target_stop", "drone_stop_id"):
        x = x / 19.0
    elif key == "drone_task_id":
        x = (x + 1.0) / 128.0
    elif key == "drone_planned_stop":
        x = (x + 1.0) / 20.0
    elif key == "drone_status":
        x = x / 6.0
    return x


def flatten_obs_batch(obs_batch: Sequence[Dict[str, np.ndarray]], obs_keys: Sequence[str]) -> np.ndarray:
    parts = []
    for key in obs_keys:
        arr = np.stack([preprocess_obs_array(key, obs[key]) for obs in obs_batch], axis=0)
        parts.append(arr.reshape(arr.shape[0], -1))
    return np.concatenate(parts, axis=1).astype(np.float32)


def stack_action_masks(obs_batch: Sequence[Dict[str, np.ndarray]]) -> np.ndarray:
    return np.stack([obs["action_mask"].astype(np.float32) for obs in obs_batch], axis=0)


class MaskedCategorical:
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        mask = (mask > 0.5)
        if mask.ndim != logits.ndim:
            raise ValueError(f"mask/logits dims mismatch: {mask.shape} vs {logits.shape}")
        safe_mask = mask.clone()
        no_valid = safe_mask.sum(dim=-1) == 0
        if torch.any(no_valid):
            safe_mask[no_valid, -1] = True
        masked_logits = torch.where(safe_mask, logits, torch.full_like(logits, -1e9))
        self.dist = Categorical(logits=masked_logits)

    def sample(self) -> torch.Tensor:
        return self.dist.sample()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(action)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy()

    def mode(self) -> torch.Tensor:
        return torch.argmax(self.dist.logits, dim=-1)


# ============================================================
# Model
# ============================================================
def orthogonal_init(layer: nn.Module, gain: float = math.sqrt(2.0)) -> nn.Module:
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, nactions: int, hidden_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.backbone = nn.Sequential(
            orthogonal_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            nn.Dropout(dropout),
            orthogonal_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            nn.Dropout(dropout),
            orthogonal_init(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.Tanh(),
        )
        feat_dim = hidden_dim // 2
        self.truck_head = orthogonal_init(nn.Linear(feat_dim, nactions), gain=0.01)
        self.drone_head = orthogonal_init(nn.Linear(feat_dim, nactions), gain=0.01)
        self.value_head = orthogonal_init(nn.Linear(feat_dim, 1), gain=1.0)

    def forward(self, obs_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(obs_vec)
        truck_logits = self.truck_head(feat)
        drone_logits = self.drone_head(feat)
        value = self.value_head(feat).squeeze(-1)
        return truck_logits, drone_logits, value

    def get_action_and_value(
        self,
        obs_vec: torch.Tensor,
        action_mask: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ):
        truck_logits, drone_logits, value = self.forward(obs_vec)
        truck_dist = MaskedCategorical(truck_logits, action_mask[:, 0, :])
        drone_dist = MaskedCategorical(drone_logits, action_mask[:, 1, :])

        if action is None:
            if deterministic:
                truck_action = truck_dist.mode()
                drone_action = drone_dist.mode()
            else:
                truck_action = truck_dist.sample()
                drone_action = drone_dist.sample()
            action = torch.stack([truck_action, drone_action], dim=-1)

        logprob = truck_dist.log_prob(action[:, 0]) + drone_dist.log_prob(action[:, 1])
        entropy = truck_dist.entropy() + drone_dist.entropy()
        return action, logprob, entropy, value

    def get_value(self, obs_vec: torch.Tensor) -> torch.Tensor:
        return self.forward(obs_vec)[2]


# ============================================================
# Env factory
# ============================================================
def make_env(cfg: TrainConfig, seed: int):
    reward_cfg = RewardConfig(
        time_coef=cfg.time_coef,
        task_done_bonus=cfg.task_done_bonus,
        early_finish_coef=cfg.early_finish_coef,
        deadline_miss_coef=cfg.deadline_miss_coef,
    )

    unc_cfg = UncertaintyConfig(
        enabled=True,
        fixed_seed=seed + 1000,
        wind_enabled=cfg.enable_wind,
        wind_intensity=cfg.wind_intensity,
        battery_enabled=cfg.enable_internal_failures,
        battery_intensity=cfg.battery_intensity,
        payload_enabled=cfg.enable_internal_failures,
        payload_intensity=cfg.payload_intensity,
    )

    env_cfg = EnvConfig(
        ndrones=1,
        fixed_scale=cfg.fixed_scale,
        road_network_type=cfg.road_network_type,
        allow_dynamic_tasks=True,
        dynamic_task_ratio=cfg.dynamic_task_ratio,
        dynamic_generate_online=cfg.dynamic_generate_online,
        heter_task_enabled=cfg.heter_task_enabled,
        urgent_task_ratio=cfg.urgent_task_ratio,
        critical_task_ratio=cfg.critical_task_ratio,
        reward=reward_cfg,
        uncertainty=unc_cfg,
    )
    env = TruckMultiDroneCleanEnv(env_cfg)
    obs, _ = env.reset(seed=seed)
    return env, obs


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate_policy(
    model: ActorCritic,
    train_cfg: TrainConfig,
    obs_keys: Sequence[str],
    device: torch.device,
    episodes: int,
) -> Dict[str, float]:
    env, obs = make_env(train_cfg, seed=train_cfg.seed + 99999)
    returns, times, comps, succs, decisions, new_tasks = [], [], [], [], [], []

    ep_ret = 0.0
    finished = 0
    while finished < episodes:
        obs_vec_np = flatten_obs_batch([obs], obs_keys)
        mask_np = stack_action_masks([obs])
        obs_vec = torch.as_tensor(obs_vec_np, dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(mask_np, dtype=torch.float32, device=device)
        action, _, _, _ = model.get_action_and_value(obs_vec, mask_t, deterministic=True)
        act_np = action.squeeze(0).cpu().numpy()
        next_obs, reward, terminated, truncated, info = env.step(act_np)
        ep_ret += float(reward)
        obs = next_obs
        if terminated or truncated:
            returns.append(ep_ret)
            times.append(float(info.get("time_h", 0.0)))
            comps.append(float(info.get("completion_rate", 0.0)))
            succs.append(float(info.get("success", 0.0)))
            decisions.append(float(info.get("decision_count", 0.0)))
            new_tasks.append(float(info.get("episode_new_task_spawned", 0.0)))
            finished += 1
            ep_ret = 0.0
            obs, _ = env.reset(seed=train_cfg.seed + 99999 + finished)
    env.close()
    return {
        "eval/episodic_return_mean": float(np.mean(returns)) if returns else 0.0,
        "eval/time_h_mean": float(np.mean(times)) if times else 0.0,
        "eval/completion_rate_mean": float(np.mean(comps)) if comps else 0.0,
        "eval/success_rate": float(np.mean(succs)) if succs else 0.0,
        "eval/decision_count_mean": float(np.mean(decisions)) if decisions else 0.0,
        "eval/new_tasks_spawned_mean": float(np.mean(new_tasks)) if new_tasks else 0.0,
    }


# ============================================================
# Main training
# ============================================================
def main(cfg: TrainConfig | None = None):
    cfg = TrainConfig() if cfg is None else cfg
    ensure_dir(cfg.save_dir)
    set_seed(cfg.seed, cfg.torch_deterministic)
    device = torch.device(cfg.device)

    if cfg.wandb_mode == "disabled":
        os.environ["WANDB_MODE"] = "disabled"

    run = wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        name=cfg.run_name,
        mode=cfg.wandb_mode,
        config=dataclasses.asdict(cfg),
        save_code=False,
    )

    envs = []
    obs_batch = []
    for i in range(cfg.num_envs):
        env, obs = make_env(cfg, seed=cfg.seed + i)
        envs.append(env)
        obs_batch.append(obs)

    obs_keys = [k for k in obs_batch[0].keys() if k not in OBS_EXCLUDE_KEYS]
    obs_dim = flatten_obs_batch(obs_batch, obs_keys).shape[1]
    nactions = int(envs[0].nactions)

    model = ActorCritic(obs_dim=obs_dim, nactions=nactions, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)

    wandb.config.update({
        "obs_dim": obs_dim,
        "nactions": nactions,
        "param_count": count_parameters(model),
    }, allow_val_change=True)

    print(f"obs_dim={obs_dim}, nactions={nactions}, params={count_parameters(model):,}, device={device}")

    T, N = cfg.rollout_steps, cfg.num_envs
    obs_buf = np.zeros((T, N, obs_dim), dtype=np.float32)
    mask_buf = np.zeros((T, N, 2, nactions), dtype=np.float32)
    act_buf = np.zeros((T, N, 2), dtype=np.int64)
    logp_buf = np.zeros((T, N), dtype=np.float32)
    rew_buf = np.zeros((T, N), dtype=np.float32)
    done_buf = np.zeros((T, N), dtype=np.float32)
    val_buf = np.zeros((T, N), dtype=np.float32)

    episode_returns = np.zeros(N, dtype=np.float32)
    episode_lengths = np.zeros(N, dtype=np.int32)
    recent_returns = deque(maxlen=100)
    recent_success = deque(maxlen=100)
    recent_time_h = deque(maxlen=100)
    recent_completion = deque(maxlen=100)

    global_step = 0
    best_eval_success = -1.0
    best_eval_time = float("inf")
    start_time = time.time()

    for update in range(1, cfg.total_updates + 1):
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1.0) / max(float(cfg.total_updates), 1.0)
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        for t in range(T):
            obs_vec_np = flatten_obs_batch(obs_batch, obs_keys)
            mask_np = stack_action_masks(obs_batch)

            obs_buf[t] = obs_vec_np
            mask_buf[t] = mask_np

            obs_t = torch.as_tensor(obs_vec_np, dtype=torch.float32, device=device)
            mask_t = torch.as_tensor(mask_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                action_t, logprob_t, _, value_t = model.get_action_and_value(obs_t, mask_t, deterministic=False)

            action_np = action_t.cpu().numpy()
            logprob_np = logprob_t.cpu().numpy()
            value_np = value_t.cpu().numpy()

            act_buf[t] = action_np
            logp_buf[t] = logprob_np
            val_buf[t] = value_np

            next_obs_batch = []
            for i, env in enumerate(envs):
                next_obs, reward, terminated, truncated, info = env.step(action_np[i])
                done = bool(terminated or truncated)
                rew_buf[t, i] = float(reward)
                done_buf[t, i] = 1.0 if done else 0.0

                episode_returns[i] += float(reward)
                episode_lengths[i] += 1
                global_step += 1

                if done:
                    ep_log = {
                        "episode/return": float(episode_returns[i]),
                        "episode/length": int(episode_lengths[i]),
                        "episode/time_h": float(info.get("time_h", 0.0)),
                        "episode/completion_rate": float(info.get("completion_rate", 0.0)),
                        "episode/success": float(info.get("success", 0.0)),
                        "episode/decision_count": float(info.get("decision_count", 0.0)),
                        "episode/invalid_action_count": float(info.get("invalid_action_count", 0.0)),
                        "episode/claim_conflict_count": float(info.get("claim_conflict_count", 0.0)),
                        "episode/new_task_spawned": float(info.get("episode_new_task_spawned", 0.0)),
                        "episode/wait_step_count": float(info.get("episode_wait_step_count", 0.0)),
                        "episode/wait_step_time_h": float(info.get("episode_wait_step_time_h", 0.0)),
                        "global_step": global_step,
                    }
                    wandb.log(ep_log, step=global_step)
                    recent_returns.append(float(episode_returns[i]))
                    recent_success.append(float(info.get("success", 0.0)))
                    recent_time_h.append(float(info.get("time_h", 0.0)))
                    recent_completion.append(float(info.get("completion_rate", 0.0)))
                    episode_returns[i] = 0.0
                    episode_lengths[i] = 0
                    next_obs, _ = env.reset(seed=cfg.seed + 100000 + update * N + i)
                next_obs_batch.append(next_obs)
            obs_batch = next_obs_batch

        next_obs_vec_np = flatten_obs_batch(obs_batch, obs_keys)
        next_obs_t = torch.as_tensor(next_obs_vec_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            next_value = model.get_value(next_obs_t).cpu().numpy()

        adv_buf = np.zeros_like(rew_buf, dtype=np.float32)
        lastgaelam = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            if t == T - 1:
                next_nonterminal = 1.0 - done_buf[t]
                next_values = next_value
            else:
                next_nonterminal = 1.0 - done_buf[t + 1]
                next_values = val_buf[t + 1]
            delta = rew_buf[t] + cfg.gamma * next_values * next_nonterminal - val_buf[t]
            lastgaelam = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * lastgaelam
            adv_buf[t] = lastgaelam
        ret_buf = adv_buf + val_buf

        b_obs = torch.as_tensor(obs_buf.reshape(T * N, obs_dim), dtype=torch.float32, device=device)
        b_mask = torch.as_tensor(mask_buf.reshape(T * N, 2, nactions), dtype=torch.float32, device=device)
        b_actions = torch.as_tensor(act_buf.reshape(T * N, 2), dtype=torch.long, device=device)
        b_logprobs = torch.as_tensor(logp_buf.reshape(T * N), dtype=torch.float32, device=device)
        b_advantages = torch.as_tensor(adv_buf.reshape(T * N), dtype=torch.float32, device=device)
        b_returns = torch.as_tensor(ret_buf.reshape(T * N), dtype=torch.float32, device=device)
        b_values = torch.as_tensor(val_buf.reshape(T * N), dtype=torch.float32, device=device)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        batch_size = T * N
        minibatch_size = batch_size // cfg.num_minibatches
        b_inds = np.arange(batch_size)

        clipfracs, approx_kls, pg_losses, v_losses, entropies = [], [], [], [], []

        for _epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = model.get_action_and_value(
                    b_obs[mb_inds], b_mask[mb_inds], action=b_actions[mb_inds], deterministic=False
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    approx_kls.append(float(approx_kl.item()))
                    clipfracs.append(float(clipfrac))

                mb_adv = b_advantages[mb_inds]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + (newvalue - b_values[mb_inds]).clamp(-cfg.clip_coef, cfg.clip_coef)
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                pg_losses.append(float(pg_loss.item()))
                v_losses.append(float(v_loss.item()))
                entropies.append(float(entropy_loss.item()))

            if cfg.target_kl is not None and len(approx_kls) > 0 and np.mean(approx_kls) > cfg.target_kl:
                break

        if update % cfg.log_every_updates == 0:
            sps = int(global_step / max(time.time() - start_time, 1e-6))
            log_data = {
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "charts/SPS": sps,
                "losses/policy_loss": float(np.mean(pg_losses)) if pg_losses else 0.0,
                "losses/value_loss": float(np.mean(v_losses)) if v_losses else 0.0,
                "losses/entropy": float(np.mean(entropies)) if entropies else 0.0,
                "losses/approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
                "losses/clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
                "rollout/reward_mean": float(rew_buf.mean()),
                "rollout/reward_std": float(rew_buf.std()),
                "rollout/value_mean": float(val_buf.mean()),
                "global_step": global_step,
                "update": update,
            }
            if recent_returns:
                log_data.update({
                    "recent/episode_return_mean": float(np.mean(recent_returns)),
                    "recent/success_rate": float(np.mean(recent_success)),
                    "recent/time_h_mean": float(np.mean(recent_time_h)),
                    "recent/completion_rate_mean": float(np.mean(recent_completion)),
                })
            wandb.log(log_data, step=global_step)
            print(
                f"[upd {update:4d}] step={global_step:8d} "
                f"pg={log_data['losses/policy_loss']:.4f} "
                f"v={log_data['losses/value_loss']:.4f} "
                f"ent={log_data['losses/entropy']:.4f} "
                f"succ={log_data.get('recent/success_rate', 0.0):.3f} "
                f"comp={log_data.get('recent/completion_rate_mean', 0.0):.3f} "
                f"time={log_data.get('recent/time_h_mean', 0.0):.3f}h"
            )

        if update % cfg.eval_every_updates == 0:
            eval_metrics = evaluate_policy(
                model=model,
                train_cfg=cfg,
                obs_keys=obs_keys,
                device=device,
                episodes=cfg.eval_episodes,
            )
            eval_metrics["global_step"] = global_step
            eval_metrics["update"] = update
            wandb.log(eval_metrics, step=global_step)
            print(
                f"[eval {update:4d}] ret={eval_metrics['eval/episodic_return_mean']:.3f} "
                f"succ={eval_metrics['eval/success_rate']:.3f} "
                f"comp={eval_metrics['eval/completion_rate_mean']:.3f} "
                f"time={eval_metrics['eval/time_h_mean']:.3f}h"
            )

            improved = False
            if eval_metrics["eval/success_rate"] > best_eval_success + 1e-9:
                improved = True
            elif abs(eval_metrics["eval/success_rate"] - best_eval_success) <= 1e-9 and eval_metrics["eval/time_h_mean"] < best_eval_time:
                improved = True

            if improved:
                best_eval_success = eval_metrics["eval/success_rate"]
                best_eval_time = eval_metrics["eval/time_h_mean"]
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_cfg": dataclasses.asdict(cfg),
                    "obs_keys": list(obs_keys),
                    "obs_dim": obs_dim,
                    "nactions": nactions,
                    "global_step": global_step,
                    "update": update,
                    "best_eval_success": best_eval_success,
                    "best_eval_time": best_eval_time,
                }
                best_path = Path(cfg.save_dir) / "best_model.pt"
                torch.save(ckpt, best_path)
                wandb.save(str(best_path))

        if update % cfg.save_every_updates == 0:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_cfg": dataclasses.asdict(cfg),
                "obs_keys": list(obs_keys),
                "obs_dim": obs_dim,
                "nactions": nactions,
                "global_step": global_step,
                "update": update,
                "best_eval_success": best_eval_success,
                "best_eval_time": best_eval_time,
            }
            save_path = Path(cfg.save_dir) / f"checkpoint_update_{update}.pt"
            torch.save(ckpt, save_path)

    final_ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_cfg": dataclasses.asdict(cfg),
        "obs_keys": list(obs_keys),
        "obs_dim": obs_dim,
        "nactions": nactions,
        "global_step": global_step,
        "update": cfg.total_updates,
        "best_eval_success": best_eval_success,
        "best_eval_time": best_eval_time,
    }
    final_path = Path(cfg.save_dir) / "final_model.pt"
    torch.save(final_ckpt, final_path)
    wandb.save(str(final_path))

    for env in envs:
        env.close()
    run.finish()
    print(f"Training finished. Final model saved to: {final_path}")


if __name__ == "__main__":
    cfg = TrainConfig(
        save_dir=r"checkpoints_masked_ppo",
        run_name=f"masked_ppo_seed42_{int(time.time())}",
        fixed_scale=None,
        num_envs=8,
        rollout_steps=256,
        total_updates=1500,
        eval_every_updates=25,
        save_every_updates=50,
        wandb_mode="online",
    )
    main(cfg)


