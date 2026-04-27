#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EDCDP-lite: a runnable first-stage implementation of the paper's
Event-Driven Constrained Diffusion Planner idea for the existing TDRP env.

This version is intentionally conservative and engineering-oriented:
- Uses heterogeneous token encoding for tasks, stops, truck, drones, and environment.
- Uses the environment-provided action_mask as the hard feasibility support.
- Trains a masked discrete planner with Advantage-Weighted Regression (AWR).
- Supports multiple drones through the env's MultiDiscrete([nactions] * nagents).

It does NOT yet implement the full H-step reverse diffusion planner in the paper.
That should be added after this file runs stably as an EDCDP-lite baseline.
"""
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
except Exception:
    wandb = None

try:
    from tdrp.envs.configs import EnvConfig, RewardConfig, UncertaintyConfig
    from tdrp.envs.env import TruckMultiDroneCleanEnv
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Cannot import TDRP environment. Install with `python -m pip install -e .[rl]` "
        "or run this file from the repository root."
    ) from e


# ============================================================
# Config
# ============================================================
@dataclass
class EDCDPLiteConfig:
    save_dir: str = "checkpoints_edcdp_lite"

    project: str = "tadpop-online"
    entity: str | None = None
    run_name: str = "edcdp_lite"
    wandb_mode: str = "online"  # online / offline / disabled

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_deterministic: bool = True

    # Environment
    ndrones: int = 3
    fixed_scale: str | None = "XS"
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

    # Reward
    time_coef: float = 5.0
    task_done_bonus: float = 14.0
    early_finish_coef: float = 2.0
    deadline_miss_coef: float = 7.0

    # Rollout/training
    num_envs: int = 4
    rollout_steps: int = 256
    total_updates: int = 1000
    update_epochs: int = 4
    num_minibatches: int = 8

    gamma: float = 0.995
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    max_grad_norm: float = 0.5

    # AWR
    awr_temperature: float = 1.0
    awr_max_weight: float = 20.0
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    # Network
    hidden_dim: int = 256
    token_dim: int = 128
    transformer_layers: int = 2
    transformer_heads: int = 4
    dropout: float = 0.0

    # Logging/eval/save
    log_every_updates: int = 1
    eval_every_updates: int = 25
    eval_episodes: int = 10
    save_every_updates: int = 50


# ============================================================
# Utilities
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


def to_tensor_batch(obs_batch: Sequence[Dict[str, np.ndarray]], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    keys = obs_batch[0].keys()
    for k in keys:
        arr = np.stack([obs[k] for obs in obs_batch], axis=0)
        if arr.dtype.kind in "iu":
            out[k] = torch.as_tensor(arr, dtype=torch.float32, device=device)
        else:
            out[k] = torch.as_tensor(arr, dtype=torch.float32, device=device)
    return out


def stack_action_masks(obs_batch: Sequence[Dict[str, np.ndarray]]) -> np.ndarray:
    return np.stack([obs["action_mask"].astype(np.float32) for obs in obs_batch], axis=0)


class MaskedCategorical:
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        mask = mask > 0.5
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
# Environment factory
# ============================================================
def make_env(cfg: EDCDPLiteConfig, seed: int) -> Tuple[TruckMultiDroneCleanEnv, Dict[str, np.ndarray]]:
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
        ndrones=cfg.ndrones,
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
# Network
# ============================================================
def mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
        nn.ReLU(),
    )


class HeterogeneousStateEncoder(nn.Module):
    """Token encoder for the current env observation.

    Tokens:
    - one token per task slot
    - one token per stop slot
    - one token per drone
    - one truck token
    - one global/environment token
    """

    def __init__(self, token_dim: int, hidden_dim: int, dropout: float = 0.0,
                 nheads: int = 4, nlayers: int = 2):
        super().__init__()
        self.token_dim = token_dim

        # Dimensions are determined by the handcrafted feature concatenations below.
        self.task_proj = mlp(22, hidden_dim, token_dim, dropout)
        self.stop_proj = mlp(3, hidden_dim, token_dim, dropout)
        self.drone_proj = mlp(13, hidden_dim, token_dim, dropout)
        self.truck_proj = mlp(6, hidden_dim, token_dim, dropout)
        self.global_proj = mlp(1 + 1 + 1 + 2 + 8 + 8 + 1, hidden_dim, token_dim, dropout)

        self.type_embed = nn.Embedding(5, token_dim)  # task, stop, drone, truck, global
        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.out_norm = nn.LayerNorm(token_dim)

    @staticmethod
    def _norm_idx(x: torch.Tensor, denom: float, offset: float = 0.0) -> torch.Tensor:
        return (x + offset) / denom

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = obs["task_mask"].shape[0]
        D = obs["drone_status"].shape[1]

        # ---------------- tasks: [B, T, 22]
        task_type = obs["task_type"].unsqueeze(-1)
        task_mask = obs["task_mask"].unsqueeze(-1)
        task_done = obs["task_done"].unsqueeze(-1)
        task_claimed = obs["task_claimed"].unsqueeze(-1)
        task_spawned = obs["task_spawned"].unsqueeze(-1)
        task_scalar = torch.stack([
            obs["point_service_h"],
            obs["task_priority"],
            obs["task_deadline_h"],
            obs["task_reward_weight"],
            obs["task_urgency"],
            obs["task_service_factor"],
            obs["task_risk"],
        ], dim=-1)
        task_feat = torch.cat([
            task_type,
            task_mask,
            obs["point_xy"],
            obs["line_start_xy"],
            obs["line_end_xy"],
            task_scalar,
            task_done,
            task_claimed,
            task_spawned,
            obs["line_start_xy"] - obs["line_end_xy"],
            obs["point_xy"] - obs["line_start_xy"],
        ], dim=-1)
        task_tok = self.task_proj(task_feat) + self.type_embed.weight[0]
        task_valid = obs["task_mask"] > 0.5

        # ---------------- stops: [B, S, 3]
        stop_feat = torch.cat([obs["stop_xy"], obs["stop_mask"].unsqueeze(-1)], dim=-1)
        stop_tok = self.stop_proj(stop_feat) + self.type_embed.weight[1]
        stop_valid = obs["stop_mask"] > 0.5

        # ---------------- drones: [B, D, 13]
        drone_feat = torch.cat([
            obs["drone_status"].unsqueeze(-1) / 6.0,
            obs["drone_stop_id"].unsqueeze(-1) / 19.0,
            (obs["drone_task_id"].unsqueeze(-1) + 1.0) / 128.0,
            obs["drone_task_side"].unsqueeze(-1),
            obs["drone_eta_h"].unsqueeze(-1),
            obs["drone_batt_ratio"].unsqueeze(-1),
            obs["drone_xy"],
            obs["drone_payload_remain"].unsqueeze(-1),
            obs["drone_payload_failed"].unsqueeze(-1),
            (obs["drone_planned_stop"].unsqueeze(-1) + 1.0) / 20.0,
            obs["response_mode"].view(B, 1, 1).expand(B, D, 1),
        ], dim=-1)
        drone_tok = self.drone_proj(drone_feat) + self.type_embed.weight[2]
        drone_valid = torch.ones(B, D, dtype=torch.bool, device=task_tok.device)

        # ---------------- truck: [B, 1, 6]
        truck_feat = torch.cat([
            obs["truck_rel_pos"],
            obs["truck_stop"] / 19.0,
            obs["truck_busy"],
            obs["truck_eta_h"],
            obs["truck_target_stop"] / 19.0,
        ], dim=-1).unsqueeze(1)
        truck_tok = self.truck_proj(truck_feat) + self.type_embed.weight[3]
        truck_valid = torch.ones(B, 1, dtype=torch.bool, device=task_tok.device)

        # ---------------- global/env: [B, 1, 21]
        wind_flat = obs["wind_samples"].reshape(B, -1)
        kernel_flat = obs["wind_kernel_xy"].reshape(B, -1)
        global_feat = torch.cat([
            obs["time_progress"],
            obs["done_ratio"],
            obs["scale_id"] / 4.0,
            obs["wind_vec"],
            wind_flat,
            kernel_flat,
            obs["response_mode"],
        ], dim=-1).unsqueeze(1)
        global_tok = self.global_proj(global_feat) + self.type_embed.weight[4]
        global_valid = torch.ones(B, 1, dtype=torch.bool, device=task_tok.device)

        tokens = torch.cat([task_tok, stop_tok, drone_tok, truck_tok, global_tok], dim=1)
        valid = torch.cat([task_valid, stop_valid, drone_valid, truck_valid, global_valid], dim=1)
        pad_mask = ~valid
        enc = self.encoder(tokens, src_key_padding_mask=pad_mask)
        enc = self.out_norm(enc)

        denom = valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (enc * valid.unsqueeze(-1).float()).sum(dim=1) / denom

        # Return pooled context plus encoded drone tokens for per-UAV action heads.
        n_tasks = task_tok.shape[1]
        n_stops = stop_tok.shape[1]
        drone_enc = enc[:, n_tasks + n_stops:n_tasks + n_stops + D]
        return pooled, drone_enc, enc


class EDCDPLitePlanner(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int, nactions: int, dropout: float,
                 nheads: int, nlayers: int):
        super().__init__()
        self.nactions = nactions
        self.encoder = HeterogeneousStateEncoder(
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            nheads=nheads,
            nlayers=nlayers,
        )
        self.truck_head = nn.Sequential(
            nn.Linear(token_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, nactions)
        )
        self.drone_head = nn.Sequential(
            nn.Linear(token_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, nactions)
        )
        self.value_head = nn.Sequential(
            nn.Linear(token_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled, drone_enc, _ = self.encoder(obs)
        B, D, _ = drone_enc.shape
        truck_logits = self.truck_head(pooled).unsqueeze(1)
        pooled_expand = pooled.unsqueeze(1).expand(B, D, pooled.shape[-1])
        drone_logits = self.drone_head(torch.cat([pooled_expand, drone_enc], dim=-1))
        logits = torch.cat([truck_logits, drone_logits], dim=1)
        value = self.value_head(pooled).squeeze(-1)
        return logits, value

    def get_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        action_mask: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        B, Agt, _ = logits.shape
        dists = [MaskedCategorical(logits[:, i, :], action_mask[:, i, :]) for i in range(Agt)]
        if action is None:
            acts = [dist.mode() if deterministic else dist.sample() for dist in dists]
            action = torch.stack(acts, dim=1)
        logp = torch.stack([dists[i].log_prob(action[:, i]) for i in range(Agt)], dim=1).sum(dim=1)
        ent = torch.stack([dists[i].entropy() for i in range(Agt)], dim=1).sum(dim=1)
        return action, logp, ent, value

    def get_value(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(obs)[1]


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate_policy(model: EDCDPLitePlanner, cfg: EDCDPLiteConfig, device: torch.device,
                    episodes: int) -> Dict[str, float]:
    env, obs = make_env(cfg, seed=cfg.seed + 99999)
    returns, times, comps, succs, decisions = [], [], [], [], []
    ep_ret = 0.0
    finished = 0
    while finished < episodes:
        obs_t = to_tensor_batch([obs], device)
        mask_t = torch.as_tensor(stack_action_masks([obs]), dtype=torch.float32, device=device)
        action, _, _, _ = model.get_action_and_value(obs_t, mask_t, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
        ep_ret += float(reward)
        obs = next_obs
        if terminated or truncated:
            returns.append(ep_ret)
            times.append(float(info.get("time_h", 0.0)))
            comps.append(float(info.get("completion_rate", 0.0)))
            succs.append(float(info.get("success", 0.0)))
            decisions.append(float(info.get("decision_count", 0.0)))
            finished += 1
            ep_ret = 0.0
            obs, _ = env.reset(seed=cfg.seed + 99999 + finished)
    env.close()
    return {
        "eval/return_mean": float(np.mean(returns)) if returns else 0.0,
        "eval/time_h_mean": float(np.mean(times)) if times else 0.0,
        "eval/completion_rate_mean": float(np.mean(comps)) if comps else 0.0,
        "eval/success_rate": float(np.mean(succs)) if succs else 0.0,
        "eval/decision_count_mean": float(np.mean(decisions)) if decisions else 0.0,
    }


# ============================================================
# Main training
# ============================================================
def main(cfg: EDCDPLiteConfig | None = None) -> None:
    cfg = EDCDPLiteConfig() if cfg is None else cfg
    ensure_dir(cfg.save_dir)
    set_seed(cfg.seed, cfg.torch_deterministic)
    device = torch.device(cfg.device)

    if cfg.wandb_mode == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
    if wandb is not None:
        run = wandb.init(
            project=cfg.project,
            entity=cfg.entity,
            name=cfg.run_name,
            mode=cfg.wandb_mode,
            config=dataclasses.asdict(cfg),
            save_code=False,
        )
    else:
        run = None

    envs: List[TruckMultiDroneCleanEnv] = []
    obs_batch: List[Dict[str, np.ndarray]] = []
    for i in range(cfg.num_envs):
        env, obs = make_env(cfg, seed=cfg.seed + i)
        envs.append(env)
        obs_batch.append(obs)

    nactions = int(envs[0].nactions)
    nagents = int(envs[0].nagents)
    model = EDCDPLitePlanner(
        token_dim=cfg.token_dim,
        hidden_dim=cfg.hidden_dim,
        nactions=nactions,
        dropout=cfg.dropout,
        nheads=cfg.transformer_heads,
        nlayers=cfg.transformer_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)

    print(f"EDCDP-lite: nactions={nactions}, nagents={nagents}, ndrones={cfg.ndrones}, "
          f"params={count_parameters(model):,}, device={device}")
    if wandb is not None:
        wandb.config.update({
            "nactions": nactions,
            "nagents": nagents,
            "param_count": count_parameters(model),
        }, allow_val_change=True)

    T, N = cfg.rollout_steps, cfg.num_envs
    act_buf = np.zeros((T, N, nagents), dtype=np.int64)
    logp_buf = np.zeros((T, N), dtype=np.float32)
    rew_buf = np.zeros((T, N), dtype=np.float32)
    done_buf = np.zeros((T, N), dtype=np.float32)
    val_buf = np.zeros((T, N), dtype=np.float32)
    obs_buf: List[List[Dict[str, np.ndarray]]] = []
    mask_buf: List[np.ndarray] = []

    episode_returns = np.zeros(N, dtype=np.float32)
    episode_lengths = np.zeros(N, dtype=np.int32)
    recent_returns, recent_success, recent_time_h, recent_completion = (
        deque(maxlen=100), deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
    )

    global_step = 0
    best_eval_success = -1.0
    best_eval_time = float("inf")
    start_time = time.time()

    for update in range(1, cfg.total_updates + 1):
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1.0) / max(float(cfg.total_updates), 1.0)
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        obs_buf.clear()
        mask_buf.clear()

        for t in range(T):
            obs_snapshot = [{k: np.array(v, copy=True) for k, v in obs.items()} for obs in obs_batch]
            obs_buf.append(obs_snapshot)
            mask_np = stack_action_masks(obs_batch)
            mask_buf.append(mask_np.copy())

            obs_t = to_tensor_batch(obs_batch, device)
            mask_t = torch.as_tensor(mask_np, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_t, logprob_t, _, value_t = model.get_action_and_value(obs_t, mask_t, deterministic=False)

            action_np = action_t.cpu().numpy()
            act_buf[t] = action_np
            logp_buf[t] = logprob_t.cpu().numpy()
            val_buf[t] = value_t.cpu().numpy()

            next_obs_batch = []
            for i, env in enumerate(envs):
                next_obs, reward, terminated, truncated, info = env.step(action_np[i])
                done = bool(terminated or truncated)
                rew_buf[t, i] = float(reward)
                done_buf[t, i] = 1.0 if done else 0.0
                global_step += 1

                episode_returns[i] += float(reward)
                episode_lengths[i] += 1
                if done:
                    recent_returns.append(float(episode_returns[i]))
                    recent_success.append(float(info.get("success", 0.0)))
                    recent_time_h.append(float(info.get("time_h", 0.0)))
                    recent_completion.append(float(info.get("completion_rate", 0.0)))
                    if wandb is not None:
                        wandb.log({
                            "episode/return": float(episode_returns[i]),
                            "episode/length": int(episode_lengths[i]),
                            "episode/time_h": float(info.get("time_h", 0.0)),
                            "episode/completion_rate": float(info.get("completion_rate", 0.0)),
                            "episode/success": float(info.get("success", 0.0)),
                            "episode/decision_count": float(info.get("decision_count", 0.0)),
                            "episode/invalid_action_count": float(info.get("invalid_action_count", 0.0)),
                            "episode/claim_conflict_count": float(info.get("claim_conflict_count", 0.0)),
                            "global_step": global_step,
                        }, step=global_step)
                    episode_returns[i] = 0.0
                    episode_lengths[i] = 0
                    next_obs, _ = env.reset(seed=cfg.seed + 100000 + update * N + i)
                next_obs_batch.append(next_obs)
            obs_batch = next_obs_batch

        # Bootstrap and GAE.
        with torch.no_grad():
            next_value = model.get_value(to_tensor_batch(obs_batch, device)).cpu().numpy()
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

        flat_obs = [obs_buf[t][n] for t in range(T) for n in range(N)]
        flat_mask = np.concatenate(mask_buf, axis=0)
        flat_actions = act_buf.reshape(T * N, nagents)
        flat_adv = adv_buf.reshape(T * N)
        flat_ret = ret_buf.reshape(T * N)

        # Normalize advantages for value stability, but use unclipped positive AWR weights.
        norm_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        awr_weights = np.exp(np.clip(norm_adv / max(cfg.awr_temperature, 1e-6), -10.0, math.log(cfg.awr_max_weight)))

        batch_size = T * N
        minibatch_size = max(1, batch_size // cfg.num_minibatches)
        indices = np.arange(batch_size)
        policy_losses, value_losses, entropies = [], [], []

        for _epoch in range(cfg.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]
                mb_obs = [flat_obs[int(i)] for i in mb_idx]
                mb_obs_t = to_tensor_batch(mb_obs, device)
                mb_mask_t = torch.as_tensor(flat_mask[mb_idx], dtype=torch.float32, device=device)
                mb_act_t = torch.as_tensor(flat_actions[mb_idx], dtype=torch.long, device=device)
                mb_ret_t = torch.as_tensor(flat_ret[mb_idx], dtype=torch.float32, device=device)
                mb_w_t = torch.as_tensor(awr_weights[mb_idx], dtype=torch.float32, device=device)

                _, logp, entropy, value = model.get_action_and_value(mb_obs_t, mb_mask_t, action=mb_act_t)
                policy_loss = -(mb_w_t.detach() * logp).mean()
                value_loss = 0.5 * (value - mb_ret_t).pow(2).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_loss.item()))

        if update % cfg.log_every_updates == 0:
            sps = int(global_step / max(time.time() - start_time, 1e-6))
            log_data = {
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "charts/SPS": sps,
                "losses/awr_policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
                "losses/value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
                "losses/entropy": float(np.mean(entropies)) if entropies else 0.0,
                "rollout/reward_mean": float(rew_buf.mean()),
                "rollout/reward_std": float(rew_buf.std()),
                "rollout/value_mean": float(val_buf.mean()),
                "rollout/adv_mean": float(flat_adv.mean()),
                "rollout/adv_std": float(flat_adv.std()),
                "global_step": global_step,
                "update": update,
            }
            if recent_returns:
                log_data.update({
                    "recent/return_mean": float(np.mean(recent_returns)),
                    "recent/success_rate": float(np.mean(recent_success)),
                    "recent/time_h_mean": float(np.mean(recent_time_h)),
                    "recent/completion_rate_mean": float(np.mean(recent_completion)),
                })
            if wandb is not None:
                wandb.log(log_data, step=global_step)
            print(
                f"[EDCDP-lite upd {update:4d}] step={global_step:8d} "
                f"awr={log_data['losses/awr_policy_loss']:.4f} "
                f"v={log_data['losses/value_loss']:.4f} "
                f"ent={log_data['losses/entropy']:.4f} "
                f"succ={log_data.get('recent/success_rate', 0.0):.3f} "
                f"comp={log_data.get('recent/completion_rate_mean', 0.0):.3f} "
                f"time={log_data.get('recent/time_h_mean', 0.0):.3f}h"
            )

        if update % cfg.eval_every_updates == 0:
            eval_metrics = evaluate_policy(model, cfg, device, cfg.eval_episodes)
            eval_metrics["global_step"] = global_step
            eval_metrics["update"] = update
            if wandb is not None:
                wandb.log(eval_metrics, step=global_step)
            print(
                f"[EDCDP-lite eval {update:4d}] ret={eval_metrics['eval/return_mean']:.3f} "
                f"succ={eval_metrics['eval/success_rate']:.3f} "
                f"comp={eval_metrics['eval/completion_rate_mean']:.3f} "
                f"time={eval_metrics['eval/time_h_mean']:.3f}h"
            )
            improved = (
                eval_metrics["eval/success_rate"] > best_eval_success + 1e-9 or
                (abs(eval_metrics["eval/success_rate"] - best_eval_success) <= 1e-9 and
                 eval_metrics["eval/time_h_mean"] < best_eval_time)
            )
            if improved:
                best_eval_success = eval_metrics["eval/success_rate"]
                best_eval_time = eval_metrics["eval/time_h_mean"]
                best_path = Path(cfg.save_dir) / "best_edcdp_lite.pt"
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg": dataclasses.asdict(cfg),
                    "nactions": nactions,
                    "nagents": nagents,
                    "global_step": global_step,
                    "update": update,
                    "best_eval_success": best_eval_success,
                    "best_eval_time": best_eval_time,
                }, best_path)
                if wandb is not None:
                    wandb.save(str(best_path))

        if update % cfg.save_every_updates == 0:
            save_path = Path(cfg.save_dir) / f"edcdp_lite_update_{update}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg": dataclasses.asdict(cfg),
                "nactions": nactions,
                "nagents": nagents,
                "global_step": global_step,
                "update": update,
            }, save_path)

    final_path = Path(cfg.save_dir) / "final_edcdp_lite.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": dataclasses.asdict(cfg),
        "nactions": nactions,
        "nagents": nagents,
        "global_step": global_step,
        "update": cfg.total_updates,
        "best_eval_success": best_eval_success,
        "best_eval_time": best_eval_time,
    }, final_path)
    if wandb is not None:
        wandb.save(str(final_path))
        run.finish()
    for env in envs:
        env.close()
    print(f"EDCDP-lite training finished. Final model saved to: {final_path}")


if __name__ == "__main__":
    cfg = EDCDPLiteConfig(
        run_name=f"edcdp_lite_seed42_{int(time.time())}",
        fixed_scale="XS",
        ndrones=3,
        num_envs=4,
        rollout_steps=128,
        total_updates=500,
        eval_every_updates=25,
        save_every_updates=50,
        wandb_mode="online",
    )
    main(cfg)
