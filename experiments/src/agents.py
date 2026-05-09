from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


DEVICE = torch.device("cpu")


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]] = deque(maxlen=capacity)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        bootstrap_mask: np.ndarray,
    ) -> None:
        self.buffer.append((obs.copy(), action, reward, next_obs.copy(), done, bootstrap_mask.copy()))

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        obs, actions, rewards, next_obs, dones, masks = zip(*batch)
        return (
            torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=DEVICE),
            torch.as_tensor(np.asarray(actions), dtype=torch.int64, device=DEVICE),
            torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=DEVICE),
            torch.as_tensor(np.asarray(next_obs), dtype=torch.float32, device=DEVICE),
            torch.as_tensor(np.asarray(dones), dtype=torch.float32, device=DEVICE),
            torch.as_tensor(np.asarray(masks), dtype=torch.float32, device=DEVICE),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x)
        return self.policy(hidden), self.value(hidden).squeeze(-1)


class EnsembleQNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, ensemble_size: int = 5, hidden_dim: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, action_dim) for _ in range(ensemble_size)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(x)
        outputs = [head(hidden) for head in self.heads]
        return torch.stack(outputs, dim=1)


@dataclass
class AgentConfig:
    gamma: float = 0.98
    lr: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 20000
    warmup_steps: int = 200
    target_update_interval: int = 80
    gradient_update_interval: int = 1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 4000
    intrinsic_scale: float = 0.04
    rnd_scale: float = 0.03
    ensemble_size: int = 5
    lambda_aleatoric: float = 0.18
    alpha_gate: float = 0.14
    sigma0_sq: float = 0.20
    aleatoric_decay: float = 0.95
    probe_horizon: int = 3
    epistemic_prior_floor: float = 0.35


def valid_action_indices_from_obs(obs: np.ndarray, action_dim: int) -> List[int]:
    visible_queue = action_dim - 2
    valid_actions: List[int] = []
    available_cpu = max(0.0, float(obs[0])) * 8.0
    available_mem = max(0.0, float(obs[1])) * 8.0
    for idx in range(visible_queue):
        cpu_idx = 7 + (4 * idx)
        mem_idx = cpu_idx + 1
        duration_idx = 7 + (4 * idx) + 2
        if duration_idx < len(obs) and float(obs[duration_idx]) > 1e-6:
            cpu = max(0.0, float(obs[cpu_idx])) * 8.0
            mem = max(0.0, float(obs[mem_idx])) * 8.0
            if cpu > available_cpu or mem > available_mem:
                continue
            valid_actions.append(idx)
    valid_actions.append(action_dim - 2)
    if float(obs[4]) > 1e-6:
        valid_actions.append(action_dim - 1)
    return valid_actions


def valid_action_mask_tensor(obs_batch: torch.Tensor, action_dim: int) -> torch.Tensor:
    batch_size = obs_batch.shape[0]
    mask = torch.zeros((batch_size, action_dim), dtype=torch.bool, device=obs_batch.device)
    visible_queue = action_dim - 2
    available_cpu = torch.clamp(obs_batch[:, 0], min=0.0) * 8.0
    available_mem = torch.clamp(obs_batch[:, 1], min=0.0) * 8.0
    for idx in range(visible_queue):
        cpu_idx = 7 + (4 * idx)
        mem_idx = cpu_idx + 1
        duration_idx = 7 + (4 * idx) + 2
        if duration_idx < obs_batch.shape[1]:
            cpu = torch.clamp(obs_batch[:, cpu_idx], min=0.0) * 8.0
            mem = torch.clamp(obs_batch[:, mem_idx], min=0.0) * 8.0
            mask[:, idx] = (obs_batch[:, duration_idx] > 1e-6) & (cpu <= available_cpu) & (mem <= available_mem)
    mask[:, action_dim - 2] = True
    mask[:, action_dim - 1] = obs_batch[:, 4] > 1e-6
    return mask


def mask_logits(logits: torch.Tensor, valid_actions: List[int]) -> torch.Tensor:
    masked = logits.clone()
    invalid = torch.ones_like(masked, dtype=torch.bool)
    invalid[valid_actions] = False
    masked[invalid] = torch.finfo(masked.dtype).min
    return masked


def masked_argmax(values: np.ndarray, valid_actions: List[int]) -> int:
    best_action = valid_actions[0]
    best_value = float(values[best_action])
    for action in valid_actions[1:]:
        value = float(values[action])
        if value > best_value:
            best_action = action
            best_value = value
    return int(best_action)


class HeuristicAgent:
    name = "heuristic_sjf_bestfit"

    def __init__(self, action_defer_index: int, action_reject_index: int) -> None:
        self.defer_index = action_defer_index
        self.reject_index = action_reject_index

    def begin_episode(self, training: bool) -> None:
        return None

    def select_action(self, obs: np.ndarray, action_mask: List[Tuple[float, float, float, float]], training: bool) -> int:
        best_idx = None
        best_key = None
        available_cpu = max(0.0, obs[0]) * 8.0
        available_mem = max(0.0, obs[1]) * 8.0
        queue_frac = max(0.0, min(1.0, obs[4]))
        for idx, slot in enumerate(action_mask):
            cpu, mem, duration, _wait = slot
            if duration <= 0:
                continue
            if cpu <= available_cpu and mem <= available_mem:
                fit_score = (cpu + mem) / max(1e-6, available_cpu + available_mem)
                key = (duration, -fit_score)
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = idx
        if best_idx is not None:
            return best_idx
        visible_tasks = [(idx, slot) for idx, slot in enumerate(action_mask) if slot[2] > 0]
        if not visible_tasks:
            return self.defer_index
        if queue_frac > 0.7:
            reject_idx = max(visible_tasks, key=lambda item: (item[1][2], item[1][3]))[0]
            return reject_idx
        return self.defer_index

    def observe_transition(self, *args, **kwargs) -> Dict[str, float]:
        return {}

    def end_episode(self, *args, **kwargs) -> Dict[str, float]:
        return {}


class PolicyGradientAgent:
    name = "deeprm_inspired_pg"

    def __init__(self, state_dim: int, action_dim: int, seed: int, config: Optional[AgentConfig] = None) -> None:
        set_global_seeds(seed)
        self.config = config or AgentConfig()
        self.net = ActorCritic(state_dim, action_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr)
        self.memory: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = []
        self.action_dim = action_dim
        self.entropy_coef = 0.01
        self.value_coef = 0.5

    def begin_episode(self, training: bool) -> None:
        self.memory = []

    def select_action(self, obs: np.ndarray, action_mask: List[Tuple[float, float, float, float]], training: bool) -> int:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits, value = self.net(obs_tensor)
        logits = logits.squeeze(0)
        valid_actions = valid_action_indices_from_obs(obs, self.action_dim)
        probs = torch.softmax(mask_logits(logits, valid_actions), dim=-1)
        dist = Categorical(probs=probs)
        if training:
            action = int(dist.sample().item())
        else:
            action = int(torch.argmax(probs).item())
        self.last_log_prob = dist.log_prob(torch.as_tensor(action, device=DEVICE))
        self.last_value = value.squeeze(0)
        self.last_entropy = dist.entropy()
        return action

    def observe_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        training: bool,
        adapt: bool,
        episode_step: Optional[int] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, float]:
        if training or adapt:
            self.memory.append((self.last_log_prob, self.last_value, self.last_entropy, float(reward)))
        return {}

    def end_episode(self, training: bool, adapt: bool) -> Dict[str, float]:
        if not (training or adapt) or not self.memory:
            return {}
        returns = []
        running = 0.0
        for *_unused, reward in reversed(self.memory):
            running = reward + self.config.gamma * running
            returns.append(running)
        returns = torch.as_tensor(list(reversed(returns)), dtype=torch.float32, device=DEVICE)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        actor_losses = []
        value_losses = []
        entropies = []
        for (log_prob, value, entropy, _reward), ret in zip(self.memory, returns):
            advantage = ret - value
            actor_losses.append(-(log_prob * advantage.detach()))
            value_losses.append(advantage.pow(2))
            entropies.append(entropy)
        loss = torch.stack(actor_losses).mean()
        loss = loss + self.value_coef * torch.stack(value_losses).mean()
        loss = loss - self.entropy_coef * torch.stack(entropies).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()
        return {"pg_loss": float(loss.item())}


class QControlAgent:
    def __init__(
        self,
        name: str,
        state_dim: int,
        action_dim: int,
        seed: int,
        config: Optional[AgentConfig] = None,
        rnd: bool = False,
        psaim: bool = False,
        no_aleatoric: bool = False,
        no_gate: bool = False,
        no_freezing: bool = False,
    ) -> None:
        set_global_seeds(seed)
        self.name = name
        self.config = config or AgentConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.visible_queue = action_dim - 2
        self.defer_index = action_dim - 2
        self.reject_index = action_dim - 1
        self.rnd = rnd
        self.psaim = psaim
        self.no_aleatoric = no_aleatoric
        self.no_gate = no_gate
        self.no_freezing = no_freezing
        self.global_step = 0
        self.epsilon = self.config.epsilon_start
        self.buffer = ReplayBuffer(self.config.buffer_size)
        self.training_mode = True
        self.last_signal: Dict[str, float] = {}
        self.pending_transitions: Deque[Dict[str, Any]] = deque()
        self.completed_signal_rows: List[Dict[str, float]] = []
        self.transition_stats: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        if self.psaim:
            self.net = EnsembleQNetwork(state_dim, action_dim, ensemble_size=self.config.ensemble_size).to(DEVICE)
            self.target_net = copy.deepcopy(self.net).to(DEVICE)
            self.aleatoric_ema = np.zeros(action_dim, dtype=np.float32)
            self.aleatoric_seen = np.zeros(action_dim, dtype=np.float32)
        else:
            self.net = MLP(state_dim, action_dim).to(DEVICE)
            self.target_net = copy.deepcopy(self.net).to(DEVICE)
            self.aleatoric_ema = np.zeros(action_dim, dtype=np.float32)
            self.aleatoric_seen = np.zeros(action_dim, dtype=np.float32)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr)
        if self.rnd:
            self.rnd_target = MLP(state_dim, 32, hidden_dim=64).to(DEVICE)
            self.rnd_predictor = MLP(state_dim, 32, hidden_dim=64).to(DEVICE)
            for param in self.rnd_target.parameters():
                param.requires_grad = False
            self.rnd_optimizer = torch.optim.Adam(self.rnd_predictor.parameters(), lr=self.config.lr)
            self.rnd_running_mean = 0.0
            self.rnd_running_var = 1.0
            self.rnd_updates = 0

    def begin_episode(self, training: bool) -> None:
        self.training_mode = training
        self.pending_transitions = deque()
        self.completed_signal_rows = []
        self.last_signal = {}

    def _epsilon_value(self, training: bool, adapt: bool) -> float:
        if training:
            frac = min(1.0, self.global_step / max(1, self.config.epsilon_decay_steps))
            return self.config.epsilon_start + frac * (self.config.epsilon_end - self.config.epsilon_start)
        if adapt:
            return 0.05
        return 0.01

    def _q_values(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        if self.psaim:
            return self.net(obs_tensor).mean(dim=1)
        return self.net(obs_tensor)

    def select_action(self, obs: np.ndarray, action_mask: List[Tuple[float, float, float, float]], training: bool, adapt: bool = False) -> int:
        self.epsilon = self._epsilon_value(training, adapt)
        feasible_actions = valid_action_indices_from_obs(obs, self.action_dim)
        if np.random.random() < self.epsilon:
            return int(np.random.choice(feasible_actions))
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q_values = self._q_values(obs_tensor).squeeze(0).cpu().numpy()
        return masked_argmax(q_values, feasible_actions)

    def _bootstrap_mask(self) -> np.ndarray:
        if not self.psaim:
            return np.ones((1,), dtype=np.float32)
        mask = np.random.binomial(1, 0.8, size=self.config.ensemble_size).astype(np.float32)
        if mask.sum() == 0:
            mask[np.random.randint(0, self.config.ensemble_size)] = 1.0
        return mask

    def _rnd_bonus(self, next_obs: np.ndarray) -> float:
        obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            target = self.rnd_target(obs_tensor)
        pred = self.rnd_predictor(obs_tensor)
        mse = F.mse_loss(pred, target).item()
        self.rnd_updates += 1
        delta = mse - self.rnd_running_mean
        self.rnd_running_mean += delta / self.rnd_updates
        self.rnd_running_var += delta * (mse - self.rnd_running_mean)
        std = max(1e-6, np.sqrt(self.rnd_running_var / max(1, self.rnd_updates)))
        return max(0.0, (mse - self.rnd_running_mean) / std)

    def _state_action_key(self, obs: np.ndarray, action: int) -> Tuple[int, ...]:
        return (
            int(np.clip(obs[0] * 4.0, 0, 3)),
            int(np.clip(obs[1] * 4.0, 0, 3)),
            int(np.clip(obs[4] * 4.0, 0, 3)),
            int(np.clip(obs[5] * 4.0, 0, 3)),
            int(action),
        )

    def _source_net(self) -> nn.Module:
        return self.net if self.no_freezing else self.target_net

    def _update_transition_stats(self, key: Tuple[int, ...], transition_features: np.ndarray) -> Dict[str, Any]:
        stats = self.transition_stats.setdefault(
            key,
            {
                "count": 0.0,
                "mean": np.zeros(self.state_dim, dtype=np.float32),
                "m2": np.zeros(self.state_dim, dtype=np.float32),
            },
        )
        count = float(stats["count"]) + 1.0
        delta = transition_features - stats["mean"]
        stats["mean"] = stats["mean"] + (delta / count)
        delta2 = transition_features - stats["mean"]
        stats["m2"] = stats["m2"] + (delta * delta2)
        stats["count"] = count
        variance = stats["m2"] / max(1.0, count - 1.0) if count > 1.0 else np.zeros_like(stats["mean"])
        return {
            "count": count,
            "mean": stats["mean"].copy(),
            "variance": np.maximum(variance, 0.0),
        }

    def _transition_variance(self, key: Tuple[int, ...]) -> Dict[str, Any]:
        stats = self.transition_stats.get(key)
        if stats is None:
            zeros = np.zeros(self.state_dim, dtype=np.float32)
            return {"count": 0.0, "mean": zeros.copy(), "variance": zeros.copy()}
        count = float(stats.get("count", 0.0))
        mean = np.asarray(stats.get("mean", np.zeros(self.state_dim, dtype=np.float32)), dtype=np.float32)
        m2 = np.asarray(stats.get("m2", np.zeros(self.state_dim, dtype=np.float32)), dtype=np.float32)
        variance = m2 / max(1.0, count - 1.0) if count > 1.0 else np.zeros_like(mean)
        return {"count": count, "mean": mean.copy(), "variance": np.maximum(variance, 0.0)}

    def _masked_head_max(self, q_heads: torch.Tensor, obs: np.ndarray) -> torch.Tensor:
        valid_actions = valid_action_indices_from_obs(obs, self.action_dim)
        action_indices = torch.as_tensor(valid_actions, dtype=torch.int64, device=q_heads.device)
        return q_heads[:, action_indices].max(dim=1).values

    def _probe_action(self, q_heads: torch.Tensor, obs: np.ndarray) -> int:
        valid_actions = valid_action_indices_from_obs(obs, self.action_dim)
        mean_values = q_heads.mean(dim=0).detach().cpu().numpy()
        return masked_argmax(mean_values, valid_actions)

    def _epistemic_value(self, q_heads: torch.Tensor, obs: np.ndarray, action: int) -> float:
        key = self._state_action_key(obs, action)
        stats = self._transition_variance(key)
        disagreement = max(0.0, float(torch.var(q_heads[:, action], unbiased=False).item()))
        q_scale = max(1.0, float(torch.mean(torch.abs(q_heads[:, action])).item()))
        normalized_disagreement = np.sqrt(disagreement) / (q_scale + np.sqrt(disagreement) + 1e-6)
        novelty = 1.0 / np.sqrt(stats["count"] + 1.0)
        prior_floor = float(np.clip(self.config.epistemic_prior_floor, 0.0, 1.0))
        calibrated_disagreement = max(prior_floor, float(normalized_disagreement))
        return float(np.clip(calibrated_disagreement * novelty, 0.0, 1.0))

    def _aleatoric_value(self, key: Tuple[int, ...]) -> float:
        stats = self._transition_variance(key)
        if stats["count"] < 2.0:
            return 0.0
        mean = np.asarray(stats["mean"], dtype=np.float32)
        variance = np.asarray(stats["variance"], dtype=np.float32)
        dynamic_indices = np.asarray([0, 1, 2, 3, 4, 5] + list(range(7, self.state_dim)), dtype=np.int64)
        mean = mean[dynamic_indices]
        variance = variance[dynamic_indices]
        scale = np.maximum((np.abs(mean) + 1.0) ** 2, 1e-6)
        normalized_variance = variance / scale
        return float(np.clip(np.mean(np.sqrt(np.maximum(normalized_variance, 0.0))), 0.0, 1.0))

    def _aleatoric_baseline(self, action: int, v_ale: float) -> float:
        if self.aleatoric_seen[action] <= 0.0:
            return float(v_ale)
        return float(self.aleatoric_ema[action])

    def _aleatoric_excess(self, action: int, v_ale: float) -> float:
        baseline = self._aleatoric_baseline(action, v_ale)
        return float(max(0.0, v_ale - baseline))

    def _update_aleatoric_baseline(self, action: int, v_ale: float) -> None:
        if self.aleatoric_seen[action] <= 0.0:
            self.aleatoric_ema[action] = float(v_ale)
            self.aleatoric_seen[action] = 1.0
            return
        decay = float(np.clip(self.config.aleatoric_decay, 0.0, 0.999))
        self.aleatoric_ema[action] = (decay * self.aleatoric_ema[action]) + ((1.0 - decay) * float(v_ale))

    def _probe_gate_value(self, future_obs: List[np.ndarray]) -> float:
        if self.no_gate or not future_obs:
            return 0.0
        probe_scores: List[float] = []
        source_net = self._source_net()
        with torch.no_grad():
            for probe_obs in future_obs[: self.config.probe_horizon]:
                obs_tensor = torch.as_tensor(probe_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                q_heads = source_net(obs_tensor).squeeze(0)
                probe_action = self._probe_action(q_heads, probe_obs)
                probe_scores.append(self._epistemic_value(q_heads, probe_obs, probe_action))
        return float(np.mean(probe_scores)) if probe_scores else 0.0

    def diagnostic_signal(self, obs: np.ndarray, action: Optional[int] = None, future_obs: Optional[List[np.ndarray]] = None) -> Dict[str, float]:
        if not self.psaim:
            return {}
        source_net = self._source_net()
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q_heads = source_net(obs_tensor).squeeze(0)
        selected_action = self._probe_action(q_heads, obs) if action is None else int(action)
        key = self._state_action_key(obs, selected_action)
        v_epi = self._epistemic_value(q_heads, obs, selected_action)
        v_ale = 0.0 if self.no_aleatoric else self._aleatoric_value(key)
        gate_value = 1.0 if self.no_gate else float(
            1.0
            / (
                1.0
                + np.exp(
                    (
                        self._probe_gate_value(future_obs or [obs.copy()])
                        - self.config.alpha_gate
                    )
                    / max(0.02, self.config.sigma0_sq)
                )
            )
        )
        penalty_weight = 1.0 if self.no_gate else 1.0 - gate_value
        v_ale_excess = 0.0 if self.no_aleatoric else self._aleatoric_excess(selected_action, v_ale)
        penalty = 0.0
        if not self.no_aleatoric:
            penalty = self.config.lambda_aleatoric * np.log1p(
                (v_ale_excess * penalty_weight) / max(self.config.sigma0_sq, 1e-6)
            )
        return {
            "action": float(selected_action),
            "V_epi": float(v_epi),
            "V_ale": float(v_ale),
            "V_ale_excess": float(v_ale_excess),
            "gate_h3": float(gate_value),
            "r_int": float((v_epi * gate_value) - penalty),
        }

    def _prepare_psaim_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        episode_step: Optional[int],
        regime: Optional[str],
    ) -> Dict[str, Any]:
        source_net = self._source_net()
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q_heads = source_net(obs_tensor).squeeze(0)
        key = self._state_action_key(obs, action)
        v_epi = self._epistemic_value(q_heads, obs, action)
        v_ale = 0.0 if self.no_aleatoric else self._aleatoric_value(key)
        return {
            "obs": obs.copy(),
            "action": int(action),
            "reward": float(reward),
            "next_obs": next_obs.copy(),
            "done": bool(done),
            "bootstrap_mask": self._bootstrap_mask(),
            "key": key,
            "transition_features": next_obs.copy(),
            "future_obs": [next_obs.copy()],
            "episode_step": int(episode_step or 0),
            "regime": regime or "unknown",
            "V_epi": float(v_epi),
            "V_ale": float(v_ale),
        }

    def _record_replay_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        bootstrap_mask: np.ndarray,
    ) -> None:
        self.buffer.add(obs, action, float(reward), next_obs, done, bootstrap_mask)
        self.global_step += 1
        update_interval = max(1, int(self.config.gradient_update_interval))
        if len(self.buffer) >= self.config.warmup_steps and self.global_step % update_interval == 0:
            self._update_network()

    def _finalize_psaim_transition(self, item: Dict[str, Any], training: bool, adapt: bool) -> Dict[str, float]:
        future_epi = self._probe_gate_value(item["future_obs"])
        if self.no_gate:
            gate_value = 1.0
            penalty_weight = 1.0
        else:
            gate_temp = max(0.02, self.config.sigma0_sq)
            gate_value = float(1.0 / (1.0 + np.exp((future_epi - self.config.alpha_gate) / gate_temp)))
            penalty_weight = 1.0 - gate_value
        aleatoric_excess = 0.0 if self.no_aleatoric else self._aleatoric_excess(item["action"], item["V_ale"])
        penalty = 0.0
        if not self.no_aleatoric:
            penalty = self.config.lambda_aleatoric * np.log1p(
                (aleatoric_excess * penalty_weight) / max(self.config.sigma0_sq, 1e-6)
            )
        intrinsic = float(item["V_epi"] * gate_value - penalty)
        signal = {
            "V_epi": float(item["V_epi"]),
            "V_ale": float(item["V_ale"]),
            "V_ale_excess": float(aleatoric_excess),
            "gate_h3": float(gate_value),
            "r_int": intrinsic,
        }
        if training or adapt:
            self._update_transition_stats(item["key"], item["transition_features"])
            if not self.no_aleatoric:
                self._update_aleatoric_baseline(item["action"], item["V_ale"])
            shaped_reward = float(item["reward"] + self.config.intrinsic_scale * intrinsic)
            self._record_replay_transition(
                item["obs"],
                item["action"],
                shaped_reward,
                item["next_obs"],
                item["done"],
                item["bootstrap_mask"],
            )
        self.last_signal = signal
        self.completed_signal_rows.append(
            {
                "episode": float(item["episode_step"]),
                "regime": str(item["regime"]),
                "action": float(item["action"]),
                **signal,
            }
        )
        return signal

    def _flush_psaim_transitions(self, training: bool, adapt: bool, done: bool) -> Dict[str, float]:
        signal: Dict[str, float] = {}
        if done:
            while self.pending_transitions:
                signal = self._finalize_psaim_transition(self.pending_transitions.popleft(), training, adapt)
            return signal

        while self.pending_transitions and len(self.pending_transitions[0]["future_obs"]) >= self.config.probe_horizon:
            signal = self._finalize_psaim_transition(self.pending_transitions.popleft(), training, adapt)
        return signal

    def observe_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        training: bool,
        adapt: bool,
        episode_step: Optional[int] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, float]:
        shaped_reward = float(reward)
        signal: Dict[str, float] = {}
        if self.rnd:
            bonus = self._rnd_bonus(next_obs)
            shaped_reward += self.config.rnd_scale * bonus
            signal["r_int"] = float(bonus)
        if self.psaim:
            for pending in self.pending_transitions:
                if len(pending["future_obs"]) < self.config.probe_horizon:
                    pending["future_obs"].append(next_obs.copy())
            self.pending_transitions.append(
                self._prepare_psaim_transition(obs, action, reward, next_obs, done, episode_step, regime)
            )
            signal = self._flush_psaim_transitions(training=training, adapt=adapt, done=done)
        elif training or adapt:
            self._record_replay_transition(obs, action, shaped_reward, next_obs, done, self._bootstrap_mask())
        self.last_signal = signal
        return signal

    def _update_network(self) -> None:
        obs, actions, rewards, next_obs, dones, masks = self.buffer.sample(self.config.batch_size)
        if self.psaim:
            all_q = self.net(obs)
            q_sa = all_q.gather(2, actions.view(-1, 1, 1).expand(-1, self.config.ensemble_size, 1)).squeeze(-1)
            with torch.no_grad():
                target_heads = self.target_net(next_obs)
                valid_mask = valid_action_mask_tensor(next_obs, self.action_dim).unsqueeze(1)
                masked_heads = target_heads.masked_fill(~valid_mask, torch.finfo(target_heads.dtype).min)
                next_max = masked_heads.max(dim=2).values
                targets = rewards.unsqueeze(1) + self.config.gamma * (1.0 - dones.unsqueeze(1)) * next_max
            mask = masks
            loss = ((q_sa - targets).pow(2) * mask).sum() / mask.sum().clamp_min(1.0)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()
        else:
            q_values = self.net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                target_q = self.target_net(next_obs)
                valid_mask = valid_action_mask_tensor(next_obs, self.action_dim)
                target_max = target_q.masked_fill(~valid_mask, torch.finfo(target_q.dtype).min).max(dim=1).values
                targets = rewards + self.config.gamma * (1.0 - dones) * target_max
            loss = F.mse_loss(q_values, targets)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.rnd:
                with torch.no_grad():
                    target = self.rnd_target(next_obs)
                pred = self.rnd_predictor(next_obs)
                rnd_loss = F.mse_loss(pred, target)
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
                self.rnd_optimizer.step()

        if self.global_step % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.net.state_dict())

    def end_episode(self, training: bool, adapt: bool) -> Dict[str, float]:
        if self.psaim and self.pending_transitions:
            self._flush_psaim_transitions(training=training, adapt=adapt, done=True)
        return self.last_signal.copy()

    def consume_signal_rows(self) -> List[Dict[str, float]]:
        rows = self.completed_signal_rows[:]
        self.completed_signal_rows = []
        return rows
