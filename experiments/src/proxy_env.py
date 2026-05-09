from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Task:
    task_id: int
    cpu: int
    memory: int
    duration: int
    arrival_step: int
    remaining: int
    start_step: Optional[int] = None
    completion_step: Optional[int] = None
    dropped: bool = False
    drop_reason: str = ""
    wait_steps: int = 0


@dataclass
class StressConfig:
    """Evaluation-time perturbations used by the proxy study.

    Stressors are intentionally parameterized in one place so paper tables,
    reruns, and reviewer-control experiments all exercise the same simulator
    rather than silently diverging across scripts.
    """

    scenario: Optional[str] = None
    severity: str = "clean"
    observation_sigma: float = 0.0
    reward_bias: float = 0.0
    reward_delay: int = 0
    burst_level: int = 0
    co_tenant_mode: str = "none"
    co_tenant_mean_cpu: float = 0.0
    co_tenant_mean_memory: float = 0.0
    co_tenant_spread: float = 0.0
    duration_misreport_strength: float = 0.0


def make_stress_config(scenario: Optional[str], severity: str) -> StressConfig:
    if scenario in (None, "clean"):
        return StressConfig(scenario="clean", severity="clean")

    severity_map = {"low": 1, "medium": 2, "high": 3}
    rank = severity_map[severity]

    if scenario == "observation_noise":
        return StressConfig(
            scenario=scenario,
            severity=severity,
            observation_sigma={1: 0.08, 2: 0.16, 3: 0.30}[rank],
        )

    if scenario == "reward_corruption":
        return StressConfig(
            scenario=scenario,
            severity=severity,
            reward_bias={1: 0.50, 2: 1.00, 3: 1.50}[rank],
            reward_delay={1: 8, 2: 16, 3: 24}[rank],
        )

    if scenario == "distribution_shift":
        return StressConfig(scenario=scenario, severity=severity, burst_level=rank)

    if scenario == "co_tenant_interference":
        return StressConfig(
            scenario=scenario,
            severity=severity,
            co_tenant_mode="random_bursty",
            co_tenant_mean_cpu={1: 0.5, 2: 1.0, 3: 1.5}[rank],
            co_tenant_mean_memory={1: 0.5, 2: 1.0, 3: 1.5}[rank],
            co_tenant_spread={1: 0.35, 2: 0.65, 3: 0.95}[rank],
        )

    if scenario == "co_tenant_matched_load_control":
        return StressConfig(
            scenario=scenario,
            severity=severity,
            co_tenant_mode="matched_periodic",
            co_tenant_mean_cpu={1: 0.5, 2: 1.0, 3: 1.5}[rank],
            co_tenant_mean_memory={1: 0.5, 2: 1.0, 3: 1.5}[rank],
            co_tenant_spread=0.0,
        )

    if scenario == "duration_misreport":
        return StressConfig(
            scenario=scenario,
            severity=severity,
            # The heuristic-breaking stress corrupts only the observed duration
            # feature. Ground-truth task duration is still used for completion
            # and slowdown metrics, which isolates input-ordering fragility.
            duration_misreport_strength={1: 0.15, 2: 0.35, 3: 0.60}[rank],
        )

    raise ValueError(f"Unknown scenario: {scenario}")


class WorkloadGenerator:
    def __init__(
        self,
        seed: int,
        mode: str,
        burst_level: int = 0,
        alternating_block: int = 24,
        drift_horizon: int = 96,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.mode = mode
        self.burst_level = burst_level
        self.alternating_block = alternating_block
        self.drift_horizon = drift_horizon
        self.in_burst = False

    def regime_for_step(self, step: int) -> str:
        if self.mode == "alternating":
            return "low_entropy" if (step // self.alternating_block) % 2 == 0 else "high_entropy"
        if self.mode in {"periodic", "low_entropy"}:
            return "low_entropy"
        return "high_entropy"

    def sample_arrivals(self, step: int) -> Tuple[str, List[Tuple[int, int, int]]]:
        if self.mode == "monotonic_drift":
            high_probability = np.clip(step / max(1, self.drift_horizon - 1), 0.0, 1.0)
            regime = "high_entropy" if self.rng.random() < high_probability else "low_entropy"
            if regime == "low_entropy":
                return regime, self._sample_low_entropy(step)
            return regime, self._sample_high_entropy(step)
        regime = self.regime_for_step(step)
        if regime == "low_entropy":
            return regime, self._sample_low_entropy(step)
        return regime, self._sample_high_entropy(step)

    def _sample_low_entropy(self, step: int) -> List[Tuple[int, int, int]]:
        if self.mode == "periodic":
            periodic_pattern = {
                0: [(3, 3, 4), (2, 1, 2)],
                1: [(1, 2, 3)],
                2: [(4, 3, 5)],
                4: [(2, 4, 4), (1, 1, 2)],
            }
            return list(periodic_pattern.get(step % 6, []))
        if step % 2 == 0:
            return [(2, 2, 3)]
        return []

    def _sample_high_entropy(self, step: int) -> List[Tuple[int, int, int]]:
        level = max(1, self.burst_level or 2)
        p_on = 0.12 + 0.05 * level
        p_off = 0.15 + 0.03 * level
        if self.in_burst:
            self.in_burst = self.rng.random() > p_off
        else:
            self.in_burst = self.rng.random() < p_on
        lam = 0.80 + 0.60 * level if self.in_burst else 0.25 + 0.10 * level
        num_arrivals = min(4, int(self.rng.poisson(lam)))
        arrivals: List[Tuple[int, int, int]] = []
        for _ in range(num_arrivals):
            cpu = int(np.clip(1 + self.rng.integers(0, 3) + (1 if self.rng.random() < 0.15 * level else 0), 1, 4))
            memory = int(np.clip(1 + self.rng.integers(0, 3) + (1 if self.rng.random() < 0.12 * level else 0), 1, 4))
            duration = int(np.clip(np.round(self.rng.lognormal(mean=0.95 + 0.04 * level, sigma=0.25 + 0.06 * level)), 2, 7))
            arrivals.append((cpu, memory, duration))
        return arrivals


class ProxyAllocationEnv:
    def __init__(
        self,
        seed: int,
        workload_mode: str,
        horizon: int = 96,
        cpu_capacity: int = 8,
        memory_capacity: int = 8,
        queue_capacity: int = 12,
        visible_queue: int = 5,
        alternating_block: int = 24,
        stress: Optional[StressConfig] = None,
    ) -> None:
        self.base_seed = seed
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.horizon = horizon
        self.queue_capacity = queue_capacity
        self.visible_queue = visible_queue
        self.action_space_n = visible_queue + 2
        self.stress = stress or StressConfig(scenario="clean", severity="clean")
        self.workload_mode = workload_mode
        self.alternating_block = alternating_block
        self.rng = np.random.default_rng(seed)
        self.generator = WorkloadGenerator(
            seed=seed + 101,
            mode=workload_mode,
            burst_level=self.stress.burst_level,
            alternating_block=alternating_block,
            drift_horizon=horizon,
        )
        self.max_duration = 9
        self.max_queue_wait = horizon
        self.invalid_action_penalty = 0.50
        self.reject_penalty = 2.00
        self.defer_penalty = 0.02
        self.reset_counter = 0
        self._reset_state()

    def _reset_state(self) -> None:
        self.step_index = 0
        self.task_counter = 0
        self.queue: List[Task] = []
        self.running: List[Task] = []
        self.all_tasks: List[Task] = []
        self.utilization_trace: List[float] = []
        self.regime_trace: List[str] = []
        self.last_regime = "low_entropy"
        self.current_reserved_cpu = 0.0
        self.current_reserved_memory = 0.0

    def reset(self) -> Tuple[np.ndarray, Dict[str, float]]:
        self._reset_state()
        episode_seed = self.base_seed + 997 * self.reset_counter
        self.reset_counter += 1
        self.rng = np.random.default_rng(episode_seed)
        self.generator = WorkloadGenerator(
            seed=episode_seed + 101,
            mode=self.workload_mode,
            burst_level=self.stress.burst_level,
            alternating_block=self.alternating_block,
            drift_horizon=self.horizon,
        )
        self._refresh_external_load()
        self._inject_arrivals()
        return self._observe(), {"regime": self.last_regime}

    def _co_tenant_reservation(self) -> Tuple[float, float]:
        if self.stress.co_tenant_mode == "none":
            return 0.0, 0.0
        phase = 2.0 * np.pi * (self.step_index % 12) / 12.0
        if self.stress.co_tenant_mode == "matched_periodic":
            cpu = self.stress.co_tenant_mean_cpu * (0.75 + 0.25 * np.sin(phase))
            mem = self.stress.co_tenant_mean_memory * (0.75 + 0.25 * np.cos(phase))
            return cpu, mem
        cpu = max(0.0, self.rng.normal(self.stress.co_tenant_mean_cpu, self.stress.co_tenant_spread))
        mem = max(0.0, self.rng.normal(self.stress.co_tenant_mean_memory, self.stress.co_tenant_spread))
        if self.rng.random() < 0.20:
            cpu += self.stress.co_tenant_spread
            mem += self.stress.co_tenant_spread
        return cpu, mem

    def _refresh_external_load(self) -> None:
        self.current_reserved_cpu, self.current_reserved_memory = self._co_tenant_reservation()

    def _available_resources(self) -> Tuple[float, float]:
        used_cpu = sum(task.cpu for task in self.running)
        used_memory = sum(task.memory for task in self.running)
        return (
            self.cpu_capacity - used_cpu - self.current_reserved_cpu,
            self.memory_capacity - used_memory - self.current_reserved_memory,
        )

    def _inject_arrivals(self) -> None:
        regime, arrivals = self.generator.sample_arrivals(self.step_index)
        self.last_regime = regime
        self.regime_trace.append(regime)
        for cpu, memory, duration in arrivals:
            task = Task(
                task_id=self.task_counter,
                cpu=cpu,
                memory=memory,
                duration=duration,
                arrival_step=self.step_index,
                remaining=duration,
            )
            self.task_counter += 1
            self.all_tasks.append(task)
            if len(self.queue) < self.queue_capacity:
                self.queue.append(task)
            else:
                task.dropped = True
                task.drop_reason = "queue_full"

    def _observe(self) -> np.ndarray:
        avail_cpu, avail_mem = self._available_resources()
        used_cpu = self.cpu_capacity - max(avail_cpu, 0.0)
        used_mem = self.memory_capacity - max(avail_mem, 0.0)
        features = [
            np.clip(avail_cpu / self.cpu_capacity, -0.5, 1.0),
            np.clip(avail_mem / self.memory_capacity, -0.5, 1.0),
            np.clip(used_cpu / self.cpu_capacity, 0.0, 1.5),
            np.clip(used_mem / self.memory_capacity, 0.0, 1.5),
            len(self.queue) / self.queue_capacity,
            len(self.running) / max(1, self.queue_capacity),
            1.0 - (self.step_index / self.horizon),
        ]
        visible = self.queue[: self.visible_queue]
        for task in visible:
            reported_duration = self._reported_duration(task)
            features.extend(
                [
                    task.cpu / self.cpu_capacity,
                    task.memory / self.memory_capacity,
                    reported_duration / self.max_duration,
                    min(task.wait_steps, self.max_queue_wait) / self.max_queue_wait,
                ]
            )
        missing_slots = self.visible_queue - len(visible)
        for _ in range(missing_slots):
            features.extend([0.0, 0.0, 0.0, 0.0])
        obs = np.asarray(features, dtype=np.float32)
        if self.stress.observation_sigma > 0.0:
            telemetry_slice = slice(0, 4)
            obs[telemetry_slice] = obs[telemetry_slice] + self.rng.normal(
                0.0,
                self.stress.observation_sigma,
                size=4,
            ).astype(np.float32)
        return obs

    def _reported_duration(self, task: Task) -> float:
        strength = float(self.stress.duration_misreport_strength)
        if strength <= 0.0:
            return float(task.duration)

        # Observation-only stressor: short jobs look longer and long jobs look
        # shorter, violating the SJF ordering assumption without changing the
        # true workload, execution dynamics, or completion-time metric.
        direction = 1.0 if task.duration <= 4 else -1.0
        reported = float(task.duration) * (1.0 + direction * strength)
        return float(np.clip(reported, 1.0, self.max_duration))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        reward = 0.0
        info: Dict[str, float] = {"regime": self.last_regime, "invalid_action": 0.0}
        rejected_task = False

        if action < self.visible_queue:
            if action >= len(self.queue):
                reward -= self.invalid_action_penalty
                info["invalid_action"] = 1.0
            else:
                task = self.queue[action]
                avail_cpu, avail_mem = self._available_resources()
                if task.cpu <= avail_cpu and task.memory <= avail_mem:
                    task.start_step = self.step_index
                    self.running.append(task)
                    self.queue.pop(action)
                else:
                    reward -= self.invalid_action_penalty
                    info["invalid_action"] = 1.0
        elif action == self.visible_queue:
            reward -= self.defer_penalty
        else:
            if self.queue:
                task = self.queue.pop(0)
                task.dropped = True
                task.drop_reason = "agent_reject"
                rejected_task = True
                reward -= self.reject_penalty
            else:
                reward -= self.invalid_action_penalty
                info["invalid_action"] = 1.0

        self.step_index += 1
        completed: List[Task] = []
        for task in self.running:
            task.remaining -= 1
            if task.remaining <= 0:
                task.completion_step = self.step_index
                completed.append(task)

        if completed:
            completed_ids = {task.task_id for task in completed}
            self.running = [task for task in self.running if task.task_id not in completed_ids]
            reward += 0.25 * len(completed)

        for task in self.queue:
            task.wait_steps += 1

        unfinished_weight = 0.0
        for task in self.queue + self.running:
            unfinished_weight += 1.0 / max(1, task.duration)

        reward -= unfinished_weight
        used_cpu = sum(task.cpu for task in self.running)
        used_mem = sum(task.memory for task in self.running)
        utilization = 0.5 * ((used_cpu / self.cpu_capacity) + (used_mem / self.memory_capacity))
        reward += 0.05 * utilization
        self.utilization_trace.append(utilization)
        info["accepted"] = float(not rejected_task and action < self.visible_queue and info["invalid_action"] == 0.0)
        info["completed_count"] = float(len(completed))

        done = self.step_index >= self.horizon
        if not done:
            self._refresh_external_load()
            self._inject_arrivals()
            next_obs = self._observe()
        else:
            next_obs = self._observe()

        return next_obs, float(reward), done, info

    def episode_metrics(self) -> Dict[str, float]:
        arrival_count = max(1, len(self.all_tasks))
        slowdowns: List[float] = []
        completion_times: List[float] = []
        failures = 0
        for task in self.all_tasks:
            if task.completion_step is None:
                failures += 1
                synthetic_completion = self.horizon + max(task.remaining, task.duration // 2)
                flow_time = synthetic_completion - task.arrival_step
            else:
                flow_time = task.completion_step - task.arrival_step
            completion_times.append(float(flow_time))
            slowdowns.append(float(flow_time / max(1, task.duration)))

        mean_completion = float(np.mean(completion_times)) if completion_times else 0.0
        p95_completion = float(np.percentile(completion_times, 95)) if completion_times else 0.0
        return {
            "average_slowdown": float(np.mean(slowdowns)) if slowdowns else 0.0,
            "mean_completion_time": mean_completion,
            "p95_completion_time": p95_completion,
            "utilization": float(np.mean(self.utilization_trace)) if self.utilization_trace else 0.0,
            "task_failure_rate": failures / arrival_count,
            "tasks_arrived": float(arrival_count),
            "tasks_failed": float(failures),
        }


class RewardCorruptor:
    def __init__(self, bias: float, delay: int) -> None:
        self.bias = bias
        self.delay = delay
        self.buffer: List[float] = []

    def push(self, reward: float) -> float:
        corrupted = reward + self.bias * abs(reward)
        self.buffer.append(corrupted)
        if len(self.buffer) <= self.delay:
            return 0.0
        return self.buffer.pop(0)
