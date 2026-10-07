"""Synthetic multimodal physics-control benchmark."""
from __future__ import annotations

import hashlib
import io
import math
import random
from dataclasses import dataclass
from typing import Iterable

import torch

ACTIONS = (
    (0.0, 0.0),
    (0.12, 0.0),
    (-0.12, 0.0),
    (0.0, 0.12),
    (0.0, -0.12),
)
ACTION_NAMES = ("none", "push_x+", "push_x-", "push_y+", "push_y-")
ACTION_COUNT = len(ACTIONS)
HISTORY = 6
HORIZON = 6
DT = 0.08
RADIUS = 0.045
IMAGE_SIZE = 32
AUDIO_SIZE = 96
VOCAB_SIZE = 32
GENERATOR_VERSION = "learned-physics-e2e-v1"
NOISE = 0.025


@dataclass
class PhysicalState:
    position: torch.Tensor
    velocity: torch.Tensor


def clone_state(state: PhysicalState) -> PhysicalState:
    return PhysicalState(state.position.clone(), state.velocity.clone())


def step_physics(
    state: PhysicalState,
    impulse: tuple[float, float] = (0.0, 0.0),
) -> PhysicalState:
    position = state.position + state.velocity * DT
    velocity = state.velocity + torch.tensor(impulse, dtype=state.velocity.dtype).view(1, 2)

    for i in range(2):
        for axis in range(2):
            if float(position[i, axis]) < RADIUS:
                position[i, axis] = RADIUS + (RADIUS - position[i, axis])
                velocity[i, axis] = velocity[i, axis].abs()
            elif float(position[i, axis]) > 1.0 - RADIUS:
                position[i, axis] = (1.0 - RADIUS) - (position[i, axis] - (1.0 - RADIUS))
                velocity[i, axis] = -velocity[i, axis].abs()

    delta = position[1] - position[0]
    distance = torch.linalg.vector_norm(delta)
    if float(distance) < 2.0 * RADIUS:
        normal = delta / (distance + 1e-6)
        relative_velocity = velocity[1] - velocity[0]
        if float((relative_velocity * normal).sum()) < 0.0:
            vn0 = (velocity[0] * normal).sum()
            vn1 = (velocity[1] * normal).sum()
            velocity[0] = velocity[0] + (vn1 - vn0) * normal
            velocity[1] = velocity[1] + (vn0 - vn1) * normal

    return PhysicalState(position, velocity)


def _sample_collision_free_history(rng: random.Random) -> list[PhysicalState]:
    for _ in range(300):
        position = torch.tensor(
            [
                [rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)],
                [rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)],
            ],
            dtype=torch.float32,
        )
        if float(torch.linalg.vector_norm(position[1] - position[0])) < 0.25:
            continue
        velocity = torch.tensor(
            [
                [rng.uniform(-0.30, 0.30), rng.uniform(-0.30, 0.30)],
                [rng.uniform(-0.30, 0.30), rng.uniform(-0.30, 0.30)],
            ],
            dtype=torch.float32,
        )
        state = PhysicalState(position, velocity)
        history = [clone_state(state)]
        valid = True
        for _ in range(HISTORY - 1):
            next_state = step_physics(state)
            if any(float(v) < 0.06 or float(v) > 0.94 for v in next_state.position.flatten()):
                valid = False
                break
            if float(torch.linalg.vector_norm(next_state.position[1] - next_state.position[0])) < 0.12:
                valid = False
                break
            history.append(clone_state(next_state))
            state = next_state
        if valid:
            return history
    raise RuntimeError("failed to sample collision-free history")


def _render_image(state: PhysicalState) -> torch.Tensor:
    coords = torch.linspace(0.0, 1.0, IMAGE_SIZE)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    image = torch.zeros(IMAGE_SIZE, IMAGE_SIZE)
    for point in state.position:
        distance_sq = (xx - point[0]) ** 2 + (yy - point[1]) ** 2
        image = torch.maximum(image, torch.exp(-distance_sq / (2.0 * 0.018**2)))
    return image.unsqueeze(0)


def _render_audio(
    state: PhysicalState,
    rng: random.Random,
    *,
    deterministic: bool = False,
) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, AUDIO_SIZE)
    audio = torch.zeros(AUDIO_SIZE)
    for index, velocity in enumerate(state.velocity):
        speed = float(torch.linalg.vector_norm(velocity))
        normalized_speed = min(speed / 0.42, 1.0)
        frequency = 2.0 + 12.0 * normalized_speed
        phase = 0.55 * index if deterministic else rng.random() * 2.0 * math.pi
        audio = audio + (
            0.25 + 0.35 * normalized_speed
        ) * torch.sin(2.0 * math.pi * frequency * t + phase)
    if not deterministic:
        audio = audio + NOISE * torch.randn_like(audio)
    return audio.unsqueeze(0)


def _render_text(goal: torch.Tensor) -> torch.Tensor:
    x_bin = int(round(max(0, min(9, float(goal[0]) * 9.0))))
    y_bin = int(round(max(0, min(9, float(goal[1]) * 9.0))))
    return torch.tensor([1, 2, 10 + x_bin, 20 + y_bin, 3], dtype=torch.long)


def _future_score(
    state: PhysicalState,
    goal: torch.Tensor,
    action: tuple[float, float],
) -> float:
    future = clone_state(state)
    for step in range(HORIZON):
        future = step_physics(future, action if step == 0 else (0.0, 0.0))
    return -float(torch.linalg.vector_norm(future.position.mean(dim=0) - goal))


def optimal_action(
    state: PhysicalState,
    goal: torch.Tensor,
    *,
    min_margin: float = 0.025,
) -> int | None:
    scores = [_future_score(state, goal, action) for action in ACTIONS]
    order = sorted(range(ACTION_COUNT), key=lambda i: scores[i], reverse=True)
    if scores[order[0]] - scores[order[1]] < min_margin:
        return None
    return order[0]


def sample_episode(rng: random.Random) -> dict:
    for _ in range(1200):
        history = _sample_collision_free_history(rng)
        goal = torch.tensor(
            [rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)],
            dtype=torch.float32,
        )
        action = optimal_action(history[-1], goal)
        if action is None:
            continue
        return {
            "observations": [
                (_render_text(goal), _render_image(state), _render_audio(state, rng))
                for state in history
            ],
            "goal": goal,
            "action": action,
        }
    raise RuntimeError("failed to sample benchmark episode")


def sample_balanced_episodes(rng: random.Random, per_action: int) -> list[dict]:
    buckets = {action: [] for action in range(ACTION_COUNT)}
    while any(len(bucket) < per_action for bucket in buckets.values()):
        episode = sample_episode(rng)
        action = int(episode["action"])
        if len(buckets[action]) < per_action:
            buckets[action].append(episode)
    episodes = [episode for bucket in buckets.values() for episode in bucket]
    rng.shuffle(episodes)
    return episodes


def episode_key(episode: dict) -> tuple:
    return (
        tuple(tuple(int(x) for x in row[0].tolist()) for row in episode["observations"]),
        tuple(float(x) for x in episode["goal"]),
        int(episode["action"]),
        tuple(tuple(float(x) for x in row[1].flatten().tolist()) for row in episode["observations"]),
    )


def episode_fingerprint(episodes: Iterable[dict]) -> str:
    serial = []
    for episode in episodes:
        serial.append(
            {
                "goal": episode["goal"].tolist(),
                "action": int(episode["action"]),
                "text": [row[0].tolist() for row in episode["observations"]],
                "image": [row[1].tolist() for row in episode["observations"]],
                "audio": [row[2].tolist() for row in episode["observations"]],
            }
        )
    buffer = io.BytesIO()
    torch.save(serial, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def generator_hash() -> str:
    with open(__file__, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def make_history_pair(rng: random.Random) -> tuple[dict, dict]:
    for _ in range(1200):
        final_position = torch.tensor(
            [
                [rng.uniform(0.30, 0.70), rng.uniform(0.30, 0.70)],
                [rng.uniform(0.30, 0.70), rng.uniform(0.30, 0.70)],
            ],
            dtype=torch.float32,
        )
        if float(torch.linalg.vector_norm(final_position[1] - final_position[0])) < 0.20:
            continue
        velocity = torch.tensor(
            [
                [rng.uniform(0.10, 0.26), rng.uniform(-0.10, 0.10)],
                [rng.uniform(-0.10, 0.10), rng.uniform(0.10, 0.26)],
            ],
            dtype=torch.float32,
        )
        goal = torch.tensor(
            [rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)],
            dtype=torch.float32,
        )
        histories = []
        for sign in (1.0, -1.0):
            signed_velocity = sign * velocity
            history = []
            for index in range(HISTORY):
                state = PhysicalState(
                    position=final_position - (HISTORY - 1 - index) * signed_velocity * DT,
                    velocity=signed_velocity.clone(),
                )
                if any(float(v) < 0.06 or float(v) > 0.94 for v in state.position.flatten()):
                    break
                history.append(state)
            if len(history) != HISTORY:
                histories = []
                break
            action = optimal_action(history[-1], goal, min_margin=0.025)
            histories.append((history, action))
        if len(histories) != 2 or histories[0][1] is None or histories[1][1] is None:
            continue
        if histories[0][1] == histories[1][1]:
            continue
        episodes = []
        for history, action in histories:
            episodes.append(
                {
                    "observations": [
                        (
                            _render_text(goal),
                            _render_image(state),
                            _render_audio(state, rng, deterministic=True),
                        )
                        for state in history
                    ],
                    "goal": goal.clone(),
                    "action": int(action),
                }
            )
        return episodes[0], episodes[1]
    raise RuntimeError("failed to sample history pair")


def sample_history_pairs(rng: random.Random, count: int) -> list[tuple[dict, dict]]:
    return [make_history_pair(rng) for _ in range(count)]


def benchmark_manifest() -> dict:
    return {
        "generator_version": GENERATOR_VERSION,
        "generator_sha256": generator_hash(),
        "history_length": HISTORY,
        "horizon": HORIZON,
        "dt": DT,
        "action_names": list(ACTION_NAMES),
        "modality_policy": {
            "image": "positions only; no velocity direction or entity IDs",
            "audio": "speed magnitude only; no velocity direction or entity IDs",
            "text": "goal coordinates only; no action labels or state labels",
        },
        "physics_prior": [
            "linear momentum conservation",
            "kinetic energy conservation",
            "kinematic displacement/velocity consistency",
        ],
    }
