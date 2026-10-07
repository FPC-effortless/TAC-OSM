"""Leakage-hardened multimodal physics benchmark for the full PLM.

The model receives only text/image/audio observations and the goal.  Hidden
physical state is used by the benchmark generator to define the environment
outcome and the optimal action, but it is never passed to the model.

The benchmark deliberately contains no object identifiers or object-order
signal in audio.  Image observations contain positions only.  Audio is a
permutation-invariant sum of speed-dependent tones.  The history is required
to infer motion direction.

This generator is derived from the repository's E2E physics workload but is
versioned independently so the full-PLM experiment has its own immutable
benchmark identity.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import random
from dataclasses import dataclass
from typing import Iterable

import torch


ACTION_IMPULSES = (
    (0.0, 0.0),
    (0.12, 0.0),
    (-0.12, 0.0),
    (0.0, 0.12),
    (0.0, -0.12),
)
ACTION_NAMES = ("none", "push_x+", "push_x-", "push_y+", "push_y-")
ACTION_COUNT = len(ACTION_IMPULSES)
HISTORY = 8
HORIZON = 6
DT = 0.08
RADIUS = 0.045
IMAGE_SIZE = 32
AUDIO_SIZE = 96
VOCAB_SIZE = 32
GENERATOR_VERSION = "full-plm-physics-v1-swila-mtsk"
GOAL_MIN = 0.15
GOAL_MAX = 0.85


@dataclass(frozen=True)
class Episode:
    text: torch.Tensor
    image: torch.Tensor
    audio: torch.Tensor
    goal: torch.Tensor
    action: int
    hidden_state: "PhysicalState"

    @property
    def batch(self) -> dict[str, torch.Tensor]:
        return {
            "text": self.text.unsqueeze(0),
            "image": self.image.unsqueeze(0),
            "audio": self.audio.unsqueeze(0),
        }


@dataclass(frozen=True)
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
    velocity = state.velocity + torch.tensor(
        impulse, dtype=state.velocity.dtype
    ).view(1, 2)
    for i in range(2):
        for axis in range(2):
            if float(position[i, axis]) < RADIUS:
                position[i, axis] = RADIUS + (RADIUS - position[i, axis])
                velocity[i, axis] = velocity[i, axis].abs()
            elif float(position[i, axis]) > 1.0 - RADIUS:
                position[i, axis] = (1.0 - RADIUS) - (
                    position[i, axis] - (1.0 - RADIUS)
                )
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


def _sample_history(rng: random.Random) -> tuple[list[PhysicalState], PhysicalState]:
    for _ in range(500):
        position = torch.tensor(
            [
                [rng.uniform(0.20, 0.80), rng.uniform(0.20, 0.80)],
                [rng.uniform(0.20, 0.80), rng.uniform(0.20, 0.80)],
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
            state = step_physics(state)
            if any(
                float(v) < 0.06 or float(v) > 0.94
                for v in state.position.flatten()
            ):
                valid = False
                break
            if float(torch.linalg.vector_norm(state.position[1] - state.position[0])) < 0.12:
                valid = False
                break
            history.append(clone_state(state))
        if valid:
            return history, state
    raise RuntimeError("failed to sample valid physical history")


def _render_image(state: PhysicalState) -> torch.Tensor:
    coords = torch.linspace(0.0, 1.0, IMAGE_SIZE)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    image = torch.zeros(IMAGE_SIZE, IMAGE_SIZE)
    for point in state.position:
        distance_sq = (xx - point[0]) ** 2 + (yy - point[1]) ** 2
        image = torch.maximum(
            image,
            torch.exp(-distance_sq / (2.0 * 0.018**2)),
        )
    return image.unsqueeze(0)


def _render_audio(state: PhysicalState) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, AUDIO_SIZE)
    audio = torch.zeros(AUDIO_SIZE)
    # Sum is explicitly permutation invariant.
    for velocity in state.velocity:
        speed = float(torch.linalg.vector_norm(velocity))
        normalized_speed = min(speed / 0.42, 1.0)
        frequency = 2.0 + 12.0 * normalized_speed
        amplitude = 0.25 + 0.35 * normalized_speed
        audio = audio + amplitude * torch.sin(2.0 * math.pi * frequency * t)
    return audio.unsqueeze(0)


def _render_text(goal: torch.Tensor) -> torch.Tensor:
    x_bin = int(round(max(0.0, min(9.0, float(goal[0]) * 9.0))))
    y_bin = int(round(max(0.0, min(9.0, float(goal[1]) * 9.0))))
    return torch.tensor([1, 2, 10 + x_bin, 20 + y_bin, 3], dtype=torch.long)


def _future_score(
    state: PhysicalState,
    goal: torch.Tensor,
    action: tuple[float, float],
) -> float:
    future = clone_state(state)
    for step in range(HORIZON):
        future = step_physics(
            future, action if step == 0 else (0.0, 0.0)
        )
    return -float(torch.linalg.vector_norm(future.position.mean(dim=0) - goal))


def optimal_action(
    state: PhysicalState,
    goal: torch.Tensor,
    *,
    min_margin: float = 0.025,
) -> int | None:
    scores = [_future_score(state, goal, action) for action in ACTION_IMPULSES]
    order = sorted(range(ACTION_COUNT), key=lambda i: scores[i], reverse=True)
    if scores[order[0]] - scores[order[1]] < min_margin:
        return None
    return order[0]


def sample_episode(rng: random.Random) -> Episode:
    for _ in range(1200):
        history, final_state = _sample_history(rng)
        goal = torch.tensor(
            [
                rng.uniform(GOAL_MIN, GOAL_MAX),
                rng.uniform(GOAL_MIN, GOAL_MAX),
            ],
            dtype=torch.float32,
        )
        action = optimal_action(final_state, goal)
        if action is None:
            continue
        return Episode(
            text=torch.stack([_render_text(goal) for _ in history]),
            image=torch.stack([_render_image(state) for state in history]),
            audio=torch.stack([_render_audio(state) for state in history]),
            goal=goal,
            action=action,
            hidden_state=final_state,
        )
    raise RuntimeError("failed to sample benchmark episode")


def sample_balanced_episodes(rng: random.Random, per_action: int) -> list[Episode]:
    buckets: dict[int, list[Episode]] = {i: [] for i in range(ACTION_COUNT)}
    while any(len(v) < per_action for v in buckets.values()):
        episode = sample_episode(rng)
        if len(buckets[episode.action]) < per_action:
            buckets[episode.action].append(episode)
    episodes = [e for values in buckets.values() for e in values]
    rng.shuffle(episodes)
    return episodes


def step_environment_from_state(
    state: PhysicalState,
    goal: torch.Tensor,
    action: int,
) -> tuple[
    PhysicalState,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    float,
    bool,
    int | None,
]:
    if not 0 <= int(action) < ACTION_COUNT:
        raise ValueError("invalid action")
    next_state = step_physics(state, ACTION_IMPULSES[int(action)])
    distance = float(
        torch.linalg.vector_norm(next_state.position.mean(dim=0) - goal)
    )
    # Continuous post-action outcome evidence is observable consequence only.
    # The binary success label is retained outside the model as a supervision
    # and evaluation variable; it is never supplied as model input.
    outcome_evidence = -distance
    target = optimal_action(state, goal)
    reward = bool(target is not None and int(action) == int(target))
    target_after = optimal_action(next_state, goal)
    observation = (
        _render_text(goal),
        _render_image(next_state),
        _render_audio(next_state),
    )
    return next_state, observation, outcome_evidence, reward, target_after


def step_environment(
    episode: Episode, action: int
) -> tuple[
    PhysicalState,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    float,
    bool,
    int | None,
]:
    return step_environment_from_state(episode.hidden_state, episode.goal, action)


def episode_key(episode: Episode) -> tuple:
    return (
        tuple(int(x) for x in episode.text.flatten().tolist()),
        tuple(round(float(x), 6) for x in episode.goal),
        int(episode.action),
        tuple(round(float(x), 6) for x in episode.image.flatten().tolist()),
        tuple(round(float(x), 6) for x in episode.audio.flatten().tolist()),
    )


def episode_fingerprint(episodes: Iterable[Episode]) -> str:
    rows = []
    for episode in episodes:
        rows.append({
            "text": episode.text.tolist(),
            "image": episode.image.tolist(),
            "audio": episode.audio.tolist(),
            "goal": episode.goal.tolist(),
            "action": int(episode.action),
        })
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def generator_hash() -> str:
    with open(__file__, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def make_history_pair(rng: random.Random) -> tuple[Episode, Episode]:
    """Two histories with identical current sensor surface but different action."""
    for _ in range(1600):
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
            [rng.uniform(GOAL_MIN, GOAL_MAX), rng.uniform(GOAL_MIN, GOAL_MAX)],
            dtype=torch.float32,
        )
        candidates: list[Episode] = []
        valid = True
        for sign in (1.0, -1.0):
            signed = sign * velocity
            history: list[PhysicalState] = []
            for index in range(HISTORY):
                state = PhysicalState(
                    position=final_position - (HISTORY - 1 - index) * signed * DT,
                    velocity=signed.clone(),
                )
                if any(float(v) < 0.06 or float(v) > 0.94 for v in state.position.flatten()):
                    valid = False
                    break
                history.append(state)
            if not valid:
                break
            action = optimal_action(history[-1], goal)
            if action is None:
                valid = False
                break
            candidates.append(Episode(
                text=torch.stack([_render_text(goal) for _ in history]),
                image=torch.stack([_render_image(state) for state in history]),
                audio=torch.stack([_render_audio(state) for state in history]),
                goal=goal.clone(),
                action=int(action),
                hidden_state=history[-1],
            ))
        if valid and len(candidates) == 2 and candidates[0].action != candidates[1].action:
            return candidates[0], candidates[1]
    raise RuntimeError("failed to construct history pair")


def sample_history_pairs(rng: random.Random, count: int) -> list[tuple[Episode, Episode]]:
    return [make_history_pair(rng) for _ in range(count)]


def benchmark_manifest() -> dict:
    return {
        "generator_version": GENERATOR_VERSION,
        "generator_sha256": generator_hash(),
        "history_length": HISTORY,
        "horizon": HORIZON,
        "dt": DT,
        "action_count": ACTION_COUNT,
        "action_names": list(ACTION_NAMES),
        "modalities": {
            "text": "goal coordinates only; no action/state labels",
            "image": "position fields only; no velocity or entity ID",
            "audio": "permutation-invariant sum of speed-dependent tones; no object index",
        },
        "hidden_state_use": "benchmark environment only; never model input",
        "model_outcome_input": "continuous post-action distance-to-goal evidence; binary success is never an input",
        "training_supervision": "action label plus external post-action success label; model receives only post-action outcome evidence",
        "physics_prior_boundary": [
            "Hamilton canonical equations on a learned canonical latent",
            "conservation of learned Hamiltonian on passive intervals",
        ],
    }


__all__ = [
    "ACTION_IMPULSES", "ACTION_NAMES", "ACTION_COUNT", "HISTORY", "HORIZON",
    "DT", "VOCAB_SIZE", "Episode", "PhysicalState", "sample_episode",
    "sample_balanced_episodes", "step_environment_from_state", "step_environment", "episode_key",
    "episode_fingerprint", "generator_hash", "make_history_pair",
    "sample_history_pairs", "benchmark_manifest", "optimal_action",
]
