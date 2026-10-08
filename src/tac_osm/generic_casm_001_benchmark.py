from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


FAMILY_NAMES = ("hadamard_affine", "concat_affine", "gated_sum")
INPUT_DIM = 8
OUTPUT_DIM = 8
SUPPORT_ROWS = 8
QUERY_ROWS = 4


@dataclass(frozen=True)
class OperatorEpisode:
    support_u: torch.Tensor
    support_v: torch.Tensor
    support_y: torch.Tensor
    query_u: torch.Tensor
    query_v: torch.Tensor
    query_y: torch.Tensor
    family: int
    parameter_fingerprint: str


def apply_operator(
    family: int,
    u: torch.Tensor,
    v: torch.Tensor,
    params: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    if family == 0:
        W, b = params
        return torch.tanh((u * v) @ W.T + b)
    if family == 1:
        W, b = params
        return torch.tanh(torch.cat([u, v], dim=-1) @ W.T + b)
    if family == 2:
        A, B, b = params
        gate = torch.sigmoid(u @ A.T)
        return torch.tanh((gate * v + (1.0 - gate) * u) @ B.T + b)
    raise ValueError(f"unknown family {family}")


def sample_params(g: torch.Generator, family: int):
    def normal(*shape: int) -> torch.Tensor:
        return 0.6 * torch.randn(*shape, generator=g)

    if family == 0:
        return normal(OUTPUT_DIM, INPUT_DIM), normal(OUTPUT_DIM)
    if family == 1:
        return normal(OUTPUT_DIM, INPUT_DIM * 2), normal(OUTPUT_DIM)
    return (
        normal(INPUT_DIM, INPUT_DIM),
        normal(OUTPUT_DIM, INPUT_DIM),
        normal(OUTPUT_DIM),
    )


def parameter_hash(params: tuple[torch.Tensor, ...]) -> str:
    h = hashlib.sha256()
    for p in params:
        h.update(p.contiguous().numpy().tobytes())
    return h.hexdigest()


def sample_episode(g: torch.Generator, family: int) -> OperatorEpisode:
    params = sample_params(g, family)
    support_u = torch.randn(SUPPORT_ROWS, INPUT_DIM, generator=g)
    support_v = torch.randn(SUPPORT_ROWS, INPUT_DIM, generator=g)
    query_u = torch.randn(QUERY_ROWS, INPUT_DIM, generator=g)
    query_v = torch.randn(QUERY_ROWS, INPUT_DIM, generator=g)
    support_y = apply_operator(family, support_u, support_v, params)
    query_y = apply_operator(family, query_u, query_v, params)
    perm = torch.randperm(SUPPORT_ROWS, generator=g)
    support_u = support_u[perm]
    support_v = support_v[perm]
    support_y = support_y[perm]
    return OperatorEpisode(
        support_u=support_u,
        support_v=support_v,
        support_y=support_y,
        query_u=query_u,
        query_v=query_v,
        query_y=query_y,
        family=family,
        parameter_fingerprint=parameter_hash(params),
    )


def generate_pool(seed: int, count: int, eval_pool: bool) -> list[OperatorEpisode]:
    g = torch.Generator().manual_seed(
        (7_000_000 if eval_pool else 1_000_000) + seed * 100_003
    )
    out = []
    for n in range(count):
        family = n % len(FAMILY_NAMES)
        out.append(sample_episode(g, family))
    return out


def pool_fingerprint(pool: list[OperatorEpisode]) -> str:
    h = hashlib.sha256()
    for ep in pool:
        h.update(ep.parameter_fingerprint.encode())
        for tensor in (
            ep.support_u, ep.support_v, ep.support_y,
            ep.query_u, ep.query_v, ep.query_y,
        ):
            h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def benchmark_manifest() -> dict:
    return {
        "families": list(FAMILY_NAMES),
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "support_rows": SUPPORT_ROWS,
        "query_rows": QUERY_ROWS,
        "train_eval_rng_namespaces": ["1_000_000", "7_000_000"],
        "operator_parameters_are_episode_local": True,
        "support_row_order_randomized": True,
        "operator_id_in_model_input": False,
        "query_target_not_input": True,
    }
