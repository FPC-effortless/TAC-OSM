"""Full learned PLM core with a SwiLA-derived MTSK state kernel.

This module is the main construction path for the full PLM experiment.  It is
not a GRU controller and it does not contain a fixed operator table.

Architecture:
    multimodal representation
        -> MTSK (SwiLA-derived fast/medium/slow recurrent state)
        -> learned CDL relevance routing over persistent expert state
        -> learned CASM program synthesis/execution
        -> learned action policy
        -> environment outcome
        -> learned verifier
        -> learned repair / recomputation
        -> verifier-gated persistent state commit
        -> OSM prediction + slow/fast learning signals

SwiLA is used as an external architectural antecedent, not as a claim about
PLM.  The implementation below is an independent, dependency-light adaptation
of the paper's mixture-of-linear-regressors recurrence: expert routing,
responsibility-weighted delta updates, temporal/sticky priors, and fixed-size
recurrent state.  See the experiment contract for the exact adaptation
boundary.

The only privileged world knowledge allowed by this experiment is the
registered physical-law prior in the optional Hamiltonian loss.  No task
ontology, object IDs, fixed operators, gold action, or hard-coded retrieval
rule is represented in the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FullPLMConfig:
    vocab_size: int = 32
    d_model: int = 96
    num_heads: int = 4
    num_experts: int = 6
    fast_experts: int = 2
    medium_experts: int = 2
    slow_experts: int = 2
    cdl_top_k: int = 2
    casm_operators: int = 4
    casm_top_k: int = 1
    casm_depth: int = 3
    action_count: int = 5
    temporal_prior: float = 0.35
    halt_threshold: float = 0.80
    verifier_threshold: float = 0.60

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads:
            raise ValueError("d_model must divide num_heads")
        if self.fast_experts + self.medium_experts + self.slow_experts != self.num_experts:
            raise ValueError("timescale expert counts must sum to num_experts")
        if self.cdl_top_k < 1 or self.cdl_top_k > self.num_experts:
            raise ValueError("invalid cdl_top_k")
        if self.casm_top_k < 1 or self.casm_top_k > self.casm_operators:
            raise ValueError("invalid casm_top_k")
        if self.casm_depth < 1:
            raise ValueError("casm_depth must be >= 1")


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        width = max(16, d_model // 2)
        self.embedding = nn.Embedding(vocab_size, width)
        self.conv = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=3, padding=1, groups=1),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1, groups=1),
            nn.GELU(),
        )
        self.proj = nn.Linear(width, d_model)

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.embedding(tokens).transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        x = x.mean(dim=1)
        return self.proj(x)


class ImageEncoder(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.GELU(),
            # Preserve a small spatial grid instead of global-average pooling.
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.proj(self.features(image))


class AudioEncoder(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 24, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(24, 48, 7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(48, 64, 7, stride=2, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, audio: Tensor) -> Tensor:
        return self.proj(self.features(audio))


class MultimodalRepresentation(nn.Module):
    """Learned cross-modal representation; no task-specific ontology."""

    def __init__(self, config: FullPLMConfig) -> None:
        super().__init__()
        self.text = TextEncoder(config.vocab_size, config.d_model)
        self.image = ImageEncoder(config.d_model)
        self.audio = AudioEncoder(config.d_model)
        self.fuse = nn.Sequential(
            nn.Linear(3 * config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.LayerNorm(config.d_model),
        )

    def forward(self, text: Tensor, image: Tensor, audio: Tensor) -> Tensor:
        return self.fuse(
            torch.cat([self.text(text), self.image(image), self.audio(audio)], dim=-1)
        )


def _inverse_sigmoid(value: float) -> float:
    eps = 1e-5
    x = min(1.0 - eps, max(eps, value))
    return float(torch.log(torch.tensor(x / (1.0 - x))))


class SwiLAMTSK(nn.Module):
    """SwiLA-derived persistent state split across fast/medium/slow experts.

    The state is W[b, head, expert, output_dim, key_dim].  It is an actual
    recurrent state tensor, not a history cache.  Expert responsibilities are
    inferred from prediction error and learned priors.

    Timescales are architectural state dynamics, not encoded knowledge about
    the benchmark.  Their role is only to provide distinct retention regimes
    for the MTSK hypothesis.
    """

    def __init__(self, config: FullPLMConfig) -> None:
        super().__init__()
        d = config.d_model // config.num_heads
        self.config = config
        self.head_dim = d
        h = config.num_heads
        j = config.num_experts

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.key_prior = nn.Linear(config.d_model, h * j * d)
        self.query_prior = nn.Linear(config.d_model, h * j * d)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.write_proj = nn.Linear(config.d_model + config.action_count + 1, config.d_model)
        self.input_decay = nn.Linear(config.d_model, h * j * d)

        # Fast, medium, slow retention priors. These are the MTSK architecture,
        # not benchmark-specific physical facts.
        base_decay = (
            [0.70] * config.fast_experts
            + [0.92] * config.medium_experts
            + [0.99] * config.slow_experts
        )
        base_beta = (
            [0.25] * config.fast_experts
            + [0.08] * config.medium_experts
            + [0.015] * config.slow_experts
        )
        self.decay_logits = nn.Parameter(
            torch.tensor([_inverse_sigmoid(x) for x in base_decay], dtype=torch.float32)
        )
        self.beta_logits = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(base_beta, dtype=torch.float32)))
        )

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(
            batch,
            self.config.num_heads,
            self.config.num_experts,
            self.head_dim,
            self.head_dim,
            device=device,
            dtype=dtype,
        )

    def _step(
        self,
        x_t: Tensor,
        state: Tensor,
        previous_resp: Optional[Tensor],
        *,
        update_gate: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        b = x_t.shape[0]
        h = self.config.num_heads
        j = self.config.num_experts
        d = self.head_dim

        q = self.q_proj(x_t).view(b, h, d)
        k = self.k_proj(x_t).view(b, h, d)
        v = self.v_proj(x_t).view(b, h, d)

        prior_k = self.key_prior(x_t).view(b, h, j, d)
        prior_q = self.query_prior(x_t).view(b, h, j, d)
        pi_k = prior_k.softmax(dim=2)

        pred = torch.einsum("bhjde,bhe->bhjd", state, k)
        delta = v.unsqueeze(2) - pred
        posterior_logits = torch.log(pi_k.clamp_min(1e-8)) - 0.5 * delta.square()

        responsibilities = posterior_logits.softmax(dim=2)
        if previous_resp is not None and self.config.temporal_prior > 0.0:
            responsibilities = (
                (1.0 - self.config.temporal_prior) * responsibilities
                + self.config.temporal_prior * previous_resp
            )
            responsibilities = responsibilities / responsibilities.sum(dim=2, keepdim=True).clamp_min(1e-8)

        base_decay = torch.sigmoid(self.decay_logits).view(1, 1, j, 1, 1)
        input_decay = self.input_decay(x_t).view(b, h, j, d, 1).sigmoid()
        decay = (0.5 * base_decay + 0.5 * input_decay).clamp(1e-4, 0.9999)
        beta = F.softplus(self.beta_logits).view(1, 1, j, 1, 1)

        if update_gate is None:
            gate = 1.0
        else:
            gate = update_gate.view(b, 1, 1, 1, 1)

        new_state = (
            decay * state
            + gate
            * beta
            * responsibilities.unsqueeze(-1)
            * delta.unsqueeze(-1)
            * k.unsqueeze(2).unsqueeze(3)
        )

        pi_q = prior_q.softmax(dim=2)
        expert_reads = torch.einsum("bhjde,bhe->bhjd", new_state, q)
        read = (pi_q * expert_reads).sum(dim=2).reshape(b, -1)
        read = self.out_proj(read)

        # For diagnostics and CDL, preserve the expert-wise persistent views.
        expert_views = expert_reads.permute(0, 2, 1, 3).reshape(
            b, j, h * d
        )
        return read, new_state, expert_views, responsibilities

    def forward_sequence(
        self, x: Tensor, state: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError("x must be [batch,time,d_model]")
        b = x.shape[0]
        if state is None:
            state = self.initial_state(b, x.device, x.dtype)

        outputs = []
        expert_views = []
        load_losses = []
        previous: Optional[Tensor] = None
        current = state
        for t in range(x.shape[1]):
            read, current, experts, responsibilities = self._step(
                x[:, t], current, previous
            )
            outputs.append(read)
            expert_views.append(experts)
            mean_pi = responsibilities.mean(dim=(0, 1, 3))
            load_losses.append((mean_pi * mean_pi.clamp_min(1e-8).log()).sum())
            previous = responsibilities
        y = torch.stack(outputs, dim=1)
        e = torch.stack(expert_views, dim=1)
        load = torch.stack(load_losses).mean()
        return y, current, e, load

    def commit_verified(
        self,
        current_state: Tensor,
        context: Tensor,
        action: Tensor,
        outcome: Tensor,
        verifier_probability: Tensor,
    ) -> Tensor:
        """Verified experience write; failed verification gets zero gate."""
        if action.ndim == 1:
            action = F.one_hot(action.long(), num_classes=self.config.action_count).to(context.dtype)
        else:
            action = action.to(context.dtype)
        if action.shape[-1] != self.config.action_count:
            raise ValueError("action encoding width mismatch")
        x = torch.cat(
            [context, action, outcome.float().view(-1, 1)], dim=-1
        )
        write_input = self.write_proj(x)
        gate = (
            verifier_probability
            if self.training
            else (verifier_probability >= self.config.verifier_threshold).to(context.dtype)
        )
        _read, new_state, _experts, _resp = self._step(
            write_input,
            current_state,
            None,
            update_gate=gate,
        )
        return new_state


class LearnedCDL(nn.Module):
    """Learned relevance routing over persistent MTSK expert views."""

    def __init__(self, config: FullPLMConfig) -> None:
        super().__init__()
        self.config = config
        self.score = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
        )

    def forward(self, query: Tensor, expert_views: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, j, d = expert_views.shape
        q = query.unsqueeze(1).expand(-1, j, -1)
        logits = self.score(torch.cat([q, expert_views], dim=-1)).squeeze(-1)
        k = min(self.config.cdl_top_k, j)
        _, top_indices = torch.topk(logits, k=k, dim=1)
        soft_weights = F.softmax(logits, dim=1)
        hard_mask = torch.zeros_like(soft_weights).scatter_(1, top_indices, 1.0)
        hard_weights = hard_mask * soft_weights
        hard_weights = hard_weights / hard_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        weights_all = hard_weights + soft_weights - soft_weights.detach()
        selected = torch.gather(
            expert_views, 1, top_indices.unsqueeze(-1).expand(-1, -1, d)
        )
        weights = torch.gather(weights_all, 1, top_indices)
        return selected, weights, logits



class LearnedStructuralWorkspace(nn.Module):
    """Learn a graph over the routed persistent structures.

    Nodes are the sparse CDL outputs plus one query node. Edge weights and node
    updates are learned from content; no fixed topology or relation labels are
    supplied. This is the structural substrate consumed by CASM.
    """

    def __init__(self, config: FullPLMConfig) -> None:
        super().__init__()
        d = config.d_model
        self.edge = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        self.message = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.norm = nn.LayerNorm(d)

    def forward(
        self, query: Tensor, selected: Tensor, weights: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        nodes = torch.cat([selected, query.unsqueeze(1)], dim=1)
        b, n, d = nodes.shape
        left = nodes.unsqueeze(2).expand(-1, -1, n, -1)
        right = nodes.unsqueeze(1).expand(-1, n, -1, -1)
        q = query.unsqueeze(1).unsqueeze(1).expand(-1, n, n, -1)
        edge_logits = self.edge(torch.cat([left, right, q], dim=-1)).squeeze(-1)
        eye = torch.eye(n, device=nodes.device, dtype=nodes.dtype).unsqueeze(0)
        edge_logits = edge_logits.masked_fill(eye.bool(), -1e4)
        adjacency = edge_logits.softmax(dim=-1)
        message_input = torch.cat(
            [
                nodes.unsqueeze(2).expand(-1, -1, n, -1),
                nodes.unsqueeze(1).expand(-1, n, -1, -1),
            ],
            dim=-1,
        )
        messages = self.message(message_input)
        updated = nodes + (adjacency.unsqueeze(-1) * messages).sum(dim=2)
        updated = self.norm(updated)
        graph_state = updated[:, :-1]
        query_state = updated[:, -1]
        return graph_state, query_state, adjacency


class LearnedCASM(nn.Module):
    """Learned structural computation with routing, composition, and halting.

    The operator bank contains generic learned program primitives. Their
    semantics are learned rather than hard-coded as Boolean functions.
    """

    def __init__(self, config: FullPLMConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.workspace = LearnedStructuralWorkspace(config)
        self.operators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * d, 2 * d),
                    nn.GELU(),
                    nn.Linear(2 * d, d),
                    nn.GELU(),
                )
                for _ in range(config.casm_operators)
            ]
        )
        self.selector = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, config.casm_operators),
        )
        self.halt = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def forward(
        self, query: Tensor, selected: Tensor, weights: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        graph, graph_query, adjacency = self.workspace(query, selected, weights)
        current = (graph * weights.unsqueeze(-1)).sum(dim=1) + graph_query
        traces = []
        halt_probs = []
        op_probs = []
        executed = []

        for _ in range(self.config.casm_depth):
            joint = torch.cat([current, query], dim=-1)
            logits = self.selector(joint)
            probs = F.softmax(logits, dim=-1)
            top_k = min(self.config.casm_top_k, len(self.operators))
            _, op_indices = torch.topk(probs, k=top_k, dim=-1)
            sparse_probs = torch.zeros_like(probs).scatter(
                1, op_indices, torch.gather(probs, 1, op_indices)
            )
            sparse_probs = sparse_probs / sparse_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            # Execute only operators selected by at least one sample in this
            # batch. This preserves sparse operator work instead of evaluating
            # the complete bank for every token.
            delta = torch.zeros_like(current)
            for op_index, operator in enumerate(self.operators):
                sample_mask = sparse_probs[:, op_index] > 0
                if sample_mask.any():
                    delta[sample_mask] = (
                        delta[sample_mask]
                        + sparse_probs[sample_mask, op_index].unsqueeze(-1)
                        * operator(joint[sample_mask])
                    )
            halt_prob = torch.sigmoid(self.halt(joint)).squeeze(-1)
            current = current + (1.0 - halt_prob).unsqueeze(-1) * delta
            traces.append(current)
            halt_probs.append(halt_prob)
            op_probs.append(probs)
            executed.append(1.0 - halt_prob)

        return (
            current,
            torch.stack(halt_probs, dim=1),
            torch.stack(op_probs, dim=1),
            torch.stack(traces, dim=1),
            adjacency,
            torch.stack(executed, dim=1),
        )


class LearnedOSM(nn.Module):
    """Learned latent world model used for prediction and credit assignment."""

    def __init__(self, config: FullPLMConfig) -> None:
        super().__init__()
        d = config.d_model
        self.action = nn.Embedding(config.action_count, d)
        self.outcome = nn.Linear(1, d)
        self.transition = nn.Sequential(
            nn.Linear(3 * d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, d),
        )
        self.hamiltonian = nn.Sequential(
            nn.Linear(d, d),
            nn.Tanh(),
            nn.Linear(d, d // 2),
            nn.Tanh(),
            nn.Linear(d // 2, 1),
        )

    def predict(self, state: Tensor, action: Tensor, outcome: Tensor) -> Tensor:
        return self.transition(
            torch.cat(
                [state, self.action(action.long()), self.outcome(outcome.float().view(-1, 1))],
                dim=-1,
            )
        )

    def energy(self, state: Tensor) -> Tensor:
        return self.hamiltonian(state).squeeze(-1)


class LearnedVerifier(nn.Module):
    def __init__(self, config: FullPLMConfig) -> None:
        super().__init__()
        d = config.d_model
        self.net = nn.Sequential(
            nn.Linear(2 * d + config.action_count + 1, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def forward(
        self,
        pre_state: Tensor,
        post_state: Tensor,
        action_prob: Tensor,
        outcome: Tensor,
    ) -> Tensor:
        x = torch.cat(
            [
                pre_state,
                post_state,
                action_prob,
                outcome.float().view(-1, 1),
            ],
            dim=-1,
        )
        return self.net(x).squeeze(-1)


class LearnedRepair(nn.Module):
    def __init__(self, config: FullPLMConfig) -> None:
        super().__init__()
        d = config.d_model
        self.net = nn.Sequential(
            nn.Linear(d + config.action_count + 1, d),
            nn.GELU(),
            nn.Linear(d, config.action_count),
        )

    def forward(self, state: Tensor, action_prob: Tensor, outcome: Tensor) -> Tensor:
        return self.net(
            torch.cat(
                [state, action_prob, outcome.float().view(-1, 1)], dim=-1
            )
        )


@dataclass
class PLMStep:
    action_logits: Tensor
    action: Tensor
    representation: Tensor
    memory_read: Tensor
    expert_views: Tensor
    cdl_weights: Tensor
    cdl_logits: Tensor
    casm_state: Tensor
    halt_probs: Tensor
    operator_probs: Tensor
    verifier_logit: Optional[Tensor] = None
    repair_logits: Optional[Tensor] = None
    next_memory_state: Optional[Tensor] = None


class FullLearnedPLM(nn.Module):
    """The complete learned PLM computation graph."""

    def __init__(self, config: FullPLMConfig = FullPLMConfig()) -> None:
        super().__init__()
        self.config = config
        self.representation = MultimodalRepresentation(config)
        self.mtsk = SwiLAMTSK(config)
        self.cdl = LearnedCDL(config)
        self.casm = LearnedCASM(config)
        self.osm = LearnedOSM(config)
        self.verifier = LearnedVerifier(config)
        self.repair = LearnedRepair(config)

        d = config.d_model
        self.goal = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.action_head = nn.Sequential(
            nn.Linear(2 * d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, config.action_count),
        )
        self.plan_head = nn.Sequential(
            nn.Linear(3 * d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, 1),
        )
        self.physics_latent = nn.Sequential(
            nn.Linear(d, d),
            nn.Tanh(),
            nn.Linear(d, d),
        )

    def encode_history(
        self,
        text: Tensor,
        image: Tensor,
        audio: Tensor,
    ) -> Tensor:
        # Inputs are [B,T,...].
        b, t = text.shape[:2]
        z = self.representation(
            text.reshape(b * t, *text.shape[2:]),
            image.reshape(b * t, *image.shape[2:]),
            audio.reshape(b * t, *audio.shape[2:]),
        )
        return z.view(b, t, -1)

    def forward(
        self,
        text: Tensor,
        image: Tensor,
        audio: Tensor,
        *,
        memory_state: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        z_seq = self.encode_history(text, image, audio)
        mem_seq, state, expert_seq, load_loss = self.mtsk.forward_sequence(
            z_seq, memory_state
        )
        query = mem_seq[:, -1]
        selected, weights, cdl_logits = self.cdl(query, expert_seq[:, -1])
        (
            casm_state,
            halt_probs,
            operator_probs,
            traces,
            adjacency,
            execution_activity,
        ) = self.casm(query, selected, weights)
        base_action_logits = self.action_head(
            torch.cat([casm_state, query], dim=-1)
        )
        action_ids = torch.arange(
            self.config.action_count, device=query.device, dtype=torch.long
        )
        plan_state = query.unsqueeze(1).expand(-1, self.config.action_count, -1).reshape(-1, query.shape[-1])
        plan_actions = action_ids.unsqueeze(0).expand(query.shape[0], -1).reshape(-1)
        zero_outcome = torch.zeros(plan_actions.shape[0], device=query.device)
        predicted_next = self.osm.predict(plan_state, plan_actions, zero_outcome).view(
            query.shape[0], self.config.action_count, -1
        )
        goal_vec = self.goal(query)
        goal_expand = goal_vec.unsqueeze(1).expand(-1, self.config.action_count, -1)
        planner_input = torch.cat([
            casm_state.unsqueeze(1).expand(-1, self.config.action_count, -1),
            predicted_next,
            goal_expand,
        ], dim=-1)
        planning_logits = self.plan_head(planner_input).squeeze(-1)
        action_logits = base_action_logits + planning_logits
        action_prob = F.softmax(action_logits, dim=-1)
        return {
            "z_seq": z_seq,
            "memory_seq": mem_seq,
            "memory_state": state,
            "expert_seq": expert_seq,
            "query": query,
            "cdl_selected": selected,
            "cdl_weights": weights,
            "cdl_logits": cdl_logits,
            "casm_state": casm_state,
            "casm_traces": traces,
            "halt_probs": halt_probs,
            "operator_probs": operator_probs,
            "casm_adjacency": adjacency,
            "casm_execution_activity": execution_activity,
            "action_logits": action_logits,
            "action_prob": action_prob,
            "load_loss": load_loss,
        }

    def verify_and_update(
        self,
        forward_output: dict[str, Tensor],
        *,
        action: Tensor,
        outcome: Tensor,
        post_text: Optional[Tensor] = None,
        post_image: Optional[Tensor] = None,
        post_audio: Optional[Tensor] = None,
    ) -> PLMStep:
        if post_text is not None:
            post_z = self.representation(post_text, post_image, post_audio)
        else:
            post_z = forward_output["query"].detach()

        action_prob = F.softmax(forward_output["action_logits"], dim=-1)
        # The verifier receives only post-action evidence and the proposed
        # action. It does not receive the benchmark target.
        verifier_logit = self.verifier(
            forward_output["query"], post_z, action_prob, outcome
        )
        verifier_prob = verifier_logit.sigmoid()
        repair_logits = self.repair(
            post_z, action_prob, outcome
        )
        action_encoded = F.one_hot(
            action.long(), num_classes=self.config.action_count
        ).to(post_z.dtype)
        next_state = self.mtsk.commit_verified(
            forward_output["memory_state"],
            post_z,
            action_encoded,
            outcome,
            verifier_prob,
        )
        return PLMStep(
            action_logits=forward_output["action_logits"],
            action=action,
            representation=forward_output["z_seq"][:, -1],
            memory_read=forward_output["query"],
            expert_views=forward_output["expert_seq"][:, -1],
            cdl_weights=forward_output["cdl_weights"],
            cdl_logits=forward_output["cdl_logits"],
            casm_state=forward_output["casm_state"],
            halt_probs=forward_output["halt_probs"],
            operator_probs=forward_output["operator_probs"],
            verifier_logit=verifier_logit,
            repair_logits=repair_logits,
            next_memory_state=next_state,
        )

    def physics_prior_loss(
        self,
        representation: Tensor,
        dt: float,
        *,
        passive_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Generic Hamiltonian prior on the learned latent, not state labels."""
        canonical = self.physics_latent(representation)
        q, p = canonical.chunk(2, dim=-1)
        energy = self.osm.energy(canonical)

        grad_canonical = torch.autograd.grad(
            energy.sum(),
            canonical,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        grad_q, grad_p = grad_canonical.chunk(2, dim=-1)
        # Approximate latent derivatives with a finite difference in the
        # learned representation trajectory when at least two frames exist.
        dq = (q[:, 1:] - q[:, :-1]) / dt
        dp = (p[:, 1:] - p[:, :-1]) / dt
        rhs_q = grad_p[:, :-1]
        rhs_p = -grad_q[:, :-1]

        dyn = (dq - rhs_q).square().mean() + (dp - rhs_p).square().mean()
        conservation = (energy[:, 1:] - energy[:, :-1]).square().mean()
        if passive_mask is not None:
            m = passive_mask[:, 1:].to(dyn.dtype)
            denom = m.sum().clamp_min(1.0)
            conservation = ((energy[:, 1:] - energy[:, :-1]).square() * m).sum() / denom
        return dyn + conservation


def full_plm_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "FullPLMConfig",
    "MultimodalRepresentation",
    "SwiLAMTSK",
    "LearnedCDL",
    "LearnedStructuralWorkspace",
    "LearnedCASM",
    "LearnedOSM",
    "LearnedVerifier",
    "LearnedRepair",
    "FullLearnedPLM",
    "PLMStep",
    "full_plm_parameter_count",
]