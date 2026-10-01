"""The loop itself: ``S_t -> R_t -> C_t -> A_t -> O_t -> V_t -> S_{t+1}``.

This is the one place the loop runs, and it is deliberately small. Five
components are injected; nothing is constructed inside. Every variant in
``docs/ABLATION_PLAN.md`` is produced by injecting different implementations
into this same method, so there is no per-experiment code path anywhere in the
model.

## The step

::

    task    = environment.next_task(state)                    # O_t source
    read    = state.read(task.query)                          # S_t
    decide  = router.route(task.query, state, candidates)     # R_t
    exec    = executor.execute(structure, inputs)             # C_t
    action  = decide.selected                                 # A_t
    outcome = environment.transition(state, action, query)    # O_t
    verdict = verifier.verify(computation, outcome)           # V_t
    write   = state.write(update) if passed else repair       # S_{t+1}

The ordering is the anti-leakage boundary: ``transition`` is called *after*
routing, so no router can observe the outcome it is selecting for, and the
verifier runs after ``transition``, so verification is post-hoc by
construction.

## Why most of the loop is deterministic

Per the integration plan, the first model is mostly deterministic —
deterministic state, learned router, deterministic executor, deterministic
environment, deterministic verifier, deterministic write — with learned
versions replacing deterministic ones one at a time. The rationale is the
CASM Phase 1.5A failure: training loss reached ~0.0012 while the intended
relation went unsolved, and a low-loss model in the wrong basin is
indistinguishable from a working one until the capability is measured
directly. If all six components are learned simultaneously and the loop
fails, there is no way to localise the failure. So exactly one component is
learned in v0, and it is the one the portfolio has positive evidence for:
routing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

from . import (
    Candidate,
    Computation,
    Environment,
    Outcome,
    PersistentState,
    Query,
    RepairController,
    RelevanceRouter,
    RoutingDecision,
    StateRead,
    StateUpdate,
    StateWrite,
    Step,
    Structure,
    VerificationResult,
)
from .executor import program_from_descriptor, relevance_program
from .environment import parse_query as parse_query_text
from .router import LearnedRelationalRouter

__all__ = ["ModelConfig", "TacOsmModel", "Episode", "run_episode",
           "RewardContext", "RewardFn", "baseline_reward"]


@dataclass(frozen=True)
class RewardContext:
    """Everything the loop has in hand at the moment of the learning update.

    The reward hook receives this and returns the scalar passed to
    ``router.update``. ``Task`` carries ``target_action`` — the gold index —
    so a hook that wants to anchor a margin to gold can. The router's own
    ``update`` never sees this object: the hook's return value is a bare
    ``float``, exactly as the outcome reward always was.

    The hook runs *after* the transition and *only when ``config.learn`` is
    true*, so it can widen neither the router's observable inputs nor the
    evaluation path.

    ``router`` is included because the F3 surrogate is computed from the
    router's own scoring function, at the weights the update is about to move.
    Handing the router to the hook does not widen the router's inputs — the
    hook is not the router, and ``router.score`` is a function of the
    permitted inputs it already reads — but it *is* the reason a surrogate
    arm can only run where the router is a learned one: a hook that needs
    ``score`` has nothing to call on an oracle or a static arm.
    """

    task: Any
    candidates: Sequence[Candidate]
    selected: int
    decision: RoutingDecision
    outcome: Outcome
    query: Query
    state: PersistentState
    router: Any


@runtime_checkable
class RewardFn(Protocol):
    """The reward hook's contract: one step's context, one scalar out."""

    def __call__(self, ctx: RewardContext) -> float:
        """Return the reward for this step. Must be a plain finite float."""
        ...


def baseline_reward(ctx: RewardContext) -> float:
    """``float(outcome.success)`` — the MATCHED-001 protocol, verbatim.

    This is what the loop hardcoded before F3, and ``ModelConfig.reward_fn =
    None`` still calls it. It is named and exported so that:

    * a measurement can assert the baseline arm's hook *is* this function,
      rather than trusting that "no hook" means "the old behaviour";
    * the F3 pre-registration's reproduction gate has a symbol to point at.

    The value is ``1[a_t = gold]``: the environment sets
    ``target_action = gold_index`` and scores ``success = action ==
    target_action``, so the outcome reward is already a gold-anchored scalar.
    That fact is the reason the F3 leakage boundary is stated in terms of
    signal *density* rather than the absence of gold — see
    ``docs/TACOSM-SURROGATE-001.md``.
    """
    return float(ctx.outcome.success)


@dataclass
class ModelConfig:
    """Loop-level settings that are not component switches.

    Component switches live on the component configs; these are the properties
    of the traversal itself. Keeping them apart is what lets two ablation
    cells differ only in a component while the traversal stays identical.
    """

    n_steps: int = 12
    learn: bool = True
    seed: int = 0
    write_on_success: bool = True
    #: The F3 reward hook. ``None`` is the baseline and must be bit-for-bit
    #: identical to the pre-F3 hardcoded ``float(outcome.success)`` call — see
    #: :func:`baseline_reward`. Consulted only when ``learn`` is true.
    reward_fn: RewardFn | None = None

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        if self.reward_fn is not None and not callable(self.reward_fn):
            raise TypeError(
                f"reward_fn must be a callable RewardFn or None, got "
                f"{type(self.reward_fn).__name__}"
            )


@dataclass
class Episode:
    """One complete traversal, or as much of one as ran.

    ``steps`` is the trajectory, and carrying it on one object is what makes
    path-versus-final verification comparable without re-running anything.
    """

    steps: list[Step] = field(default_factory=list)
    successes: int = 0
    writes: int = 0
    repairs: int = 0
    terminated: str = ""

    @property
    def n(self) -> int:
        return len(self.steps)

    @property
    def accuracy(self) -> float:
        return self.successes / self.n if self.n else 0.0


class TacOsmModel:
    """The loop. Injected components, one traversal method, no per-arm branches.

    The model does not know which router it has, whether the executor is
    flattened, or whether the verifier is path-level. That ignorance is the
    integration boundary working in the other direction: an ablation result is
    a property of the injected set, not of the model.
    """

    def __init__(
        self,
        *,
        state: PersistentState,
        router: RelevanceRouter,
        executor: Any,
        verifier: Any,
        repair: RepairController | None,
        environment: Environment,
        computation_selector: Any | None = None,
        config: ModelConfig | None = None,
    ) -> None:
        self.state = state
        self.router = router
        self.executor = executor
        self.verifier = verifier
        self.repair = repair
        self.environment = environment
        # Optional explicit CASM computation-selection boundary. Legacy runs
        # keep the original relevance_program path unchanged.
        self.computation_selector = computation_selector
        self.config = config if config is not None else ModelConfig()
        self._rng = random.Random(self.config.seed)
        self.episodes: list[Episode] = []
        self._last_query: Query | None = None
        self._last_context: tuple[int, ...] = ()

    # -- the loop ---------------------------------------------------------- #

    def step(self, step_index: int) -> Step:
        """One complete traversal, ``S_t -> ... -> S_{t+1}``."""
        # O_t source: the environment presents a task and writes world facts
        # into the store the loop will read.
        task = self.environment.next_task(self.state)
        query = task.public()
        self._last_query = query
        self._last_context = query.context

        # S_t: read the candidate structures addressable by this query.
        read = self.state.read(query)

        # R_t: select one candidate. Routing sees only query, state, and
        # candidates — never the target action or the outcome.
        #
        # The oracle arm is the exception the interface boundary permits: it
        # reads ``target_action`` and therefore has to be *bound* here, at the
        # one point in the loop where the task is already in hand and the
        # decision has not yet been made. Binding it in the builder would move
        # the answer into a component that runs before the task exists; binding
        # it here keeps the oracle's privilege inside the step, auditable, and
        # unavailable to every other arm.
        self._bind_oracle(task)
        decision = self.router.route(query, self.state, task.candidates)

        # C_t: execute the selected candidate's structure. When state is
        # disabled the persistence families have no reference — the relation
        # they ask for is simply not available — so the step is recorded as
        # undefined rather than crashing. A ``-state`` ablation cell then still
        # runs, and the missing capability shows up in the measurements rather
        # than in a traceback.
        selected = task.candidates[decision.selected]
        try:
            structure = self._structure_for(selected, step_index, read)
        except ValueError as exc:
            return self._undefined_step(step_index, query, decision, task, str(exc))
        inputs = self._inputs_for(read, selected)
        execution = self.executor.execute(structure, inputs)

        # A_t: the selected candidate index is the action.
        action = decision.selected

        # O_t: the environment reveals the outcome, after routing.
        outcome = self.environment.transition(self.state, action, query)

        computation = Computation(
            structure=structure,
            action=action,
            trace=self._trace_of(execution),
        )

        # V_t: verification is post-hoc by construction.
        verification = self.verifier.verify(computation, outcome)

        # S_{t+1}: commit only on verification, or repair.
        write: StateWrite | None = None
        repair_result = None
        if verification.passed:
            if self.config.write_on_success:
                write = self.state.write(self._update_for(task, outcome, step_index))
        elif self.repair is not None:
            repair_result = self.repair.repair(computation, verification)

        # Learning boundary:
        # - successor routers own their outcome update;
        # - legacy routers retain the frozen RewardContext/reward_fn contract.
        # This preserves historical experiments while keeping the successor
        # path independent of the legacy scalar reward API.
        if self.config.learn:
            # Successor learners may consume the full post-verification signal.
            # This method is called only after routing, execution, outcome and
            # verification, so a verifier-derived label cannot leak into the
            # decision that produced it.
            verifier_learner = getattr(self.router, "learn_from_verifier", None)
            if callable(verifier_learner):
                verifier_learner(
                    query=query,
                    state=self.state,
                    candidates=task.candidates,
                    selected=decision.selected,
                    outcome=outcome,
                    verification=verification,
                    scores=decision.scores,
                )
            elif self.router.__class__.__name__ == "RepresentationEnergyRouter":
                learner = getattr(self.router, "learn_from_outcome", None)
                if callable(learner):
                    learner(
                        query=query,
                        state=self.state,
                        candidates=task.candidates,
                        selected=decision.selected,
                        success=bool(outcome.success),
                        scores=decision.scores,
                    )
            elif isinstance(self.router, LearnedRelationalRouter):
                ctx = RewardContext(
                    task=task,
                    candidates=task.candidates,
                    selected=decision.selected,
                    decision=decision,
                    outcome=outcome,
                    query=query,
                    state=self.state,
                    router=self.router,
                )
                reward = self._reward(ctx)
                self.router.update(
                    query=query,
                    candidates=task.candidates,
                    selected=decision.selected,
                    reward=reward,
                    probs=decision.scores,
                    state=self.state,
                )

        return Step(
            step=step_index,
            query=query,
            decision=decision,
            computation=computation,
            outcome=outcome,
            verification=verification,
            write=write,
            repair=repair_result,
            provenance={
                "family": task.family,
                "router": decision.provenance,
                "executor": execution.provenance,
                "success": bool(outcome.success),
                "wrote": bool(write and write.committed),
                "repaired": repair_result is not None,
            },
        )

    def _undefined_step(self, step_index: int, query: Query, decision: Any,
                        task: Any, reason: str) -> Step:
        """Record a step whose computation could not be formed at all.

        Reached when the relation a family asks for is not available under the
        current switches — the ``-state`` ablation, whose persistence families
        lose their reference. Failing closed here would make the cell
        unrunnable and the ablation impossible; recording the step keeps the
        missing capability visible in the measurements, where an ablation
        result belongs.
        """
        outcome = self.environment.transition(self.state, decision.selected, query)
        return Step(
            step=step_index,
            query=query,
            decision=decision,
            computation=Computation(
                structure=Structure(key="undefined", spec=None, provenance="unavailable"),
                action=decision.selected,
                trace=(),
            ),
            outcome=outcome,
            verification=VerificationResult(passed=False, feedback=f"undefined:{reason}"),
            write=None,
            repair=None,
            provenance={
                "family": task.family,
                "router": decision.provenance,
                "executor": "unavailable",
                "success": bool(outcome.success),
                "wrote": False,
                "repaired": False,
                "undefined": True,
            },
        )

    def run(self) -> Episode:
        """Run a full episode and record it."""
        episode = Episode()
        for i in range(self.config.n_steps):
            s = self.step(i)
            episode.steps.append(s)
            if s.outcome.success:
                episode.successes += 1
            if s.write is not None and s.write.committed:
                episode.writes += 1
            if s.repair is not None:
                episode.repairs += 1
        episode.terminated = "complete"
        self.episodes.append(episode)
        return episode


    # -- adapters ---------------------------------------------------------- #

    def _reward(self, ctx: RewardContext) -> float:
        """The one point the F3 hook is consulted.

        ``None`` is the pre-F3 behaviour — :func:`baseline_reward`, the
        MATCHED-001 protocol. A hook is never asked twice for one step, is
        given only the context, and its return value is passed to the router
        untouched. The router therefore still receives a bare ``float``, so
        no hook can widen the router's own inputs; the hook can only change
        the value of the scalar the router was already consuming.

        Runtime finite-ness is checked here rather than at hook construction
        time so that a NaN-emitting hook fails at the step where it fired,
        with the step's provenance in hand, instead of poisoning a mean.
        """
        fn = self.config.reward_fn
        if fn is None:
            return baseline_reward(ctx)
        reward = fn(ctx)
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise TypeError(
                f"reward_fn must return a plain float, got "
                f"{type(reward).__name__}"
            )
        value = float(reward)
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"reward_fn returned a non-finite reward: {value!r}")
        return value

    def _bind_oracle(self, task: Any) -> None:
        """Hand the oracle arm its target for this step, if it is the arm.

        Any other router type is untouched, so this costs nothing outside the
        oracle arm and cannot widen any other arm's information boundary.
        """
        bind = getattr(self.router, "bind", None)
        if callable(bind):
            bind(task.target_action)

    def _structure_for(self, candidate: Candidate, step_index: int,
                       read: StateRead) -> Structure:
        """Turn a candidate descriptor into an executable structure.

        This is the integration boundary at execution: the environment speaks
        bit descriptors, the executor speaks programs, and this method is the
        whole of the translation. Nothing else in the loop knows both.

        The structure is the relevance circuit for this candidate against the
        marked positions. The reference is the query's public bits for
        ``relational`` tasks and the *state-read* bits for the persistence
        families, whose target is not in the observation. ``replay`` carries
        both, and the address wins — see :meth:`_reference_for`. Routing and
        execution are then two *independent* computations of one relation
        — the router scores it with a linear basis, the executor computes it
        as a Boolean circuit — so verification is a genuine cross-check rather
        than a tautology.

        ``read`` is passed rather than read from ``self`` because it is
        already bound to this step's query by the caller, and the circuit must
        see the same read the router scored against.
        """
        marks = self._marks_for_step()
        reference = self._reference_for(read)
        if not reference:
            raise ValueError(
                "relevance circuit has no reference: query carries no bits and "
                "state read is empty, so the relation is undefined for this step"
            )
        if self.computation_selector is not None:
            selector = getattr(self.computation_selector, "select", None)
            if not callable(selector):
                raise TypeError("computation_selector must expose select()")
            return selector(
                reference=reference,
                candidate=candidate,
                marks=marks,
                step_index=step_index,
            )
        program = relevance_program(
            reference, candidate.descriptor, marks, max_nodes=self._max_nodes()
        )
        return Structure(
            key=candidate.key,
            spec=program,
            provenance=f"{candidate.provenance}:{step_index}",
        )

    def _reference_for(self, read: StateRead) -> tuple[int, ...]:
        """The bits the relation is measured against.

        **Address first.** When the query carries a state address, the target
        lives at that address and *nowhere else*: ``replay`` carries public
        bits *and* an address, and the relation is built against the written
        vector, not the bits. Preferring the bits in that case made the
        executor compute the relation against a vector the environment never
        used, so the verifier rejected a correct execution ~75% of the time
        on that family (defect 6; caught by the §34 audit, not by any test).

        ``relational`` carries only bits and no address, so it takes the
        public path; ``state_lookup`` carries only an address, so it takes
        the state path. A query carrying neither has no relation to compute.
        """
        bits, address = parse_query_text(self._last_query)
        if address:
            for key, value in zip(read.keys, read.values):
                if key == address and value:
                    return tuple(int(b) for b in value)
        if bits:
            return bits
        for value in read.values:
            if value:
                return tuple(int(b) for b in value)
        return ()

    def _marks_for_step(self) -> list[int]:
        """The positions the current query's context marks."""
        context = self._last_context
        return [j for j, b in enumerate(context) if b == 1]

    def _max_nodes(self) -> int:
        executor_config = getattr(self.executor, "config", None)
        return int(getattr(executor_config, "max_nodes", 10))

    def _inputs_for(self, read: StateRead, candidate: Candidate) -> list[float]:
        """The executor's input vector.

        The descriptor's bits are the primary signal; the state read is
        appended so a structural pathway can depend on persisted values. This
        is the only place persisted values enter execution, and they enter
        downstream of routing, so the information boundary of the executor
        contract holds: runtime values never reach the router.
        """
        inputs = [float(b) for b in candidate.descriptor]
        for value in read.values:
            inputs.extend(float(b) for b in value)
        return inputs

    def _trace_of(self, execution: Any) -> tuple[Any, ...]:
        """Carry the executor's trace onto the computation.

        Normalised to a flat float tuple — the output first, then the
        per-node values — because path verification needs to compare
        intermediates against the output numerically. Final verification
        ignores this entirely; path verification reads all of it. Both
        variants therefore see the same executed computation, which is what
        makes the §19 comparison interpretable.
        """
        node_values = getattr(execution, "node_values", ())
        output = float(getattr(execution, "output", 0.0))
        flat = [output]
        for z in node_values:
            flat.append(float(z))
        return tuple(flat)

    def _update_for(self, task: Any, outcome: Outcome, step_index: int) -> StateUpdate:
        """The proposed persistent write for a verified step.

        Written under the verified-only commit rule, so this is reached only
        through the verifier. The key is the task's state address when it has
        one, which is what lets a later ``replay`` task read the value back.
        """
        address = task.query.text.partition("\t")[2]
        # World facts written by the environment occupy address keys. Never
        # overwrite them with a learning write: verified experience gets its
        # own namespace, so persistence and learning remain independently
        # measurable.
        key = (
            f"experience:{task.family}:{step_index:04d}:{address}"
            if address
            else f"experience:{task.family}:{step_index:04d}"
        )
        value = self._descriptor_bits(task)
        if not value:
            # Pure lookup/replay facts are hidden in world state. The selected
            # candidate is not in the public query, so preserve that fact as a
            # post-hoc verified experience via the candidate descriptor.
            target = getattr(getattr(task, "detail", None), "written_bits", ())
            if target:
                value = tuple(int(b) for b in target)
        return StateUpdate(
            key=key,
            value=value,
            task_key=task.family,
            success_score=float(outcome.success),
            step=step_index,
        )

    def _descriptor_bits(self, task: Any) -> tuple[int, ...]:
        """The task's public bits, as the value to persist."""
        bits = task.query.text.partition("\t")[0]
        return tuple(int(b) for b in bits.split()) if bits.strip() else ()


def run_episode(model: TacOsmModel) -> Episode:
    """Run one episode and return it. Thin wrapper kept for script use."""
    return model.run()
