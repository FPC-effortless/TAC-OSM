"""DEV-052B bounded end-to-end world model.

Only NativeWorld and training target collection access synthetic hidden state.
The deployment controller takes a public record and public action receipts.
Entity routing and action constraints are explicit controls, not learned claims.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import math
import random
import time

import torch
from torch import nn
import torch.nn.functional as F

from .veritas_public_challenge import make_episode

BASE = (.85, (.75, .8, 1.))
SHIFTS = ((.65, (.75, .8, .7)), (.85, (.55, .6, .7)),
          (.7, (.6, .65, .8)), (.95, (.85, .9, .9)))
NEURAL = ("last_event", "recurrent_belief", "forecast_only", "joint",
          "frozen_writer", "policy_bc", "policy_dagger")
ARMS = ("known_bayes", "estimated_bayes", "event_table", "verify_retry",
        "all_three", "wait") + NEURAL
COMMANDS = ("done", "inspect", "service")
HIDDEN = 32
EVENT_DIM = 12


def stream(*parts):
    return random.Random(":".join(map(str, ("tac-dev052b-v1",) + parts)))


def make_world(seed, index, shifted=False):
    """Fresh synthetic oracle; healthy state is absorbing across all repairs."""
    from investigation_world.operational.models import StateAssertion
    ep = make_episode(seed, index).model_copy(deep=True)
    identity = hashlib.sha256(f"tac-dev052b-v1:{seed}:{index}".encode()).hexdigest()[:24]
    identity = "TAC-DEV052B-" + identity
    accuracy, rates = SHIFTS[index % len(SHIFTS)] if shifted else BASE
    rng = stream("world", seed, index)
    state = {"plant.slot": -1, "plant.collateral": 0}
    effects = [e for e in ep.oracle.action_effects if e.action_name != "service"]
    repair_outcomes = {}
    for entity in range(2):
        fault = rng.randrange(2)
        state.update({f"e{entity}.fault": fault, f"e{entity}.attempt": 0})
        ep.records[entity].fields = {"entity": entity, "reported_fault": rng.randrange(2)}
        successes = [int(rng.random() < q) for q in rates]
        repair_outcomes[entity] = successes
        for attempt, success in enumerate(successes, 1):
            fault *= 1-success
            template = next(e for e in ep.oracle.action_effects
                            if e.action_name == "service" and
                            e.required_parameters == {"entity": entity, "attempt": 1})
            effects.append(template.model_copy(update={
                "required_parameters": {"entity": entity, "attempt": attempt},
                "required_state": [
                    StateAssertion(object_id="plant", field_name="slot", expected_value=entity),
                    StateAssertion(object_id=f"e{entity}", field_name="attempt", expected_value=attempt-1)],
                "set_state": {f"e{entity}.fault": fault, f"e{entity}.attempt": attempt,
                              "plant.slot": -1}}))
    ep.episode_id = ep.world_id = ep.task.task_id = ep.task.world_id = ep.oracle.task_id = identity
    ep.oracle = ep.oracle.model_copy(update={"initial_state": state, "action_effects": effects,
                        "required_action_counts": {}, "max_cost": 100, "max_tool_calls": 24})
    ep.task.constraints = ["Isolate before each service; one shared slot; service releases it.",
                          "At most three service attempts and four paid inspections per entity.",
                          "Inspections may be incorrect; any repair may fail; healthy is absorbing."]
    # Actual rates and pre-sampled outcomes remain harness-only.
    ep.task.metadata = {"max_service_attempts": 3, "max_inspections": 4}
    ep.metadata = {"public_synthetic": True}
    return ep, (accuracy, tuple(rates)), repair_outcomes


class NativeWorld:
    """Evaluator-owned capability boundary: never pass this object to a policy."""
    def __init__(self, seed, index, shifted=False):
        from investigation_world.operational.runtime import OperationalRuntime
        self.seed, self.index = seed, index
        self.episode, self.rates, self.repair_outcomes = make_world(seed, index, shifted)
        self.runtime = OperationalRuntime(self.episode)
        self.calls = []
        self.reads = [0, 0]
        self.violations = 0

    def act(self, name, entity, attempt=None):
        args = {"entity": entity}
        if attempt is not None:
            args["attempt"] = attempt
        result = dict(self.runtime.act(name, **args))
        if name == "inspect":
            truth = int(self.runtime.state[f"e{entity}.fault"])
            rng = stream("sensor", self.seed, self.index, entity, self.reads[entity])
            result["fault"] = truth if rng.random() < self.rates[0] else 1-truth
            self.reads[entity] += 1
        self.calls.append({"name": name, "parameters": args, "receipt": result})
        snap = self.runtime.state
        self.violations += int(snap["plant.collateral"] != 0)
        return result

    def fault(self, entity):
        return int(self.runtime.state[f"e{entity}.fault"])

    def finish(self):
        from investigation_world.operational.models import EpisodeSubmission
        faults = sum(self.fault(e) for e in range(2))
        slot = int(self.runtime.state["plant.slot"])
        verify = self.runtime.submit(EpisodeSubmission(conclusion="DEV-052B public predictive repair"))
        return {"world": self.episode.episode_id, "seed": self.seed, "index": self.index,
                "failures": faults, "complete": int(verify.state == 1),
                "cost": self.runtime.budget.spent/10 + 4*faults,
                "native_spent": self.runtime.budget.spent,
                "blocked": sum(e.blocked for e in self.runtime.events),
                "violations": self.violations + int(slot != -1),
                "inspections": sum(c["name"] == "inspect" for c in self.calls),
                "services": sum(c["name"] == "service" for c in self.calls),
                "calls": self.calls, "verification": verify.model_dump(mode="json")}


def posterior(p, bit, accuracy):
    numerator = p*(accuracy if bit else 1-accuracy)
    denominator = numerator+(1-p)*(1-accuracy if bit else accuracy)
    return numerator/denominator if denominator else p


def bellman(p, attempts, senses, params, penalty=4.):
    """Shared exact finite-horizon public-belief planner, bounded local cache."""
    accuracy, rates = params

    @lru_cache(maxsize=None)
    def value(p, attempts, senses):
        candidates = [(penalty*p, 0)]
        if attempts < 3:
            q = round(p*(1-rates[attempts]), 8)
            candidates.append((1.2+value(q, attempts+1, senses)[0], 2))
        if senses < 4:
            positive = p*accuracy+(1-p)*(1-accuracy)
            vp = value(round(posterior(p, 1, accuracy), 8), attempts, senses+1)[0]
            vn = value(round(posterior(p, 0, accuracy), 8), attempts, senses+1)[0]
            candidates.append((.1+positive*vp+(1-positive)*vn, 1))
        # Fixed tie order inspect, service, done, as in predecessor.
        return min(candidates, key=lambda x: (x[0], {1: 0, 2: 1, 0: 2}[x[1]]))

    result = value(round(float(p), 8), attempts, senses)
    return result


@dataclass
class PublicState:
    reported: int
    attempts: int = 0
    senses: int = 0
    ticks: int = 0
    isolated: bool = False
    last_bit: int | None = None
    reading_history: list = field(default_factory=list)

    def token(self, kind="record", receipt=None):
        receipt = receipt or {}
        bit = receipt.get("fault", 0)
        return [float(kind == k) for k in ("record", "inspect", "isolate", "service")] + [
            float(bit), float("fault" in receipt), float(receipt.get("accepted", True)),
            float(self.reported), self.attempts/3., self.senses/4., self.ticks/12., 1.]

    def commit(self, name, receipt):
        # Native acceptance is receipt integrity, not hidden physical repair success.
        if not receipt.get("accepted", True):
            return False
        self.ticks += 1
        if name == "inspect":
            self.last_bit = int(receipt["fault"])
            self.reading_history.append((self.attempts, self.last_bit))
            self.senses += 1
        elif name == "isolate":
            self.isolated = True
        elif name == "service":
            self.attempts += 1
            self.isolated = False
            self.last_bit = None
        return True

    def key(self):
        return str((self.attempts, tuple(self.reading_history)))

    def legal(self, command):
        return (command == 0 or command == 1 and self.senses < 4 or
                command == 2 and self.attempts < 3)


class BeliefModel(nn.Module):
    def __init__(self, seed, arm):
        super().__init__()
        if arm not in NEURAL:
            raise ValueError(arm)
        torch.manual_seed(seed)
        self.arm = arm
        self.writer = nn.GRUCell(EVENT_DIM, HIDDEN)
        self.last_encoder = nn.Linear(EVENT_DIM, HIDDEN)
        self.fault_head = nn.Linear(HIDDEN, 1)
        self.effect_head = nn.Linear(HIDDEN, 3)
        self.sensor_head = nn.Linear(HIDDEN, 1)
        self.policy_head = nn.Linear(HIDDEN, 3)
        if arm == "frozen_writer":
            self.writer.requires_grad_(False)

    def write(self, token, old):
        if self.arm == "last_event":
            return self.last_encoder(token).tanh()
        return self.writer(token, old)

    def heads(self, h):
        return (self.fault_head(h).squeeze(-1), self.effect_head(h),
                self.sensor_head(h).squeeze(-1), self.policy_head(h))

    def sequence(self, x):
        h = x.new_zeros((x.shape[0], HIDDEN))
        outputs = []
        for t in range(x.shape[1]):
            h = self.write(x[:, t], h)
            outputs.append(h)
        return self.heads(torch.stack(outputs, 1))


class Controller:
    """Deployment: public-only inputs, no evaluator object, IDs, rates or targets."""
    def __init__(self, arm, params, model=None, calibration=None, table=None):
        self.arm, self.params, self.model = arm, params, model
        self.calibration = calibration or {"temperature": 1., "risk": 1.}
        self.table = table or {}
        self.states = {}
        self.banks = {}
        self.priors = {}
        self.predictions = {}
        self.history = {}
        self.inference_seconds = 0.

    def begin(self, entity, record):
        self.states[entity] = PublicState(int(record["reported_fault"]))
        self.priors[entity] = .5
        token = self.states[entity].token()
        self.history[entity] = [token]
        if self.model is not None:
            with torch.no_grad():
                self.banks[entity] = self.model.write(torch.tensor([token]), torch.zeros(1, HIDDEN))

    def observe(self, entity, kind, receipt):
        state = self.states[entity]
        if not state.commit(kind, receipt):
            return False
        if kind == "inspect":
            self.priors[entity] = posterior(self.priors[entity], int(receipt["fault"]), self.params[0])
        elif kind == "service":
            self.priors[entity] *= 1-self.params[1][state.attempts-1]
        token = state.token(kind, receipt)
        self.history[entity].append(token)
        if self.model is not None:
            with torch.no_grad():
                self.banks[entity] = self.model.write(torch.tensor([token]), self.banks[entity])
        return True

    def forecast(self, entity):
        p, params, policy = self.priors[entity], self.params, None
        state = self.states[entity]
        if self.arm == "event_table":
            hits, n = self.table.get(state.key(), (0, 0))
            p = (hits+2*p)/(n+2)
        if self.model is not None:
            with torch.no_grad():
                logit, effects, sensor, policy = self.model.heads(self.banks[entity])
                p = float(torch.sigmoid(logit/self.calibration["temperature"])[0])
                if self.arm != "recurrent_belief":
                    # Bound estimated observation accuracy away from singularities;
                    # orientation is known from the public binary sensor contract.
                    a = max(.5001, min(.9999, float(sensor.sigmoid()[0])))
                    params = (a, tuple(float(q) for q in effects.sigmoid()[0]))
                policy = policy[0]
        return p, params, policy

    def choose(self, entity):
        start = time.perf_counter()
        state = self.states[entity]
        p, params, logits = self.forecast(entity)
        if self.arm == "wait":
            command = 0
        elif self.arm == "all_three":
            command = 2 if state.attempts < 3 else 0
        elif self.arm == "verify_retry":
            command = 0 if state.attempts == 3 or state.last_bit == 0 else 1 if state.last_bit is None and state.senses < 4 else 2
        elif self.arm in ("policy_bc", "policy_dagger"):
            valid = torch.tensor([state.legal(i) for i in range(3)])
            command = int(logits.masked_fill(~valid, -torch.inf).argmax())
        else:
            _, command = bellman(p, state.attempts, state.senses, params, 4*self.calibration["risk"])
        if not state.legal(command):
            raise AssertionError("inadmissible public action")
        if state.isolated:
            raise AssertionError("unfinished service macro")
        self.predictions[entity] = (p, params)
        self.inference_seconds += time.perf_counter()-start
        return command


def target_at(world, entity, state, p, params):
    """TRAIN ONLY privileged data, shared with fitted non-neural baselines."""
    truth = world.fault(entity)
    repairs = [truth*(1-q) for q in world.repair_outcomes[entity]]
    label = bellman(p, state.attempts, state.senses, params)[1]
    return {"y": truth, "repair_y": repairs, "teacher": label,
            "attempts": state.attempts, "senses": state.senses,
            "key": state.key()}


def collect(spec, params=BASE, model=None, arm="policy_dagger"):
    """One bounded native trajectory per entity; no evaluation world accepted."""
    if spec["kind"] not in ("train", "validation", "calibration", "smoke"):
        raise ValueError("evaluation cannot enter training collection")
    sequences, sensors = [], []
    for index in range(spec["worlds"]):
        world = NativeWorld(spec["seed"], index)
        c = Controller(arm if model else "estimated_bayes", params, model=model)
        for rec in world.runtime.public_payload()["records"]:
            entity = int(rec["fields"]["entity"])
            c.begin(entity, rec["fields"])
            entries = []
            def capture():
                s = c.states[entity]
                entries.append({"x": c.history[entity][-1], **target_at(world, entity, s, c.priors[entity], params)})
            capture()
            if model is None:
                # Four reads at varying positions; all three repairs on every
                # entity independently of hidden truth or inspection results.
                rng = stream("exploration", spec["seed"], index, entity)
                reads = [0, 0, 0, 0]
                for _ in range(4):
                    reads[rng.randrange(4)] += 1
                commands = []
                for attempt in range(4):
                    commands.extend([1]*reads[attempt])
                    if attempt < 3:
                        commands.append(2)
            else:
                commands = None
            for step in range(8):
                command = commands[step] if commands is not None and step < len(commands) else c.choose(entity) if commands is None else 0
                if command == 0:
                    break
                primitive = ("inspect",) if command == 1 else ("isolate", "service")
                for name in primitive:
                    truth_before = world.fault(entity)
                    s = c.states[entity]
                    receipt = world.act(name, entity, s.attempts+1 if name == "service" else None)
                    if name == "inspect":
                        sensors.append(int(receipt["fault"] == truth_before))
                    c.observe(entity, name, receipt)
                    capture()
            sequences.append(entries)
    return sequences, sensors


def estimate(sequences, sensors):
    # Same labeled evidence as neural learners, not true simulator rates.
    first = [seq[0] for seq in sequences]
    rates = []
    for attempt in range(3):
        opportunities = [row for row in first if row["y"] == 1]
        successes = sum(1-row["repair_y"][attempt] for row in opportunities)
        rates.append((successes+1)/(len(opportunities)+2))
    accuracy = (sum(sensors)+1)/(len(sensors)+2)
    table = {}
    for seq in sequences:
        for row in seq:
            hits, n = table.get(row["key"], (0, 0))
            table[row["key"]] = (hits+row["y"], n+1)
    return (accuracy, tuple(rates)), table


def tensorize(sequences):
    n, length = len(sequences), max(map(len, sequences))
    x = torch.zeros(n, length, EVENT_DIM)
    y, sensor, mask = (torch.zeros(n, length) for _ in range(3))
    repairs = torch.zeros(n, length, 3)
    labels = torch.zeros(n, length, dtype=torch.long)
    legal = torch.zeros(n, length, 3, dtype=torch.bool)
    for i, seq in enumerate(sequences):
        for t, row in enumerate(seq):
            x[i, t] = torch.tensor(row["x"])
            y[i, t] = row["y"]
            repairs[i, t] = torch.tensor(row["repair_y"])
            labels[i, t] = row["teacher"]
            mask[i, t] = 1
            legal[i, t] = torch.tensor([True, row["senses"] < 4, row["attempts"] < 3])
    return x, y, repairs, labels, mask, legal


def fit(model, data, sensor_accuracy, updates, seed, optimizer=None, offset=0):
    opt = optimizer or torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                         lr=.003, weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed+991+offset)
    curve = []
    x, y, repairs, labels, mask, legal = data
    for step in range(updates):
        ids = torch.randint(x.shape[0], (128,), generator=rng)
        fault, effects, sensor, policy = model.sequence(x[ids])
        m = mask[ids]
        fault_loss = F.binary_cross_entropy_with_logits(fault, y[ids], reduction="none")
        paired = fault.sigmoid().unsqueeze(-1)*(1-effects.sigmoid())
        effect_loss = F.binary_cross_entropy(paired.clamp(1e-7, 1-1e-7), repairs[ids], reduction="none").mean(-1)
        sensor_loss = F.binary_cross_entropy_with_logits(sensor, torch.full_like(sensor, sensor_accuracy), reduction="none")
        decision_loss = F.cross_entropy(policy.masked_fill(~legal[ids], -1e9).reshape(-1, 3),
                                        labels[ids].reshape(-1), reduction="none").reshape_as(m)
        if model.arm in ("policy_bc", "policy_dagger"):
            loss_each = decision_loss
        elif model.arm == "recurrent_belief":
            loss_each = fault_loss
        else:
            loss_each = fault_loss+effect_loss+sensor_loss
            if model.arm in ("joint", "frozen_writer"):
                loss_each = loss_each+decision_loss
        loss = (loss_each*m).sum()/m.sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        if step == 0 or step == updates-1:
            curve.append({"update": offset+step+1, "loss": float(loss.detach()), "gradient_norm": float(grad)})
    return opt, curve


def score_predictions(model, data, temperature=1.):
    with torch.no_grad():
        logit, effects, _, _ = model.sequence(data[0])
        m = data[4].bool()
        p = torch.sigmoid(logit/temperature)[m].clamp(1e-7, 1-1e-7)
        y = data[1][m]
        paired = (torch.sigmoid(logit/temperature).unsqueeze(-1)*(1-effects.sigmoid()))[m]
        return {"nll": float(F.binary_cross_entropy(p, y)), "brier": float(((p-y)**2).mean()),
                "repair_brier": float(((paired-data[2][m])**2).mean()), "events": int(m.sum())}


def rollout(spec, index, arm, params, model=None, calibration=None, table=None, collect_student=False):
    world = NativeWorld(spec["seed"], index, spec.get("shifted", False))
    true_params = world.rates
    # Known-config ceiling is the only arm supplied the shifted generator rates.
    policy_params = true_params if arm == "known_bayes" else params
    c = Controller(arm, policy_params, model, calibration, table)
    decisions = []
    prediction_pairs = []
    recoveries = failures_serviced = 0
    for rec in world.runtime.public_payload()["records"]:
        entity = int(rec["fields"]["entity"])
        c.begin(entity, rec["fields"])
        failed_before = False
        for _ in range(8):
            command = c.choose(entity)
            p, learned_params = c.predictions[entity]
            truth = world.fault(entity)  # evaluator diagnostic AFTER policy choice
            prediction_pairs.append((p, truth))
            decisions.append({"entity": entity, "command": COMMANDS[command], "p_fault": p,
                              "repair_risk": [p*(1-q) for q in learned_params[1]],
                              "sensor_accuracy": learned_params[0], "attempts": c.states[entity].attempts,
                              "senses": c.states[entity].senses})
            if command == 0:
                break
            for name in (("inspect",) if command == 1 else ("isolate", "service")):
                s = c.states[entity]
                receipt = world.act(name, entity, s.attempts+1 if name == "service" else None)
                c.observe(entity, name, receipt)
            if command == 2:
                if failed_before and world.fault(entity) == 0:
                    recoveries += 1
                failed_before = bool(world.fault(entity))
                failures_serviced += int(failed_before)
        else:
            raise AssertionError("bounded entity action horizon exhausted")
    result = world.finish()
    eps = 1e-7
    result.update({"arm": arm, "decisions": decisions,
                   "forecast_nll_sum": sum(-math.log(max(eps, min(1-eps, p)) if y else max(eps, 1-min(1-eps, p))) for p, y in prediction_pairs),
                   "forecast_brier_sum": sum((p-y)**2 for p, y in prediction_pairs),
                   "forecast_entropy_sum": sum(-max(eps, p)*math.log(max(eps, p))-max(eps, 1-p)*math.log(max(eps, 1-p)) for p, _ in prediction_pairs),
                   "prediction_count": len(prediction_pairs),
                   "recovered_after_failed_attempt": recoveries, "failed_attempts": failures_serviced,
                   "inference_seconds": c.inference_seconds})
    return result


def summarize(rows):
    n = len(rows)
    forecasts = sum(r["prediction_count"] for r in rows)
    return {"worlds": n, "cost_per_entity": sum(r["cost"] for r in rows)/(2*n),
            "residual_faults": sum(r["failures"] for r in rows),
            "completed_worlds": sum(r["complete"] for r in rows),
            "blocked_actions": sum(r["blocked"] for r in rows),
            "trajectory_violations": sum(r["violations"] for r in rows),
            "mean_inspections": sum(r["inspections"] for r in rows)/n,
            "mean_services": sum(r["services"] for r in rows)/n,
            "forecast_nll": sum(r["forecast_nll_sum"] for r in rows)/forecasts,
            "forecast_brier": sum(r["forecast_brier_sum"] for r in rows)/forecasts,
            "forecast_entropy": sum(r["forecast_entropy_sum"] for r in rows)/forecasts,
            "recoveries": sum(r["recovered_after_failed_attempt"] for r in rows),
            "failed_attempts": sum(r["failed_attempts"] for r in rows),
            "inference_seconds": sum(r["inference_seconds"] for r in rows)}


def strip_timing(obj):
    if isinstance(obj, dict):
        return {k: strip_timing(v) for k, v in obj.items() if not k.endswith("seconds")}
    if isinstance(obj, list):
        return [strip_timing(v) for v in obj]
    return obj
