"""Frozen CODEVALUE003 state adapters. No training, corpus loading or scoring.

TAC070 has the same state equations as 040, but its distinct source class is
explicitly checked here. The reviewed 043 implementation is inherited without
weakening its original constructor. Guards run at observation-chunk boundaries.
All states include cached last logits and meter every retained tensor.
"""
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
PARAMS = {"tac": 499623, "gru": 500165, "ssm": 499782}
CONDITIONS = ("carry", "reset", "donor", "corrupt")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def _load(name, path):
    path = Path(path).resolve()
    if name in sys.modules:
        if Path(sys.modules[name].__file__).resolve() != path:
            raise ValueError("source module collision: " + name)
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


TRAIN = _load("langpersist096_train002", ROOT / "research/commercial-value/codevalue002/train.py")
POSITION = _load("langpersist096_position042", ROOT / "research/mathematical-engineering/langmemory042/candidate.py")
_old_candidate = sys.modules.get("candidate")
sys.modules["candidate"] = POSITION
try:
    STREAM = _load("langpersist096_stream043", ROOT / "research/mathematical-engineering/langstream043/incremental.py")
finally:
    if _old_candidate is None:
        sys.modules.pop("candidate", None)
    else:
        sys.modules["candidate"] = _old_candidate


def implementation_sha256():
    h = hashlib.sha256(str(torch.__version__).encode())
    for path in (Path(__file__), Path(TRAIN.__file__), Path(STREAM.__file__),
                 Path(POSITION.__file__), *(Path(sys.modules[n].__file__) for n in
                 ("scaled_recurrence", "context_transition", "recurrence_only", "model094"))):
        h.update(str(path.relative_to(ROOT)).encode()); h.update(path.read_bytes())
    return h.hexdigest()


class IncrementalTAC070096(STREAM.IncrementalTAC043):
    """Source070 bridge with a new explicit constructor, native positions only."""
    def __init__(self, model):
        if type(model) is not TRAIN.ScaledRecurrenceTAC070 or model.scale != "p500k" or model.parameter_count() != PARAMS["tac"] or model.upper_state_width != 48 or model.upper_concept_graph:
            raise ValueError("only exact registered source070 p500k is supported")
        if model.embedding.weight.dtype != torch.float32 or model.embedding.weight.device.type != "cpu":
            raise ValueError("CPU FP32 source070 required")
        self.model = model
        self.position_mode = "native"
        self.implementation_sha256 = implementation_sha256()
        self._binding = self._signature()
        self.weights_sha256 = self._weight_hash()
        self._check_model()


@dataclass(frozen=True)
class State096:
    arm: str
    owners: tuple[str, ...]
    clock: int
    checkpoint_sha256: str
    implementation_sha256: str
    tensors: dict[str, torch.Tensor]

    def tensor_bytes(self):
        return sum(x.numel()*x.element_size() for x in self.tensors.values())


def _construct_model(arm, saved, expected_sha):
    """Internal construction primitive; fixture tests make no training claims."""
    model = TRAIN.make(arm).cpu().float().eval()
    model.load_state_dict(saved["model"], strict=True)
    if not all(bool(p.isfinite().all()) for p in model.parameters()):
        raise ValueError("nonfinite checkpoint")
    model._checkpoint_sha256_096 = expected_sha
    return model


def make_model(arm, checkpoint_path, expected_sha):
    """Load only the fixed nine sealed originals, using local evidence only."""
    proof = _load("langpersist096_checkpoint_provenance", Path(__file__).with_name("checkpoints.py"))
    if arm not in PARAMS or expected_sha not in {v for (a, _), v in proof.CHECKPOINT_SHA256.items() if a == arm}:
        raise ValueError("unregistered checkpoint hash or arm")
    records = {}
    for path, digest in {**proof.METADATA_SHA256, **proof.FACTORY_SHA256}.items():
        local = ROOT / path
        if sha256(local) != digest:
            raise ValueError("pinned local source/evidence differs: " + path)
        if path in proof.METADATA_SHA256:
            records[local.name] = json.loads(local.read_text())
    rows = proof.validate_original_records(records["FINAL_MANIFEST.json"], records["SEAL.json"], records["SOURCE_RECEIPT.json"])
    row = next(r for r in rows if r["arm"] == arm and r["checkpoint_sha256"] == expected_sha)
    if sha256(checkpoint_path) != expected_sha:
        raise ValueError("checkpoint bytes differ from sealed original")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    verified = proof.verify_checkpoint_metadata(saved, row)
    model = _construct_model(arm, saved, expected_sha)
    model._source_metadata_096 = {**row, "payload_verification": verified}
    return model


class FrozenAdapter096:
    def __init__(self, model, arm):
        if arm not in PARAMS or sum(p.numel() for p in model.parameters()) != PARAMS[arm] or any(m.training for m in model.modules()):
            raise ValueError("registered frozen eval model required")
        expected_type = {"tac": TRAIN.ScaledRecurrenceTAC070, "gru": TRAIN.GRU500, "ssm": TRAIN.SimpleSSM500}[arm]
        if type(model) is not expected_type or any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("exact source CPU FP32 model required")
        self.model, self.arm = model, arm
        self.checkpoint = model._checkpoint_sha256_096
        self.implementation = implementation_sha256()
        self.engine = IncrementalTAC070096(model) if arm == "tac" else None
        self.signature = tuple((id(p), p._version) for p in model.parameters())
        self.shapes = STREAM.SHAPES if arm == "tac" else ({"hidden": (2, 211), "last_logits": (256,)} if arm == "gru" else {"hidden": (128,), "last_logits": (256,)})

    def validate(self, state):
        if type(state) is not State096 or (state.arm, state.checkpoint_sha256, state.implementation_sha256) != (self.arm, self.checkpoint, self.implementation):
            raise ValueError("state/model binding differs")
        if type(state.clock) is not int or not 0 <= state.clock <= 100000 or not state.owners or len(set(state.owners)) != len(state.owners) or any(type(x) is not str or not x for x in state.owners):
            raise ValueError("invalid owner/clock metadata")
        if set(state.tensors) != set(self.shapes) or self.signature != tuple((id(p), p._version) for p in self.model.parameters()):
            raise ValueError("state schema or frozen weights changed")
        for name, shape in self.shapes.items():
            x = state.tensors[name]
            if not isinstance(x, torch.Tensor) or x.shape != (len(state.owners), *shape) or x.dtype != torch.float32 or x.device.type != "cpu" or x.requires_grad or not bool(x.isfinite().all()):
                raise ValueError("invalid complete state: " + name)

    def start(self, owners):
        owners = tuple(owners)
        state = State096(self.arm, owners, 0, self.checkpoint, self.implementation,
                         {k: torch.zeros((len(owners), *v)) for k, v in self.shapes.items()})
        self.validate(state)
        return state

    @torch.inference_mode()
    def observe(self, state, observed):
        self.validate(state)
        x = torch.as_tensor(observed)
        if x.ndim != 2 or x.shape[0] != len(state.owners) or not 1 <= x.shape[1] <= 1024 or x.dtype not in (torch.int64, torch.uint8) or x.min() < 0 or x.max() > 255:
            raise ValueError("only bounded observed [B,T] byte tensors allowed")
        x = x.long()
        if self.arm == "tac":
            s = STREAM.InferenceState(state.owners, state.clock, self.engine.weights_sha256, "native", self.engine.implementation_sha256, state.tensors)
            self.engine.validate(s, state.owners)
            if state.clock == 0:
                logits, s = self.engine.prefill(x, owners=state.owners)
            else:
                mask = torch.ones(x.shape[0], dtype=torch.bool)
                for token in x.unbind(1):
                    logits, s = self.engine._advance(token, s, mask)
            tensors = s.tensors
        elif self.arm == "gru":
            out, hidden = self.model.gru(self.model.embedding(x), state.tensors["hidden"].transpose(0, 1).contiguous())
            logits = self.model.head(out[:, -1])
            tensors = {"hidden": hidden.transpose(0, 1).contiguous(), "last_logits": logits.clone()}
        else:
            recurrent = sys.modules["model094"].Recurrent094State(state.owners, state.clock, state.tensors["hidden"])
            out, s = TRAIN.SelectiveSSMLanguage094.forward(self.model, x, state=recurrent, owners=state.owners)
            logits = out[:, -1]
            tensors = {"hidden": s.value, "last_logits": logits.clone()}
        result = State096(self.arm, state.owners, state.clock+x.shape[1], self.checkpoint, self.implementation, tensors)
        self.validate(result)
        return logits, result

    def step(self, state, observed_byte):
        return self.observe(state, torch.as_tensor(observed_byte).reshape(-1, 1))


def acquire(model, arm, observed, owners):
    adapter = FrozenAdapter096(model, arm)
    return adapter.observe(adapter.start(owners), observed)[1]


def intervene(state, condition, donors=None, seed=96096):
    if condition not in CONDITIONS:
        raise ValueError("unregistered intervention")
    if condition == "donor":
        idx = torch.as_tensor(donors)
        if idx.dtype != torch.int64 or idx.shape != (len(state.owners),) or bool((idx < 0).any()) or bool((idx >= len(state.owners)).any()) or bool((idx == torch.arange(len(idx))).any()):
            raise ValueError("legal nonself donor indices required")
    values = {}
    generator = torch.Generator().manual_seed(seed)
    for name, x in state.tensors.items():
        if condition == "reset":
            values[name] = torch.zeros_like(x)
        elif condition == "donor":
            values[name] = x.index_select(0, idx).clone()
        elif condition == "corrupt":
            flat = x.reshape(len(state.owners), -1)
            values[name] = torch.stack([row[torch.randperm(row.numel(), generator=generator)] for row in flat]).reshape_as(x)
        else:
            values[name] = x.clone()
    return State096(state.arm, state.owners, state.clock, state.checkpoint_sha256, state.implementation_sha256, values)


def select_state(state, indices):
    indices = list(indices)
    return State096(state.arm, tuple(state.owners[i] for i in indices), state.clock,
                    state.checkpoint_sha256, state.implementation_sha256,
                    {k: v[indices].clone() for k, v in state.tensors.items()})


def join_states(states):
    first = states[0]
    if any((s.arm, s.clock, s.checkpoint_sha256, s.implementation_sha256) != (first.arm, first.clock, first.checkpoint_sha256, first.implementation_sha256) for s in states):
        raise ValueError("incompatible state chunks")
    return State096(first.arm, sum((s.owners for s in states), ()), first.clock,
                    first.checkpoint_sha256, first.implementation_sha256,
                    {k: torch.cat([s.tensors[k] for s in states]) for k in first.tensors})


def _tensor_digest(tensors):
    h = hashlib.sha256()
    for name, x in sorted(tensors.items()):
        h.update(name.encode()); h.update(str(tuple(x.shape)).encode()); h.update(x.contiguous().numpy().tobytes())
    return h.hexdigest()


def save_state(path, state, cost, acquisition_sha256):
    meta = dict(schema="langpersist096.state.v1", arm=state.arm, owners=list(state.owners), clock=state.clock,
                checkpoint_sha256=state.checkpoint_sha256, implementation_sha256=state.implementation_sha256,
                acquisition_sha256=acquisition_sha256, tensor_sha256=_tensor_digest(state.tensors), cost=dict(cost))
    meta["cost"]["state_payload_bytes"] = state.tensor_bytes()
    meta["cost"]["serialized_metadata_bytes"] = 0
    for _ in range(5):
        encoded = json.dumps(meta, sort_keys=True, separators=(",", ":"), allow_nan=False)
        size = len(encoded.encode())
        if size == meta["cost"]["serialized_metadata_bytes"]:
            break
        meta["cost"]["serialized_metadata_bytes"] = size
    with Path(path).open("xb") as handle:
        torch.save({"metadata": encoded, "tensors": state.tensors}, handle)
    return meta


def load_state(path, adapter):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or set(saved) != {"metadata", "tensors"}:
        raise ValueError("state file schema differs")
    meta = json.loads(saved["metadata"])
    if meta["schema"] != "langpersist096.state.v1" or meta["clock"] != 128 or _tensor_digest(saved["tensors"]) != meta["tensor_sha256"]:
        raise ValueError("state provenance/digest differs")
    state = State096(meta["arm"], tuple(meta["owners"]), meta["clock"], meta["checkpoint_sha256"], meta["implementation_sha256"], saved["tensors"])
    adapter.validate(state)
    if state.tensor_bytes() != meta["cost"]["state_payload_bytes"] or len(saved["metadata"].encode()) != meta["cost"]["serialized_metadata_bytes"]:
        raise ValueError("state accounting differs")
    return state, meta
