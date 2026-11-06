"""DECISION-SYS1-002: causal SCANIA feature adapter and coarsened censor loss.

Publisher TRAIN only. No download, fitting of a neural model, or access to TEST.
Time is anonymized operating time, NOT days. The labels never enter Readout.
Exact event times at 6/12/24/48 remain excluded pending publisher boundary proof.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import numbers
from typing import Iterable, Mapping

import numpy as np
import torch
from torch.nn import functional as F

FEATURES = tuple(f'{v}_{i}' for v, n in [(167,10),(272,10),(291,11),
                  (158,10),(459,20),(397,36)] for i in range(n)) + tuple(
                  f'{v}_0' for v in (171,666,427,837,309,835,370,100))
CUTOFFS = (6.0, 12.0, 24.0, 48.0)
SPLIT_SALT = 'WIL-DECISION-SYS1-002-TRAIN-INTERNAL-v1'


def vehicle_key(value: object) -> str:
    key = str(value)
    if not key.isascii() or not key.isdecimal() or str(int(key)) != key:
        raise ValueError('vehicle_id must be a canonical nonnegative integer string')
    return key


def split_of(vehicle_id: object) -> str:
    """Immutable, label-independent 80/20 FIT/DEV partition of publisher TRAIN.

    These are development partitions, NOT newly untouched scientific TEST.
    """
    key = vehicle_key(vehicle_id)
    digest = hashlib.sha256(f'{SPLIT_SALT}:{key}'.encode()).digest()
    return 'fit' if int.from_bytes(digest[:8], 'big') < (4 * 2**64)//5 else 'dev'


@dataclass(frozen=True)
class Readout:
    vehicle_id: str
    time_step: float
    values: tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(self, "vehicle_id", vehicle_key(self.vehicle_id))
        object.__setattr__(self, "time_step", float(self.time_step))
        object.__setattr__(self, "values", tuple(float(v) for v in self.values))
        if not math.isfinite(self.time_step) or self.time_step < 0:
            raise ValueError('invalid operating time')
        if len(self.values) != 105 or np.isinf(self.values).any():
            raise ValueError('expected 105 finite-or-missing measurements')

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]):
        expected = set(FEATURES) | {'vehicle_id', 'time_step'}
        if set(row) != expected:
            raise ValueError('observation schema mismatch; labels/extra fields forbidden')
        vals = tuple(float('nan') if row[k] in ('', None) else float(row[k]) for k in FEATURES)
        return cls(vehicle_key(row['vehicle_id']), float(row['time_step']), vals)

    @property
    def example_id(self) -> str:
        return hashlib.sha256(f'{self.vehicle_id}:{self.time_step.hex()}'.encode()).hexdigest()


class FitOnlyStatistics:
    """Streaming observed-value moments; only the frozen FIT identity set may fit.

    Missing values become the FIT mean (zero after standardization). No forward
    or backward fill, clipping, label conditioning, or inferred outcome feedback.
    """
    def __init__(self, fit_ids: Iterable[str]):
        self.fit_ids = frozenset(vehicle_key(v) for v in fit_ids)
        if not self.fit_ids or any(split_of(v) != 'fit' for v in self.fit_ids):
            raise ValueError('only frozen TRAIN-internal FIT vehicles can fit transforms')
        self.count = np.zeros(105, dtype=np.int64)
        self.mean = np.zeros(105, dtype=np.float64)
        self.m2 = np.zeros(105, dtype=np.float64)
        self.frozen = False
        self.rows = 0

    def update(self, row: Readout) -> None:
        if self.frozen:
            raise ValueError('transform already frozen')
        if row.vehicle_id not in self.fit_ids:
            raise ValueError('non-FIT vehicle cannot update transform')
        x = np.asarray(row.values, dtype=np.float64)
        valid = np.isfinite(x)
        n = self.count[valid] + 1
        delta = x[valid] - self.mean[valid]
        new_mean = self.mean[valid] + delta / n
        new_m2 = self.m2[valid] + delta * (x[valid] - new_mean)
        if not np.isfinite(new_mean).all() or not np.isfinite(new_m2).all():
            raise ValueError('moment overflow')
        self.count[valid], self.mean[valid], self.m2[valid] = n, new_mean, new_m2
        self.rows += 1

    def freeze(self) -> None:
        if self.rows == 0 or np.any(self.count == 0):
            raise ValueError('every feature needs observed FIT data')
        self.scale = np.sqrt(np.maximum(self.m2 / self.count, 0))
        self.scale[self.scale == 0] = 1.0
        self.frozen = True
        for a in (self.count, self.mean, self.m2, self.scale):
            a.flags.writeable = False

    def transform(self, row: Readout, previous_time: float | None) -> np.ndarray:
        if not self.frozen:
            raise ValueError('freeze training statistics first')
        if previous_time is not None and (not math.isfinite(previous_time)
                or previous_time < 0 or row.time_step <= previous_time):
            raise ValueError('duplicate/out-of-order time within vehicle')
        x = np.asarray(row.values, dtype=np.float64)
        missing = np.isnan(x)
        z = (np.where(missing, self.mean, x) - self.mean) / self.scale
        elapsed = 0.0 if previous_time is None else row.time_step - previous_time
        out = np.concatenate((z, missing.astype(np.float64),
                              [math.log1p(elapsed), math.log1p(row.time_step)]))
        if not np.isfinite(out).all():
            raise ValueError('nonfinite transformed input')
        return out

    def receipt(self) -> dict:
        if not self.frozen:
            raise ValueError('freeze first')
        payload = {'feature_order': FEATURES, 'fit_vehicle_ids': sorted(self.fit_ids),
                   'rows': self.rows, 'counts': self.count.tolist(),
                   'mean': self.mean.tolist(), 'scale': self.scale.tolist()}
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        return {**payload, 'sha256': hashlib.sha256(encoded).hexdigest()}


class CausalFeatureStream:
    def __init__(self, stats: FitOnlyStatistics):
        if not stats.frozen:
            raise ValueError('frozen transforms required')
        self.stats = stats
        self.last_time: dict[str, float] = {}

    def transform(self, row: Readout) -> np.ndarray:
        out = self.stats.transform(row, self.last_time.get(row.vehicle_id))
        self.last_time[row.vehicle_id] = row.time_step
        return out


@dataclass(frozen=True)
class SurvivalTarget:
    values: tuple[float, float, float, float]
    observed: tuple[bool, bool, bool, bool]
    known_class: int | None
    status: str


def survival_target(readout_time: float, study_end: float, repaired: int) -> SurvivalTarget:
    """Coarsen follow-up to complete bins; never label unknown future as negative.

    Right censoring inside a bin contributes survival through preceding completed
    bins ONLY. This sacrifices partial-bin information, without assuming a within-
    bin hazard. The model therefore fits a coarsened discrete-time likelihood.
    Missingness/independent-censoring assumptions still require real-source audit.
    """
    if not isinstance(repaired, numbers.Integral) or isinstance(repaired, bool) or repaired not in (0, 1):
        raise ValueError('repair indicator must be integer 0 or 1')
    if not all(math.isfinite(t) and t >= 0 for t in (readout_time, study_end)):
        raise ValueError('invalid target times')
    remaining = study_end - readout_time
    empty = SurvivalTarget((0.,)*4, (False,)*4, None, 'no_observed_horizon')
    if remaining < 0 or (repaired == 1 and remaining == 0):
        raise ValueError('reading at/after repair or after study end excluded')
    if repaired == 1 and remaining in CUTOFFS:
        return SurvivalTarget(empty.values, empty.observed, None, 'unqualified_exact_boundary')
    if repaired == 1:
        idx = sum(remaining > c for c in CUTOFFS)
        vals = tuple(float(i == idx) for i in range(4))
        obs = tuple(i <= idx for i in range(4))
        return SurvivalTarget(vals, obs, 0 if idx == 4 else 4-idx, 'event')
    obs = tuple(remaining >= c for c in CUTOFFS)
    return SurvivalTarget(empty.values, obs, 0 if all(obs) else None,
                          'censored' if any(obs) else 'no_observed_horizon')


def censor_aware_loss(logits: torch.Tensor, targets: list[SurvivalTarget],
                      reduction: str = 'mean') -> torch.Tensor:
    """Mean per-informative-readout negative log likelihood, not per-hazard mean.

    Repeated readouts need an explicitly registered vehicle weighting/sampling
    scheme in the eventual trainer. No claim that this is a fleet-averaged metric.
    """
    if logits.ndim != 2 or logits.shape[1] != 4 or logits.shape[0] != len(targets):
        raise ValueError('target/logit shape mismatch')
    if not torch.isfinite(logits).all():
        raise ValueError('nonfinite logits')
    y = logits.new_tensor([t.values for t in targets])
    mask = torch.tensor([t.observed for t in targets], dtype=torch.bool, device=logits.device)
    rows = torch.where(mask, F.binary_cross_entropy_with_logits(logits, y, reduction='none'), 0).sum(-1)
    if reduction == 'none':
        return rows
    if reduction != 'mean':
        raise ValueError('unsupported reduction')
    informative = mask.any(-1)
    if not informative.any():
        raise ValueError('batch contains no observed likelihood terms')
    return rows[informative].mean()
