"""Offline scalar calibration of frozen categorical logits; no neural training.

Guo et al., ICML 2017. Real-source provenance/replay remain external prerequisites.
"""
from dataclasses import dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class LogitBatch:
    logits: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...]
    documents: tuple[str, ...]
    role: str

    def validate(self):
        if self.role not in ('calibration', 'score'):
            raise ValueError('only declared calibration/score FIT/DEV roles')
        if not self.logits or not (len(self.logits) == len(self.labels) == len(self.documents)):
            raise ValueError('nonempty aligned frozen cases required')
        vocab = len(self.logits[0])
        if vocab < 2:
            raise ValueError('categorical vocabulary required')
        for row, y, doc in zip(self.logits, self.labels, self.documents):
            if len(row) != vocab or any(not math.isfinite(x) for x in row):
                raise ValueError('finite same-vocabulary logits required')
            if type(y) is not int or not 0 <= y < vocab or not isinstance(doc, str) or not doc:
                raise ValueError('valid target and source document identity required')
        return vocab


def loss_derivatives(batch: LogitBatch, inverse_temperature: float):
    """Mean CE, dCE/da and d²CE/da²=mean categorical variance.

    Subtracting each unscaled row maximum preserves the mathematical law and
    improves numerical stability. This executable FP64 implementation is not a
    whole-model IEEE/AD proof.
    """
    batch.validate()
    a = inverse_temperature
    if not math.isfinite(a) or a <= 0:
        raise ValueError('positive finite inverse temperature required')
    losses, gradients, curvatures = [], [], []
    for row, y in zip(batch.logits, batch.labels):
        maximum = max(row)
        centered = [x - maximum for x in row]
        scaled = [a * x for x in centered]
        if not all(math.isfinite(x) for x in scaled):
            raise ValueError('logit range exceeds finite calibration arithmetic')
        exps = [math.exp(x) for x in scaled]
        total = math.fsum(exps)
        probabilities = [x / total for x in exps]
        mean = math.fsum(p * x for p, x in zip(probabilities, centered))
        variance = math.fsum(p * (x - mean)**2 for p, x in zip(probabilities, centered))
        losses.append(math.log(total) - scaled[y])
        gradients.append(mean - centered[y])
        curvatures.append(variance)
    n = len(losses)
    result = tuple(math.fsum(xs) / n for xs in (losses, gradients, curvatures))
    if not all(math.isfinite(x) for x in result):
        raise ValueError('calibration arithmetic not finite')
    return result


@dataclass(frozen=True)
class ScalarCalibration:
    inverse_temperature: float
    calibration_documents: frozenset[str]
    calibration_sha256: str
    calibration_nll: float
    identity_nll: float
    vocabulary: int


def fit_temperature(batch: LogitBatch) -> ScalarCalibration:
    vocab = batch.validate()
    if batch.role != 'calibration':
        raise ValueError('scoring labels cannot select temperature')
    low, high = .25, 4.
    # Monotone exact-real gradient. Fixed search, no score-set selection.
    for _ in range(64):
        mid = (low + high) / 2
        if loss_derivatives(batch, mid)[1] > 0:
            high = mid
        else:
            low = mid
    candidates = (1., .25, 4., (low + high) / 2)
    identity = loss_derivatives(batch, 1.)[0]
    losses = [(loss_derivatives(batch, a)[0], a) for a in candidates]
    # Identity wins exact numerical ties. This prevents needless flat-law changes.
    best_loss, a = min(losses, key=lambda pair: (pair[0], pair[1] != 1., pair[1]))
    raw = json.dumps([batch.logits, batch.labels, batch.documents],
                     separators=(',', ':'), allow_nan=False).encode()
    return ScalarCalibration(a, frozenset(batch.documents), hashlib.sha256(raw).hexdigest(),
                             best_loss, identity, vocab)


def score_temperature(batch: LogitBatch, fitted: ScalarCalibration, *, codec: str):
    vocab = batch.validate()
    if batch.role != 'score' or set(batch.documents) & fitted.calibration_documents:
        raise ValueError('scoring documents must be disjoint from calibration')
    if vocab != fitted.vocabulary:
        raise ValueError('calibration and scoring vocabularies differ')
    if codec not in ('mapped128', 'byte256') or vocab != {'mapped128': 128, 'byte256': 256}[codec]:
        raise ValueError('explicit original codec/vocabulary contract required')
    base = loss_derivatives(batch, 1.)[0]
    nll = loss_derivatives(batch, fitted.inverse_temperature)[0]
    count = len(batch.labels)
    top1 = top5 = 0
    for row, y in zip(batch.logits, batch.labels):
        order = sorted(range(vocab), key=lambda i: (-row[i], i))
        top1 += y == order[0]
        top5 += y in order[:5]
    return dict(nll_nats=nll, identity_nll_nats=base,
                perplexity=math.exp(nll) if nll < 709 else None,
                top1=top1/count, top5=top5/count, targets=count,
                source_documents=len(set(batch.documents)),
                bits_per_symbol=nll/math.log(2),
                raw_bits_per_byte=nll/math.log(2) if codec == 'byte256' else None,
                inverse_temperature=fitted.inverse_temperature,
                calibration_sha256=fitted.calibration_sha256,
                status='INTERNAL_EXPOSED_FIT_DEV_NOT_INDEPENDENTLY_QUALIFIED')
