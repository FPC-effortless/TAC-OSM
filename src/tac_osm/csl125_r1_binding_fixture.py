"""No-fit identity-disjoint binding evaluator, NOT a learned model capability run.

Events are prediction-time visible; oracle targets exist only in independent
scoring functions. Synthetic TRAIN/DEV fixtures are diagnostic test inputs,
NOT authenticated real-world evidence, not the protected TEST set, and not
training authorization. Preserves all prior R9/R10/R11 failures.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Literal

ABSENT = 0

@dataclass(frozen=True)
class Event:
    kind: Literal['write', 'query']
    key: str
    value: int | None = None

    def __post_init__(self):
        if self.kind not in ('write','query') or not self.key:
            raise ValueError('invalid event')
        if self.kind == 'query' and self.value is not None:
            raise ValueError('query cannot expose its expected value')
        if self.kind == 'write' and (type(self.value) is not int or self.value <= ABSENT):
            raise ValueError('write requires a positive integer payload')


def six_query_history(keys: tuple[str, str, str], values: tuple[int, int, int],
                      before_order: tuple[int,int,int]=(0,1,2),
                      after_order: tuple[int,int,int]=(0,1,2)) -> tuple[Event, ...]:
    a,b,c=keys
    old,untouched,replaced=values
    if (sorted(before_order) != [0,1,2] or sorted(after_order) != [0,1,2]):
        raise ValueError('independent per-phase query orders must be permutations')
    if len(set(keys)) != 3 or len(set(values)) != 3 or any(v <= ABSENT for v in values):
        raise ValueError('distinct keys/values required')
    pre=tuple(Event('query',keys[k]) for k in before_order)
    post=tuple(Event('query',keys[k]) for k in after_order)
    return (Event('write', a, old), Event('write', b, untouched),
            *pre, Event('write', a, replaced), *post)


def exact_oracle(events: tuple[Event, ...]) -> tuple[int, ...]:
    """External keyed store used for gold construction, never fed to a model."""
    state: dict[str, int] = {}
    results: list[int] = []
    for event in events:
        if event.kind == 'write':
            assert event.value is not None
            state[event.key]=event.value
        else:
            results.append(state.get(event.key, ABSENT))
    return tuple(results)


def always_absent(events: tuple[Event, ...]) -> tuple[int, ...]:
    return (ABSENT,)*sum(event.kind=='query' for event in events)


def latest_value_without_identity(events: tuple[Event, ...]) -> tuple[int, ...]:
    last = ABSENT
    result=[]
    for event in events:
        if event.kind=='write':
            assert event.value is not None
            last=event.value
        else:
            result.append(last)
    return tuple(result)


def score(predictions: tuple[tuple[int, ...], ...],
          histories: tuple[tuple[Event, ...], ...]) -> dict[str, int | float]:
    if len(predictions) != len(histories) or not histories:
        raise ValueError('nonempty exactly matched batches required')
    total=0; correct=0; complete=0; absent_correct=0; absent_total=0
    for predicted, events in zip(predictions,histories):
        gold=exact_oracle(events)
        if len(predicted)!=len(gold) or len(gold)!=6 or any(type(v) is not int for v in predicted):
            raise ValueError('exact six-query integer predictions required')
        matched=[p==y for p,y in zip(predicted,gold)]
        correct+=sum(matched);total+=len(gold);complete+=int(all(matched))
        absent_total+=sum(y==ABSENT for y in gold)
        absent_correct+=sum(p==y for p,y in zip(predicted,gold) if y==ABSENT)
    return {'histories':len(histories),'complete_histories':complete,
            'exact_answers':correct,'query_count':total,
            'absent_query_count':absent_total,'absent_exact':absent_correct,
            'per_query_accuracy':correct/total,
            'complete_history_accuracy':complete/len(histories)}


def make_partition(partition: Literal['TRAIN','DEV'], *, count:int=128,
                   seed:int=19051) -> tuple[tuple[Event, ...], ...]:
    """Identities disjoint across partitions; value alphabet intentionally shared."""
    if partition not in ('TRAIN','DEV') or type(count) is not int or count<1 or count>1000:
        raise ValueError('unsupported diagnostic partition or count')
    prefix='T' if partition=='TRAIN' else 'D'
    # Independent key namespaces guarantee disjoint identity sets even if a
    # caller uses the same seed. This is not a substitute for corpus provenance.
    rng=random.Random(seed)
    rows=[]
    for i in range(count):
        keys=tuple(f'{prefix}{i:04d}-{j}' for j in rng.sample(range(8),3))
        values=tuple(rng.sample(range(1,9),3))
        before=list(range(3));after=list(range(3))
        rng.shuffle(before);rng.shuffle(after)
        rows.append(six_query_history(keys, values, tuple(before), tuple(after)))
    return tuple(rows)


def fixture_digest(histories: tuple[tuple[Event, ...], ...]) -> str:
    value=[[(ev.kind,ev.key,ev.value) for ev in row] for row in histories]
    return hashlib.sha256(json.dumps(value,separators=(',',':'),sort_keys=False).encode()).hexdigest()


def emit_report() -> dict:
    train,dev=make_partition('TRAIN'),make_partition('DEV')
    keys=lambda rows:set(ev.key for row in rows for ev in row)
    if keys(train)&keys(dev):raise AssertionError('TRAIN/DEV identity leakage')
    result={'stage':'CONSTRUCTED_ORACLE_AND_NEGATIVE_CONTROLS_ONLY',
            'model_calls':0,'optimizer_updates':0,'source_qualified':False,
            'train_hash':fixture_digest(train),'dev_hash':fixture_digest(dev),
            'train_dev_key_overlap':0,'histories_each':len(dev),
            'scores':{},'note':'Neither perfect symbolic store nor synthetic fixture demonstrates CSL learned memory.'}
    for name, method in [('symbolic_exact_oracle',exact_oracle),
                         ('always_absent',always_absent),
                         ('latest_value_no_identity',latest_value_without_identity)]:
        result['scores'][name]=score(tuple(method(row) for row in dev),dev)
    return result

if __name__=='__main__':
    print(json.dumps(emit_report(),indent=2))
