"""Source-frozen grouped chronological split; no postrepair feature leakage."""
import csv
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

SOURCE_SHA = '6062eaa4c758d07820e7a13c4c31f783970dbf5598fa554afee2de5a36d58258'
MISSING = ('', 'na', 'n/a', 'null', 'missing', 'nan')


def normalize(value):
    return ' '.join(value.strip().casefold().split())


def date(value):
    for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y'):
        try: return datetime.datetime.strptime(value.strip(), fmt).date()
        except ValueError: pass
    return None


def number(value):
    try: result = float(value.strip())
    except (ValueError, TypeError): return None
    return result if np.isfinite(result) else None


def source(path):
    path = Path(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_SHA
    return list(csv.DictReader(path.open()))


def split(rows):
    # No outcome values are read in this function.
    excluded = {normalize(r['water_point_name']) for r in rows[:10]}
    groups = {}
    invalid = []
    for i, row in enumerate(rows):
        group = normalize(row['water_point_name']); tick = date(row['chlorination_rehab_date'])
        if group in excluded or not group or tick is None:
            invalid.append(i); continue
        groups.setdefault(group, []).append((i, tick))
    ordered = sorted(max(d for _, d in group) for group in groups.values())
    first = ordered[int(.6*len(ordered))-1]; second = ordered[int(.8*len(ordered))-1]
    result = {name: [] for name in ('train', 'validation', 'test', 'purged')}
    hashes = {}
    for group, visits in groups.items():
        low = min(d for _, d in visits); high = max(d for _, d in visits)
        name = ('train' if high <= first else 'validation' if low > first and high <= second
                else 'test' if low > second else 'purged')
        group_hash = hashlib.sha256(group.encode()).hexdigest()
        result[name].extend(i for i, _ in visits)
        for i, _ in visits: hashes[i] = group_hash
    return result, hashes, dict(reported_date_boundaries=[str(first), str(second)],
        excluded_preview_groups=len(excluded), excluded_rows=len(invalid),
        rows={k: len(v) for k, v in result.items()},
        groups={k: len({hashes[i] for i in v}) for k, v in result.items()},
        assignment_sha256=hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest())


def columns(rows, arm):
    # Before functionality is explicitly named as pre-intervention; replacement
    # fields name recorded actions. Other timing-ambiguous inspection fields and
    # ALL after-functionality/bucket/strokes/text/location/date fields are absent.
    categorical = ['functional_status_before']
    numeric = []
    if arm == 'repair':
        for name in rows[0]:
            if name.startswith('replaced_'):
                (categorical if name.endswith('_tick') else numeric).append(name)
        categorical.append('chlorination_done')
    return categorical, numeric


def fit_schema(rows, train_ids, arm):
    categorical, numeric = columns(rows, arm)
    vocabulary = {name: sorted({normalize(rows[i][name]) for i in train_ids
                    if normalize(rows[i][name]) not in MISSING}) for name in categorical}
    medians = {}
    for name in numeric:
        values = [number(rows[i][name]) for i in train_ids]
        values = [v for v in values if v is not None and v >= 0]
        medians[name] = float(np.median(values)) if values else 0.
    return dict(arm=arm, categorical=categorical, numeric=numeric, vocabulary=vocabulary,
                numeric_imputation=medians, numeric_transform='log1p nonnegative count/length; explicit missing flag')


def features(rows, ids, schema):
    result = []
    for i in ids:
        vector = []
        for name in schema['categorical']:
            value = normalize(rows[i][name]); vocab = schema['vocabulary'][name]
            vector.extend(float(value == known) for known in vocab)
            vector += [float(value in MISSING), float(value not in vocab and value not in MISSING)]
        for name in schema['numeric']:
            value = number(rows[i][name]); missing = value is None or value < 0
            value = schema['numeric_imputation'][name] if missing else value
            vector += [float(np.log1p(value)), float(missing)]
        result.append(vector)
    return np.asarray(result, dtype=np.float64)


def labeled(rows, ids):
    valid, targets, rejected = [], [], []
    for i in ids:
        value = number(rows[i]['time_to_fill_20l_bucket'])
        if value is None or value <= 0:
            rejected.append(i); continue
        valid.append(i); targets.append(value)
    return valid, np.asarray(targets, dtype=np.float64), rejected
