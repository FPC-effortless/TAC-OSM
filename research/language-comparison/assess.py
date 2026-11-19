"""Check a frozen language comparison's reported endpoints without running models."""
import argparse
import json
import math
from pathlib import Path

BASELINES = ('nanochat', 'gru', 'mamba3')
MODELS = ('tac_osm', *BASELINES)


def check_number(value, *, positive=False):
    if (type(value) not in (float, int) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError('Expected a finite measurement in the allowed range')
    return value


def at_most(value, limit):
    return value <= limit or math.isclose(value, limit, rel_tol=0, abs_tol=1e-12)


def assess(record):
    seeds = record['heldout_bits_per_byte']
    if len(seeds) < 3 or any(type(row['seed']) is not int for row in seeds):
        raise ValueError('Expected at least three integer seed IDs')
    if len({row['seed'] for row in seeds}) != len(seeds):
        raise ValueError('Duplicate seed IDs')
    differences = {baseline: [] for baseline in BASELINES}
    for row in seeds:
        scores = row['scores']
        if set(scores) != set(MODELS):
            raise ValueError('Each seed needs all four model scores')
        candidate = check_number(scores['tac_osm'])
        for baseline in BASELINES:
            differences[baseline].append(candidate - check_number(scores[baseline]))

    profiles = record['inference_profiles']
    if len(profiles) < 2 or len({row['id'] for row in profiles}) != len(profiles):
        raise ValueError('Expected at least two unique registered inference profiles')
    inference = []
    for profile in profiles:
        times = profile['median_end_to_end_ms']
        if set(times) != set(MODELS):
            raise ValueError('Each inference profile needs all four model times')
        candidate = check_number(times['tac_osm'], positive=True)
        ratios = {name: candidate / check_number(times[name], positive=True)
                  for name in BASELINES}
        inference.append({'id': profile['id'], 'time_ratio': ratios,
                          'passes': all(at_most(ratio, 0.8) for ratio in ratios.values())})

    quality = {
        'beats_nanochat': at_most(max(differences['nanochat']), -0.01),
        'beats_gru': at_most(max(differences['gru']), -0.01),
        'matches_or_beats_mamba3': at_most(max(differences['mamba3']), 0.01),
    }
    return {
        'scope': 'Threshold arithmetic on supplied measurements',
        'passes': all(quality.values()) and all(row['passes'] for row in inference),
        'quality': quality,
        'paired_loss_delta': differences,
        'inference': inference,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('measurements', type=Path)
    args = parser.parse_args()
    result = assess(json.loads(args.measurements.read_text()))
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result['passes'] else 1)
