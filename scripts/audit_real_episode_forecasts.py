"""Episode-paired real-trajectory forecasting audit for TAC-OSM variants.

Input JSON: {"episodes": [{"id": "...", "actual": [[...], ...],
"current": [[...], ...], "predictions": {"model": [[...], ...]},
"actions": [[...], ...] (optional)}]}. No synthetic data generated here.
All arrays are genuine aligned one-step physical transitions.
"""
import argparse
import json
import math
import random
from pathlib import Path


def _mae(a, b):
    if len(a) != len(b) or not a:
        raise ValueError("nonempty matching state dimensions required")
    return sum(abs(float(x)-float(y)) for x, y in zip(a, b))/len(a)


def evaluate(payload, motion_threshold, bootstrap, seed):
    episodes = payload["episodes"]
    if not episodes:
        raise ValueError("at least one independent real episode required")
    seen = set()
    per_episode = []
    models = None
    for ep in episodes:
        identifier = str(ep["id"])
        if identifier in seen:
            raise ValueError("duplicate episode identifier")
        seen.add(identifier)
        current, actual, predictions = ep["current"], ep["actual"], ep["predictions"]
        previous = ep.get("previous")
        if previous is not None and len(previous) != len(current):
            raise ValueError("previous state count mismatch")
        if models is None:
            models = tuple(sorted(predictions))
        if tuple(sorted(predictions)) != models or not current or len(current) != len(actual):
            raise ValueError("model names and aligned transition counts must match")
        reference_names = ("persistence", "constant_velocity") if previous is not None else ("persistence",)
        sums = {name: {"all": [], "motion": [], "static": []} for name in reference_names+models}
        for t, (x, y) in enumerate(zip(current, actual)):
            if not all(math.isfinite(float(v)) for v in x+y):
                raise ValueError("nonfinite physical state")
            motion = _mae(x, y)
            category = "motion" if motion > motion_threshold else "static"
            candidates = {"persistence": x}
            if previous is not None:
                before = previous[t]
                if len(before) != len(x) or not all(math.isfinite(float(v)) for v in before):
                    raise ValueError("nonfinite or misaligned previous physical state")
                candidates["constant_velocity"] = [2*float(v)-float(u) for v,u in zip(x,before)]
            for name in models:
                if len(predictions[name]) != len(current):
                    raise ValueError("prediction count mismatch")
                candidates[name] = predictions[name][t]
            for name, estimate in candidates.items():
                error = _mae(y, estimate)
                if not math.isfinite(error):
                    raise ValueError("nonfinite prediction")
                sums[name]["all"].append(error)
                sums[name][category].append(error)
        per_episode.append({"id": identifier, "errors": sums})
    rng = random.Random(seed)
    result = {"episode_count": len(per_episode), "motion_threshold": motion_threshold,
              "bootstrap_replicates": bootstrap, "bootstrap_seed": seed,
              "comparisons": {}}
    for name in models:
        result["comparisons"][name] = {}
        for stratum in ("all", "motion", "static"):
            paired = []
            for ep in per_episode:
                p = ep["errors"]["persistence"][stratum]
                m = ep["errors"][name][stratum]
                if p:
                    paired.append((sum(m)/len(m), sum(p)/len(p)))
            if not paired:
                result["comparisons"][name][stratum] = {"status": "no_samples"}
                continue
            differences = [m-p for m,p in paired]
            samples = []
            for _ in range(bootstrap):
                draws = [differences[rng.randrange(len(differences))] for _ in differences]
                samples.append(sum(draws)/len(draws))
            samples.sort()
            result["comparisons"][name][stratum] = {
                "episodes": len(paired),
                "model_episode_mean_mae": sum(m for m,_ in paired)/len(paired),
                "persistence_episode_mean_mae": sum(p for _,p in paired)/len(paired),
                "paired_difference_model_minus_persistence": sum(differences)/len(differences),
                "bootstrap_95pct_CI": [samples[int(.025*(bootstrap-1))],
                                         samples[int(.975*(bootstrap-1))]]}
    if all("constant_velocity" in ep["errors"] for ep in per_episode):
        result["constant_velocity"] = {}
        for stratum in ("all", "motion", "static"):
            paired = []
            for ep in per_episode:
                baseline = ep["errors"]["persistence"][stratum]
                velocity = ep["errors"]["constant_velocity"][stratum]
                if baseline:
                    paired.append((sum(velocity)/len(velocity), sum(baseline)/len(baseline)))
            if paired:
                result["constant_velocity"][stratum] = {
                    "episodes": len(paired),
                    "velocity_episode_mean_mae": sum(v for v, _ in paired)/len(paired),
                    "persistence_episode_mean_mae": sum(p for _, p in paired)/len(paired),
                }
            else:
                result["constant_velocity"][stratum] = {"status": "no_samples"}
    result["constant_velocity_available_for_all_episodes"] = all("constant_velocity" in ep["errors"] for ep in per_episode)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--motion-threshold", type=float, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=333)
    args = parser.parse_args()
    if args.motion_threshold < 0 or not math.isfinite(args.motion_threshold) or args.bootstrap < 100:
        parser.error("threshold must be nonnegative and bootstrap >= 100")
    report = evaluate(json.loads(Path(args.episodes_json).read_text()),
                      args.motion_threshold, args.bootstrap, args.seed)
    Path(args.output_json).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
