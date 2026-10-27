"""Original LATFUSE008 graph instrumentation, called only after authorization.

No CLI, optimizer or source download. Offline writer/oracle traces are separate
from native inputs. Source-derived traces are diagnostics, not model state.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import random
import subprocess
import sys

from tac_osm.historical_read_controls import WriteIdentity, select_intervening_writes
from tac_osm.history_audit_authorization import AuditBlocked


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_implementation(root, commit):
    """Reject changed, extra/ignored or symlinked Python before importing it."""
    root = Path(root).resolve()
    git = lambda *args: subprocess.check_output(["git", "-C", str(root), *args])
    if git("rev-parse", "HEAD").decode().strip() != commit:
        raise AuditBlocked("original implementation commit mismatch")
    names = git("ls-tree", "-r", "--name-only", commit, "scripts", "src").decode().splitlines()
    expected = {p for p in names if p.endswith(".py")}
    actual = {p.relative_to(root).as_posix() for folder in ("scripts", "src")
              for p in (root / folder).rglob("*.py")}
    if expected != actual:
        raise AuditBlocked("extra or missing original Python source")
    for p in sorted(expected):
        path = root / p
        if any(q.is_symlink() for q in (path, *path.parents) if q != root.parent):
            raise AuditBlocked("symlinked original Python source")
        if path.read_bytes() != git("show", f"{commit}:{p}"):
            raise AuditBlocked("modified original implementation: " + p)
    return root


def load_original_modules(root):
    # Keep the current guard/selector already loaded, but resolve the ORIGINAL
    # scripts and model graph from the verified pinned checkout. No exec strings.
    import tac_osm
    import scripts
    tac_osm.__path__.insert(0, str(root / "src/tac_osm"))
    scripts.__path__ = [str(root / "scripts"), *scripts.__path__]
    fit = importlib.import_module("scripts.latfuse008_paired_real_training")
    source = importlib.import_module("scripts.latfuse002_real_fit")
    for name, module in list(sys.modules.items()):
        if (name.startswith("tac_osm.latent_fusion_") or
                name.startswith("scripts.latfuse")) and getattr(module, "__file__", None):
            if not Path(module.__file__).resolve().is_relative_to(root):
                raise AuditBlocked("mixed original model/loader implementations")
    return source, fit


def tensor_sha(tensor):
    t = tensor.detach().cpu().contiguous()
    header = json.dumps([str(t.dtype), list(t.shape)], separators=(",", ":")).encode()
    return hashlib.sha256(header + t.numpy().tobytes()).hexdigest()


def verify_causal_features(d, source, stop_hour):
    """Rebuild actual authorized prefix normalization/last/age from raw data.

    No fields beyond the permitted query prefix are needed. Normalizer fitting
    stops before March 2014, not at the later source loader's temporary cutoff.
    """
    import torch
    import datetime as dt
    cutoff = source.hour(dt.datetime(2014, 3, 1))
    raw = d["raw"][:, :stop_hour + 1].clone()
    if not cutoff <= stop_hour or raw.shape[1] != stop_hour + 1:
        raise AuditBlocked("normalizer/prediction-time prefix unavailable")
    raw[:, :, :6] = torch.where(torch.isfinite(raw[:, :, :6]),
                               torch.log1p(raw[:, :, :6].clamp_min(0)), raw[:, :, :6])
    past = raw[:, :cutoff]
    mask = torch.isfinite(past)
    n = mask.sum((0, 1)).clamp_min(1)
    center = torch.where(mask, past, 0).sum((0, 1)) / n
    spread = (torch.where(mask, past - center, 0).square().sum((0, 1)) / n).clamp_min(1e-6).sqrt()
    torch.testing.assert_close(center, d["center"], rtol=1e-6, atol=2e-6)
    torch.testing.assert_close(spread, d["spread"], rtol=1e-6, atol=2e-6)
    normalized = (raw - center) / spread
    valid = torch.isfinite(normalized)
    values = torch.where(valid, normalized, 0)
    torch.testing.assert_close(values, d["values"][:, :stop_hour + 1], rtol=1e-6, atol=2e-6)
    if not torch.equal(valid, d["valid"][:, :stop_hour + 1]):
        raise AuditBlocked("actual source missingness differs")
    recent = torch.zeros((12, 13))
    age = torch.full_like(recent, 256.)
    for t in range(stop_hour + 1):
        if not torch.equal(recent, d["last"][:, t]):
            raise AuditBlocked("future/noncausal last-value cache")
        age = torch.where(valid[:, t], 0., (age + 1).clamp(max=256))
        if not torch.equal(age, d["age"][:, t]):
            raise AuditBlocked("future/noncausal source age cache")
        recent = torch.where(valid[:, t], values[:, t], recent)


def trace_case(model, d, source, *, site, anchor, lag, control_count, control_seed):
    """Trace one owner; no labels, pair ordering or raw targets are arguments.

    Capture projection calls made by actual step(), not a guessed historical
    re-encoding. Reconstruct its update and its native event read separately.
    """
    import torch
    import torch.nn.functional as F
    from tac_osm.latent_fusion_004_factorial import head_weights
    from scripts.latfuse011_real_memory_falsifiers import read_query, projected_read

    slots = torch.tensor([site], dtype=torch.long)
    state = model.initial(1)
    captured = {}
    handles = []
    for name in ("key_map", "val_map", "retention_map", "write_map", "write_router"):
        def hook(module, args, output, name=name):
            captured.setdefault(name, []).append((args[0].detach().clone(), output.detach().clone()))
        handles.append(getattr(model, name).register_forward_hook(hook))
    trace, keys, values, writers = [], {}, {}, []
    update_error = 0.
    try:
        for hour in range(anchor - 23, anchor + lag + 1):
            captured.clear()
            inputs = [d[k][slots, hour] for k in ("values", "valid", "age", "last")]
            previous = state.associative.clone()
            state = model.step(state, *inputs, slots)
            if any(len(captured.get(name, [])) != 1 for name in (
                    "key_map", "val_map", "retention_map", "write_map", "write_router")):
                raise AuditBlocked("actual writer invocation missing or ambiguous")
            encoded = captured["key_map"][0][0]
            if any(not torch.equal(encoded, got[0][0]) for got in captured.values()):
                raise AuditBlocked("writer projections used different actual encoder inputs")
            key = F.normalize(captured["key_map"][0][1].reshape(1, 2, 8), dim=-1, eps=1e-8)
            value = torch.tanh(captured["val_map"][0][1])
            route, eligible = head_weights(captured["write_router"][0][1],
                model.log_head_temperature, head_count=2, soft=True)
            eligible = eligible & inputs[1].any(-1)[:, None]
            alpha = torch.where(eligible, torch.sigmoid(captured["retention_map"][0][1]), 1.)
            beta = torch.sigmoid(captured["write_map"][0][1]) * route * eligible
            old = previous.reshape(1, 16, 2, 8).permute(0, 2, 1, 3)
            retained = alpha[:, :, None, None] * old
            directional = beta[:, :, None, None] * (
                value[:, None, :, None] - retained @ key.unsqueeze(-1)) * key[:, :, None, :]
            reconstructed = (retained + directional).permute(0, 2, 1, 3).contiguous().reshape(1, 16, 16)
            error = float((reconstructed - state.associative).abs().max())
            if error > 2e-6 or not torch.isfinite(reconstructed).all():
                raise AuditBlocked("original write/source-to-matrix correspondence failed")
            update_error = max(update_error, error)
            actual_write = bool(directional.ne(0).any())
            if actual_write:
                writers.append(WriteIdentity(str(site), hour))
                keys[hour] = key.clone()
                values[hour] = value.clone()
            trace.append({"owner": site, "hour": hour,
                "input_sha256": [tensor_sha(x) for x in inputs],
                "encoded_sha256": tensor_sha(encoded), "key_sha256": tensor_sha(key),
                "source_observed": bool(inputs[1].any()), "directional_write_nonzero": actual_write,
                "matrix_before_sha256": tensor_sha(previous),
                "matrix_after_sha256": tensor_sha(state.associative),
                "reconstruction_max_abs_error": error})
    finally:
        for handle in handles:
            handle.remove()
    selection = select_intervening_writes(writers, target=WriteIdentity(str(site), anchor),
        query_hour=anchor + lag, count=control_count, seed=control_seed)
    clock = source.calendar([anchor])[:, None, :]
    native_key, route = read_query(model, state, torch.tensor([anchor]))
    read = projected_read(state.associative, native_key, route)
    native = model.historic_event_logits(state, slots, clock)[0, 0]
    decomposed = model.historic_event_head(read)[0, 0]
    if not torch.allclose(native, decomposed, rtol=0, atol=2e-6):
        raise AuditBlocked("native read-to-prediction decomposition failed")
    zeroed = state._replace(associative=torch.zeros_like(state.associative))
    zero_probability = float(model.historic_event_logits(zeroed, slots, clock)[0, 0].sigmoid())
    oracle = projected_read(state.associative, keys[anchor], route)
    controls = []
    for hour in selection["selected_hours"]:
        control_read = projected_read(state.associative, keys[hour], route)
        controls.append({"hour": hour, "age_hours": anchor + lag - hour,
            "key_sha256": tensor_sha(keys[hour]),
            "privileged_event_probability": float(model.historic_event_head(control_read)[0, 0].sigmoid())})
    query = anchor + lag
    future_clock = source.calendar([query + 6])[:, None, :]
    mean, sd = model.forecast(state, slots, future_clock)
    last = torch.where(d["valid"][site, query], d["values"][site, query], d["last"][site, query])
    # No historical truth is consulted by this producer. Targets are scored by
    # its caller only after both original architectures finish prediction.
    result = {"owner": site, "anchor_hour": anchor, "query_hour": query,
        "native_probability": float(native.sigmoid()), "zero_read_probability": zero_probability,
        "oracle_probability_PRIVILEGED": float(model.historic_event_head(oracle)[0, 0].sigmoid()),
        "oracle_value_cosine_geometry_only": float(F.cosine_similarity(oracle, values[anchor], dim=-1)),
        "native_value_cosine_geometry_only": float(F.cosine_similarity(read, values[anchor], dim=-1)),
        "control_selection": selection, "controls_PRIVILEGED": controls,
        "native_query_key_sha256": tensor_sha(native_key),
        "native_read_sha256": tensor_sha(read), "native_logit_sha256": tensor_sha(native),
        "future_forecast_mean": mean[0, 0].tolist(), "future_forecast_sd": sd[0, 0].tolist(),
        "causal_current_last": last.tolist(), "trace": trace,
        "memory_update_max_abs_error": update_error,
        "prediction_equivalence_max_abs_error": float((native - decomposed).abs())}
    return result


def paired_gain_ci(sites, baseline_errors, native_errors, *, draws, seed):
    """Station bootstrap keeps all original pair members together."""
    groups = [[i for i, s in enumerate(sites) if s == station] for station in sorted(set(sites))]
    if len(groups) < 3:
        raise AuditBlocked("insufficient original source-station clusters")
    rng = random.Random(seed)
    differences = [a - b for a, b in zip(baseline_errors, native_errors)]
    samples = []
    for _ in range(draws):
        idx = [i for _ in groups for i in rng.choice(groups)]
        samples.append(sum(differences[i] for i in idx) / len(idx))
    samples.sort()
    return [samples[int(.025 * (draws - 1))], samples[int(.975 * (draws - 1))]]


def run_original_audit(plan, root, source_path):
    """Only the authorized CLI calls this internal source-scoring function."""
    import torch
    if (plan["lags"] != [64] or plan["history_hours"] != 24 or
            plan["control_count"] != 4 or plan["control_seed"] != 14014):
        raise AuditBlocked("unsupported frozen audit registration")
    root = verify_implementation(root, plan["implementation_commit"])
    if sha_file(source_path) != plan["original_source_sha256"]:
        raise AuditBlocked("original source archive digest mismatch")
    for row in plan["models"]:
        if sha_file(root / row["path"]) != row["sha256"]:
            raise AuditBlocked("original checkpoint changed")
    source, fit = load_original_modules(root)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    d = fit.source_observations(Path(source_path))
    if d["raw"].shape != (12, source.hour(source.VAL_END), 13):
        raise AuditBlocked("source loader crossed original FIT numeric boundary")
    rows = fit.select_pairs(d, fit.EVAL_START, fit.EVAL_STOP, 2)
    cohort_sha = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    if len(rows) != 48 or cohort_sha != plan["source_eval_cohort_sha256"]:
        raise AuditBlocked("frozen original source cohort mismatch")
    max_query = max(t + 64 for _, t, _ in rows)
    verify_causal_features(d, source, max_query)
    results = []
    with torch.no_grad():
        for seed in (80701, 80702):
            spec = {r["arm"]: r for r in plan["models"] if r["seed"] == seed}
            if set(spec) != {"assoc_historic_query", "fast_slow_historic_query"}:
                raise AuditBlocked("incomplete original model matrix")
            if spec["assoc_historic_query"]["sample_order_sha256"] != spec["fast_slow_historic_query"]["sample_order_sha256"]:
                raise AuditBlocked("original comparator data order mismatch")
            models = {}
            before = {}
            for arm, row in spec.items():
                m = fit.create_model(seed, arm)
                checkpoint = torch.load(root / row["path"], map_location="cpu", weights_only=False)
                m.load_state_dict(checkpoint["original_model"], strict=True)
                m.eval()
                models[arm] = m
                before[arm] = {k: tensor_sha(v) for k, v in m.state_dict().items()}
            predictions = []
            for site, anchor, _ in rows:
                # Actual inference boundary: no raw/label dictionary and no
                # future feature row is reachable by either architecture.
                prefix = {key: d[key][:, :anchor + 65] for key in ("values", "valid", "age", "last")}
                item = trace_case(models["assoc_historic_query"], prefix, source,
                    site=site, anchor=anchor, lag=64, control_count=4, control_seed=14014)
                logit, forecast, _, _ = fit.forward_case(models["fast_slow_historic_query"],
                    prefix, [(site, anchor, 0)], 64, with_truth=False)
                item["nonassoc_probability"] = float(logit[0].sigmoid())
                item["nonassoc_forecast_mean"] = forecast[0].tolist()
                predictions.append(item)
            # Predictions of BOTH original architectures are complete before
            # consulting the historical labels/future outcomes for scoring.
            errors = {name: [] for name in ("native", "nonassoc", "zero_read", "prior", "current_last")}
            privileged = {name: [] for name in ("oracle", "control_0", "control_1", "control_2", "control_3")}
            for item, (site, anchor, label) in zip(predictions, rows):
                raw = float(d["raw"][site, anchor, 0])
                if not (raw >= 150 if label else raw <= 35):
                    raise AuditBlocked("frozen source outcome identity differs")
                item["source_label_AFTER_prediction"] = label
                last_pm = torch.expm1(torch.tensor(item["causal_current_last"][0]) * d["spread"][0] + d["center"][0])
                probabilities = {"native": item["native_probability"],
                    "nonassoc": item["nonassoc_probability"], "zero_read": item["zero_read_probability"],
                    "prior": .5, "current_last": float(last_pm >= 150)}
                for name, probability in probabilities.items():
                    errors[name].append((probability - label) ** 2)
                privileged["oracle"].append((item["oracle_probability_PRIVILEGED"] - label) ** 2)
                for rank, control in enumerate(item["controls_PRIVILEGED"]):
                    privileged[f"control_{rank}"].append((control["privileged_event_probability"] - label) ** 2)
                t = anchor + 64 + 6
                valid = d["valid"][site, t]
                actual = d["values"][site, t]
                item["forecast_normalized_abs_error_sum"] = {
                    name: float((torch.tensor(pred) - actual).abs()[valid].sum())
                    for name, pred in (("native", item["future_forecast_mean"]),
                                       ("nonassoc", item["nonassoc_forecast_mean"]),
                                       ("persistence", item["causal_current_last"]))}
                item["forecast_observed_channel_count"] = int(valid.sum())
            sites = [row[0] for row in rows]
            brier = {name: sum(values) / len(values) for name, values in errors.items()}
            comparisons = {}
            for name in errors.keys() - {"native"}:
                ci = paired_gain_ci(sites, errors[name], errors["native"],
                    draws=4000, seed=14014)
                gain = (brier[name] - brier["native"]) / max(brier[name], 1e-12)
                comparisons[name] = {"relative_brier_gain": gain, "station_bootstrap_95_loss_gain": ci,
                                     "exploratory_gate_pass": gain >= .05 and ci[0] > 0}
            for arm, m in models.items():
                if before[arm] != {k: tensor_sha(v) for k, v in m.state_dict().items()} or sha_file(root / spec[arm]["path"]) != spec[arm]["sha256"]:
                    raise AuditBlocked("original checkpoint/model bytes changed")
            denominator = sum(p["forecast_observed_channel_count"] for p in predictions)
            forecast_mae = {arm: sum(p["forecast_normalized_abs_error_sum"][arm] for p in predictions)
                            / max(denominator, 1) for arm in ("native", "nonassoc", "persistence")}
            results.append({"seed": seed, "Brier": brier, "comparisons": comparisons,
                            "privileged_Brier_not_native_skill": {k: sum(v) / len(v) for k, v in privileged.items()},
                            "original_plus6_forecast_normalized_MAE": forecast_mae,
                            "cases": predictions, "unchanged_checkpoints": spec})
    return {"status": "MEASURED_FROZEN_FIT_DIAGNOSTIC", "models": results,
        "source_sha256": plan["original_source_sha256"], "cohort_sha256": cohort_sha,
        "checks": {"actual_write_and_native_read_correspondence": "PASS",
            "source_prefix_normalizer_and_last_age": "PASS", "checkpoint_immutability": "PASS",
            "raw_labels_and_future_features_absent_from_inference_mapping": "PASS",
            "end_to_end_reloaded_source_future_poison": "INCONCLUSIVE_NOT_IMPLEMENTED",
            "whole_model_PNDS_IEEE_AD_equivalence": "OPEN"},
        "independent_model_holdout": False, "complete_training_FLOP_parity_established": False,
        "scientific_qualification": False}
