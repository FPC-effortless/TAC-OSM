#!/usr/bin/env python3
"""S1MOTION001: preregistered authentic ALOHA System1 motion-proxy pilot.

The secondary label is high measured joint-motion, NOT adverse injury/failure.
No execution, causal command, independent external-transfer or safety claims.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tac_osm.variants.system1_parallel import System1ParallelWorld

REV = "6728831ef9674b74f3ce20bf517890e9ee20b0d9"
SOURCE = "lerobot/aloha_static_candy"
PARQUET = "data/chunk-000/file-000.parquet"
FIT = tuple(range(25))
DEV = tuple(range(25, 35))
SEEDS = (26001, 26002, 26003)
WINDOW = 32
BATCH = 8
UPDATES = 192
P = 14


def sha256(blob):
    return hashlib.sha256(blob).hexdigest()


def source_episodes(cache_dir: Path):
    """Retrieve a pinned real publisher snapshot; ignore TEST episode values."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        f"https://huggingface.co/api/datasets/{SOURCE}/revision/{REV}",
        headers={"User-Agent": "TAC-OSM-S1MOTION001-registered/1.0"})
    with urllib.request.urlopen(req, timeout=90) as response:
        metadata = json.load(response)
    if metadata.get("sha") != REV or PARQUET not in {
            s["rfilename"] for s in metadata.get("siblings", [])}:
        raise RuntimeError("real publisher commit/layout mismatch")
    file = cache_dir / "publisher-aloha-static-candy.parquet"
    if not file.exists():
        req = urllib.request.Request(
            f"https://huggingface.co/datasets/{SOURCE}/resolve/{REV}/{PARQUET}",
            headers={"User-Agent": "TAC-OSM-S1MOTION001-registered/1.0"})
        with urllib.request.urlopen(req, timeout=120) as response:
            file.write_bytes(response.read())
    source_hash = sha256(file.read_bytes())
    if not file.stat().st_size:
        raise RuntimeError("empty publisher data")
    table = pq.read_table(file, columns=[
        "episode_index", "frame_index", "timestamp", "observation.state", "action"])
    cols = table.to_pydict()
    out = {ep: [] for ep in FIT + DEV}
    for ep, fr, ts, x, u in zip(*(cols[k] for k in [
            "episode_index", "frame_index", "timestamp", "observation.state", "action"])):
        ep = int(ep)
        if ep not in out:
            continue  # TEST=35..49: no value normalization, training or scoring
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        if (x.shape != (P,) or u.shape != (P,) or
                not np.isfinite(x).all() or not np.isfinite(u).all() or
                not math.isfinite(float(ts))):
            raise RuntimeError(f"invalid observed publisher row: episode {ep}")
        if len(out[ep]) < 257:
            out[ep].append((int(fr), float(ts), x, u))
    for ep, rows in out.items():
        if len(rows) != 257:
            raise RuntimeError(f"insufficient real episode frames: {ep}")
        for a, b in zip(rows, rows[1:]):
            if b[0] != a[0] + 1 or not (0 < b[1] - a[1] < .2):
                raise RuntimeError(f"nonchronological publisher frames: {ep}")
    return out, dict(dataset=SOURCE, revision=REV, parquet_path=PARQUET,
                     full_parquet_sha256=source_hash, publisher_episode_count=50,
                     fit_episode_ids=list(FIT), dev_episode_ids=list(DEV),
                     test_ids_excluded_from_values=list(range(35, 50)),
                     downloaded_file_contains_protected_test_rows=True)


def prepare(fit_source, dev_source):
    """Fit all transformations and the proxy threshold using ONLY FIT."""
    fit_x = np.concatenate([np.stack([r[2] for r in fit_source[e]])
                            for e in FIT], axis=0)
    fit_u = np.concatenate([np.stack([r[3] for r in fit_source[e]])
                            for e in FIT], axis=0)
    xm = fit_x.mean(0); xs = np.maximum(fit_x.std(0), 1e-3)
    um = fit_u.mean(0); us = np.maximum(fit_u.std(0), 1e-3)
    fit_motion = np.concatenate([
        ((np.diff(np.stack([r[2] for r in fit_source[e]]), axis=0)) ** 2).mean(-1)
        for e in FIT])
    threshold = float(np.quantile(fit_motion, .90))
    if threshold <= 0 or not np.isfinite(threshold):
        raise RuntimeError("FIT-derived high-motion threshold unavailable")
    standardized_deltas = np.concatenate([
        np.diff(np.stack([r[2] for r in fit_source[e]]), axis=0) / xs
        for e in FIT])
    delta_scale = np.maximum(np.quantile(np.abs(standardized_deltas), .95, axis=0), 1e-4)
    if not np.isfinite(delta_scale).all():
        raise RuntimeError("invalid FIT-only scales")

    def build(rows):
        x = np.stack([r[2] for r in rows]).astype(np.float64)
        u = np.stack([r[3] for r in rows]).astype(np.float64)
        norm_x = (x - xm) / xs
        norm_u = (u - um) / us
        # At t, use only the action logged at t-1 (no same-frame actuation claim).
        # t=1..255, targets x[t+1] are revealed only AFTER the forecast.
        event = ((x[2:257] - x[1:256]) ** 2).mean(-1) >= threshold
        return dict(x=torch.from_numpy(norm_x[1:256]).float(),
                    prior_u=torch.from_numpy(norm_u[0:255]).float(),
                    y=torch.from_numpy(norm_x[2:257]).float(),
                    event=torch.from_numpy(event.astype(np.float32)),
                    target_physical=x[2:257], observed_physical=x[1:256],
                    preceding_physical=x[0:255],
                    episode_frames=255)
    fit = {e: build(fit_source[e]) for e in FIT}
    dev = {e: build(dev_source[e]) for e in DEV}
    if not (len(fit) == 25 and len(dev) == 10):
        raise RuntimeError("source split size mismatch")
    normalization = dict(state_mean=xm.tolist(), state_sd=xs.tolist(),
                         action_mean=um.tolist(), action_sd=us.tolist(),
                         high_motion_threshold_raw_squared_units=threshold,
                         train_delta_scale_normalized=delta_scale.tolist())
    return fit, dev, normalization


class MatchedGRU(nn.Module):
    """28D input -> GRU160 -> 29D output, 95,869 trainable parameters."""
    def __init__(self, scale):
        super().__init__()
        self.recurrent = nn.GRU(28, 160, batch_first=True)
        self.readout = nn.Linear(160, 29)
        self.register_buffer("delta_scale", scale.detach().clone())

    def forward(self, x, prior_u):
        h, _ = self.recurrent(torch.cat((x, prior_u), dim=-1))
        raw = self.readout(h)
        pred = x + self.delta_scale * raw[..., :14].tanh()
        return pred, raw[..., 14], F.softplus(raw[..., 15:]) + 1e-4


def outputs(model, x, prior_u):
    if isinstance(model, System1ParallelWorld):
        result = model(x, prior_u)
        # The output named adverse_event_* is used ONLY for observed motion proxy.
        return result.next_observation, result.adverse_event_logit, result.residual_scale
    return model(x, prior_u)


def objective(model, x, u, y, event):
    pred, logit, scale = outputs(model, x, u)
    mse = (pred - y).square().mean()
    binary = F.binary_cross_entropy_with_logits(logit, event)
    normal = (0.5 * ((pred - y) / scale).square() + scale.log()).mean()
    return mse + .1 * binary + .01 * normal


def windows(fit, seed):
    rng = np.random.default_rng(seed)
    plans = []
    for _ in range(UPDATES):
        selected = [FIT[int(rng.integers(0, len(FIT)))] for _ in range(BATCH)]
        begins = [int(rng.integers(0, 255 - WINDOW + 1)) for _ in range(BATCH)]
        plans.append((selected, begins))
    return plans


def batch(fit, selected, begins):
    return tuple(torch.stack([fit[e][key][s:s + WINDOW] for e, s in
                        zip(selected, begins)], 0)
                 for key in ("x", "prior_u", "y", "event"))


def fit_model(arm, seed, train, selected_windows, scale, save_dir):
    torch.manual_seed(seed)
    if arm == "system1":
        model = System1ParallelWorld()
        model.set_train_delta_scale(scale)
    else:
        model = MatchedGRU(scale)
    model.train()
    params = sum(v.numel() for v in model.parameters() if v.requires_grad)
    expected = 95907 if arm == "system1" else 95869
    if params != expected:
        raise RuntimeError(f"matched model parameter count changed {arm}: {params}")
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    start = time.perf_counter()
    recent = 0.0
    for selected, begins in selected_windows:
        x, u, y, event = batch(train, selected, begins)
        opt.zero_grad(set_to_none=True)
        loss = objective(model, x, u, y, event)
        if not torch.isfinite(loss).item():
            raise RuntimeError("nonfinite real-source training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        recent = float(loss.detach().item())
    elapsed = time.perf_counter() - start
    # Freeze all model / optimizer / torch RNG BEFORE opening the DEV score loop.
    payload = dict(architecture=arm, seed=seed, model=model.state_dict(),
                   optimizer=opt.state_dict(), torch_rng=torch.random.get_rng_state(),
                   updates=UPDATES, batch=BATCH, frames=WINDOW)
    buf = io.BytesIO()
    torch.save(payload, buf)
    snapshot = buf.getvalue()
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / f"{arm}-{seed}.pt").write_bytes(snapshot)
    return model.eval(), dict(seed=seed, arm=arm, params=params,
                              checkpoint_sha256=sha256(snapshot),
                              optimizer_updates=UPDATES,
                              frames_consumed=UPDATES * BATCH * WINDOW,
                              fitting_seconds=elapsed, final_fit_loss=recent)


def auroc(y, p):
    y = np.asarray(y).astype(bool); p = np.asarray(p)
    n1 = int(y.sum()); n0 = len(y) - n1
    if not n0 or not n1:
        return None
    # Stable midranks make ties mathematically reproducible.
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    ranks = np.empty(len(y), dtype=np.float64)
    i = 0
    while i < len(y):
        j = i + 1
        while j < len(y) and sorted_p[j] == sorted_p[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(y, p):
    y = np.asarray(y).astype(np.float64)
    if y.sum() == 0:
        return None
    order = np.argsort(-np.asarray(p), kind="mergesort")
    y = y[order]
    precision = np.cumsum(y) / (1 + np.arange(len(y)))
    return float((precision * y).sum() / y.sum())


@torch.no_grad()
def score(model, dev, normalization):
    xm = np.asarray(normalization["state_mean"])
    xs = np.asarray(normalization["state_sd"])
    per_episode = []
    ys = []; ps = []
    for ep in DEV:
        z = dev[ep]
        pred, logits, _ = outputs(model, z["x"][None], z["prior_u"][None])
        physical = pred[0].numpy().astype(np.float64) * xs + xm
        actual = z["target_physical"]
        observed = z["observed_physical"]
        velocity = 2 * observed - z["preceding_physical"]
        motion = z["event"].numpy().astype(bool)
        probs = torch.sigmoid(logits[0]).numpy()
        ys.extend(motion.astype(int).tolist()); ps.extend(probs.tolist())
        terms = dict(tac=float(np.mean((physical - actual) ** 2)),
                     persistence=float(np.mean((observed - actual) ** 2)),
                     velocity=float(np.mean((velocity - actual) ** 2)))
        moving = dict(tac=float(np.mean((physical[motion] - actual[motion]) ** 2))
                          if motion.any() else None,
                      velocity=float(np.mean((velocity[motion] - actual[motion]) ** 2))
                          if motion.any() else None)
        per_episode.append(dict(episode=int(ep), frames=len(actual),
                    proxy_high_motion_cases=int(motion.sum()),
                    raw_joint_MSE=terms, raw_joint_high_motion_MSE=moving,
                    proxy_Brier=float(np.mean((probs - motion) ** 2))))
    y = np.asarray(ys); p = np.asarray(ps)
    losses = {a: float(np.mean([r["raw_joint_MSE"][a] for r in per_episode]))
              for a in ("tac", "velocity", "persistence")}
    moving_weight = sum(z["proxy_high_motion_cases"] for z in per_episode)
    moving = {a: float(sum(z["raw_joint_high_motion_MSE"][a] * z["proxy_high_motion_cases"]
                          for z in per_episode if z["proxy_high_motion_cases"])
                       / moving_weight) if moving_weight else None
              for a in ("tac", "velocity")}
    return dict(raw_joint_MSE=losses, raw_joint_high_motion_MSE=moving,
                proxy_prevalence=float(y.mean()), proxy_AUROC=auroc(y,p),
                proxy_AUPRC=auprc(y,p), proxy_Brier=float(np.mean((p-y)**2)),
                per_episode=per_episode, eval_real_frames=len(y))


def flop_proxy(model, fit):
    """PyTorch profiler's MATMUL subset only; not complete training FLOPs."""
    try:
        from torch.profiler import profile, ProfilerActivity
        first = batch(fit, [FIT[0]] * BATCH, [0] * BATCH)
        model.train()
        model.zero_grad(set_to_none=True)
        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
            objective(model, *first).backward()
        model.zero_grad(set_to_none=True)
        model.eval()
        values = [int(e.flops) for e in prof.key_averages() if e.flops]
        return dict(reported_operator_flops=sum(values),
                    coverage="incomplete_matmul_conv_subset_includes_autograd")
    except Exception as exc:
        model.eval()
        return dict(reported_operator_flops=None, coverage="unavailable",
                    profiler_error=type(exc).__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="/tmp/s1motion001-aloha")
    p.add_argument("--out", default="research/diagnostics/s1motion001-original")
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    episodes, source = source_episodes(Path(a.cache))
    fit_src = {e: episodes[e] for e in FIT}
    dev_src = {e: episodes[e] for e in DEV}
    train, dev, norm = prepare(fit_src, dev_src)
    directory = ROOT / a.out
    # Train and freeze ALL original checkpoint/optimizer bytes before DEV scoring.
    trained = {}
    fit_records = []
    scale = torch.tensor(norm["train_delta_scale_normalized"], dtype=torch.float32)
    for seed in SEEDS:
        sampling = windows(train, seed)
        for arm in ("system1", "gru"):
            model, details = fit_model(arm, seed, train, sampling, scale, directory)
            trained[(seed, arm)] = model
            fit_records.append(details)
            print("S1MOTION001_FIT_FROZEN", json.dumps(details, sort_keys=True), flush=True)
    manifest = dict(protocol="research/parallel-six/S1MOTION001-PROTOCOL.md",
        source=source, fitted_only_real_source=True,
        trained_seeds=list(SEEDS), fit_records=fit_records,
        train_and_dev_actor_disjoint=True,
        blind_external_site=False, actual_adverse_label_available=False,
        proxy_definition="FIT_90th_percentile_next_measured_joint_motion",
        new_merged_model_family=False, normalization=norm,
        no_original_historical_weights_changed=True,
        test_episode_ids_not_processed=True)
    (directory / "before-dev-score.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    print("S1MOTION001_ALL_MODELS_FROZEN_BEFORE_DEV=YES", flush=True)
    results = []
    for seed in SEEDS:
        paired = {}
        for arm in ("system1", "gru"):
            model = trained[(seed, arm)]
            paired[arm] = score(model, dev, norm)
            print("S1MOTION001_DEV_SCORE", json.dumps(dict(seed=seed, arm=arm,
                  **{k:v for k,v in paired[arm].items() if k!="per_episode"}),
                  sort_keys=True), flush=True)
        s = paired["system1"]; g = paired["gru"]
        s_mse = s["raw_joint_MSE"]["tac"]
        v_mse = s["raw_joint_MSE"]["velocity"]
        p_mse = s["raw_joint_MSE"]["persistence"]
        high = s["raw_joint_high_motion_MSE"]
        pass_gate = (s_mse <= .99 * min(v_mse, p_mse) and
            s_mse < g["raw_joint_MSE"]["tac"] and
            high["tac"] is not None and high["velocity"] is not None and
            high["tac"] <= 1.05 * high["velocity"])
        results.append(dict(seed=seed, scores=paired,
                           registered_predictive_gate_pass=bool(pass_gate)))
    # FLOP coverage is partial; do not use its subset as complete compute equivalence.
    for record in fit_records:
        record["training_flops_partial"] = flop_proxy(
            trained[(record["seed"], record["arm"])], train)
    result = dict(**manifest, fit_records=fit_records,
                  dev_original_first_scores=results,
                  full_three_seed_predictive_gate_pass=all(
                      z["registered_predictive_gate_pass"] for z in results),
                  fully_equal_training_flops_not_demonstrated=True,
                  qualified_real_adverse_risk=False,
                  qualified_independent_system1_skill=False,
                  qualified_device_latency=False)
    (directory / "first-dev-scores.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("S1MOTION001_FIRST_DEV_SCORE_COMPLETE=YES",flush=True)
    print("S1MOTION001_ORIGINAL_PREDICTIVE_GATE="+(
        "PASS" if result["full_three_seed_predictive_gate_pass"] else "FAIL"),flush=True)
    print("S1MOTION001_REAL_WORLD_QUALIFIED=NO",flush=True)


if __name__ == "__main__":
    main()
