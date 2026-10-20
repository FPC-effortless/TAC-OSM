"""Recorded-only audit for ME023. No training, source download or test opening."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scripts.me022_native_readout import predictions, scores, save_json
from scripts.me023_predictive_research import C, sha, load_source, make_model, state_diagnostic


def audit(root, parent, rebuild_source=False):
    root = Path(root)
    result = json.loads((root/"results.json").read_text())
    assert result["contract"] == C
    assert len(result["models"]) == 12
    assert {(r["arm"], r["seed"]) for r in result["models"]} == {(a,s) for a in C["arms"] for s in C["seeds"]}
    for row in result["models"]:
        assert sha(root/row["checkpoint"]) == row["checkpoint_sha256"]
        assert sha(root/row["standalone"]) == row["standalone_sha256"]
    if not rebuild_source:
        data = load_source(parent)
    else:
        from scripts.me014b_exact_action_memory import read_early, FIT_CUT, DEV_CUT
        from scripts.me019_authentic_history import original_128_windows
        from scripts.me022_replay import fingerprints
        people, source = read_early()
        assert source["zip_sha256"] == C["source_sha256"]
        data = {"train": original_128_windows(people, 0, FIT_CUT, 12000),
                "dev": original_128_windows(people, FIT_CUT, DEV_CUT, 6000), "source": source}
        expected = json.loads((Path(__file__).resolve().parents[1]/
            "research/mathematical-engineering/me022/data-fingerprints.json").read_text())
        assert fingerprints(data) == expected
        assert not source["future_test_labels_parsed"]
    train, dev = data["train"], data["dev"]
    torch.set_num_threads(1)
    sampler = {}
    errors, export_errors, mean_errors = [], [], []
    matrices = []
    stream_state_errors, stream_logit_errors = [], []
    for row in result["models"]:
        ck = torch.load(root/row["checkpoint"], weights_only=True)
        assert ck["step"] == C["updates"] == row["steps"]
        assert ck["contract"] == C
        assert ck["arm"] == row["arm"] and ck["seed"] == row["seed"]
        assert row["parameters"] == C["parameters"]
        assert row["train_cpu_seconds"] <= C["cpu_seconds_ceiling_per_fit"]
        assert row["sampled_windows"] == C["updates"]*C["batch"]
        assert row["historical_event_presentations"] == row["sampled_windows"]*128
        assert row["supervised_response_presentations"] == row["sampled_windows"]*(4 if row["arm"].endswith("prefix") else 1)
        assert [t["step"] for t in row["trace"]] == C["trace_steps"]
        if row["seed"] in sampler:
            assert torch.equal(sampler[row["seed"]], ck["sampler_rng"])
        sampler[row["seed"]] = ck["sampler_rng"]
        net = make_model(row["arm"], ck["duration_mean"], ck["duration_scale"])
        net.load_state_dict(ck["state_dict"], strict=True)
        script = torch.jit.script(net).eval()
        p = predictions(script, dev)
        old = np.load(root/f'{row["arm"]}-seed{row["seed"]}-predictions.npz')
        error = float(np.max(np.abs(p-old["probability"])))
        assert error <= 1e-7
        errors.append(error)
        for metric, value in scores(p, dev["y"].numpy()).items():
            assert abs(row["dev"][metric]-value) <= 1e-9
        exported = torch.jit.load(str(root/row["standalone"])).eval()
        e = float(np.max(np.abs(predictions(exported, dev)-p)))
        assert e <= 1e-7
        export_errors.append(e)
        # Separate engineering export; original fitting-source graph is retained.
        with torch.no_grad():
            state = torch.zeros(32, 64)
            for t in range(128):
                state = script.stream_step(dev["items"][:32, t], dev["metadata"][:32, t], state)
            window_state = script.encode(dev["items"][:32], dev["metadata"][:32])
            se = float((state-window_state).abs().max())
            assert se <= 1e-6
            prior = (dev["metadata"][:32, -32:, 1].sum(-1, keepdim=True)+1)/34
            streamed = script.readout(state, dev["action"][:32], dev["duration"][:32], prior)
            direct = script(dev["items"][:32], dev["metadata"][:32], dev["action"][:32], dev["duration"][:32])
            le = float((streamed-direct).abs().max())
            assert le <= 1e-6
            stream_state_errors.append(se); stream_logit_errors.append(le)
            torch.jit.save(script, str(root/f'{row["arm"]}-seed{row["seed"]}.streaming.jit.pt'))
        state_metrics, mean, mp = state_diagnostic(script, train, dev)
        mean_old = torch.load(root/f'{row["arm"]}-seed{row["seed"]}-mean-state.pt', weights_only=True)["state"]
        torch.testing.assert_close(mean, mean_old, atol=0, rtol=0)
        me = float(np.max(np.abs(mp-old["mean_state_probability"])))
        assert me <= 1e-7
        mean_errors.append(me)
        for metric, value in state_metrics.items():
            assert abs(row["mean_state_dev"][metric]-value) <= 1e-9
        matrix = script.bounded_matrix().detach().double()
        singular = torch.linalg.svdvals(matrix)
        assert matrix.norm() <= C["gain"]+1e-6
        matrices.append({"arm": row["arm"], "seed": row["seed"],
                         "frobenius": float(matrix.norm()),
                         "singular_values": singular.tolist()})
    report = {"status": "PASS", "checkpoints": 12, "standalone_exports": 12,
              "predictions_per_path": 12*len(dev["y"]),
              "maximum_original_prediction_error": max(errors),
              "maximum_export_prediction_error": max(export_errors),
              "maximum_mean_state_prediction_error": max(mean_errors),
              "streaming_exports": 12, "streaming_real_probe_windows_per_model": 32,
              "maximum_streaming_state_error": max(stream_state_errors),
              "maximum_streaming_logit_error": max(stream_logit_errors),
              "exact_sampler_rng_and_mean_states": True, "new_fits": 0, "new_test_outcomes": 0,
              "interaction_spectra": matrices}
    save_json(root/"replay.json", report)
    print(json.dumps({k:v for k,v in report.items() if k != "interaction_spectra"}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/me023")
    parser.add_argument("--parent", default="results/me022")
    parser.add_argument("--rebuild-source", action="store_true")
    args = parser.parse_args()
    audit(args.root, args.parent, args.rebuild_source)
