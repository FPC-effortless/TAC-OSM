#!/usr/bin/env python3
"""ARC-AGI-3 public OFFLINE development baseline. No hidden evaluator access.

Only the official arc_agi toolkit creates game environments. This policy uses
visible frames, level/status fields and the published action space; it is not a
trained TAC-OSM model or a benchmark-success claim.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

def enum_name(value):
    if not isinstance(getattr(value, "name", None), str):
        raise ValueError("expected named public enum action/state")
    return value.name

def public_frame(obs):
    data = getattr(obs, "frame", None)
    if hasattr(data, "tolist"):
        data = data.tolist()
    if not isinstance(data, (list, tuple)) or not 1 <= len(data) <= 64:
        raise ValueError("missing or oversized public frame")
    for layer in data:
        if not isinstance(layer, (list, tuple)) or not 1 <= len(layer) <= 64:
            raise ValueError("invalid public grid")
        widths = set()
        for row in layer:
            if not isinstance(row, (list, tuple)) or not 1 <= len(row) <= 64:
                raise ValueError("invalid public grid row")
            widths.add(len(row))
            if any(type(v) is not int or not 0 <= v <= 15 for v in row):
                raise ValueError("public grid palette must be integer 0..15")
        if len(widths) != 1:
            raise ValueError("ragged public grid")
    return data

def obs_digest(obs):
    levels = getattr(obs, "levels_completed", None)
    if type(levels) is not int or levels < 0:
        raise ValueError("invalid levels_completed")
    payload = {"frame": public_frame(obs), "state": enum_name(obs.state),
               "levels_completed": levels}
    return hashlib.sha256(canonical(payload).encode()).hexdigest()

class BoundedExplore:
    def __init__(self, capacity=256):
        if not 1 <= capacity <= 10000:
            raise ValueError("invalid memory capacity")
        self.capacity = capacity
        self.per_state = OrderedDict()
        self.global_visits = {}

    def choose(self, key, actions):
        choices = [a for a in actions if enum_name(a) != "RESET"]
        if not choices:
            raise ValueError("no legal non-reset action available")
        if key not in self.per_state:
            if len(self.per_state) >= self.capacity:
                self.per_state.popitem(last=False)
            self.per_state[key] = {}
        else:
            self.per_state.move_to_end(key)
        counts = self.per_state[key]
        a = min(choices, key=lambda x: (
            counts.get(enum_name(x), 0),
            self.global_visits.get(enum_name(x), 0), enum_name(x)))
        name = enum_name(a)
        counts[name] = counts.get(name, 0) + 1
        self.global_visits[name] = self.global_visits.get(name, 0) + 1
        return a, {"state_visits": counts[name], "global_visits": self.global_visits[name]}

def verify_chain(entries):
    prev = "0" * 64
    for idx, record in enumerate(entries):
        if record.get("index") != idx or record.get("previous_digest") != prev:
            return False
        payload = {k: v for k, v in record.items() if k != "digest"}
        expected = hashlib.sha256(canonical(payload).encode()).hexdigest()
        if record.get("digest") != expected:
            return False
        prev = expected
    return True

def run_public_game(arcade, game_id, max_actions=80, capacity=256):
    if not 1 <= max_actions <= 10000:
        raise ValueError("invalid action budget")
    policy = BoundedExplore(capacity=capacity)
    env = arcade.make(game_id)
    if env is None:
        raise ValueError("public OFFLINE environment unavailable")
    obs = env.observation_space
    if obs is None:
        obs = env.reset()  # one initial observation only
    if obs is None:
        raise ValueError("initial public observation unavailable")
    ledger = []
    for index in range(max_actions):
        if enum_name(obs.state) in ("WIN", "GAME_OVER"):
            break
        before = obs_digest(obs)
        action, visits = policy.choose(before, list(env.action_space))
        data = None
        if action.is_complex():
            layer = public_frame(obs)[0]
            data = {"x": len(layer[0]) // 2, "y": len(layer) // 2}
        after = env.step(action, data=data)
        if after is None:
            raise ValueError("game step returned no public observation")
        row = {"index": index, "previous_digest": ledger[-1]["digest"] if ledger else "0" * 64,
               "observation_sha256": before, "action": enum_name(action),
               "action_data": data, "visits": visits,
               "next_observation_sha256": obs_digest(after),
               "state": enum_name(after.state), "levels_completed": after.levels_completed}
        row["digest"] = hashlib.sha256(canonical(row).encode()).hexdigest()
        ledger.append(row)
        obs = after
    return {"schema": "arc3-public-dev-v1", "mode": "offline-only",
            "game_id": game_id, "policy": "bounded-least-visited-v1",
            "action_budget": max_actions, "memory_capacity": capacity,
            "actions_taken": len(ledger), "terminal_state": enum_name(obs.state),
            "levels_completed": obs.levels_completed,
            "ledger_verified": verify_chain(ledger), "ledger": ledger,
            "official_score": "NOT_EVALUATED", "trained_model": False}

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", required=True)
    p.add_argument("--environments", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-actions", type=int, default=80)
    p.add_argument("--memory-capacity", type=int, default=256)
    args = p.parse_args(argv)
    if not args.environments.is_dir():
        p.error("local public development directory required; no online fallback")
    if args.out.exists():
        p.error("output exists; refusing to overwrite previous evidence")
    try:
        from arc_agi import Arcade, OperationMode
        arcade = Arcade(operation_mode=OperationMode.OFFLINE,
                        environments_dir=str(args.environments.resolve()),
                        recordings_dir=str((args.out.parent / "arc_local_recordings").resolve()))
        result = run_public_game(arcade, args.game, args.max_actions, args.memory_capacity)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2, sort_keys=True)
            output.write("\n")
        print(canonical({k: v for k, v in result.items() if k != "ledger"}))
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print(canonical({"status": "BLOCKED_OR_FAILED", "reason": str(exc)}), file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
