"""Constructed harness tests only: NO official ARC game or scientific score."""
from enum import Enum
import importlib.util
from pathlib import Path
import sys
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts" / "arc3_public_baseline.py"
spec = importlib.util.spec_from_file_location("arc3_public_baseline", PATH)
arc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = arc
spec.loader.exec_module(arc)

class Action(Enum):
    ACTION1 = 1
    ACTION2 = 2
    RESET = 3
    def is_complex(self):
        return False

class State(Enum):
    NOT_FINISHED = 1
    WIN = 2
    GAME_OVER = 3

class Obs:
    def __init__(self, state=State.NOT_FINISHED):
        self.state = state
        self.frame = [[[0, 1], [2, 3]]]
        self.levels_completed = int(state == State.WIN)

class FakeEnvironment:
    def __init__(self):
        self.observation_space = Obs()
        self.action_space = [Action.ACTION1, Action.ACTION2, Action.RESET]
        self.history = []
    def step(self, action, data=None):
        self.history.append((action, data))
        if len(self.history) == 3:
            self.observation_space = Obs(State.WIN)
        return self.observation_space

class FakeArcade:
    def __init__(self, env):
        self.env = env
    def make(self, game_id):
        return self.env

class PublicBaselineFixtureTests(unittest.TestCase):
    def test_episode_stops_with_hashed_ledger(self):
        env = FakeEnvironment()
        result = arc.run_public_game(FakeArcade(env), "constructed", 8)
        self.assertEqual(result["actions_taken"], 3)
        self.assertEqual(result["terminal_state"], "WIN")
        self.assertEqual(result["official_score"], "NOT_EVALUATED")
        self.assertTrue(arc.verify_chain(result["ledger"]))
        self.assertEqual(env.history[:2], [(Action.ACTION1, None), (Action.ACTION2, None)])
    def test_tampered_ledger_is_rejected(self):
        result = arc.run_public_game(FakeArcade(FakeEnvironment()), "constructed")
        result["ledger"][0]["action"] = "ACTION7"
        self.assertFalse(arc.verify_chain(result["ledger"]))
    def test_fixed_capacity(self):
        policy = arc.BoundedExplore(capacity=2)
        for i in range(6):
            policy.choose(str(i), [Action.ACTION1])
        self.assertEqual(len(policy.per_state), 2)
    def test_cannot_choose_reset(self):
        with self.assertRaises(ValueError):
            arc.BoundedExplore().choose("id", [Action.RESET])
    def test_invalid_observation(self):
        obs = Obs()
        obs.frame = [[[32]]]
        with self.assertRaises(ValueError):
            arc.obs_digest(obs)
    def test_bad_level_type(self):
        obs = Obs()
        obs.levels_completed = "0"
        with self.assertRaises(ValueError):
            arc.obs_digest(obs)
    def test_missing_environment(self):
        with self.assertRaises(ValueError):
            arc.run_public_game(FakeArcade(None), "unavailable")
    def test_bad_action_budget(self):
        with self.assertRaises(ValueError):
            arc.run_public_game(FakeArcade(FakeEnvironment()), "constructed", 0)
    def test_no_public_actions(self):
        env = FakeEnvironment()
        env.action_space = [Action.RESET]
        with self.assertRaises(ValueError):
            arc.run_public_game(FakeArcade(env), "constructed")
    def test_missing_initial_frame(self):
        env = FakeEnvironment()
        env.observation_space = None
        env.reset = lambda: None
        with self.assertRaises(ValueError):
            arc.run_public_game(FakeArcade(env), "constructed")

if __name__ == "__main__":
    unittest.main()
