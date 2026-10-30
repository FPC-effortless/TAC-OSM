import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('langpersist096_run',ROOT/'research/mathematical-engineering/langpersist096/run.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)


class RunBoundaryTests(unittest.TestCase):
    def test_future_poison_uses_fixed_offsets_and_private_source_copies(self):
        original=bytes(range(24));sources={'f':original}
        records=[{'file':'f','acquisition_start':0,'acquisition_end':4,'query_start':4,'query_end':8,'target_offset':8,'target_end':12}]
        a=np.frombuffer(original[:4],dtype=np.uint8)[None]
        q=np.frombuffer(original[4:8],dtype=np.uint8)[None]
        poisoned,audit=study.poison_fixed_sources(sources,records,a,q)
        np.testing.assert_array_equal(poisoned,q)
        self.assertNotEqual(audit['original_source_hashes'],audit['poisoned_source_hashes'])
        self.assertEqual(sources['f'],original)
        self.assertFalse(audit['eligibility_recomputed'])

    def test_overlapping_future_source_is_not_falsely_accepted(self):
        original=bytes(range(24));sources={'f':original}
        records=[{'file':'f','acquisition_start':0,'acquisition_end':4,'query_start':4,'query_end':8,'target_offset':6,'target_end':10}]
        with self.assertRaises(ValueError):
            study.poison_fixed_sources(sources,records,np.frombuffer(original[:4],dtype=np.uint8)[None],np.frombuffer(original[4:8],dtype=np.uint8)[None])

    def test_replay_compares_predictions_not_nondeterministic_cost_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);a=folder/'a.npz';b=folder/'b.npz'
            np.savez(a,greedy=np.zeros((2,3),dtype=np.uint8),metadata=np.array(json.dumps({'seconds':1})))
            np.savez(b,greedy=np.zeros((2,3),dtype=np.uint8),metadata=np.array(json.dumps({'seconds':2})))
            study.verify_replay(a,b,('greedy',))
            np.savez(b,greedy=np.ones((2,3),dtype=np.uint8),metadata=np.array('{}'))
            with self.assertRaises(ValueError): study.verify_replay(a,b,('greedy',))

    def test_actor_provenance_cannot_silently_change_checkpoint_or_future_boundary(self):
        row={'arm':'tac','checkpoint_sha256':'a'*64}
        meta={**row,'optimizer_steps':0,'acquisition_sha256':'b'*64,'query_sha256':'c'*64}
        study.verify_actor_metadata(meta,row,'b'*64,'c'*64)
        for key,value in [('arm','gru'),('checkpoint_sha256','d'*64),('optimizer_steps',1),('query_sha256','e'*64)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                study.verify_actor_metadata({**meta,key:value},row,'b'*64,'c'*64)


if __name__=='__main__': unittest.main()
