import hashlib
import math
import unittest

import torch

from score_bytes import score_documents


def doc(name, raw):
    return name, hashlib.sha256(raw).hexdigest(), raw


def uniform(ids):
    return torch.zeros((*ids.shape, 256))


class ScoreBytesTest(unittest.TestCase):
    def test_bytes_and_window_boundaries(self):
        calls = []
        def capture(ids):
            calls.append(ids.tolist()[0])
            return uniform(ids)
        raw = b'x' * 256 + b'yz'
        r = score_documents(capture, [doc('a', raw), doc('b', 'éa'.encode()), doc('c', b'')])
        self.assertEqual(calls, [[120]*256, [121, 0], [195, 169]])
        self.assertEqual([d['targets'] for d in r['documents']], [257, 2, 0])
        self.assertEqual(r['target_bytes'], 259)
        self.assertAlmostEqual(r['bits_per_byte'], 8., places=12)

    def test_next_byte_offset_and_global_weighting(self):
        def current(ids):
            logits = uniform(ids).double()
            return logits.scatter_(-1, ids.unsqueeze(-1), math.log(255))
        # Each repeated byte has probability 1/2. A changed byte has probability 1/510.
        r = score_documents(current, [doc('first', b'aa'), doc('second', b'ababa')])
        self.assertAlmostEqual(r['bits_per_byte'], (1 + 4 * math.log2(510)) / 5, places=12)

    def test_refuse_changed_or_duplicate_inputs(self):
        with self.assertRaises(ValueError):
            score_documents(uniform, [('x', 'wrong', b'abc')])
        with self.assertRaises(ValueError):
            score_documents(uniform, [doc('x', b'abc'), doc('x', b'def')])
        with self.assertRaises(ValueError):
            score_documents(uniform, [doc('empty', b'x')])

    def test_refuse_invalid_model_output(self):
        for invalid in (lambda ids: uniform(ids)[:, :, :255],
                        lambda ids: torch.full((*ids.shape, 256), float('nan'))):
            with self.assertRaises(ValueError):
                score_documents(invalid, [doc('x', b'ab')])


if __name__ == '__main__':
    unittest.main()
