"""No-fit scan algebra/numerical tests; no language-quality claim."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/mathematical-engineering/langmath095"))
from scans import diagonal_scan, matrix_scan

try:
    import torch
except ImportError:
    torch = None


def oracle(p, q, initial, mask, matrix):
    state = initial.copy()
    history = []
    for t in range(p.shape[1]):
        offered = state @ p[:, t] + q[:, t] if matrix else state * p[:, t] + q[:, t]
        admit = mask[:, t].reshape(p.shape[0], *([1] * (state.ndim - 1)))
        state = np.where(admit, offered, state)
        history.append(state.copy())
    return np.stack(history, axis=1) if history else np.empty((p.shape[0], 0, *initial.shape[1:]), dtype=p.dtype)


class NumpyScanTests(unittest.TestCase):
    def inputs(self, length, matrix=True, dtype=np.float64):
        rng = np.random.default_rng(95095 + length)
        shape = (2, length, 3, 3) if matrix else (2, length, 3)
        p = (rng.normal(size=shape) * .15).astype(dtype)
        qshape = (2, length, 2, 3) if matrix else shape
        q = rng.normal(size=qshape).astype(dtype)
        initial = rng.normal(size=(2, 2, 3) if matrix else (2, 3)).astype(dtype)
        mask = (np.arange(length)[None, :] + np.arange(2)[:, None]) % 3 != 1
        return p, q, initial, mask

    def test_oracle_shapes_dtypes_masks_and_no_mutation(self):
        for matrix, scan in ((True, matrix_scan), (False, diagonal_scan)):
            for dtype in (np.float32, np.float64):
                for length in (0, 1, 2, 3, 7, 17, 65):
                    p, q, initial, mask = self.inputs(length, matrix, dtype)
                    originals = [x.copy() for x in (p, q, initial, mask)]
                    for admit in (mask, np.zeros_like(mask), np.ones_like(mask)):
                        expected = oracle(p, q, initial, admit, matrix)
                        for algorithm in ("sequential", "tree", "chunked"):
                            for size in (1, 3, 16, 128):
                                with self.subTest(matrix=matrix, dtype=dtype, length=length,
                                                  algorithm=algorithm, size=size):
                                    got = scan(p, q, initial=initial, write_mask=admit,
                                               algorithm=algorithm, chunk_size=size)
                                    self.assertEqual(got.shape, expected.shape)
                                    self.assertEqual(got.dtype, dtype)
                                    tol = 2e-6 if dtype == np.float32 else 2e-13
                                    np.testing.assert_allclose(got, expected, atol=tol, rtol=tol)
                    for x, before in zip((p, q, initial, mask), originals):
                        np.testing.assert_array_equal(x, before)

    def test_noncommuting_chronology(self):
        p = np.array([[[[1., 2.], [0., 1.]], [[2., 0.], [3., 1.]],
                        [[0., 1.], [1., 0.]]]])
        q = np.array([[[[1., 2.]], [[-3., 1.]], [[.5, -2.]]]])
        initial = np.array([[[2., -1.]]])
        self.assertFalse(np.array_equal(p[:, 0] @ p[:, 1], p[:, 1] @ p[:, 0]))
        expected = oracle(p, q, initial, np.ones((1, 3), bool), True)
        for algorithm in ("sequential", "tree", "chunked"):
            np.testing.assert_array_equal(matrix_scan(p, q, initial=initial,
                                                       algorithm=algorithm, chunk_size=2), expected)

    def test_every_chunk_boundary_and_zero_default(self):
        for matrix, scan in ((True, matrix_scan), (False, diagonal_scan)):
            p, q, initial, mask = self.inputs(11, matrix)
            for algorithm in ("sequential", "tree", "chunked"):
                allstates = scan(p, q, initial=initial, write_mask=mask, algorithm=algorithm)
                for split in range(12):
                    left = scan(p[:, :split], q[:, :split], initial=initial,
                                write_mask=mask[:, :split], algorithm=algorithm)
                    carried = left[:, -1] if split else initial
                    right = scan(p[:, split:], q[:, split:], initial=carried,
                                 write_mask=mask[:, split:], algorithm=algorithm)
                    np.testing.assert_allclose(np.concatenate((left, right), axis=1),
                                               allstates, atol=2e-13, rtol=2e-13)
                np.testing.assert_allclose(scan(p, q, algorithm=algorithm),
                                           scan(p, q, initial=np.zeros_like(initial), algorithm=algorithm))

    def test_invalid_geometry_dtype_backend_and_options(self):
        p, q, initial, mask = self.inputs(3)
        invalid = [dict(algorithm="unknown"), dict(backend="unknown"),
                   dict(chunk_size=0), dict(chunk_size=True), dict(initial=initial.astype(np.float32)),
                   dict(initial=initial[:, :1]), dict(write_mask=mask.astype(np.int64)),
                   dict(write_mask=mask[:, :1])]
        for kw in invalid:
            with self.subTest(kw=kw), self.assertRaises((TypeError, ValueError)):
                matrix_scan(p, q, **kw)
        for a, b in ((p.astype(np.int64), q.astype(np.int64)), (p, q.astype(np.float32)),
                     (p[..., :2], q), (p, q[..., :2]), (p.tolist(), q)):
            with self.assertRaises((TypeError, ValueError)):
                matrix_scan(a, b)
        with self.assertRaises(ValueError):
            diagonal_scan(np.zeros((1, 3, 2)), np.zeros((1, 3, 3)))

    def test_float32_real_associativity_is_not_bitwise_associativity(self):
        p = np.full((1, 3, 1), .5, dtype=np.float32)
        q = np.array([.5, -.25, 1e-8], dtype=np.float32).reshape(1, 3, 1)
        sequential = diagonal_scan(p, q, algorithm="sequential")
        # This legal alternate parenthesization is the original Hillis-Steele
        # grouping, while our balanced tree may choose a different grouping.
        grouped = np.float32(np.float32(q[0, 0, 0] * np.float32(.25)) +
                             np.float32(np.float32(q[0, 1, 0] * np.float32(.5)) + q[0, 2, 0]))
        self.assertNotEqual(sequential[0, -1, 0], grouped)
        self.assertLess(abs(float(sequential[0, -1, 0] - grouped)), 3e-9)


@unittest.skipIf(torch is None, "optional torch dependency unavailable")
class TorchScanTests(unittest.TestCase):
    def test_forward_and_all_input_vjps(self):
        torch.manual_seed(95095)
        for matrix, scan in ((True, matrix_scan), (False, diagonal_scan)):
            pshape = (2, 7, 3, 3) if matrix else (2, 7, 3)
            qshape = (2, 7, 2, 3) if matrix else pshape
            ishape = (2, 2, 3) if matrix else (2, 3)
            original = (torch.randn(pshape, dtype=torch.float64) * .15,
                        torch.randn(qshape, dtype=torch.float64), torch.randn(ishape, dtype=torch.float64))
            mask = torch.tensor([[True, False, True, True, False, True, True]]).expand(2, -1)
            for algorithm in ("tree", "chunked"):
                for seed in (9, 95):
                    a = [x.clone().requires_grad_() for x in original]
                    b = [x.clone().requires_grad_() for x in original]
                    expected = scan(a[0], a[1], initial=a[2], write_mask=mask, algorithm="sequential")
                    actual = scan(b[0], b[1], initial=b[2], write_mask=mask,
                                  algorithm=algorithm, chunk_size=3)
                    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
                    torch.manual_seed(seed)
                    cotangent = torch.randn_like(actual)
                    ga = torch.autograd.grad((expected*cotangent).sum(), a)
                    gb = torch.autograd.grad((actual*cotangent).sum(), b)
                    for x, y in zip(ga, gb):
                        torch.testing.assert_close(x, y, atol=2e-12, rtol=2e-12)

    def test_gradcheck(self):
        torch.manual_seed(95)
        for matrix, scan in ((True, matrix_scan), (False, diagonal_scan)):
            shape = (1, 3, 2, 2) if matrix else (1, 3, 2)
            ishape = (1, 2, 2) if matrix else (1, 2)
            p = (torch.randn(shape, dtype=torch.float64)*.1).requires_grad_()
            q = torch.randn(shape, dtype=torch.float64, requires_grad=True)
            initial = torch.randn(ishape, dtype=torch.float64, requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(
                lambda p, q, s: scan(p, q, initial=s, algorithm="chunked", chunk_size=2),
                (p, q, initial)))

    def test_actual_source_scan_output_and_gradient_parity(self):
        path = ROOT / "research/mathematical-engineering/context026/context_transition.py"
        if not path.exists():
            self.skipTest("source snapshot unavailable")
        spec = importlib.util.spec_from_file_location("langmath095_original_context026", path)
        source = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source)
        torch.manual_seed(95095)
        p0 = torch.randn(2, 7, 3, 3, dtype=torch.float64)*.1
        q0 = torch.randn(2, 7, 2, 3, dtype=torch.float64)
        for algorithm in ("tree", "chunked"):
            a = [p0.clone().requires_grad_(), q0.clone().requires_grad_()]
            b = [p0.clone().requires_grad_(), q0.clone().requires_grad_()]
            expected = source.affine_scan(*a)
            actual = matrix_scan(*b, algorithm=algorithm, chunk_size=3)
            torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
            weights = torch.arange(actual.numel(), dtype=actual.dtype).sin().reshape_as(actual)
            ga = torch.autograd.grad((expected*weights).sum(), a)
            gb = torch.autograd.grad((actual*weights).sum(), b)
            for x, y in zip(ga, gb):
                torch.testing.assert_close(x, y, atol=2e-12, rtol=2e-12)


if __name__ == "__main__":
    unittest.main()
