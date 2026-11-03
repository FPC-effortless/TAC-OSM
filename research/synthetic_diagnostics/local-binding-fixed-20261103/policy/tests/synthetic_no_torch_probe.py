"""Run a selected source-only assertion in a fresh Torch-blocked interpreter.

The parent may have imported Torch while collecting unrelated repository tests.
Only the isolated child's imports are evidence for this no-numerical-import gate.
"""
import importlib.abc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


def assert_isolated_no_torch(case, test_file):
    if getattr(case, "_isolated_no_torch_probe", False):
        case.assertFalse(any(n == "torch" or n.startswith("torch.") for n in sys.modules))
        return
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
         str(Path(test_file).resolve()), case.__class__.__name__, case._testMethodName],
        capture_output=True, text=True, timeout=30, check=False,
    )
    case.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    case.assertEqual(json.loads(result.stdout), {
        "status": "PASS_ISOLATED_NO_TORCH_IMPORT", "torch_import_attempts": 0,
        "tests_run": 1,
    })


def main():
    test_file = Path(sys.argv[1]).resolve()
    root = test_file.parents[1]
    # -I excludes inherited PYTHONPATH and current-directory injection. Add only
    # the repository paths needed by these exact source tests.
    sys.path[:0] = [str(root), str(root / "src")]
    attempts = []

    class BlockTorch(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "torch" or fullname.startswith("torch."):
                attempts.append(fullname)
                raise ImportError("Torch import prohibited in isolated source probe")
            return None

    if any(n == "torch" or n.startswith("torch.") for n in sys.modules):
        raise RuntimeError("isolated interpreter unexpectedly contains Torch")
    sys.meta_path.insert(0, BlockTorch())
    spec = importlib.util.spec_from_file_location("isolated_source_test", test_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case = getattr(module, sys.argv[2])(sys.argv[3])
    case._isolated_no_torch_probe = True
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=1).run(unittest.TestSuite([case]))
    if (not result.wasSuccessful() or result.testsRun != 1 or attempts
            or any(n == "torch" or n.startswith("torch.") for n in sys.modules)):
        raise SystemExit(1)
    print(json.dumps({"status": "PASS_ISOLATED_NO_TORCH_IMPORT",
                      "torch_import_attempts": len(attempts), "tests_run": result.testsRun}))


if __name__ == "__main__":
    main()
