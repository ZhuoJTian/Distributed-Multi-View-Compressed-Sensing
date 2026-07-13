"""Smoke tests for the reorganized package and experiment entry points."""

import importlib.util
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@unittest.skipUnless(
    importlib.util.find_spec("cvxpy") is not None,
    "cvxpy is not installed; install requirements.txt to run import tests",
)
class ImportTests(unittest.TestCase):
    """Verify that package modules can be imported without running experiments."""

    def test_core_modules_import(self):
        from dmvcs.algorithms import admm  # noqa: F401
        from dmvcs.data import generation  # noqa: F401
        from dmvcs.inference import view_inference  # noqa: F401
        from dmvcs.network import generation as network_generation  # noqa: F401
        from dmvcs.optimization import update_l1_cvx  # noqa: F401

    def test_experiment_modules_expose_main(self):
        modules = [
            ROOT / "scripts" / "experiments" / "compare_measurements.py",
            ROOT / "scripts" / "experiments" / "compare_blockages.py",
        ]
        for index, module_path in enumerate(modules):
            spec = importlib.util.spec_from_file_location(f"experiment_{index}", module_path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            self.assertTrue(callable(module.main))


if __name__ == "__main__":
    unittest.main()
