"""Check Ma2020 bootstrap-parameter routing without executing an experiment.

Only function definitions are loaded from the analysis source. EEG loading,
bootstrap generation, screening, classification, and target splitting are all
replaced by mocks. No scientific result or existing artifact is read or changed.
The same file can run from the research ``simulations/`` directory or the
compact repository's ``tests/`` directory.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CANDIDATES = (
    ROOT / "simulations/ci_gate_ma2020_riemann.py",
    ROOT / "analysis/ci_gate_ma2020_riemann.py",
)
SOURCE = next((path for path in SOURCE_CANDIDATES if path.is_file()), None)
if SOURCE is None:
    raise FileNotFoundError("The Ma2020 analysis source is required beside this test.")


def load_functions() -> dict:
    """Load actual function bodies without importing EEG or classifier packages."""
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef)
             or (isinstance(node, ast.ImportFrom) and node.module == "__future__")]
    namespace = {
        "__file__": str(SOURCE), "__doc__": "Mocked parameter-routing test.",
        "argparse": argparse, "hashlib": hashlib, "json": json,
        "Path": Path, "sys": sys, "time": time,
        "np": np, "pd": pd, "ROOT": ROOT,
        "DEFAULT_OUT": ROOT / "unused-mocked-output",
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


class Ma2020BootstrapBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.code = load_functions()
        # Constant arrays are indexing fixtures, not simulated EEG observations.
        self.covariances = np.tile(np.eye(2), (12, 1, 1))
        self.labels = np.arange(12) % 2
        self.bootstrap_calls = []

        def fake_bootstrap(sources, target, rng, count, shrinkage, shared_target):
            self.bootstrap_calls.append({
                "count": count, "target_rows": len(target),
                "shared_target": shared_target,
            })
            return np.arange(1, len(sources) + 1, dtype=float), np.zeros((count, len(sources)))

        source_path = Mock()
        source_path.exists.return_value = True
        self.loader = Mock(return_value=(self.covariances, self.labels))
        self.bootstrap = Mock(side_effect=fake_bootstrap)
        self.classifier = Mock(return_value=0.625)
        self.screen = Mock(return_value=np.array([0], dtype=int))
        split = Mock()
        split.split.return_value = [(np.arange(6), np.arange(6, 12))]
        self.splitter = Mock(return_value=split)
        system = {key: np.zeros(2) for key in (
            "lower_c", "upper_c", "lower_d", "lower_cj", "upper_cj", "lower_dj",
        )}
        system["jj"] = np.array([[0, 1], [1, 0]])
        self.code.update({
            "session_path": Mock(return_value=source_path),
            "extract_session_covs": self.loader,
            "riemann_estimates_and_bootstrap": self.bootstrap,
            "screen_pair_ref": self.screen,
            "train_eval_ts": self.classifier,
            "StratifiedShuffleSplit": self.splitter,
            "build_system": Mock(return_value=system),
            "rect_gate": Mock(return_value=np.array([0, 1], dtype=int)),
            "pair_gate": Mock(return_value=np.array([0, 1], dtype=int)),
            "resolve_data_root": Mock(side_effect=lambda value, parser: value),
            "resolve_cache_root": Mock(side_effect=lambda value, parser, **kwargs: value),
        })

    @staticmethod
    def arguments(*, boot=499, down_boot=17):
        return SimpleNamespace(
            target_session=3, data_root=Path("unused-data"),
            cache_root=Path("unused-cache"), tmin=0.0, tmax=4.0,
            l_freq=8.0, h_freq=30.0, shrinkage=0.1,
            boot=boot, down_boot=down_boot, splits=1, seed=20260618,
            alpha=0.05, alpha_c=0.025, alpha_d=0.025,
        )

    def downstream_counts(self, **parameters):
        self.bootstrap_calls.clear()
        self.code["run_subject_downstream"](1, self.arguments(**parameters), object())
        return [call["count"] for call in self.bootstrap_calls]

    def run_mocked_main(self, *, downstream=True, seed=None, use_default_budgets=False):
        """Use real CLI parsing and manifest writing, with all analyses mocked."""
        with tempfile.TemporaryDirectory(prefix="ma2020-budget-test-") as directory:
            temporary = Path(directory)
            output = temporary / "output"
            command = [
                str(SOURCE), "--subjects", "1-2", "--target-session", "3",
                "--data-root", str(temporary / "unused-data"),
                "--cache-root", str(temporary / "unused-cache"),
                "--out-dir", str(output), "--splits", "1",
            ]
            if not use_default_budgets:
                command.extend(["--boot", "31", "--down-boot", "7"])
            if downstream:
                command.append("--downstream")
            if seed is not None:
                command.extend(["--seed", str(seed)])
            self.last_command = list(command)
            # Creating or drawing from a real RNG is unnecessary for these tests.
            with patch.object(sys, "argv", command), \
                    patch.object(np.random, "default_rng", return_value=object()), \
                    redirect_stdout(io.StringIO()):
                self.code["main"]()
            return json.loads((output / "manifest.json").read_text())

    def test_downstream_uses_its_explicit_budget(self):
        self.assertEqual(self.downstream_counts(boot=499, down_boot=17), [17])

    def test_changing_full_screening_budget_does_not_change_downstream_budget(self):
        first = self.downstream_counts(boot=499, down_boot=17)
        second = self.downstream_counts(boot=999, down_boot=17)
        self.assertEqual(first, [17])
        self.assertEqual(second, first)

    def test_changing_downstream_budget_reaches_the_bootstrap_call(self):
        self.assertEqual(self.downstream_counts(boot=499, down_boot=17), [17])
        self.assertEqual(self.downstream_counts(boot=499, down_boot=29), [29])

    def test_full_screening_keeps_its_own_budget_for_both_target_controls(self):
        self.code["run_subject_screening"](1, self.arguments(boot=31, down_boot=7), object())
        self.assertEqual([call["count"] for call in self.bootstrap_calls], [31, 31])
        self.assertEqual([call["shared_target"] for call in self.bootstrap_calls], [True, False])

    def test_manifest_downstream_budget_equals_the_budget_passed_at_runtime(self):
        manifest = self.run_mocked_main()
        counts = [call["count"] for call in self.bootstrap_calls if call["target_rows"] == 6]
        self.assertEqual(len(counts), 2)
        self.assertEqual(counts, [manifest["downstream_boot"]] * 2)
        self.assertEqual(manifest["downstream_boot"], 7)

    def test_manifest_full_screening_budget_equals_both_target_control_budgets(self):
        manifest = self.run_mocked_main()
        counts = [call["count"] for call in self.bootstrap_calls if call["target_rows"] == 12]
        self.assertEqual(counts, [manifest["screening_boot"]] * 4)
        self.assertEqual(manifest["screening_boot"], 31)

    def test_without_downstream_no_downstream_budget_is_claimed_or_executed(self):
        manifest = self.run_mocked_main(downstream=False)
        self.assertIsNone(manifest["downstream_boot"])
        self.assertIsNone(manifest["splits"])
        self.assertEqual([call["target_rows"] for call in self.bootstrap_calls], [12] * 4)
        self.classifier.assert_not_called()
        self.screen.assert_not_called()

    def test_manifest_script_hash_matches_the_loaded_analysis_source(self):
        manifest = self.run_mocked_main()
        self.assertEqual(manifest["script_sha256"], hashlib.sha256(SOURCE.read_bytes()).hexdigest())
        self.assertEqual(manifest["script"], SOURCE.name)

    def test_default_budgets_reach_the_actual_calls_and_manifest(self):
        manifest = self.run_mocked_main(use_default_budgets=True)
        full = [call["count"] for call in self.bootstrap_calls if call["target_rows"] == 12]
        downstream = [call["count"] for call in self.bootstrap_calls if call["target_rows"] == 6]
        self.assertEqual(full, [499] * 4)
        self.assertEqual(downstream, [199] * 2)
        self.assertEqual(manifest["screening_boot"], 499)
        self.assertEqual(manifest["downstream_boot"], 199)

    def test_default_budgets_match_the_preserved_primary_protocol_record(self):
        manifest = self.run_mocked_main(use_default_budgets=True)
        candidates = (
            ROOT / "results/eeg/ma2020/protocol.json",
            ROOT / "simulation_results/ma2020_riemann_all_s1_s25_b499/manifest.json",
        )
        record_path = next((path for path in candidates if path.is_file()), None)
        expected = {"screening_boot": 499, "downstream_boot": 199}
        if record_path is not None:
            record = json.loads(record_path.read_text())
            for field, count in expected.items():
                self.assertEqual(record[field], count, str(record_path))
                self.assertEqual(manifest[field], record[field])
        else:
            for field, count in expected.items():
                self.assertEqual(manifest[field], count)

    def test_manifest_records_the_mocked_command_and_default_or_explicit_seed(self):
        for seed in (None, 20261020):
            with self.subTest(seed=seed):
                manifest = self.run_mocked_main(seed=seed)
                self.assertEqual(manifest["command"], [sys.executable, *self.last_command])
                self.assertEqual(manifest["seed"], 20260618 if seed is None else seed)

    def test_downstream_requires_only_the_mocked_analysis_functions(self):
        rows = self.code["run_subject_downstream"](1, self.arguments(), object())
        self.assertEqual(self.loader.call_count, 3)
        self.assertEqual(self.bootstrap.call_count, 1)
        self.assertEqual(self.screen.call_count, 1)
        self.assertEqual(self.classifier.call_count, 4)
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["balanced_accuracy"] for row in rows}, {0.625})


if __name__ == "__main__":
    unittest.main(verbosity=2)
