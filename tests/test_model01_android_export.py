"""Model01 no-fit Android-compatible Java export: constructed numerical/invariant tests.

Host JVM evidence only; NOT Android hardware, energy, airplane-mode, or new source.
"""
from __future__ import annotations

import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "mobile/android/model01/Model01OfflinePredictor.java"
EXPORT = ROOT / "scripts/export_model01_android_java.py"
sys.path.insert(0, str(ROOT / "specialist_models"))
import model01_reference_api as reference

HARNESS = r'''
import java.io.BufferedReader;
import java.io.InputStreamReader;
import org.tacosm.mobile.model01.Model01OfflinePredictor;
public class Model01HostHarness {
    public static void main(String[] ignored) throws Exception {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, "UTF-8"));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] cells = line.split(",", -1);
            try {
                long index = Long.parseLong(cells[0]);
                double[] values = new double[cells.length - 1];
                for (int i = 1; i < cells.length; i++) values[i - 1] = Double.parseDouble(cells[i]);
                Model01OfflinePredictor.Prediction p = Model01OfflinePredictor.predict(index, values);
                System.out.println(p.predictionWh + "," + p.interval90LowerWh + ","
                        + p.interval90UpperWh + "," + p.independentQualification + "," + p.actionAuthority);
            } catch (IllegalArgumentException error) {
                System.out.println("ERR");
            }
        }
    }
}
'''
STATE_HARNESS = r'''
import org.tacosm.mobile.model01.Model01OfflinePredictor;
public class Model01StateHarness {
    public static void main(String[] args) {
        double[] original = new double[145];
        for (int i = 0; i < 145; i++) original[i] = 40 + i;
        Model01OfflinePredictor.WindowState state =
            new Model01OfflinePredictor.WindowState(500, original);
        original[144] = 9999; // defensive input snapshot
        if (state.snapshot()[144] != 184) throw new AssertionError("input alias");
        double[] leaked = state.snapshot(); leaked[144] = 9999;
        if (state.snapshot()[144] != 184) throw new AssertionError("output alias");
        state.append(501, 212);
        if (state.snapshot()[143] != 184 || state.snapshot()[144] != 212)
            throw new AssertionError("shift/state update");
        if (state.sourceAlignedIndex() != 501) throw new AssertionError("index");
        try { state.append(503, 213); throw new AssertionError("accepted gap"); }
        catch (IllegalArgumentException expected) { }
        if (state.sourceAlignedIndex() != 501) throw new AssertionError("mutated on gap");
        try { state.append(502, Double.NaN); throw new AssertionError("accepted NaN"); }
        catch (IllegalArgumentException expected) { }
        if (state.sourceAlignedIndex() != 501) throw new AssertionError("mutated on NaN");
        if (state.predict().horizonMinutes != 60) throw new AssertionError("horizon");
        System.out.println("STATE_OK");
    }
}
'''


def request(index, observations):
    return {"schema": "model01-energy-source-aligned-v1",
            "source_aligned_step_index": index,
            "last_145_energy_wh": observations}


def line(index, values):
    return ",".join([str(index)] + [str(x) for x in values])


@unittest.skipUnless(shutil.which("javac") and shutil.which("java"), "JDK needed for host JVM checks")
class OfflineExportChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.q = reference.load_original()
        cls.tmp = tempfile.TemporaryDirectory(prefix="model01-java-")
        cls.dir = Path(cls.tmp.name)
        for name, source in (("Model01HostHarness.java", HARNESS),
                             ("Model01StateHarness.java", STATE_HARNESS)):
            (cls.dir / name).write_text(source, encoding="utf-8")
        subprocess.run(["javac", "-d", str(cls.dir), str(SRC),
                        str(cls.dir / "Model01HostHarness.java"),
                        str(cls.dir / "Model01StateHarness.java")], check=True,
                       capture_output=True, text=True, timeout=60)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_lines(self, lines):
        p = subprocess.run(["java", "-cp", str(self.dir), "Model01HostHarness"],
                           input="\n".join(lines) + "\n", text=True,
                           capture_output=True, check=True, timeout=30)
        return p.stdout.splitlines()

    def test_exact_pinned_generated_source(self):
        subprocess.run([sys.executable, str(EXPORT), "--check"],
                       capture_output=True, text=True, check=True)
        source = SRC.read_text()
        self.assertNotIn("java.net", source)
        self.assertNotIn("http://", source)
        self.assertIn(reference.ORIGINAL_MODEL_SHA256, source)

    def test_numerical_parity_with_original_frozen_python_reference(self):
        rng = random.Random(28_10_2026)
        cases = []
        for index in [144, 145, 500, 1024, 20000, 20_000_000]:
            cases.append((index, [float(20 + j % 7) for j in range(145)]))
        for _ in range(40):
            index = rng.randint(144, 120000)
            cases.append((index, [rng.random() * 1800 for _ in range(145)]))
        cases.append((144, [0.0] * 145))
        cases.append((500, [1_000_000.0] * 145))
        outputs = self.run_lines([line(i, vals) for i, vals in cases])
        self.assertEqual(len(cases), len(outputs))
        for (idx, values), result in zip(cases, outputs):
            expected = reference.predict(self.model, self.q, request(idx, values))
            fields = result.split(",")
            self.assertEqual(fields[3:], ["NOT_QUALIFIED", "ADVISORY_ONLY"])
            for got, target in zip(fields[:3],
                [expected["prediction_wh"], *expected["interval_90_cal_only_wh"]]):
                self.assertTrue(math.isclose(float(got), target, rel_tol=1e-12, abs_tol=1e-9),
                                (idx, got, target))

    def test_invalid_input_abstains_without_fallback(self):
        vals = [55.0] * 145
        bad = [(143, vals), (20_000_001, vals), (500, vals[:-1]),
               (500, vals[:-1] + [float("nan")]),
               (500, vals[:-1] + [float("inf")]),
               (500, vals[:-1] + [-1.0]),
               (500, vals[:-1] + [1_000_001.0])]
        self.assertEqual(self.run_lines([line(i, v) for i, v in bad]), ["ERR"] * len(bad))

    def test_window_state_copy_gap_and_validation(self):
        p = subprocess.run(["java", "-cp", str(self.dir), "Model01StateHarness"],
                           capture_output=True, check=True, text=True, timeout=15)
        self.assertEqual(p.stdout.strip(), "STATE_OK")


if __name__ == "__main__":
    unittest.main()
