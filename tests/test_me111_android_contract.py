"""No-fit host-JVM tests for frozen TAC ME111 Android input/state guard.

Constructed values are purely software diagnostics; not real source performance,
not physical Android/JNI ONNX execution or airplane-mode qualification.
"""
import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "mobile/android/model01/sample-app/app/src/main/java"
          / "org/tacosm/mobile/me111/Me111Contract.java")
HARNESS = r'''
import org.tacosm.mobile.me111.Me111Contract;
public class Me111ContractHarness {
    private static int assertions = 0;
    private static void ok(boolean valid) {
        assertions++;
        if (!valid) throw new AssertionError("contract check failed at " + assertions);
    }
    private static void reject(Runnable fn) {
        assertions++;
        try { fn.run(); } catch (IllegalArgumentException expected) { return; }
        throw new AssertionError("invalid state/event was accepted");
    }
    private static Me111Contract.Event event(int owner, long hour) {
        return new Me111Contract.Event(owner, hour,
            new float[13], new boolean[13], new float[13], new float[6]);
    }
    public static void main(String[] args) {
        Me111Contract.State initial = Me111Contract.initial(0, 100);
        Me111Contract.Event e = event(0, 100);
        Me111Contract.validateStep(e, initial);
        ok(initial.lastHour == 99 && initial.count == 0);
        ok(initial.memory.length == 8 && initial.evidenceSum.length == 62);
        ok(Me111Contract.MODEL_SHA256.length() == 64);
        reject(() -> Me111Contract.initial(12, 100));
        reject(() -> Me111Contract.initial(0, 0));
        reject(() -> event(0, 0));
        reject(() -> event(-1, 100));
        reject(() -> Me111Contract.validateStep(event(1, 100), initial));
        reject(() -> Me111Contract.validateStep(event(0, 99), initial));
        reject(() -> Me111Contract.validateStep(event(0, 98), initial));
        reject(() -> new Me111Contract.Event(0, 100,
            new float[12], new boolean[13], new float[13], new float[6]));
        reject(() -> new Me111Contract.Event(0, 100,
            new float[13], new boolean[12], new float[13], new float[6]));
        reject(() -> new Me111Contract.Event(0, 100,
            new float[13], new boolean[13], new float[13], new float[5]));
        float[] invalid = new float[13]; invalid[0] = Float.NaN;
        reject(() -> new Me111Contract.Event(0, 100,
            invalid, new boolean[13], new float[13], new float[6]));
        invalid[0] = Float.POSITIVE_INFINITY;
        reject(() -> new Me111Contract.Event(0, 100,
            invalid, new boolean[13], new float[13], new float[6]));
        invalid[0] = 9.0f;
        reject(() -> new Me111Contract.Event(0, 100,
            invalid, new boolean[13], new float[13], new float[6]));
        invalid[0] = 1.0f;
        reject(() -> new Me111Contract.Event(0, 100,
            invalid, new boolean[13], new float[13], new float[6]));
        boolean[] mask = new boolean[13]; mask[0] = true;
        new Me111Contract.Event(0, 100, invalid, mask, new float[13], new float[6]);
        float[] badAge = new float[13]; badAge[4] = -0.01f;
        reject(() -> new Me111Contract.Event(0, 100,
            new float[13], new boolean[13], badAge, new float[6]));
        float[] badClock = new float[6]; badClock[1] = Float.NaN;
        reject(() -> new Me111Contract.Event(0, 100,
            new float[13], new boolean[13], new float[13], badClock));
        reject(() -> new Me111Contract.State(0, 99, -1,
            new float[13], new float[13], new float[62], new float[8][8]));
        float[] sourceMass = new float[13]; sourceMass[3] = 2.0f;
        reject(() -> new Me111Contract.State(0, 99, 0,
            new float[13], sourceMass, new float[62], new float[8][8]));
        float[][] mem = new float[8][8];
        Me111Contract.State defensivelyCopied = new Me111Contract.State(0,99,0,
            new float[13],new float[13],new float[62],mem);
        mem[0][0] = 999.0f;
        ok(defensivelyCopied.memory[0][0] == 0.0f);
        float[] original = new float[13];
        Me111Contract.Event copied = new Me111Contract.Event(0,100,original,
            new boolean[13], new float[13], new float[6]);
        original[0] = 1.0f;
        ok(copied.values[0] == 0.0f);
        Me111Contract.validatePrediction(new float[6],
            new float[]{1,1,1,1,1,1}, new float[]{1,1,1,1,1,1});
        reject(() -> Me111Contract.validatePrediction(new float[6],
            new float[]{0,1,1,1,1,1}, new float[]{1,1,1,1,1,1}));
        reject(() -> Me111Contract.validatePrediction(new float[5],
            new float[]{1,1,1,1,1,1}, new float[]{1,1,1,1,1,1}));
        ok(assertions >= 22);
        System.out.println("ME111_ANDROID_HOST_CONTRACT_PASS=" + assertions);
    }
}
'''

@unittest.skipUnless(shutil.which("javac") and shutil.which("java"),
                     "JDK required; no false Android qualification")
class AndroidMe111ContractTest(unittest.TestCase):
    def test_java_frozen_input_and_state_invariants(self):
        with tempfile.TemporaryDirectory(prefix="me111-host-contract-") as temp:
            directory = pathlib.Path(temp)
            harness = directory / "Me111ContractHarness.java"
            harness.write_text(HARNESS, encoding="utf-8")
            subprocess.run(["javac", "-d", str(directory), str(SOURCE), str(harness)],
                           check=True, text=True, capture_output=True, timeout=60)
            result = subprocess.run(["java", "-cp", str(directory), "Me111ContractHarness"],
                                    check=True, text=True, capture_output=True, timeout=30)
            self.assertIn("ME111_ANDROID_HOST_CONTRACT_PASS=", result.stdout)


if __name__ == "__main__":
    unittest.main()
