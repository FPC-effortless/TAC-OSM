"""Host-JVM constructed negative controls for the frozen ME111 v2 state codec.

These checks verify serialization/detection of accidental corruption. SHA256
is NOT device authentication, a MAC, encryption, or rollback prevention.
"""
import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
JAVA = (ROOT / "mobile/android/model01/sample-app/app/src/main/java"
        / "org/tacosm/mobile/me111")
HARNESS = r'''
import java.io.IOException;
import java.security.MessageDigest;
import java.util.Arrays;
import org.tacosm.mobile.me111.Me111Contract;
import org.tacosm.mobile.me111.Me111StateCodec;
public final class StateCodecHarness {
    private static int assertions = 0;
    private static void yes(boolean condition) {
        assertions++;
        if (!condition) throw new AssertionError("state codec assertion #" + assertions);
    }
    private static void fails(byte[] bytes) {
        assertions++;
        try { Me111StateCodec.decode(bytes); }
        catch (IOException expected) { return; }
        throw new AssertionError("invalid state accepted");
    }
    private static void rehash(byte[] blob) throws Exception {
        byte[] sha = MessageDigest.getInstance("SHA-256")
            .digest(Arrays.copyOf(blob, blob.length - 32));
        System.arraycopy(sha, 0, blob, blob.length - 32, 32);
    }
    public static void main(String[] args) throws Exception {
        Me111Contract.State first = Me111Contract.initial(0, 100);
        byte[] encoded = Me111StateCodec.encode(first);
        yes(encoded.length == Me111StateCodec.BINARY_LENGTH);
        yes(encoded.length == 734);
        Me111Contract.State restored = Me111StateCodec.decode(encoded);
        yes(restored.owner == 0 && restored.lastHour == 99 && restored.count == 0);
        yes(Arrays.equals(restored.sourceSum, first.sourceSum));
        yes(Arrays.equals(restored.sourceMass, first.sourceMass));
        yes(Arrays.equals(restored.evidenceSum, first.evidenceSum));
        yes(Arrays.equals(restored.memory[7], first.memory[7]));
        yes(Arrays.equals(encoded, Me111StateCodec.encode(restored)));

        byte[] changedPayload = encoded.clone();
        changedPayload[changedPayload.length - 33] ^= 1; // only low-bit FP32 error
        fails(changedPayload);
        byte[] changedDigest = encoded.clone();
        changedDigest[changedDigest.length - 1] ^= 1;
        fails(changedDigest);
        fails(Arrays.copyOf(encoded, encoded.length - 1));
        fails(Arrays.copyOf(encoded, encoded.length + 1));
        fails(Arrays.copyOf(encoded, encoded.length - 32)); // unsigned v1 format
        fails(new byte[encoded.length]);
        byte[] alteredVersion = encoded.clone();
        alteredVersion[7] ^= 1;
        rehash(alteredVersion);
        fails(alteredVersion);
        byte[] alteredIdentity = encoded.clone();
        alteredIdentity[10] = alteredIdentity[10] == 'd' ? (byte)'e' : (byte)'d';
        rehash(alteredIdentity);
        fails(alteredIdentity);

        float[] source = new float[13];
        float[] mass = new float[13];
        float[] evidence = new float[62];
        float[][] memory = new float[8][8];
        source[0] = -0.0f;
        source[1] = 0.33333334f;
        mass[0] = 0.625f;
        evidence[4] = 4.125f;
        memory[2][3] = -0.0234375f;
        Me111Contract.State filled = new Me111Contract.State(
            5, 255, 20, source, mass, evidence, memory);
        byte[] positive = Me111StateCodec.encode(filled);
        Me111Contract.State recovered = Me111StateCodec.decode(positive);
        yes(recovered.owner == 5 && recovered.lastHour == 255 && recovered.count == 20);
        yes(Float.floatToRawIntBits(recovered.sourceSum[0])
             == Float.floatToRawIntBits(-0.0f));
        yes(Arrays.equals(filled.sourceSum, recovered.sourceSum));
        yes(Arrays.equals(filled.sourceMass, recovered.sourceMass));
        yes(Arrays.equals(filled.evidenceSum, recovered.evidenceSum));
        for (int i = 0; i < 8; i++)
            yes(Arrays.equals(filled.memory[i], recovered.memory[i]));
        yes(Arrays.equals(positive, Me111StateCodec.encode(recovered)));

        System.out.println("ME111_STATE_CODEC_HOST_JVM_PASS=" + assertions);
    }
}
'''

@unittest.skipUnless(shutil.which("javac") and shutil.which("java"),
                     "JDK required; never assume Android/JNI was qualified")
class TestStateCodec(unittest.TestCase):
    def test_sha256_fixed_length_state_integrity(self):
        with tempfile.TemporaryDirectory(prefix="me111-codec-") as root:
            root = pathlib.Path(root)
            harness = root / "StateCodecHarness.java"
            harness.write_text(HARNESS, encoding="utf-8")
            subprocess.run(
                ["javac", "-d", str(root), str(JAVA / "Me111Contract.java"),
                 str(JAVA / "Me111StateCodec.java"), str(harness)],
                check=True, capture_output=True, text=True, timeout=60)
            outcome = subprocess.run(
                ["java", "-cp", str(root), "StateCodecHarness"],
                check=True, capture_output=True, text=True, timeout=30)
            self.assertIn("ME111_STATE_CODEC_HOST_JVM_PASS=", outcome.stdout)


if __name__ == "__main__":
    unittest.main()
