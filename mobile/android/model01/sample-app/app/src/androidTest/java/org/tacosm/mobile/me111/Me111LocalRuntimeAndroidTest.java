package org.tacosm.mobile.me111;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import android.util.AtomicFile;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

import static org.junit.Assert.*;

/**
 * Emulator integration only. Constructed reference vectors from ORIGINAL ME111.
 * This is NOT physical-device, real-data accuracy or airplane-mode evidence.
 */
@RunWith(AndroidJUnit4.class)
public final class Me111LocalRuntimeAndroidTest {
    private static float[] numbers(JSONArray values) throws Exception {
        float[] result = new float[values.length()];
        for (int i = 0; i < result.length; i++)
            result[i] = (float)values.getDouble(i);
        return result;
    }

    private static void checkClose(JSONArray expected, float[] actual) throws Exception {
        assertEquals(expected.length(), actual.length);
        for (int i = 0; i < actual.length; i++) {
            double reference = expected.getDouble(i);
            double allowed = 3e-4 + 2e-4 * Math.abs(reference);
            assertTrue("Android ONNX numerical mismatch at " + i + ": " + actual[i]
                + " versus frozen PyTorch " + reference,
                Math.abs(actual[i] - reference) <= allowed);
        }
    }

    private static void checkMemory(JSONArray expected, float[][] actual) throws Exception {
        assertEquals(8, expected.length());
        for (int i = 0; i < 8; i++)
            checkClose(expected.getJSONArray(i), actual[i]);
    }

    private static JSONObject fixture(Context context) throws Exception {
        try (InputStream input = context.getAssets().open("me111_reference_vectors.json");
             ByteArrayOutputStream bytes = new ByteArrayOutputStream()) {
            byte[] chunk = new byte[8192];
            int n;
            while ((n = input.read(chunk)) != -1)
                bytes.write(chunk, 0, n);
            return new JSONObject(new String(bytes.toByteArray(), StandardCharsets.UTF_8));
        }
    }

    @Test public void testOriginalFrozenModelNumericalParityAndPrivateState() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        JSONObject expected = fixture(context);
        assertEquals("CONSTRUCTED_SOFTWARE_PARITY_ONLY", expected.getString("status"));
        assertEquals(Me111Contract.CHECKPOINT_SHA256, expected.getString("original_checkpoint_sha256"));
        assertEquals(Me111Contract.MODEL_SHA256, expected.getString("exported_onnx_sha256"));

        Me111StateStore store = new Me111StateStore(context);
        store.clear();  // test-only explicit reset inside test application.
        Me111Contract.State current = Me111Contract.initial(0, 100);
        JSONArray records = expected.getJSONArray("records");
        assertEquals(2, records.length());

        try (Me111LocalEngine engine = new Me111LocalEngine(context)) {
            for (int i = 0; i < records.length(); i++) {
                JSONObject r = records.getJSONObject(i);
                float[] values = new float[13];
                boolean[] observed = new boolean[13];
                values[0] = (float)r.getDouble("input_value_first");
                observed[0] = true;
                Me111Contract.Event event = new Me111Contract.Event(
                    0, r.getLong("hour"), values, observed, new float[13], new float[6]);
                Me111LocalEngine.Result result = engine.predict(event, current);
                checkClose(r.getJSONArray("log_location"), result.logLocation);
                checkClose(r.getJSONArray("left_scale"), result.leftScale);
                checkClose(r.getJSONArray("right_scale"), result.rightScale);
                checkClose(r.getJSONArray("source_sum"), result.nextState.sourceSum);
                checkClose(r.getJSONArray("source_mass"), result.nextState.sourceMass);
                checkClose(r.getJSONArray("evidence_sum"), result.nextState.evidenceSum);
                checkMemory(r.getJSONArray("memory"), result.nextState.memory);
                assertEquals(r.getLong("count"), result.nextState.count);
                assertEquals(r.getLong("last_hour"), result.nextState.lastHour);
                assertEquals("ADVISORY_ONLY", result.actionAuthority);
                assertEquals("FAILED_MIXED_ME111", result.scientificQualification);

                store.save(result.nextState);
                Me111Contract.State restored = store.load();
                assertNotNull(restored);
                assertEquals(result.nextState.count, restored.count);
                assertEquals(result.nextState.lastHour, restored.lastHour);
                for (int k = 0; k < 8; k++)
                    assertArrayEquals(result.nextState.memory[k], restored.memory[k], 0f);
                current = restored;
            }
            final Me111Contract.State recorded = current;
            try {
                engine.predict(new Me111Contract.Event(0, recorded.lastHour,
                    new float[13], new boolean[13], new float[13], new float[6]), recorded);
                fail("duplicate source hour accepted");
            } catch (IllegalArgumentException expectedError) { }
            try {
                engine.predict(new Me111Contract.Event(1, recorded.lastHour + 1,
                    new float[13], new boolean[13], new float[13], new float[6]), recorded);
                fail("foreign source station accepted");
            } catch (IllegalArgumentException expectedError) { }
            assertEquals(recorded.lastHour, store.load().lastHour);
        } finally {
            store.clear();
        }
    }

    @Test public void testAtomicStateInterruptedWriteAndCorruptionFailClosed() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        Me111StateStore store = new Me111StateStore(context);
        store.clear();
        try {
            Me111Contract.State initial = Me111Contract.initial(0, 100);
            store.save(initial);
            File file = new File(context.getFilesDir(), "me111-state-v1.bin");
            AtomicFile simulatedCrash = new AtomicFile(file);
            // Simulate process death mid-write: do not commit the .new candidate.
            try (FileOutputStream unfinished = simulatedCrash.startWrite()) {
                unfinished.write(new byte[]{0x5, 0x6, 0x7, 0x8});
                unfinished.flush();
            }
            // Reading must restore the last completely committed state.
            Me111Contract.State restored = store.load();
            assertNotNull(restored);
            assertEquals(initial.lastHour, restored.lastHour);
            assertEquals(initial.count, restored.count);

            // Now emulate corrupted committed bytes; never substitute a zero state.
            try (FileOutputStream corrupt = new FileOutputStream(file)) {
                corrupt.write(new byte[]{0x1, 0x2, 0x3, 0x4});
                corrupt.flush();
            }
            try {
                store.load();
                fail("corrupt persistent state was silently accepted/reset");
            } catch (IOException expected) { }
        } finally {
            store.clear();
        }
    }

    @Test public void testAppHasNoInternetPermission() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        PackageInfo info = context.getPackageManager().getPackageInfo(
            context.getPackageName(), PackageManager.GET_PERMISSIONS);
        if (info.requestedPermissions != null) {
            for (String requested : info.requestedPermissions)
                assertNotEquals(Manifest.permission.INTERNET, requested);
        }
        assertEquals(PackageManager.PERMISSION_DENIED,
            context.checkSelfPermission(Manifest.permission.INTERNET));
    }
}
