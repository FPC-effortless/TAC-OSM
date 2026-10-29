package org.tacosm.mobile.me111;

import android.content.Context;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

import static org.junit.Assert.*;

/**
 * Original frozen ME111 PyTorch-vs-Android CPU: 256 ordered events per source,
 * missing observations, 3-hour gaps and eight private-state save/restore rounds.
 *
 * All observations are CONSTRUCTED numerical diagnostics, NOT authentic sensor
 * records, a forecast-accuracy result or 256-event useful-memory evidence.
 */
@RunWith(AndroidJUnit4.class)
public final class Me111LongStreamAndroidTest {
    private static final String FIXTURE = "me111_256_reference_vectors.json";

    private static JSONObject readOracle() throws Exception {
        Context instrumentation = InstrumentationRegistry.getInstrumentation().getContext();
        try (InputStream input = instrumentation.getAssets().open(FIXTURE);
             ByteArrayOutputStream bytes = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[16 * 1024];
            int n;
            while ((n = input.read(buffer)) != -1) {
                bytes.write(buffer, 0, n);
                if (bytes.size() > 2_000_000) throw new IllegalStateException("fixture too large");
            }
            return new JSONObject(new String(bytes.toByteArray(), StandardCharsets.UTF_8));
        }
    }

    private static float[] floats(JSONArray values, int dimension) throws Exception {
        assertEquals("original fixture dimension", dimension, values.length());
        float[] result = new float[dimension];
        for (int i = 0; i < dimension; i++)
            result[i] = (float) values.getDouble(i);
        return result;
    }

    private static boolean[] flags(JSONArray values) throws Exception {
        assertEquals(13, values.length());
        boolean[] bits = new boolean[13];
        for (int i = 0; i < bits.length; i++) bits[i] = values.getBoolean(i);
        return bits;
    }

    private static void originalClose(JSONArray expected, float[] actual) throws Exception {
        assertEquals(expected.length(), actual.length);
        for (int i = 0; i < actual.length; i++) {
            double target = expected.getDouble(i);
            assertTrue("nonfinite Android neural output at " + i,
                       !Float.isNaN(actual[i]) && !Float.isInfinite(actual[i]));
            double epsilon = 3e-4 + 2e-4 * Math.abs(target);
            assertTrue("256-step original PyTorch/Android mismatch channel " + i
                       + " expected=" + target + " actual=" + actual[i],
                       Math.abs(target - actual[i]) <= epsilon);
        }
    }

    private static void compareSnapshot(JSONObject ref, Me111LocalEngine.Result got)
            throws Exception {
        originalClose(ref.getJSONArray("log_location"), got.logLocation);
        originalClose(ref.getJSONArray("left_scale"), got.leftScale);
        originalClose(ref.getJSONArray("right_scale"), got.rightScale);
        originalClose(ref.getJSONArray("source_sum"), got.nextState.sourceSum);
        originalClose(ref.getJSONArray("source_mass"), got.nextState.sourceMass);
        originalClose(ref.getJSONArray("evidence_sum"), got.nextState.evidenceSum);
        JSONArray target = ref.getJSONArray("memory");
        assertEquals(8, target.length());
        for (int i = 0; i < 8; i++)
            originalClose(target.getJSONArray(i), got.nextState.memory[i]);
        assertEquals(ref.getLong("count"), got.nextState.count);
        assertEquals(ref.getLong("last_hour"), got.nextState.lastHour);
    }

    private static void requireLosslessState(Me111Contract.State written,
                                             Me111Contract.State restored) {
        assertNotNull(restored);
        assertEquals(written.owner, restored.owner);
        assertEquals(written.lastHour, restored.lastHour);
        assertEquals(written.count, restored.count);
        assertArrayEquals(written.sourceSum, restored.sourceSum, 0f);
        assertArrayEquals(written.sourceMass, restored.sourceMass, 0f);
        assertArrayEquals(written.evidenceSum, restored.evidenceSum, 0f);
        for (int i = 0; i < 8; i++)
            assertArrayEquals(written.memory[i], restored.memory[i], 0f);
    }

    @Test public void twoOwnersAnd256OriginalSourceStateSteps() throws Exception {
        JSONObject oracle = readOracle();
        assertEquals("ORIGINAL_FROZEN_PYTORCH_256_STEP_CONSTRUCTED_ANDROID_FIXTURE",
                     oracle.getString("status"));
        assertTrue(oracle.getBoolean("not_authentic_source_evaluation"));
        assertEquals("ME111_FAILED_MIXED", oracle.getString("original_scientific_gate"));
        assertEquals(Me111Contract.CHECKPOINT_SHA256,
                     oracle.getString("original_checkpoint_sha256"));
        assertEquals(Me111Contract.MODEL_SHA256, oracle.getString("onnx_model_sha256"));
        assertEquals(256, oracle.getInt("per_owner_steps"));
        assertEquals(512, oracle.getInt("total_steps"));

        Context application = InstrumentationRegistry.getInstrumentation().getTargetContext();
        Me111StateStore store = new Me111StateStore(application);
        store.clear(); // isolated emulator test; never production state.
        int comparisons = 0;
        int preservedStates = 0;
        int processed = 0;
        try (Me111LocalEngine engine = new Me111LocalEngine(application)) {
            JSONArray streams = oracle.getJSONArray("streams");
            assertEquals(2, streams.length());
            for (int streamIndex = 0; streamIndex < streams.length(); streamIndex++) {
                store.clear(); // independent owner starts from fresh source state.
                JSONObject stream = streams.getJSONObject(streamIndex);
                int owner = stream.getInt("owner");
                assertTrue(owner == 0 || owner == 5);
                JSONArray records = stream.getJSONArray("records");
                assertEquals(256, records.length());
                Me111Contract.State current = Me111Contract.initial(owner, 100);
                long lastHour = 99;
                for (int i = 0; i < records.length(); i++) {
                    JSONObject record = records.getJSONObject(i);
                    assertEquals(i, record.getInt("step"));
                    long hour = record.getLong("hour");
                    assertTrue("timestamps must be monotonic", hour > lastHour);
                    Me111Contract.Event event = new Me111Contract.Event(
                        owner, hour,
                        floats(record.getJSONArray("values"), 13),
                        flags(record.getJSONArray("observed")),
                        floats(record.getJSONArray("ages"), 13),
                        floats(record.getJSONArray("clock"), 6));
                    Me111LocalEngine.Result output = engine.predict(event, current);
                    assertEquals("ADVISORY_ONLY", output.actionAuthority);
                    assertEquals("FAILED_MIXED_ME111", output.scientificQualification);
                    if (record.has("reference")) {
                        compareSnapshot(record.getJSONObject("reference"), output);
                        comparisons++;
                    }
                    current = output.nextState;
                    lastHour = hour;
                    processed++;

                    // Independently verify exact FP32, int64 and 8x8 matrix
                    // serialization after 32,64,...,256 events.
                    if ((i + 1) % 32 == 0) {
                        store.save(current);
                        Me111Contract.State reread = store.load();
                        requireLosslessState(current, reread);
                        current = reread; // actually carry restored state forward
                        preservedStates++;
                    }
                }
                assertEquals(stream.getLong("count"), current.count);
                assertEquals(stream.getLong("final_hour"), current.lastHour);
                try {
                    engine.predict(new Me111Contract.Event(
                        owner, current.lastHour, new float[13],
                        new boolean[13], new float[13], new float[6]), current);
                    fail("duplicate chronological event accepted after 256 steps");
                } catch (IllegalArgumentException expected) { }
            }
            assertEquals(512, processed);
            assertEquals(24, comparisons);
            assertEquals(16, preservedStates);
        } finally {
            store.clear();
        }
    }
}
