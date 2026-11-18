package org.tacosm.mobile.me111;

import android.content.Context;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.IOException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;

/**
 * Frozen ME111 local CPU ONNX inference; no network I/O or cloud fallback.
 * This is an engineering prototype; original real-source scientific gate FAIL.
 *
 * B=1,T=1 ONLY. All Python guards must be repeated here. Input tensors are
 * original standardized features: this class does NOT invent source mapping.
 * Consumer must validate as-of provenance and time cadence separately.
 */
public final class Me111LocalEngine implements AutoCloseable {
    public static final String ASSET_NAME = "tac_me111.onnx";
    private static final Set<String> EXPECTED_INPUTS = new HashSet<>(Arrays.asList(
        "values", "observed", "ages", "clock", "owners", "hours",
        "source_sum", "source_mass", "evidence_sum", "count", "memory", "last_hour"
    ));
    private static final Set<String> EXPECTED_OUTPUTS = new HashSet<>(Arrays.asList(
        "log_location", "left_scale", "right_scale",
        "next_source_sum", "next_source_mass", "next_evidence_sum",
        "next_count", "next_memory", "next_last_hour", "chronology_ok"
    ));

    private final OrtEnvironment environment;
    private final OrtSession session;

    public static final class Result {
        /** Standardized log1p units, NOT raw pollutant quantities. */
        public final float[] logLocation, leftScale, rightScale;
        public final Me111Contract.State nextState;
        public final String scientificQualification = "FAILED_MIXED_ME111";
        public final String actionAuthority = "ADVISORY_ONLY";

        Result(float[] logLocation, float[] leftScale, float[] rightScale,
               Me111Contract.State nextState) {
            Me111Contract.validatePrediction(logLocation, leftScale, rightScale);
            this.logLocation = logLocation.clone();
            this.leftScale = leftScale.clone();
            this.rightScale = rightScale.clone();
            this.nextState = nextState;
        }
    }

    public Me111LocalEngine(Context context) throws Exception {
        byte[] model = readAsset(context, ASSET_NAME);
        String actualSha = hexSha256(model);
        if (!Me111Contract.MODEL_SHA256.equals(actualSha))
            throw new SecurityException("frozen ONNX graph checksum mismatch");
        environment = OrtEnvironment.getEnvironment();
        OrtSession created;
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setIntraOpNumThreads(1);
            options.setInterOpNumThreads(1);
            // No NNAPI, GPU, remote inference or external execution provider.
            created = environment.createSession(model, options);
        }
        if (!EXPECTED_INPUTS.equals(created.getInputNames())
                || !EXPECTED_OUTPUTS.equals(created.getOutputNames())) {
            created.close();
            throw new IllegalStateException("frozen ONNX input/output schema changed");
        }
        session = created;
    }

    public synchronized Result predict(Me111Contract.Event event, Me111Contract.State state)
            throws OrtException {
        Me111Contract.validateStep(event, state);
        Map<String, OnnxTensor> inputs = new LinkedHashMap<>();
        try {
            inputs.put("values", OnnxTensor.createTensor(environment,
                    new float[][][] {{event.values.clone()}}));
            inputs.put("observed", OnnxTensor.createTensor(environment,
                    new boolean[][][] {{event.observed.clone()}}));
            inputs.put("ages", OnnxTensor.createTensor(environment,
                    new float[][][] {{event.ages.clone()}}));
            inputs.put("clock", OnnxTensor.createTensor(environment,
                    new float[][][] {{event.clock.clone()}}));
            inputs.put("owners", OnnxTensor.createTensor(environment, new long[] {event.owner}));
            inputs.put("hours", OnnxTensor.createTensor(environment, new long[][] {{event.hour}}));
            inputs.put("source_sum", OnnxTensor.createTensor(environment,
                    new float[][] {state.sourceSum.clone()}));
            inputs.put("source_mass", OnnxTensor.createTensor(environment,
                    new float[][] {state.sourceMass.clone()}));
            inputs.put("evidence_sum", OnnxTensor.createTensor(environment,
                    new float[][] {state.evidenceSum.clone()}));
            inputs.put("count", OnnxTensor.createTensor(environment,
                    new long[][] {{state.count}}));
            inputs.put("memory", OnnxTensor.createTensor(environment,
                    new float[][][] {cloneMemory(state.memory)}));
            inputs.put("last_hour", OnnxTensor.createTensor(environment,
                    new long[] {state.lastHour}));
            try (OrtSession.Result output = session.run(inputs)) {
                boolean[] chronological = (boolean[]) value(output, "chronology_ok");
                if (chronological.length != 1 || !chronological[0])
                    throw new IllegalArgumentException("ONNX chronology check rejected input");
                float[] loc = singleton((float[][])value(output, "log_location"), 6);
                float[] left = singleton((float[][])value(output, "left_scale"), 6);
                float[] right = singleton((float[][])value(output, "right_scale"), 6);
                float[] sourceSum = singleton((float[][])value(output, "next_source_sum"), 13);
                float[] sourceMass = singleton((float[][])value(output, "next_source_mass"), 13);
                float[] evidenceSum = singleton((float[][])value(output, "next_evidence_sum"), 62);
                float[][][] memory = (float[][][])value(output, "next_memory");
                long[][] nextCount = (long[][])value(output, "next_count");
                long[] nextHour = (long[])value(output, "next_last_hour");
                if (memory.length != 1 || nextCount.length != 1
                        || nextCount[0].length != 1 || nextHour.length != 1
                        || nextHour[0] != event.hour)
                    throw new IllegalStateException("invalid ONNX state shapes/chronology");
                long expected = state.count + (anyObserved(event.observed) ? 1L : 0L);
                if (nextCount[0][0] != expected)
                    throw new IllegalStateException("unexpected ONNX event count");
                Me111Contract.State next = new Me111Contract.State(
                    event.owner, nextHour[0], nextCount[0][0],
                    sourceSum, sourceMass, evidenceSum, memory[0]);
                return new Result(loc, left, right, next);
            }
        } finally {
            for (OnnxTensor tensor : inputs.values()) {
                if (tensor != null) tensor.close();
            }
        }
    }

    private static boolean anyObserved(boolean[] mask) {
        for (boolean bit : mask) if (bit) return true;
        return false;
    }

    private static Object value(OrtSession.Result output, String name) throws OrtException {
        OnnxValue result = output.get(name)
                .orElseThrow(() -> new IllegalStateException("missing ONNX output " + name));
        return result.getValue();
    }

    private static float[] singleton(float[][] values, int expected) {
        if (values.length != 1 || values[0].length != expected)
            throw new IllegalStateException("invalid frozen output shape " + expected);
        return values[0].clone();
    }

    private static float[][] cloneMemory(float[][] rows) {
        float[][] out = new float[rows.length][];
        for (int i = 0; i < rows.length; i++) out[i] = rows[i].clone();
        return out;
    }

    private static byte[] readAsset(Context context, String name) throws IOException {
        try (InputStream in = context.getAssets().open(name);
             ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192];
            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
                if (out.size() > 4_000_000)
                    throw new IOException("unexpected frozen ONNX model size");
            }
            return out.toByteArray();
        }
    }

    private static String hexSha256(byte[] bytes) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(bytes);
            StringBuilder text = new StringBuilder(64);
            for (byte b : digest) {
                text.append(Character.forDigit((b >>> 4) & 15, 16));
                text.append(Character.forDigit(b & 15, 16));
            }
            return text.toString();
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 unavailable", impossible);
        }
    }

    @Override public synchronized void close() throws OrtException {
        session.close();
        // OrtEnvironment is a process-shared singleton; do not shut it down here.
    }
}
