package org.tacosm.mobile.me111;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Debug;
import android.os.SystemClock;
import android.provider.Settings;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.Arrays;
import java.util.Locale;

/**
 * Explicitly requested, entirely on-phone CPU timing diagnostic.
 *
 * No real sensor records, no cloud, no ADB/laptop required, no file or
 * customer-state mutations. Refuses Android emulators and airplane-mode-off.
 * Does not establish real-world accuracy, independently verified radio
 * isolation, cold process startup, energy, or any scientific promotion.
 */
public final class Me111DeviceSelfTest {
    private static final int WARMUP = 20;
    private static final int MEASURED = 150;

    private Me111DeviceSelfTest() { }

    private static boolean isEmulator() {
        String fingerprint = Build.FINGERPRINT.toLowerCase(Locale.ROOT);
        String model = Build.MODEL.toLowerCase(Locale.ROOT);
        String hardware = Build.HARDWARE.toLowerCase(Locale.ROOT);
        return fingerprint.contains("generic") || fingerprint.contains("emulator")
            || fingerprint.contains("sdk") || model.contains("emulator")
            || model.contains("android sdk") || hardware.contains("ranchu")
            || hardware.contains("goldfish");
    }

    private static long totalPssBytes() {
        Debug.MemoryInfo info = new Debug.MemoryInfo();
        Debug.getMemoryInfo(info);
        return ((long) info.getTotalPss()) * 1024L;
    }

    private static double milliseconds(long nanoseconds) {
        return nanoseconds / 1_000_000.0;
    }

    private static double quantileMs(long[] sortedNanos, int percent) {
        int index = (int) Math.ceil(percent * sortedNanos.length / 100.0) - 1;
        return milliseconds(sortedNanos[Math.max(0, index)]);
    }

    /** Run only from an explicit UI tap, on a worker thread. */
    public static String run(Context context) throws Exception {
        if (isEmulator())
            throw new IllegalStateException("NOT_RUN: physical Android required, emulator refused");
        if (Settings.Global.getInt(context.getContentResolver(),
                Settings.Global.AIRPLANE_MODE_ON, 0) != 1)
            throw new IllegalStateException("NOT_RUN: enable Airplane Mode in phone Settings");
        if (context.checkSelfPermission(Manifest.permission.INTERNET)
                != PackageManager.PERMISSION_DENIED)
            throw new SecurityException("offline permission boundary violated");

        // Scratch-only state: never read, overwrite, reset or export any user
        // persisted source history. This diagnostic measures CPU portability.
        long pssBefore = totalPssBytes();
        long loadingStarted = SystemClock.elapsedRealtimeNanos();
        long loadNs;
        long[] samples = new long[MEASURED];
        long peakPss = pssBefore;
        Me111Contract.State state = Me111Contract.initial(0, 100);

        try (Me111LocalEngine engine = new Me111LocalEngine(context)) {
            loadNs = SystemClock.elapsedRealtimeNanos() - loadingStarted;
            for (int step = 0; step < WARMUP + MEASURED; step++) {
                float[] values = new float[Me111Contract.CHANNELS];
                boolean[] observed = new boolean[Me111Contract.CHANNELS];
                observed[0] = true;
                values[0] = (float) (0.05 + (step % 7) * 0.01);
                Me111Contract.Event e = new Me111Contract.Event(
                    0, 100 + step, values, observed, new float[13], new float[6]);
                long started = SystemClock.elapsedRealtimeNanos();
                Me111LocalEngine.Result prediction = engine.predict(e, state);
                long elapsedNs = SystemClock.elapsedRealtimeNanos() - started;
                state = prediction.nextState;
                if (step >= WARMUP) samples[step - WARMUP] = elapsedNs;
                if ((step & 7) == 0)
                    peakPss = Math.max(peakPss, totalPssBytes());
            }
            peakPss = Math.max(peakPss, totalPssBytes());
        }
        if (state.count != WARMUP + MEASURED || state.lastHour != 269)
            throw new IllegalStateException("constructed benchmark state not coherent");
        Arrays.sort(samples);

        JSONObject receipt = new JSONObject();
        receipt.put("status", "LOCAL_PHYSICAL_CPU_SELF_TEST_MEASURED_NOT_RELEASE_QUALIFIED");
        receipt.put("original_checkpoint_sha256", Me111Contract.CHECKPOINT_SHA256);
        receipt.put("onnx_sha256", Me111Contract.MODEL_SHA256);
        receipt.put("device_manufacturer", Build.MANUFACTURER);
        receipt.put("device_model", Build.MODEL);
        receipt.put("device_hardware", Build.HARDWARE);
        receipt.put("android_sdk", Build.VERSION.SDK_INT);
        receipt.put("supported_abis", new JSONArray(Build.SUPPORTED_ABIS));
        receipt.put("airplane_mode_setting", true);
        receipt.put("radios_independently_verified_disabled", false);
        receipt.put("network_trace", "NOT_CAPTURED");
        receipt.put("source_inputs", "CONSTRUCTED_DIAGNOSTIC_ONLY");
        receipt.put("customer_state_read_or_modified", false);
        receipt.put("me111_scientific_gate", "FAILED_MIXED_TRADEOFF");
        receipt.put("warmup_events", WARMUP);
        receipt.put("timed_events", MEASURED);
        receipt.put("session_init_ms", milliseconds(loadNs));
        receipt.put("session_init_is_cold_process", false);
        receipt.put("inference_p50_ms", quantileMs(samples, 50));
        receipt.put("inference_p95_ms", quantileMs(samples, 95));
        receipt.put("inference_max_ms", milliseconds(samples[MEASURED - 1]));
        receipt.put("app_pss_before_bytes", pssBefore);
        receipt.put("app_pss_peak_sampled_bytes", peakPss);
        receipt.put("energy_joules", JSONObject.NULL);
        receipt.put("energy_status", "NOT_MEASURED");
        receipt.put("device_restart_state_recovery", "NOT_MEASURED");
        receipt.put("independent_sensor_quality", "NOT_QUALIFIED");
        return receipt.toString(2);
    }
}
