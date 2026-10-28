package org.tacosm.mobile.me111;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Debug;
import android.os.SystemClock;
import android.provider.Settings;
import android.util.Log;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Locale;

import static org.junit.Assert.*;

/**
 * HARDWARE-ONLY local CPU benchmark. Must never run on CI emulator and must not
 * be called a real-world scientific forecast evaluation. Constructed inputs.
 *
 * Launch only on a named, physically controlled phone with airplane mode ON.
 * MODEL ENERGY requires an external instrument; battery-level proxy not accepted.
 */
@RunWith(AndroidJUnit4.class)
public final class Me111PhysicalBenchmarkAndroidTest {
    private static final String RECEIPT = "me111-physical-cpu-benchmark.json";
    private static final int WARMUP = 20;
    private static final int SAMPLES = 150;

    private static boolean emulator() {
        String fingerprint = Build.FINGERPRINT.toLowerCase(Locale.ROOT);
        String model = Build.MODEL.toLowerCase(Locale.ROOT);
        String product = Build.PRODUCT.toLowerCase(Locale.ROOT);
        return fingerprint.contains("generic") || fingerprint.contains("emulator")
            || fingerprint.contains("sdk") || model.contains("emulator")
            || model.contains("android sdk") || product.contains("sdk")
            || product.contains("emulator") || Build.HARDWARE.toLowerCase(Locale.ROOT).contains("ranchu");
    }

    private static float[] sourceValues(int i) {
        float[] values = new float[13];
        values[0] = 0.05f + (i % 7) / 100f;
        return values;
    }

    private static Me111Contract.Event event(long hour, int i) {
        boolean[] mask = new boolean[13];
        mask[0] = true;
        return new Me111Contract.Event(0, hour, sourceValues(i), mask,
            new float[13], new float[6]);
    }

    private static double ms(long nanoseconds) {
        return nanoseconds / 1_000_000.0;
    }

    private static double percentileMs(long[] sorted, int numerator) {
        int rank = (int)Math.ceil((numerator / 100.0) * sorted.length) - 1;
        return ms(sorted[Math.max(0, rank)]);
    }

    private static long privatePssBytes() {
        Debug.MemoryInfo info = new Debug.MemoryInfo();
        Debug.getMemoryInfo(info);
        return (long)info.getTotalPss() * 1024L;
    }

    @Test public void benchmarkOnNamedOfflinePhysicalAndroidOnly() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        assertFalse("PHYSICAL_DEVICE_REQUIRED; emulator never qualifies hardware", emulator());
        int airplane = Settings.Global.getInt(context.getContentResolver(),
            Settings.Global.AIRPLANE_MODE_ON, 0);
        assertEquals("AIRPLANE_MODE_REQUIRED; enable it in device Settings first", 1, airplane);
        assertEquals("APK must have zero INTERNET permission",
            PackageManager.PERMISSION_DENIED,
            context.checkSelfPermission(Manifest.permission.INTERNET));
        long pssBefore = privatePssBytes();
        long modelLoadStarted = SystemClock.elapsedRealtimeNanos();
        try (Me111LocalEngine engine = new Me111LocalEngine(context)) {
            long loadDuration = SystemClock.elapsedRealtimeNanos() - modelLoadStarted;
            Me111Contract.State state = Me111Contract.initial(0, 100);
            long hour = 100;
            long[] latencies = new long[SAMPLES];
            long peakPss = pssBefore;
            for (int i = 0; i < WARMUP + SAMPLES; i++, hour++) {
                Me111Contract.Event observation = event(hour, i);
                long started = SystemClock.elapsedRealtimeNanos();
                Me111LocalEngine.Result prediction = engine.predict(observation, state);
                long elapsed = SystemClock.elapsedRealtimeNanos() - started;
                state = prediction.nextState;
                if (i >= WARMUP) latencies[i - WARMUP] = elapsed;
                if ((i & 7) == 0) peakPss = Math.max(peakPss, privatePssBytes());
            }
            peakPss = Math.max(peakPss, privatePssBytes());
            Arrays.sort(latencies);
            assertEquals(WARMUP + SAMPLES, state.count);
            assertEquals(hour - 1, state.lastHour);
            Me111StateStore store = new Me111StateStore(context);
            // Do not reset production state; private benchmark should use a
            // dedicated test installation/account with no real customer data.
            Me111Contract.State prior = store.load();
            assertNull("benchmark demands fresh isolated test-app state", prior);
            store.save(state);
            Me111Contract.State restored = store.load();
            assertNotNull(restored);
            assertEquals(state.count, restored.count);
            assertEquals(state.lastHour, restored.lastHour);
            for (int j = 0; j < 8; j++)
                assertArrayEquals(state.memory[j], restored.memory[j], 0f);
            // Dedicated test-only state: do not leave constructed data behind.
            store.clear();

            JSONObject result = new JSONObject();
            result.put("status", "PHYSICAL_ANDROID_LOCAL_CPU_MEASURED");
            result.put("physical_device", true);
            result.put("os_sdk", Build.VERSION.SDK_INT);
            result.put("manufacturer", Build.MANUFACTURER);
            result.put("model", Build.MODEL);
            result.put("hardware", Build.HARDWARE);
            result.put("abis", new org.json.JSONArray(Build.SUPPORTED_ABIS));
            result.put("apk_permission_internet", false);
            result.put("airplane_mode_setting", true);
            result.put("wifi_and_bluetooth_radio_state", "NOT_INDEPENDENTLY_VERIFIED");
            result.put("model_sha256", Me111Contract.MODEL_SHA256);
            result.put("original_checkpoint_sha256", Me111Contract.CHECKPOINT_SHA256);
            result.put("warmup_steps", WARMUP);
            result.put("sample_steps", SAMPLES);
            result.put("model_load_ms", ms(loadDuration));
            result.put("warm_inference_p50_ms", percentileMs(latencies, 50));
            result.put("warm_inference_p95_ms", percentileMs(latencies, 95));
            result.put("warm_inference_max_ms", ms(latencies[latencies.length - 1]));
            result.put("pss_before_bytes", pssBefore);
            result.put("peak_total_pss_bytes", peakPss);
            result.put("peak_total_pss_delta_bytes", Math.max(0, peakPss - pssBefore));
            result.put("state_restored", true);
            result.put("source_data", "CONSTRUCTED_SOFTWARE_INPUTS_ONLY");
            result.put("scientific_forecasting_qualification", "FAILED_MIXED_ME111");
            result.put("energy_joules", JSONObject.NULL);
            result.put("energy_status", "NOT_MEASURED_EXTERNAL_INSTRUMENT_REQUIRED");
            result.put("network_trace", "NOT_CAPTURED_REQUIRE_EXTERNAL_INDEPENDENT_AUDIT");
            result.put("radio_isolation", "AIRPLANE_MODE_SETTING_ONLY");
            result.put("app_relaunch_and_process_restart", "NOT_MEASURED_IN_THIS_TEST");
            result.put("device_thermal_battery_conditions", "NOT_RECORDED");
            String serialized = result.toString(2) + "\n";
            try (FileOutputStream output =
                    context.openFileOutput(RECEIPT, Context.MODE_PRIVATE)) {
                output.write(serialized.getBytes(StandardCharsets.UTF_8));
                output.getFD().sync();
            }
            Log.i("ME111_PHYSICAL_BENCHMARK_UNQUALIFIED",
                  serialized.replace("\n", " "));
        }
    }
}
