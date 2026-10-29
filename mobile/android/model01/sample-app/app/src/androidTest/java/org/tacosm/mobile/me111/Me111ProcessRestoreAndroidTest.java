package org.tacosm.mobile.me111;

import android.content.Context;
import android.os.Build;
import android.os.Process;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.Locale;

import static org.junit.Assert.*;

/**
 * Second independent emulator instrumentation invocation. CI must force-stop
 * the app between seed and restore. A same-process run is explicitly rejected.
 */
@RunWith(AndroidJUnit4.class)
public final class Me111ProcessRestoreAndroidTest {
    @Test public void restartRestoresAndContinuesFrozenNativeMemory() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        String hardware = Build.HARDWARE.toLowerCase(Locale.ROOT);
        assertTrue("isolated emulator-only test, not device qualification",
                   hardware.contains("ranchu") || hardware.contains("goldfish"));
        android.content.SharedPreferences meta = context.getSharedPreferences(
            "me111-process-test", Context.MODE_PRIVATE);
        int firstPid = meta.getInt("source_pid", -1);
        assertTrue("missing original process receipt", firstPid > 0);
        assertNotEquals("must be a new process after a force-stop",
                        firstPid, Process.myPid());
        assertEquals(Me111Contract.MODEL_SHA256,
                     meta.getString("model_sha256", ""));

        Me111StateStore store = new Me111StateStore(context);
        try {
            Me111Contract.State loaded = store.load();
            assertNotNull("state vanished across actual process restart", loaded);
            assertEquals(3, loaded.owner);
            assertEquals(100L, loaded.lastHour);
            assertEquals(1L, loaded.count);
            float[] values = new float[13];
            boolean[] observed = new boolean[13];
            values[0] = 0.125f;
            observed[0] = true;
            Me111Contract.Event second = new Me111Contract.Event(
                3, 101, values, observed, new float[13], new float[6]);
            Me111Contract.State continued;
            try (Me111LocalEngine engine = new Me111LocalEngine(context)) {
                continued = engine.predict(second, loaded).nextState;
            }
            assertEquals(3, continued.owner);
            assertEquals(101L, continued.lastHour);
            assertEquals(2L, continued.count);
            store.save(continued);
            Me111Contract.State verified = store.load();
            assertEquals(2L, verified.count);
            assertEquals(101L, verified.lastHour);
            for (int i = 0; i < 8; i++)
                assertArrayEquals(continued.memory[i], verified.memory[i], 0f);
        } finally {
            store.clear();
            meta.edit().clear().commit();
        }
    }
}
