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
 * First independent instrumentation invocation: create an on-device snapshot.
 * CI force-stops the target package BEFORE running RestoreAndroidTest.
 * Constructed software diagnostics, never actual customer sensor data.
 */
@RunWith(AndroidJUnit4.class)
public final class Me111ProcessSeedAndroidTest {
    @Test public void seedOwnedRecurrentStateThenExit() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        String hardware = Build.HARDWARE.toLowerCase(Locale.ROOT);
        assertTrue("isolated emulator-only restart test, not physical device",
                   hardware.contains("ranchu") || hardware.contains("goldfish"));
        Me111StateStore store = new Me111StateStore(context);
        store.clear();
        context.getSharedPreferences("me111-process-test", Context.MODE_PRIVATE)
               .edit().clear().commit();
        Me111Contract.State initial = Me111Contract.initial(3, 100);
        float[] values = new float[13];
        boolean[] mask = new boolean[13];
        values[0] = 0.0625f;
        mask[0] = true;
        Me111Contract.Event event = new Me111Contract.Event(
            3, 100, values, mask, new float[13], new float[6]);
        Me111Contract.State written;
        try (Me111LocalEngine engine = new Me111LocalEngine(context)) {
            written = engine.predict(event, initial).nextState;
        }
        assertEquals(1L, written.count);
        assertEquals(100L, written.lastHour);
        store.save(written);
        assertEquals(written.count, store.load().count);
        assertTrue(context.getSharedPreferences("me111-process-test",
                       Context.MODE_PRIVATE)
            .edit()
            .putInt("source_pid", Process.myPid())
            .putString("model_sha256", Me111Contract.MODEL_SHA256)
            .commit());
    }
}
