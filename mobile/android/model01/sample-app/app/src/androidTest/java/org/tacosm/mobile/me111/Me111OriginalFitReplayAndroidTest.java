package org.tacosm.mobile.me111;

import android.content.Context;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

import static org.junit.Assert.*;

/**
 * Actual Android ONNX CPU native inference using ORIGINAL published 2013
 * FIT sensor rows and exact original PyTorch frozen-checkpoint oracle.
 *
 * This validates offline raw-to-forecast software and state algebra only:
 * NOT 2015 REPORT / protected TEST, authentic predictive quality, physical
 * mobile benchmark, health-risk accuracy, or release authorization.
 */
@RunWith(AndroidJUnit4.class)
public final class Me111OriginalFitReplayAndroidTest {
    private static JSONObject reference(Context test) throws Exception {
        try (InputStream input = test.getAssets().open("me111_original_fit_expected.json");
             ByteArrayOutputStream bytes = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192];
            int n;
            while ((n=input.read(buffer)) != -1) bytes.write(buffer, 0, n);
            return new JSONObject(new String(bytes.toByteArray(), StandardCharsets.UTF_8));
        }
    }

    private static void close(JSONArray ref, float[] actual, String field) throws Exception {
        assertEquals(field+" length",ref.length(),actual.length);
        for(int i=0;i<actual.length;i++) {
            double expected=ref.getDouble(i);
            assertTrue(field+":"+i+" original="+expected+" local="+actual[i],
                Float.isFinite(actual[i]) &&
                Math.abs(expected-actual[i])<= 3e-4 + 2e-4*Math.abs(expected));
        }
    }

    @Test public void testRealFitSourceRawToNativeForecastAndState() throws Exception {
        Context test=InstrumentationRegistry.getInstrumentation().getContext();
        Context app=InstrumentationRegistry.getInstrumentation().getTargetContext();
        JSONObject ref=reference(test);
        assertEquals("ORIGINAL_AUTHENTIC_FIT_INPUT_256_EVENT_PYTORCH_SOFTWARE_ORACLE",
                     ref.getString("status"));
        assertEquals(Me111Contract.CHECKPOINT_SHA256,ref.getString("checkpoint_sha256"));
        assertEquals(Me111Contract.MODEL_SHA256,ref.getString("onnx_sha256"));
        assertEquals(Me111FrozenFitStats.PUBLISHER_ARCHIVE_SHA256,
                     ref.getString("original_source_sha256"));
        assertEquals("FAILED_MIXED_TRADEOFF",ref.getString("original_scientific_gate"));
        assertEquals(256,ref.getInt("steps"));
        Me111StateStore persisted=new Me111StateStore(app);
        persisted.clear();
        try (InputStream csv=test.getAssets().open("me111-original-fit-first256.csv");
             Me111LocalEngine engine=new Me111LocalEngine(app)) {
            Me111OriginalFitReplay.Report report=Me111OriginalFitReplay.run(csv,engine);
            assertEquals(0,report.owner);
            assertEquals(256,report.events);
            assertEquals(255L,report.finalHour);
            assertEquals(ref.getLong("count"),report.observedCount);
            close(ref.getJSONArray("log_location"),report.logLocation,"forecast");
            close(ref.getJSONArray("left_scale"),report.leftScale,"left_scale");
            close(ref.getJSONArray("right_scale"),report.rightScale,"right_scale");
            close(ref.getJSONArray("source_sum"),report.finalState.sourceSum,"source_sum");
            close(ref.getJSONArray("source_mass"),report.finalState.sourceMass,"source_mass");
            close(ref.getJSONArray("evidence_sum"),report.finalState.evidenceSum,"evidence");
            JSONArray memories=ref.getJSONArray("memory");
            for(int i=0;i<8;i++)
                close(memories.getJSONArray(i),report.finalState.memory[i],"memory["+i+"]");
            assertNull("scratch source replay must never write customer state",
                       persisted.load());
        } finally {
            persisted.clear();
        }
    }

    @Test public void testTruncatedRawArchiveFailsWithoutPersistedState() throws Exception {
        Context app=InstrumentationRegistry.getInstrumentation().getTargetContext();
        Me111StateStore persisted=new Me111StateStore(app);
        persisted.clear();
        String header="year,month,day,hour,station,wd,PM2.5,PM10,SO2,NO2,CO,O3,TEMP,PRES,DEWP,RAIN,WSPM\n";
        String first="2013,3,1,0,Aotizhongxin,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA\n";
        try (InputStream csv=new ByteArrayInputStream((header+first)
                  .getBytes(StandardCharsets.UTF_8));
             Me111LocalEngine engine=new Me111LocalEngine(app)) {
            try {
                Me111OriginalFitReplay.run(csv,engine);
                fail("truncated source was incorrectly accepted");
            } catch (IOException expected) {
                assertTrue(expected.getMessage().contains("FIT row"));
            }
            assertNull("incomplete source replay silently committed memory",persisted.load());
        } finally {
            persisted.clear();
        }
    }
}
