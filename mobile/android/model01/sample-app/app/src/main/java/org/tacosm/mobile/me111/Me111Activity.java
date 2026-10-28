package org.tacosm.mobile.me111;

import android.app.Activity;
import android.app.AlertDialog;
import android.os.Bundle;
import android.view.View;
import android.view.Gravity;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.text.InputType;

import java.io.IOException;
import java.util.Locale;

/** Local source-aligned ME111 engineering demonstrator. No source authorization
 * or independent outcome validation. Prefilled zero input is CONSTRUCTED only.
 * No device actions, cloud endpoint, hidden fallback, or automatic retraining.
 */
public final class Me111Activity extends Activity {
    private Me111StateStore store;
    private Me111Contract.State state;
    private boolean stateCorrupt;
    private EditText owner, hour, values, mask, ages, clock;
    private TextView result;

    private TextView text(String title) {
        TextView label = new TextView(this);
        label.setText(title);
        label.setTextSize(15);
        label.setPadding(8, 14, 8, 14);
        return label;
    }

    private EditText field(LinearLayout panel, String name, String defaults) {
        panel.addView(text(name));
        EditText field = new EditText(this);
        field.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE);
        field.setText(defaults);
        field.setMinLines(1);
        panel.addView(field);
        return field;
    }

    private static String repeated(int n, String token) {
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < n; i++) {
            if (i != 0) out.append(',');
            out.append(token);
        }
        return out.toString();
    }

    @Override public void onCreate(Bundle stateBundle) {
        super.onCreate(stateBundle);
        store = new Me111StateStore(this);
        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setGravity(Gravity.CENTER_HORIZONTAL);
        panel.setPadding(24, 20, 24, 20);
        ScrollView scroll = new ScrollView(this);
        scroll.addView(panel);
        setContentView(scroll);
        panel.addView(text("TAC-OSM 104,663-PARAMETER LOCAL CPU INFERENCE\n"
            + "RESEARCH/ENGINEERING DEMO ONLY: ME111 original coverage gate FAILED. "
            + "The prefilled inputs are CONSTRUCTED diagnostics, not observed air quality. "
            + "Do not treat this as a calibrated health or physical-action prediction. "
            + "Provide source-authorized, timestamp-aligned, normalized 13-channel features."));
        owner = field(panel, "Original source station ID (0–11)", "0");
        hour = field(panel, "Monotone source-naive hour (not phone wall-clock)", "100");
        values = field(panel, "13 standardized feature values (masked=zero)", repeated(13, "0"));
        mask = field(panel, "13 observed flags: 0 or 1", repeated(13, "0"));
        ages = field(panel, "13 normalized source ages [0,1]", repeated(13, "0"));
        clock = field(panel, "6 original-source clock features", repeated(6, "0"));

        Button run = new Button(this);
        run.setText("Run frozen TAC model offline and save local state");
        panel.addView(run);
        run.setOnClickListener((View v) -> runOnce());

        Button reset = new Button(this);
        reset.setText("Reset LOCAL recurrent state (explicit)");
        panel.addView(reset);
        reset.setOnClickListener((View v) -> new AlertDialog.Builder(this)
            .setTitle("Delete device-local ME111 state?")
            .setMessage("This cannot restore previous state or infer missing observations.")
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Delete local state", (dialog, choice) -> {
                store.clear();
                state = null;
                stateCorrupt = false;
                result.setText("Local state explicitly reset; next source-aligned event can initialize.");
            }).show());

        result = text("MODEL NOT SCIENTIFICALLY QUALIFIED. "
            + "Physical Android p50/p95, peak RAM, energy and airplane-mode: NOT RUN.");
        panel.addView(result);
        try {
            state = store.load();
            if (state != null) {
                owner.setText(Integer.toString(state.owner));
                hour.setText(Long.toString(state.lastHour + 1));
                result.setText("Recovered private local state; source-aligned next hour required. "
                    + "Model remains scientifically unqualified.");
            }
        } catch (IOException error) {
            stateCorrupt = true;
            result.setText("ABSTAIN: corrupt local recurrent state. Explicitly reset; "
                + "never infer from unknown stored memory.");
        }
    }

    private static float[] floats(EditText field, int count) {
        String[] cells = field.getText().toString().trim().split(",", -1);
        if (cells.length != count)
            throw new IllegalArgumentException("incorrect input channel count");
        float[] data = new float[count];
        for (int i = 0; i < count; i++) data[i] = Float.parseFloat(cells[i].trim());
        return data;
    }

    private static boolean[] flags(EditText field) {
        String[] cells = field.getText().toString().trim().split(",", -1);
        if (cells.length != 13)
            throw new IllegalArgumentException("13 observation flags required");
        boolean[] result = new boolean[13];
        for (int i = 0; i < 13; i++) {
            String flag = cells[i].trim();
            if (!flag.equals("0") && !flag.equals("1"))
                throw new IllegalArgumentException("nonbinary observation flag");
            result[i] = flag.equals("1");
        }
        return result;
    }

    private void runOnce() {
        if (stateCorrupt) {
            result.setText("ABSTAIN LOCALLY: state invalid; explicit reset required.");
            return;
        }
        try {
            int stationId = Integer.parseInt(owner.getText().toString().trim());
            long sourceHour = Long.parseLong(hour.getText().toString().trim());
            Me111Contract.Event input = new Me111Contract.Event(stationId, sourceHour,
                floats(values, 13), flags(mask), floats(ages, 13), floats(clock, 6));
            Me111Contract.State previous = state == null
                ? Me111Contract.initial(stationId, sourceHour) : state;
            Me111Contract.validateStep(input, previous);
            // No internet/SDK remote endpoint. Full ONNX CPU graph and state are local.
            Me111LocalEngine.Result predicted;
            try (Me111LocalEngine engine = new Me111LocalEngine(getApplicationContext())) {
                predicted = engine.predict(input, previous);
            }
            // Fail closed: advance RAM state only AFTER atomic private-disk commit.
            store.save(predicted.nextState);
            state = predicted.nextState;
            hour.setText(Long.toString(sourceHour + 1));
            StringBuilder output = new StringBuilder();
            output.append("LOCAL STATE STORED; no cloud.\n");
            for (int i = 0; i < 6; i++) {
                output.append(String.format(Locale.US,
                    "channel %d standardized-log median %.6f; L %.6f / R %.6f\n",
                    i, predicted.logLocation[i],
                    predicted.leftScale[i], predicted.rightScale[i]));
            }
            output.append("Source hour ").append(state.lastHour).append("; observation count ")
                .append(state.count).append(".\n");
            output.append("SCIENTIFIC GATE: FAILED/MIXED. ADVISORY ONLY. "
                + "No independent or physical-device qualification.");
            result.setText(output.toString());
        } catch (Exception error) {
            // Never silently reset corrupt state, interpolate a gap or contact a server.
            result.setText("ABSTAIN LOCALLY: " + error.getClass().getSimpleName()
                + ". No fallback; observation/state not advanced.");
        }
    }
}
