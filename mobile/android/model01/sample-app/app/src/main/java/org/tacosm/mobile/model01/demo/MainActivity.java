package org.tacosm.mobile.model01.demo;

import android.app.Activity;
import android.content.Intent;
import org.tacosm.mobile.me111.Me111Activity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.text.InputType;
import java.util.Locale;
import org.tacosm.mobile.model01.Model01OfflinePredictor;

/** Minimal research demo. No network permission, remote processing or hidden fallback.
 * Original UCI-origin step phase is not equivalent to an arbitrary phone clock.
 * This is NOT a verified predictor for a new household.
 */
public final class MainActivity extends Activity {
    private static final String PREFS = "model01-local-window-v1";
    private EditText indexInput;
    private EditText readingsInput;
    private EditText nextReadingInput;
    private TextView result;
    private Model01OfflinePredictor.WindowState state;

    private TextView label(String value) {
        TextView text = new TextView(this);
        text.setText(value);
        text.setTextSize(15);
        text.setPadding(0, 12, 0, 12);
        return text;
    }

    private static String constructDemoWindow() {
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < 145; i++) {
            if (i > 0) result.append(',');
            result.append(90 + (i % 5));
        }
        return result.toString();
    }

    private static double[] parseReadings(String csv) {
        String[] parts = csv.trim().split(",", -1);
        if (parts.length != 145) {
            throw new IllegalArgumentException("exactly 145 readings required");
        }
        double[] values = new double[145];
        for (int i = 0; i < 145; i++) {
            values[i] = Double.parseDouble(parts[i].trim());
        }
        return values;
    }

    private static String csv(double[] values) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < values.length; i++) {
            if (i > 0) builder.append(',');
            builder.append(values[i]);
        }
        return builder.toString();
    }

    @Override public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(26, 20, 26, 20);
        ScrollView scroll = new ScrollView(this);
        scroll.addView(panel);
        setContentView(scroll);

        Button neural = new Button(this);
        neural.setText("Open frozen 104,663-param TAC neural offline demo");
        panel.addView(neural);
        neural.setOnClickListener((View v) ->
            startActivity(new Intent(this, Me111Activity.class)));
        panel.addView(label("OFFLINE RESEARCH DEMO — original UCI source phase only. "
            + "No independent new-household qualification; predictions are advisory."));
        panel.addView(label("Source-aligned 10-minute step index (not wall-clock time)"));
        indexInput = new EditText(this);
        indexInput.setInputType(InputType.TYPE_CLASS_NUMBER);
        indexInput.setText("500");
        panel.addView(indexInput);

        panel.addView(label("Exactly 145 previously observed Wh readings, oldest first, comma-separated. "
            + "Preloaded example is CONSTRUCTED, not real energy evidence."));
        readingsInput = new EditText(this);
        readingsInput.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE);
        readingsInput.setMinLines(4);
        readingsInput.setText(constructDemoWindow());
        panel.addView(readingsInput);

        Button predict = new Button(this);
        predict.setText("Predict entirely offline");
        panel.addView(predict);
        predict.setOnClickListener((View unused) -> {
            try {
                long index = Long.parseLong(indexInput.getText().toString().trim());
                state = new Model01OfflinePredictor.WindowState(index,
                    parseReadings(readingsInput.getText().toString()));
                saveState();
                show(state.predict());
            } catch (RuntimeException invalid) {
                result.setText("ABSTAIN LOCALLY: invalid index/readings. No fallback.");
            }
        });

        panel.addView(label("Next actual contiguous 10-minute Wh reading (demo state update)"));
        nextReadingInput = new EditText(this);
        nextReadingInput.setInputType(InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_DECIMAL);
        panel.addView(nextReadingInput);
        Button append = new Button(this);
        append.setText("Append next contiguous observation locally");
        panel.addView(append);
        append.setOnClickListener((View unused) -> {
            try {
                if (state == null) throw new IllegalArgumentException("no local state");
                double observedWh = Double.parseDouble(nextReadingInput.getText().toString().trim());
                state.append(state.sourceAlignedIndex() + 1, observedWh);
                indexInput.setText(Long.toString(state.sourceAlignedIndex()));
                readingsInput.setText(csv(state.snapshot()));
                saveState();
                show(state.predict());
            } catch (RuntimeException invalid) {
                result.setText("ABSTAIN LOCALLY: missing/invalid state or reading. No fallback.");
            }
        });
        result = label("NOT QUALIFIED; NO DEVICE MEASUREMENTS YET");
        panel.addView(result);
        restoreState();
    }

    private void saveState() {
        if (state == null) return;
        boolean saved = getSharedPreferences(PREFS, MODE_PRIVATE).edit()
            .putLong("index", state.sourceAlignedIndex())
            .putString("readings", csv(state.snapshot()))
            .commit(); // synchronous local persistence before acknowledging state update
        if (!saved) throw new IllegalStateException("device-local state persistence failed");
    }

    private void restoreState() {
        String data = getSharedPreferences(PREFS, MODE_PRIVATE).getString("readings", null);
        if (data == null) return;
        try {
            long index = getSharedPreferences(PREFS, MODE_PRIVATE).getLong("index", -1);
            Model01OfflinePredictor.WindowState restored =
                new Model01OfflinePredictor.WindowState(index, parseReadings(data));
            state = restored;
            indexInput.setText(Long.toString(index));
            readingsInput.setText(csv(restored.snapshot()));
            result.setText("Local observations restored. Only source-phase-aligned demo inputs supported.");
        } catch (RuntimeException invalid) {
            state = null;
            result.setText("ABSTAIN: corrupt local state; enter 145 valid observations. No fallback.");
        }
    }

    private void show(Model01OfflinePredictor.Prediction prediction) {
        result.setText(String.format(Locale.US,
            "LOCAL FORECAST: %.3f Wh in %d min\nHistorical CAL interval: [%.3f, %.3f] Wh\n"
            + "NOT QUALIFIED; interval not guaranteed; advisory only. "
            + "Android latency, RAM, energy and airplane-mode verification NOT RUN.",
            prediction.predictionWh, prediction.horizonMinutes,
            prediction.interval90LowerWh, prediction.interval90UpperWh));
    }
}
