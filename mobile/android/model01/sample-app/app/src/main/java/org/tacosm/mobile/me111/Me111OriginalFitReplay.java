package org.tacosm.mobile.me111;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Fully offline scratch replay of the FIRST 256 hourly rows from a locally
 * selected UCI PRSA station CSV. Exact publisher schema, complete history
 * from source hour zero, and FIT-only date range are mandatory.
 *
 * Critically, an arbitrary user-provided CSV is NOT verified against the
 * UCI publisher source archive. This is a training-FIT historical demo and
 * neither a real-time monitor nor an independent evaluation of forecasts.
 * Never touches the persistent user ME111 state file.
 */
public final class Me111OriginalFitReplay {
    public static final int EVENTS = 256;
    public static final String SOURCE_STATUS =
        "USER_SELECTED_UCI_COMPATIBLE_CSV_NOT_SOURCE_HASH_VERIFIED";
    private static final int MAX_LINE_CHARS = 8192;
    private static final String[] REQUIRED = {
        "year", "month", "day", "hour", "station", "wd",
        "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
        "TEMP", "PRES", "DEWP", "RAIN", "WSPM"
    };

    public static final class Report {
        public final int owner;
        public final int events;
        public final long finalHour;
        public final long observedCount;
        public final float[] logLocation, leftScale, rightScale;
        public final Me111Contract.State finalState;

        private Report(int owner, Me111Contract.State state, Me111LocalEngine.Result last) {
            this.owner = owner;
            this.events = EVENTS;
            this.finalHour = state.lastHour;
            this.observedCount = state.count;
            this.finalState = state;
            this.logLocation = last.logLocation.clone();
            this.leftScale = last.leftScale.clone();
            this.rightScale = last.rightScale.clone();
        }

        public String display() {
            StringBuilder text = new StringBuilder();
            text.append("OFFLINE ORIGINAL PRSA FIT-ONLY DEMO.\n")
                .append("USER CSV SOURCE IDENTITY: NOT HASH-VERIFIED.\n")
                .append("MODEL'S ORIGINAL REAL-SOURCE SCIENTIFIC GATE: FAILED/MIXED.\n")
                .append("NO customer state saved. No live sensing or health guidance.\n")
                .append("Publisher: Chen (2017), UCI Beijing Multi-Site Air Quality, CC BY 4.0.\n")
                .append("Station ").append(Me111SourceFeatures.SITES[owner])
                .append("; processed ").append(events).append(" chronological 2013 FIT rows; ")
                .append("final original source hour ").append(finalHour)
                .append("; observed updates ").append(observedCount).append(".\n");
            for (int i = 0; i < 6; i++) {
                text.append(String.format(Locale.US,
                    "channel %d standardized log1p location %.6f; L %.6f; R %.6f\n",
                    i, logLocation[i], leftScale[i], rightScale[i]));
            }
            text.append("These are uncalibrated research predictions, not raw pollutant units.");
            return text.toString();
        }
    }

    private Me111OriginalFitReplay() { }

    private static String boundedLine(BufferedReader reader) throws IOException {
        // readLine() would allocate an unbounded string for a hostile selected
        // document. Reject oversized rows while they are being read.
        StringBuilder out = new StringBuilder(160);
        int next;
        while ((next = reader.read()) != -1) {
            if (next == 10) break;
            if (next == 13) continue;
            if (out.length() >= MAX_LINE_CHARS)
                throw new IOException("overlong source CSV record");
            out.append((char) next);
        }
        if (next == -1 && out.length() == 0) return null;
        return out.toString();
    }

    private static Map<String, Integer> header(String line) throws IOException {
        if (line == null || line.length() > MAX_LINE_CHARS)
            throw new IOException("missing/oversized original PRSA CSV header");
        String[] cells = line.replace("\uFEFF", "").split(",", -1);
        Map<String, Integer> columns = new HashMap<>();
        Set<String> seen = new HashSet<>();
        for (int i = 0; i < cells.length; i++) {
            String name = cells[i].trim();
            if (!seen.add(name)) throw new IOException("duplicate CSV column " + name);
            columns.put(name, i);
        }
        for (String name : REQUIRED)
            if (!columns.containsKey(name))
                throw new IOException("not an original PRSA CSV: missing " + name);
        return columns;
    }

    private static String cell(String[] row, Map<String, Integer> columns,
                               String name) throws IOException {
        int at = columns.get(name);
        if (at >= row.length)
            throw new IOException("malformed PRSA CSV row: missing " + name);
        return row[at].trim();
    }

    private static int number(String[] row, Map<String, Integer> columns,
                              String name) throws IOException {
        try {
            return Integer.parseInt(cell(row, columns, name));
        } catch (NumberFormatException invalid) {
            throw new IOException("malformed PRSA calendar " + name, invalid);
        }
    }

    public static Report run(InputStream source, Me111LocalEngine engine)
            throws IOException, ai.onnxruntime.OrtException {
        if (source == null || engine == null)
            throw new IllegalArgumentException("source file and local engine required");
        // A user-selected content provider may itself be network-backed.
        // The Activity enforces airplane-mode BEFORE allowing the file picker;
        // no app network permission or cloud inference is available.
        BufferedReader reader = new BufferedReader(
            new InputStreamReader(source, StandardCharsets.UTF_8), 8192);
        Map<String, Integer> columns = header(boundedLine(reader));
        Me111SourceFeatures encoder = Me111FrozenFitStats.newEncoder();
        Me111SourceFeatures.History history = null;
        Me111Contract.State state = null;
        Me111LocalEngine.Result prediction = null;
        int owner = -1;
        int rowWidth = columns.size();
        for (int i = 0; i < EVENTS; i++) {
            String line = boundedLine(reader);
            if (line == null || line.isEmpty() || line.length() > MAX_LINE_CHARS
                    || line.indexOf('"') >= 0)
                throw new IOException("missing/unsupported original hourly FIT row " + i);
            String[] cells = line.split(",", -1);
            if (cells.length != rowWidth)
                throw new IOException("PRSA CSV row width mismatch");
            String station = cell(cells, columns, "station");
            int stationId = Me111SourceFeatures.stationId(station);
            if (history == null) {
                owner = stationId;
                history = Me111SourceFeatures.beginningAtOriginalArchiveStart(owner);
                state = Me111Contract.initial(owner, 0);
            }
            if (owner != stationId) throw new IOException("cross-station source rows");
            LocalDateTime timestamp;
            try {
                timestamp = LocalDateTime.of(
                    number(cells, columns, "year"),
                    number(cells, columns, "month"),
                    number(cells, columns, "day"),
                    number(cells, columns, "hour"), 0
                );
            } catch (java.time.DateTimeException invalid) {
                throw new IOException("invalid original source calendar", invalid);
            }
            if (timestamp.getYear() != 2013)
                throw new IOException("only the original public 2013 FIT prefix is eligible");
            String[] numeric = new String[11];
            for (int j = 0; j < numeric.length; j++)
                numeric[j] = cell(cells, columns, Me111SourceFeatures.RAW_NAMES[j]);
            Me111Contract.Event event = encoder.encode(
                history, station, timestamp, numeric, cell(cells, columns, "wd"));
            prediction = engine.predict(event, state);
            state = prediction.nextState;
        }
        if (state == null || prediction == null || state.lastHour != EVENTS - 1)
            throw new IOException("incomplete original 256-event FIT prefix");
        return new Report(owner, state, prediction);
    }
}
