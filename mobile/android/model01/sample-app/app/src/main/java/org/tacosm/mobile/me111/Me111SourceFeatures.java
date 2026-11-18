package org.tacosm.mobile.me111;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.Arrays;

/**
 * Source-faithful ME100/ME111 feature preprocessor for publicly documented
 * PRSA Beijing hourly rows, using FIT-only moments from the exact original
 * archive. This is NOT a model, a source-rights check, or a scientific
 * prediction qualification. No networking or Android framework is needed.
 *
 * Matches original me100_sensor_data.py:
 *   raw 11 numeric channels, 16-point wind sin/cos -> 13 channels;
 *   first six pollutants log1p, then FIT (mean,std), clip [-8,8];
 *   calendar sin/cos 24h, 7-day, 366-day; masked value=0;
 *   age=log1p(min(time-lastObserved,256))/log(257).
 *
 * Each History belongs to ONE source and must begin with the first observation
 * in an uninterrupted, lawful chronological archive. Starting mid-stream will
 * yield unknown ages until 256h and cannot be called exact-source parity.
 */
public final class Me111SourceFeatures {
    public static final String[] SITES = {
        "Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan",
        "Gucheng", "Huairou", "Nongzhanguan", "Shunyi", "Tiantan",
        "Wanliu", "Wanshouxigong"
    };
    public static final String[] RAW_NAMES = {
        "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
        "TEMP", "PRES", "DEWP", "RAIN", "WSPM"
    };
    private static final String[] WIND = {
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
    };
    private static final LocalDateTime EPOCH = LocalDateTime.of(2013, 3, 1, 0, 0);
    private static final double TAU = 2 * Math.PI;
    private static final double LOG257 = Math.log(257);
    private final double[] means, scales;

    public static final class History {
        public final int owner;
        private long lastHour = -1;
        private final long[] lastSeen = new long[Me111Contract.CHANNELS];
        private History(int owner) {
            if (owner < 0 || owner >= SITES.length)
                throw new IllegalArgumentException("unknown original source station");
            this.owner = owner;
            Arrays.fill(lastSeen, -1_000_000_000L);
        }
        public long lastHour() { return lastHour; }
    }

    public Me111SourceFeatures(double[] fitMean, double[] fitStd) {
        if (fitMean == null || fitStd == null
                || fitMean.length != Me111Contract.CHANNELS
                || fitStd.length != Me111Contract.CHANNELS)
            throw new IllegalArgumentException("exact thirteen FIT-only source moments required");
        means = fitMean.clone();
        scales = fitStd.clone();
        for (int i = 0; i < means.length; i++) {
            if (!Double.isFinite(means[i]) || !Double.isFinite(scales[i])
                    || scales[i] < 1e-4)
                throw new IllegalArgumentException("invalid original FIT moment");
        }
    }

    public static int stationId(String station) {
        for (int i = 0; i < SITES.length; i++)
            if (SITES[i].equals(station)) return i;
        throw new IllegalArgumentException("unknown original station");
    }

    public static History beginningAtOriginalArchiveStart(int owner) {
        return new History(owner);
    }

    /**
     * Numeric channel strings must come from an authorized source row. "NA"
     * or empty means missing, not an imputed number. The wind field is a
     * publisher tag (N, NNE, etc.) or NA. At most one station per History.
     */
    public Me111Contract.Event encode(History history, String station,
                                     LocalDateTime timestamp, String[] numeric,
                                     String windTag) {
        if (history == null || stationId(station) != history.owner)
            throw new IllegalArgumentException("cross-source history rejected");
        if (numeric == null || numeric.length != RAW_NAMES.length || windTag == null
                || timestamp == null || timestamp.getMinute() != 0
                || timestamp.getSecond() != 0 || timestamp.getNano() != 0)
            throw new IllegalArgumentException("source schema or hour malformed");
        long hour = ChronoUnit.HOURS.between(EPOCH, timestamp);
        if (hour < 0 || hour > 20_000_000 || hour <= history.lastHour)
            throw new IllegalArgumentException("source hours must strictly increase");
        // The original UCI loader reads the full hourly archive from epoch
        // zero. Without contiguous prior rows, channel ages cannot be derived
        // faithfully; refuse an attractive but misleading partial replay.
        if ((history.lastHour == -1 && hour != 0)
                || (history.lastHour >= 0 && hour != history.lastHour + 1))
            throw new IllegalArgumentException(
                "complete original hourly source history required for exact ages");

        float[] raw = new float[Me111Contract.CHANNELS];
        boolean[] observed = new boolean[Me111Contract.CHANNELS];
        for (int i = 0; i < numeric.length; i++) {
            String cell = numeric[i] == null ? "" : numeric[i].trim();
            if (cell.isEmpty() || cell.equals("NA")) continue;
            float number;
            try { number = Float.parseFloat(cell); }
            catch (NumberFormatException error) {
                throw new IllegalArgumentException("not an original finite numeric reading", error);
            }
            if (!Float.isFinite(number) || (i < 6 && number < 0f))
                throw new IllegalArgumentException("nonfinite/negative original pollutant");
            raw[i] = i < 6 ? (float) Math.log1p((double) number) : number;
            observed[i] = true;
        }
        if (!windTag.trim().equals("NA") && !windTag.trim().isEmpty()) {
            int index = -1;
            for (int i = 0; i < WIND.length; i++)
                if (WIND[i].equals(windTag.trim())) { index = i; break; }
            if (index < 0) throw new IllegalArgumentException("unknown original wind tag");
            double angle = TAU * index / 16.0;
            raw[11] = (float)Math.sin(angle);
            raw[12] = (float)Math.cos(angle);
            observed[11] = observed[12] = true;
        }
        float[] normalized = new float[Me111Contract.CHANNELS];
        float[] ages = new float[Me111Contract.CHANNELS];
        // First validate every field and its model range; only then advance
        // History, so malformed input cannot silently consume source hours.
        for (int i = 0; i < normalized.length; i++) {
            if (observed[i]) {
                double z = (raw[i] - means[i]) / scales[i];
                normalized[i] = (float)Math.max(-8.0, Math.min(8.0, z));
                if (!Float.isFinite(normalized[i]))
                    throw new IllegalArgumentException("invalid transformed source value");
            }
            long previous = observed[i] ? hour : history.lastSeen[i];
            long distance = Math.max(0, Math.min(256, hour - previous));
            ages[i] = (float)(Math.log1p(distance) / LOG257);
        }
        double[] cyc = {
            timestamp.getHour() / 24.0,
            (timestamp.getDayOfWeek().getValue() - 1) / 7.0,
            (timestamp.getDayOfYear() - 1) / 366.0
        };
        float[] clocks = new float[Me111Contract.CLOCK_FIELDS];
        for (int i = 0; i < 3; i++) {
            clocks[i] = (float)Math.sin(TAU * cyc[i]);
            clocks[i + 3] = (float)Math.cos(TAU * cyc[i]);
        }
        Me111Contract.Event event = new Me111Contract.Event(
            history.owner, hour, normalized, observed, ages, clocks);
        for (int i = 0; i < normalized.length; i++)
            if (observed[i]) history.lastSeen[i] = hour;
        history.lastHour = hour;
        return event;
    }
}
