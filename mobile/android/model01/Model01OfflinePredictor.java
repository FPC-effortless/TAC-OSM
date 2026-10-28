// Generated from the pinned original real-source ridge checkpoint. DO NOT HAND EDIT.
// Generate/check: python scripts/export_model01_android_java.py [--check]
// Classical model, NOT TAC-OSM neural; NOT independent scientific/device qualification.
package org.tacosm.mobile.model01;

/** No dependencies, I/O, sockets, cloud processing, embeddings or hidden fallback. */
public final class Model01OfflinePredictor {
    public static final String CHECKPOINT_SHA256 = "cb756a734b026602bb13f6af11b08711e2029f78aad1e95bb469d55131bc6c82";
    public static final String SOURCE_SHA256 = "2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a";
    public static final String INDEPENDENT_QUALIFICATION = "NOT_QUALIFIED";
    public static final String ACTION_AUTHORITY = "ADVISORY_ONLY";
    public static final String INTERVAL_STATUS = "EMPIRICAL_HISTORICAL_CAL_NOT_GUARANTEED";
    public static final int HORIZON_MINUTES = 60;
    public static final int HISTORY_LENGTH = 145;
    private static final long MAX_SOURCE_INDEX = 20_000_000L;
    private static final double[] WEIGHTS = {98.35237213131624, 18.67573607498697, 1.450332858085561, 19.777725589356056, 2.5235918489055407, -18.05731600017427, 16.284123345157074};
    private static final double[] MEANS = {98.45219718019965, 98.45116805598435, 98.44293506226201, 98.29679942369044, 0.0046895183546518, 0.0004102796934287378};
    private static final double[] SCALES = {108.92417912447513, 108.92468413900774, 108.92853486710413, 108.95874934776015, 0.7071984489378856, 0.7069994299205787};
    private static final double CAL_Q_90_WH = 92.532708685843;

    private Model01OfflinePredictor() { }

    private static void requireReading(double reading) {
        if (Double.isNaN(reading) || Double.isInfinite(reading)
                || reading < 0.0 || reading > 1_000_000.0) {
            throw new IllegalArgumentException("invalid observed energy Wh");
        }
    }

    private static void requireIndex(long index) {
        if (index < 144 || index > MAX_SOURCE_INDEX) {
            throw new IllegalArgumentException("unverified source-aligned index");
        }
    }

    /** Caller must establish the original dataset's ten-minute phase origin;
     * arbitrary device wall-clock timestamps are NOT interchangeable. */
    public static Prediction predict(long sourceAlignedIndex, double[] last145EnergyWh) {
        requireIndex(sourceAlignedIndex);
        if (last145EnergyWh == null || last145EnergyWh.length != HISTORY_LENGTH) {
            throw new IllegalArgumentException("exactly 145 historical readings are required");
        }
        for (double reading : last145EnergyWh) {
            requireReading(reading);
        }
        final double angle = 2.0 * Math.PI * (sourceAlignedIndex % 144L) / 144.0;
        final double[] features = {
            last145EnergyWh[144], last145EnergyWh[143], last145EnergyWh[138],
            last145EnergyWh[0], Math.sin(angle), Math.cos(angle)
        };
        double sum = 0.0;
        for (int j = 0; j < 6; j++) {
            sum += WEIGHTS[j + 1] * (features[j] - MEANS[j]) / SCALES[j];
        }
        double point = Math.max(0.0, WEIGHTS[0] + sum);
        if (Double.isNaN(point) || Double.isInfinite(point)) {
            throw new IllegalArgumentException("nonfinite prediction; abstain locally");
        }
        return new Prediction(point, point - CAL_Q_90_WH, point + CAL_Q_90_WH);
    }

    public static final class Prediction {
        public final double predictionWh;
        public final double interval90LowerWh;
        public final double interval90UpperWh;
        public final int horizonMinutes = HORIZON_MINUTES;
        public final String independentQualification = INDEPENDENT_QUALIFICATION;
        public final String intervalStatus = INTERVAL_STATUS;
        public final String actionAuthority = ACTION_AUTHORITY;

        private Prediction(double point, double low, double high) {
            this.predictionWh = point;
            this.interval90LowerWh = low;
            this.interval90UpperWh = high;
        }
    }

    /** Local, bounded, source-aligned observation buffer (NOT TAC recurrent memory).
     * Persist index+snapshot in app-private storage for restart continuity.
     * Missing/late observations require abstention, not silent interpolation. */
    public static final class WindowState {
        private long sourceAlignedIndex;
        private final double[] readings;

        public WindowState(long sourceAlignedIndex, double[] last145EnergyWh) {
            // Full validation before retaining a defensive copy.
            Model01OfflinePredictor.predict(sourceAlignedIndex, last145EnergyWh);
            this.sourceAlignedIndex = sourceAlignedIndex;
            this.readings = last145EnergyWh.clone();
        }

        public long sourceAlignedIndex() { return sourceAlignedIndex; }
        public double[] snapshot() { return readings.clone(); }
        public Prediction predict() {
            return Model01OfflinePredictor.predict(sourceAlignedIndex, readings);
        }

        public void append(long nextSourceAlignedIndex, double observedWh) {
            if (sourceAlignedIndex >= MAX_SOURCE_INDEX
                    || nextSourceAlignedIndex != sourceAlignedIndex + 1) {
                throw new IllegalArgumentException("source index gap/duplicate: abstain");
            }
            requireReading(observedWh);
            System.arraycopy(readings, 1, readings, 0, HISTORY_LENGTH - 1);
            readings[HISTORY_LENGTH - 1] = observedWh;
            sourceAlignedIndex = nextSourceAlignedIndex;
        }
    }
}
