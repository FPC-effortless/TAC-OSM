package org.tacosm.mobile.me111;

import java.util.Arrays;

/**
 * Trust-boundary validation for frozen ME111 single-entity, single-step inference.
 * Local-only, dependency-free. Not an authentication or data-licensing mechanism.
 *
 * Inputs MUST already be original-source-aligned standardized features.
 * No implicit clock mapping, missing-data fabrication, network, or remote fallback.
 */
public final class Me111Contract {
    public static final String MODEL_SHA256 =
        "dd65519a9e4f3388966538af53119b56e9292b46a9209d63ad4927b87f0a176b";
    public static final String CHECKPOINT_SHA256 =
        "337b882e9a10631dc85700846ef2b15ca8908699499668186ad3a248dc108449";
    public static final int CHANNELS = 13;
    public static final int CLOCK_FIELDS = 6;
    public static final int EVIDENCE_DIM = 62;
    public static final int MEMORY_DIM = 8;
    public static final int FORECAST_CHANNELS = 6;
    private static final long MAX_HOUR = 20_000_000L;

    private Me111Contract() { }

    public static final class Event {
        public final int owner;
        public final long hour;
        public final float[] values, ages, clock;
        public final boolean[] observed;

        public Event(int owner, long hour, float[] values, boolean[] observed,
                     float[] ages, float[] clock) {
            this.owner = owner;
            this.hour = hour;
            this.values = copy(values, CHANNELS, "values");
            this.observed = copy(observed, CHANNELS, "observed");
            this.ages = copy(ages, CHANNELS, "ages");
            this.clock = copy(clock, CLOCK_FIELDS, "clock");
            validateEvent(this);
        }
    }

    public static final class State {
        public final int owner;
        public final long lastHour, count;
        public final float[] sourceSum, sourceMass, evidenceSum;
        public final float[][] memory;

        public State(int owner, long lastHour, long count,
                     float[] sourceSum, float[] sourceMass, float[] evidenceSum,
                     float[][] memory) {
            this.owner = owner;
            this.lastHour = lastHour;
            this.count = count;
            this.sourceSum = copy(sourceSum, CHANNELS, "sourceSum");
            this.sourceMass = copy(sourceMass, CHANNELS, "sourceMass");
            this.evidenceSum = copy(evidenceSum, EVIDENCE_DIM, "evidenceSum");
            if (memory == null || memory.length != MEMORY_DIM)
                throw new IllegalArgumentException("memory rows must equal eight");
            this.memory = new float[MEMORY_DIM][];
            for (int i = 0; i < MEMORY_DIM; i++)
                this.memory[i] = copy(memory[i], MEMORY_DIM, "memory");
            validateState(this);
        }
    }

    public static State initial(int owner, long firstHour) {
        if (owner < 0 || owner >= 12 || firstHour < 1 || firstHour > MAX_HOUR)
            throw new IllegalArgumentException("unknown owner or source-hour origin");
        return new State(owner, firstHour - 1, 0,
            new float[CHANNELS], new float[CHANNELS], new float[EVIDENCE_DIM],
            new float[MEMORY_DIM][MEMORY_DIM]);
    }

    public static void validateStep(Event event, State state) {
        validateEvent(event);
        validateState(state);
        if (event.owner != state.owner || event.hour <= state.lastHour)
            throw new IllegalArgumentException("foreign station or stale/duplicate hour");
        if (state.count == Long.MAX_VALUE)
            throw new IllegalArgumentException("state count overflow");
    }

    private static void validateEvent(Event event) {
        if (event.owner < 0 || event.owner >= 12 || event.hour < 1 || event.hour > MAX_HOUR)
            throw new IllegalArgumentException("owner/hour source contract violated");
        for (int i = 0; i < CHANNELS; i++) {
            float value = event.values[i], age = event.ages[i];
            requireFinite(value, "values"); requireFinite(age, "ages");
            if (Math.abs(value) > 8.00001f || age < 0 || age > 1)
                throw new IllegalArgumentException("source normalization/age contract violated");
            if (!event.observed[i] && value != 0f)
                throw new IllegalArgumentException("masked readings must be explicitly zero");
        }
        for (float value : event.clock) {
            requireFinite(value, "clock");
            if (Math.abs(value) > 100f)
                throw new IllegalArgumentException("unqualified clock scale");
        }
    }

    private static void validateState(State state) {
        if (state.owner < 0 || state.owner >= 12
                || state.lastHour < 0 || state.lastHour > MAX_HOUR
                || state.count < 0 || state.count > MAX_HOUR)
            throw new IllegalArgumentException("invalid state owner/hour/count");
        for (int i = 0; i < CHANNELS; i++) {
            requireFinite(state.sourceSum[i], "sourceSum");
            requireFinite(state.sourceMass[i], "sourceMass");
            if (state.sourceMass[i] < -1e-6f || state.sourceMass[i] > 1.00001f
                    || Math.abs(state.sourceSum[i]) > 1.00001f)
                throw new IllegalArgumentException("invalid evidence mass/mean");
        }
        for (float v : state.evidenceSum) {
            requireFinite(v, "evidenceSum");
            if (Math.abs(v) > (double)state.count + 1.0)
                throw new IllegalArgumentException("unbounded evidence sum");
        }
        for (float[] row : state.memory)
            for (float v : row) {
                requireFinite(v, "associative memory");
                if (Math.abs(v) > 8f)
                    throw new IllegalArgumentException("unbounded associative state");
            }
    }

    public static void validatePrediction(float[] location, float[] left, float[] right) {
        if (location == null || left == null || right == null
                || location.length != FORECAST_CHANNELS
                || left.length != FORECAST_CHANNELS
                || right.length != FORECAST_CHANNELS)
            throw new IllegalArgumentException("invalid output dimensions");
        for (int i = 0; i < FORECAST_CHANNELS; i++) {
            requireFinite(location[i], "location");
            requireFinite(left[i], "left scale");
            requireFinite(right[i], "right scale");
            if (!(left[i] > 0f && right[i] > 0f))
                throw new IllegalArgumentException("nonpositive predictive scale");
        }
    }

    private static void requireFinite(float f, String what) {
        if (Float.isNaN(f) || Float.isInfinite(f))
            throw new IllegalArgumentException("nonfinite " + what);
    }

    private static float[] copy(float[] v, int n, String what) {
        if (v == null || v.length != n)
            throw new IllegalArgumentException("invalid " + what + " dimension");
        return Arrays.copyOf(v, n);
    }

    private static boolean[] copy(boolean[] v, int n, String what) {
        if (v == null || v.length != n)
            throw new IllegalArgumentException("invalid " + what + " dimension");
        return Arrays.copyOf(v, n);
    }
}
