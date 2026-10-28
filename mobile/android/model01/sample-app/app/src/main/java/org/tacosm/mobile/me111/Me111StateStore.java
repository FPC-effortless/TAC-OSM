package org.tacosm.mobile.me111;

import android.content.Context;
import android.util.AtomicFile;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

/**
 * App-private, atomic local ME111 state. Never uploads, syncs or guesses state.
 * A corrupt or model-mismatched state causes local abstention until explicit reset.
 */
public final class Me111StateStore {
    private static final int MAGIC = 0x4D453131; // ME11
    private static final int VERSION = 1;
    private final AtomicFile file;

    public Me111StateStore(Context context) {
        this.file = new AtomicFile(new File(context.getFilesDir(), "me111-state-v1.bin"));
    }

    public synchronized Me111Contract.State load() throws IOException {
        // AtomicFile.exists() also detects a recoverable backup after interrupted write.
        // Checking only the base path could silently discard surviving state.
        if (!file.exists()) return null;
        try (DataInputStream in = new DataInputStream(file.openRead())) {
            if (in.readInt() != MAGIC || in.readInt() != VERSION
                    || !Me111Contract.MODEL_SHA256.equals(in.readUTF()))
                throw new IOException("stored local state version/model mismatch");
            int owner = in.readInt();
            long lastHour = in.readLong();
            long count = in.readLong();
            float[] sum = floats(in, 13);
            float[] mass = floats(in, 13);
            float[] evidence = floats(in, 62);
            float[][] memory = new float[8][];
            for (int i = 0; i < 8; i++) memory[i] = floats(in, 8);
            if (in.read() != -1)
                throw new IOException("trailing bytes in local state");
            try {
                return new Me111Contract.State(owner, lastHour, count,
                                               sum, mass, evidence, memory);
            } catch (IllegalArgumentException invalid) {
                throw new IOException("invalid persisted numerical state", invalid);
            }
        }
    }

    public synchronized void save(Me111Contract.State state) throws IOException {
        // Revalidate without mutation before any on-disk changes.
        Me111Contract.State checked = new Me111Contract.State(
            state.owner, state.lastHour, state.count, state.sourceSum,
            state.sourceMass, state.evidenceSum, state.memory);
        FileOutputStream raw = file.startWrite();
        try {
            DataOutputStream out = new DataOutputStream(raw);
            out.writeInt(MAGIC);
            out.writeInt(VERSION);
            out.writeUTF(Me111Contract.MODEL_SHA256);
            out.writeInt(checked.owner);
            out.writeLong(checked.lastHour);
            out.writeLong(checked.count);
            writeFloats(out, checked.sourceSum);
            writeFloats(out, checked.sourceMass);
            writeFloats(out, checked.evidenceSum);
            for (float[] row : checked.memory) writeFloats(out, row);
            out.flush();
            file.finishWrite(raw); // AtomicFile sync/replace, not an external service.
        } catch (IOException | RuntimeException exception) {
            file.failWrite(raw);
            throw exception;
        }
    }

    /** Explicit user reset only, not an inference-error fallback. */
    public synchronized void clear() {
        file.delete();
    }

    private static float[] floats(DataInputStream in, int length) throws IOException {
        float[] items = new float[length];
        for (int i = 0; i < items.length; i++) items[i] = in.readFloat();
        return items;
    }

    private static void writeFloats(DataOutputStream out, float[] items)
            throws IOException {
        for (float item : items) out.writeFloat(item);
    }
}
