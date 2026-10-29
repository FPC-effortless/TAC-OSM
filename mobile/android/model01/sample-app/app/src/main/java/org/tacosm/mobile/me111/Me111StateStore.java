package org.tacosm.mobile.me111;

import android.content.Context;
import android.util.AtomicFile;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * Atomic device-private ME111 recurrent-state persistence with v2 integrity.
 *
 * An old unsigned v1 file is NEVER silently interpreted as a fresh state.
 * The application must abstain and require explicit user reset or audited
 * event-log recovery. SHA256 checks accidental corruption, not malicious
 * device tampering, rollback or authentication.
 */
public final class Me111StateStore {
    // Keep historical pathname so an old unsigned v1 file is detected and
    // REJECTED, rather than silently lost by switching to a new pathname.
    private static final String STATE_FILE = "me111-state-v1.bin";
    private final AtomicFile file;

    public Me111StateStore(Context context) {
        this.file = new AtomicFile(new File(context.getFilesDir(), STATE_FILE));
    }

    public synchronized Me111Contract.State load() throws IOException {
        File base = file.getBaseFile();
        // android.util.AtomicFile may retain a .bak or .new during a process
        // death. openRead() handles its native recovery; never reset if a
        // potentially recoverable or corrupt file exists.
        if (!base.exists()
                && !new File(base.getPath() + ".bak").exists()
                && !new File(base.getPath() + ".new").exists()) return null;
        try (InputStream input = file.openRead()) {
            byte[] bytes = new byte[Me111StateCodec.BINARY_LENGTH];
            int read = 0;
            while (read < bytes.length) {
                int n = input.read(bytes, read, bytes.length - read);
                if (n == -1) break;
                if (n == 0) continue;
                read += n;
            }
            if (read != bytes.length || input.read() != -1)
                throw new IOException("truncated/oversized or obsolete ME111 state; abstain");
            return Me111StateCodec.decode(bytes);
        }
    }

    public synchronized void save(Me111Contract.State state) throws IOException {
        // Validate all values and compute integrity bytes before AtomicFile
        // starts a write; no partial mutation of the previous committed state.
        byte[] encoded = Me111StateCodec.encode(state);
        FileOutputStream raw = file.startWrite();
        try {
            raw.write(encoded);
            file.finishWrite(raw);
        } catch (IOException | RuntimeException error) {
            file.failWrite(raw);
            throw error;
        }
    }

    /** User-authorized explicit reset only; never an inference-error fallback. */
    public synchronized void clear() {
        file.delete();
    }
}
