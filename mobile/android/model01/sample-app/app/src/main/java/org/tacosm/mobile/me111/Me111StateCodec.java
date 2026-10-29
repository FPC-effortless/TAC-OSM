package org.tacosm.mobile.me111;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/**
 * Dependency-free, fixed-length, SHA256-checked persistent ME111 state codec.
 *
 * Version 2 intentionally does NOT silently import the old unsigned v1 file.
 * Such state must be explicitly reset or independently recovered from a trusted
 * original event log. SHA256 detects accidental/inadvertent mutation but is
 * NOT keyed authentication, rollback prevention, encryption or device attestation.
 *
 * The storage layer, not this codec, must provide atomic app-private writes.
 */
public final class Me111StateCodec {
    private static final int MAGIC = 0x4D453131; // ME11
    private static final int VERSION = 2;
    private static final int HASH_BYTES = 32;
    private static final int PAYLOAD_BYTES =
        4 + 4 + 2 + 64 + 4 + 8 + 8
            + 4 * (Me111Contract.CHANNELS * 2
                   + Me111Contract.EVIDENCE_DIM
                   + Me111Contract.MEMORY_DIM * Me111Contract.MEMORY_DIM);
    public static final int BINARY_LENGTH = PAYLOAD_BYTES + HASH_BYTES;

    private Me111StateCodec() { }

    public static byte[] encode(Me111Contract.State state) throws IOException {
        // A fresh copy forces all numeric/shape validations before serialization.
        Me111Contract.State checked = new Me111Contract.State(
            state.owner, state.lastHour, state.count,
            state.sourceSum, state.sourceMass, state.evidenceSum, state.memory);
        ByteArrayOutputStream buffer = new ByteArrayOutputStream(PAYLOAD_BYTES);
        try (DataOutputStream output = new DataOutputStream(buffer)) {
            output.writeInt(MAGIC);
            output.writeInt(VERSION);
            output.writeUTF(Me111Contract.MODEL_SHA256);
            output.writeInt(checked.owner);
            output.writeLong(checked.lastHour);
            output.writeLong(checked.count);
            floats(output, checked.sourceSum);
            floats(output, checked.sourceMass);
            floats(output, checked.evidenceSum);
            for (float[] row : checked.memory) floats(output, row);
            output.flush();
        }
        byte[] payload = buffer.toByteArray();
        if (payload.length != PAYLOAD_BYTES)
            throw new IOException("unexpected fixed ME111 state payload length");
        byte[] bytes = new byte[BINARY_LENGTH];
        System.arraycopy(payload, 0, bytes, 0, PAYLOAD_BYTES);
        byte[] hash = digest(bytes, 0, PAYLOAD_BYTES);
        System.arraycopy(hash, 0, bytes, PAYLOAD_BYTES, HASH_BYTES);
        return bytes;
    }

    public static Me111Contract.State decode(byte[] bytes) throws IOException {
        if (bytes == null || bytes.length != BINARY_LENGTH) {
            // Old v1 files have no digest. Never treat them as empty/new states.
            throw new IOException("unsupported, truncated or oversized ME111 state; explicit reset required");
        }
        byte[] hash = digest(bytes, 0, PAYLOAD_BYTES);
        byte[] provided = new byte[HASH_BYTES];
        System.arraycopy(bytes, PAYLOAD_BYTES, provided, 0, HASH_BYTES);
        if (!MessageDigest.isEqual(hash, provided))
            throw new IOException("ME111 state checksum mismatch; abstain locally");

        try (DataInputStream input = new DataInputStream(
                new ByteArrayInputStream(bytes, 0, PAYLOAD_BYTES))) {
            if (input.readInt() != MAGIC || input.readInt() != VERSION
                    || !Me111Contract.MODEL_SHA256.equals(input.readUTF()))
                throw new IOException("ME111 state format or model version changed");
            int owner = input.readInt();
            long hour = input.readLong();
            long count = input.readLong();
            float[] sums = readFloats(input, Me111Contract.CHANNELS);
            float[] masses = readFloats(input, Me111Contract.CHANNELS);
            float[] evidence = readFloats(input, Me111Contract.EVIDENCE_DIM);
            float[][] memory = new float[Me111Contract.MEMORY_DIM][];
            for (int row = 0; row < memory.length; row++)
                memory[row] = readFloats(input, Me111Contract.MEMORY_DIM);
            if (input.read() != -1)
                throw new IOException("unexpected trailing ME111 state payload");
            try {
                return new Me111Contract.State(
                    owner, hour, count, sums, masses, evidence, memory);
            } catch (IllegalArgumentException invalid) {
                throw new IOException("ME111 persisted numerical state rejected", invalid);
            }
        }
    }

    private static byte[] digest(byte[] source, int offset, int length) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update(source, offset, length);
            return digest.digest();
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 unavailable", impossible);
        }
    }

    private static float[] readFloats(DataInputStream input, int count)
            throws IOException {
        float[] values = new float[count];
        for (int i = 0; i < count; i++) values[i] = input.readFloat();
        return values;
    }

    private static void floats(DataOutputStream output, float[] values)
            throws IOException {
        for (float v : values) output.writeFloat(v);
    }
}
