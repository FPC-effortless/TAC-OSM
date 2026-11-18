package org.tacosm.mobile.installprobe;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.pm.Signature;
import android.os.Build;
import android.os.Bundle;
import android.os.Process;
import android.system.Os;
import android.system.OsConstants;
import android.graphics.Typeface;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.security.MessageDigest;
import java.util.Arrays;

/**
 * Purposefully independent, Java-only, no INTERNET/no native library package.
 * It identifies a phone APK installer/ABI/signing conflict WITHOUT installing
 * or modifying TAC-OSM's neural app, touching user files or uploading data.
 *
 * This reports only software/device technical metadata. No model inference.
 */
public final class ProbeActivity extends Activity {
    private static final String MODEL_PACKAGE = "org.tacosm.mobile.model01.demo";

    private static String hex(byte[] digest) {
        StringBuilder out = new StringBuilder(digest.length * 2);
        for (byte b : digest) {
            int n = b & 0xff;
            out.append(Character.forDigit(n >>> 4, 16));
            out.append(Character.forDigit(n & 15, 16));
        }
        return out.toString();
    }

    private String installedModel() {
        try {
            PackageManager packages = getPackageManager();
            PackageInfo info;
            if (Build.VERSION.SDK_INT >= 28) {
                info = packages.getPackageInfo(
                    MODEL_PACKAGE, PackageManager.GET_SIGNING_CERTIFICATES);
            } else {
                info = packages.getPackageInfo(
                    MODEL_PACKAGE, PackageManager.GET_SIGNATURES);
            }
            String output = "INSTALLED (version " + info.versionName + ")";
            Signature[] signatures;
            if (Build.VERSION.SDK_INT >= 28 && info.signingInfo != null) {
                signatures = info.signingInfo.hasMultipleSigners()
                    ? info.signingInfo.getApkContentsSigners()
                    : info.signingInfo.getSigningCertificateHistory();
            } else {
                signatures = info.signatures;
            }
            if (signatures != null && signatures.length > 0) {
                MessageDigest digest = MessageDigest.getInstance("SHA-256");
                output += "\nInstalled model signing cert SHA256: "
                    + hex(digest.digest(signatures[0].toByteArray()));
            }
            return output;
        } catch (PackageManager.NameNotFoundException missing) {
            return "NOT INSTALLED";
        } catch (Exception error) {
            return "Installed-model query unavailable: "
                + error.getClass().getSimpleName();
        }
    }

    private String report() {
        long pageSize = -1;
        try { pageSize = Os.sysconf(OsConstants._SC_PAGESIZE); }
        catch (Exception ignored) { /* show UNKNOWN */ }
        StringBuilder result = new StringBuilder();
        result.append("TAC INSTALL PROBE PASS\n");
        result.append("This tiny APK installed and launched.\n\n");
        result.append("Manufacturer: ").append(Build.MANUFACTURER).append('\n');
        result.append("Model: ").append(Build.MODEL).append('\n');
        result.append("Android SDK: ").append(Build.VERSION.SDK_INT).append('\n');
        result.append("Android version: ").append(Build.VERSION.RELEASE).append('\n');
        result.append("Supported ABIs: ")
              .append(Arrays.toString(Build.SUPPORTED_ABIS)).append('\n');
        result.append("64-bit ABIs: ")
              .append(Arrays.toString(Build.SUPPORTED_64_BIT_ABIS)).append('\n');
        result.append("Probe process is 64-bit: ").append(Process.is64Bit())
              .append('\n');
        result.append("System memory page size: ").append(pageSize)
              .append(" bytes\n");
        result.append("TAC neural app: ").append(installedModel()).append('\n');
        result.append("Probe app Internet permission: ")
              .append(checkSelfPermission(android.Manifest.permission.INTERNET)
                          == PackageManager.PERMISSION_DENIED
                          ? "DENIED" : "UNEXPECTEDLY_GRANTED").append("\n\n");
        result.append("No neural inference was run.\n");
        result.append("No files or accounts were read; no data uploaded.\n");
        result.append("If this works but the larger TAC APK fails, investigate ");
        result.append("package integrity, installed signing certificate, ");
        result.append("native libraries or Samsung's installer policy.\n");
        return result.toString();
    }

    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        final String diagnostic = report();
        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        content.setPadding(32, 24, 32, 24);

        TextView title = new TextView(this);
        title.setText("TAC Installation Diagnostic");
        title.setTextSize(22f);
        title.setTypeface(null, Typeface.BOLD);
        content.addView(title);

        TextView info = new TextView(this);
        info.setText(diagnostic);
        info.setTextIsSelectable(true);
        info.setTextSize(15f);
        content.addView(info);

        Button copy = new Button(this);
        copy.setText("Copy diagnostic (only if you choose)");
        copy.setOnClickListener((View button) -> {
            ClipboardManager board =
                (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            if (board != null)
                board.setPrimaryClip(ClipData.newPlainText(
                    "TAC installer diagnostics", diagnostic));
        });
        content.addView(copy);
        ScrollView viewport = new ScrollView(this);
        viewport.addView(content);
        setContentView(viewport);
    }
}
