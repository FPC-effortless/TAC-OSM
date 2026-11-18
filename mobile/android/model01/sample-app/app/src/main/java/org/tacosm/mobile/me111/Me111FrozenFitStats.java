package org.tacosm.mobile.me111;

/**
 * Original ME111 13-channel FIT moments, exact values from the pinned
 * 2013-2014 only me100_sensor_data.py loader; no REPORT or sealed data.
 *
 * Original UCI publisher: Chen, S. (2017), Beijing Multi-Site Air Quality,
 * DOI:10.24432/C5RK5G, licensed CC BY 4.0. Historical research inputs only.
 *
 * The source loader/asset identity is independently checked in CI against
 * Git commit 3fc2fbe8e4974491349caecd8d6171e738810b54, never inferred
 * from the outputs of a deployed or user-provided source document.
 */
public final class Me111FrozenFitStats {
    public static final String PUBLISHER_ARCHIVE_SHA256 =
        "d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8";
    public static final String SOURCE_LOADER_SHA256 =
        "57f4d9086251d6dd8f12feb675ebe96de1223b3016906ad867cf3795fd0c74cc";
    private static final double[] MEAN = {
        3.955594778060913, 4.328571796417236, 2.488572120666504,
        3.756563186645508, 6.816250324249268, 3.4224164485931396,
        14.724853515625, 1009.3157348632812, 3.987563133239746,
        0.05722920224070549, 1.6201766729354858,
        0.04757897928357124, 0.09777148813009262
    };
    private static final double[] STD = {
        1.0774638652801514, 0.9826381206512451,
        1.067042350769043, 0.7692899107933044,
        0.7946207523345947, 1.348153829574585,
        10.934796333312988, 10.019977569580078,
        13.479767799377441, 0.7550337314605713,
        1.2372828722000122, 0.7111272811889648,
        0.6946041584014893
    };

    private Me111FrozenFitStats() { }

    public static double[] mean() { return MEAN.clone(); }
    public static double[] std() { return STD.clone(); }
    public static Me111SourceFeatures newEncoder() {
        return new Me111SourceFeatures(mean(), std());
    }
}
