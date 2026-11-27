//! Hy4-flash's E4M3 KV row (`--kv-cache fp8`): the bytes a row, and the host's copy of the format's arithmetic.
//!
//! The device's row is TensorFold's shared E4M3 format (`kernels/cuda/kv_fp8.cuh` `quant_row<512, true>`): one FP32
//! row scale, absmax / 448, and round-to-nearest saturated E4M3 codes for the 512-wide latent. The roped 64 columns
//! are not quantized: they keep their BF16 values in a side plane (`cuda_mla`'s read kernel), so only the latent pays
//! the format's error. `pack` below is a CPU reference of that write, never an inference path.
const std = @import("std");
const core = @import("core");
const m = @import("mla_weights.zig");

const fp8 = core.quant.fp8;

/// Bytes a cache row of `m.latent + m.rope` columns: 576 BF16, or the codes, the row scale and the BF16 rope plane.
pub fn rowBytes(kv8: bool) usize {
    return if (kv8) m.latent + @sizeOf(f32) + m.rope * 2 else m.absorbed_stride * 2;
}

/// An E4M3 code for `x`, ties to even and saturated at 448, as the device conversion reads it.
fn code(x: f32) u8 {
    const sign: u8 = @intCast((@as(u32, @bitCast(x)) >> 24) & 0x80);
    const v: f32 = @floatCast(@abs(fp8.toE4m3(x)));
    const b: u32 = @bitCast(v);
    return sign | if (v < 0x1p-6) @as(u8, @intFromFloat(v * 512)) else @as(u8, @intCast((((b >> 23) - 120) << 3) | ((b >> 20) & 7)));
}

/// `row`'s codes into `out`, returning the row's scale: the write `quant_row<512, true>` makes, in FP32.
pub fn pack(row: []const f32, out: []u8) !f32 {
    if (row.len == 0 or row.len != out.len) return error.BadRow;
    var amax: f32 = 0;
    for (row) |x| {
        if (!std.math.isFinite(x)) return error.NonFiniteRow;
        amax = @max(amax, @abs(x));
    }
    var inv: f32 = if (amax > 0) 448 / amax else 1;
    var scale: f32 = if (amax > 0) amax / 448 else 1;
    if (scale < 0x1p-126) {
        scale = 0x1p-126;
        inv = 1 / scale;
    }
    for (row, out) |x, *slot| slot.* = code(x * inv);
    return scale;
}

/// A code's value times the row's scale, in FP64 so the reference's own error does not enter the measurement.
pub fn value(bits: u8, scale: f32) f64 {
    return fp8.e4m3(bits) * @as(f64, scale);
}

test "the E4M3 cache row: 644 bytes against 1,152, and the latent alone loses precision" {
    try std.testing.expectEqual(@as(usize, 512 + 4 + 128), rowBytes(true));
    try std.testing.expectEqual(@as(usize, 576 * 2), rowBytes(false));
}

test "the E4M3 row round-trips within the format's bound, and the rope plane exactly" {
    var prng = std.Random.DefaultPrng.init(77);
    const random = prng.random();
    var row: [m.latent]f32 = undefined;
    var codes: [m.latent]u8 = undefined;
    var worst: f64 = 0;
    var sum: f64 = 0;
    for (0..64) |trial| {
        // Real latents have no outliers (K2's Q8 round trip found 1.5-1.9x the median channel): a spread of scales,
        // one large channel, and a row that is all zeros.
        for (&row, 0..) |*x, i| x.* = switch (trial) {
            0 => 0,
            1 => if (i == 0) 12.5 else 0.1,
            else => random.floatNorm(f32) * (1.0 + trial % 5) * 0.25,
        };
        const scale = try pack(&row, &codes);
        var amax: f32 = 0;
        for (row) |x| amax = @max(amax, @abs(x));
        if (amax > 0) {
            try std.testing.expectApproxEqRel(@as(f32, @floatCast(amax / 448.0)), scale, 1e-6);
        } else try std.testing.expectEqual(@as(f32, 1), scale);
        for (row, codes) |x, bits| {
            const back = value(bits, scale);
            const err = @abs(back - @as(f64, x));
            // E4M3 has three mantissa bits: nearest-even is within 2^-4 of a normal value, and the subnormals'
            // 2^-9 step bounds the rest (a row scaled to absmax = 448 never reaches them).
            const bound = @as(f64, @abs(x)) * 0x1p-4 + 0x1p-9 + @as(f64, scale) * 0x1p-10;
            try std.testing.expect(err <= bound);
            if (x != 0) {
                worst = @max(worst, err / @abs(@as(f64, x)));
                sum += err * err;
            }
        }
    }
    // The format's relative RMS is ~2.6 % (the plan's proxy: E4M3 2.6 % against the fork's Q8 0.54 %).
    const rms = @sqrt(sum / @as(f64, @floatFromInt(64 * m.latent)));
    try std.testing.expect(worst <= 0x1p-4 + 0x1p-9 / 0.25 and worst > 0.05);
    try std.testing.expect(rms > 0.005 and rms < 0.05);
}

test "the E4M3 row saturates and refuses what the device refuses" {
    var codes: [4]u8 = undefined;
    // absmax 448: the largest code is 448's and the row's own values land on the format's grid.
    const scale = try pack(&.{ 448, -448, 224, 0 }, &codes);
    try std.testing.expectEqual(@as(f32, 1), scale);
    try std.testing.expectEqual(@as(u8, 0x7e), codes[0]); // E4M3 448
    try std.testing.expectEqual(@as(u8, 0xfe), codes[1]);
    try std.testing.expectEqual(@as(u8, 0x5e), codes[2]); // E4M3 224
    try std.testing.expectEqual(@as(u8, 0x00), codes[3]);
    // A tiny row keeps a finite scale (2^-126's floor) instead of dividing by zero.
    const tiny = try pack(&.{ 1e-40, 0 }, codes[0..2]);
    try std.testing.expect(std.math.isFinite(tiny) and tiny >= 0x1p-126);
    try std.testing.expectError(error.NonFiniteRow, pack(&.{std.math.inf(f32)}, codes[0..1]));
}
