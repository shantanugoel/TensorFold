//! Dense Hy4-flash projections on CUDA: the pack's own EXL3 group, or the BF16 weight beside it.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");

/// One projection as the layer holds it: `n` rows out, `k` columns in, the bytes uploaded once at load.
pub const Projection = struct {
    n: usize,
    k: usize,
    weight: union(enum) { bf16: u64, exl3: cuda.exl3.Weight },
};

const DType = core.safetensors.DType;

fn name(buf: []u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
    return std.fmt.bufPrint(buf, fmt, args);
}

/// `<base>` as the pack stores it: an EXL3 group when `<base>.trellis` is there, else the BF16 `<base>.weight`.
pub fn read(own: *cuda.exl3_upload.Uploader, ck: *core.Checkpoint, comptime fmt: []const u8, args: anytype, n: usize, k: usize, dtype: DType) !Projection {
    var buf: [208]u8 = undefined;
    const base = try name(&buf, fmt, args);
    var trellis_buf: [224]u8 = undefined;
    if (!ck.has(try name(&trellis_buf, "{s}.trellis", .{base}))) {
        var weight_buf: [224]u8 = undefined;
        const t = try ck.expect(try name(&weight_buf, "{s}.weight", .{base}), dtype, &.{ n, k });
        return .{ .n = n, .k = k, .weight = .{ .bf16 = try own.dev(t.bytes) } };
    }
    return group(own, ck, base, n, k);
}

/// `base`'s EXL3 group: the trellis in the stored layout, one codebook marker, fp16 scales, no bias.
fn group(own: *cuda.exl3_upload.Uploader, ck: *core.Checkpoint, base: []const u8, n: usize, k: usize) !Projection {
    var b1: [224]u8 = undefined;
    var b2: [224]u8 = undefined;
    const t = try ck.get(try name(&b1, "{s}.trellis", .{base}));
    if (t.dtype != .i16 or t.rank != 3 or k % 128 != 0 or n % 128 != 0) return error.BadHy4DenseTrellis;
    const g = try core.exl3_layout.Group.fromShape(t.shape[0..t.rank]);
    if (g.k() != k or g.n() != n or t.bytes.len != g.trellisBytes()) return error.BadHy4DenseTrellis;
    var codebook: ?core.exl3_load.Codebook = null;
    for (core.exl3_load.markers) |marker| {
        const marker_name = try name(&b2, "{s}.{s}", .{ base, marker.name });
        if (!ck.has(marker_name)) continue;
        if (codebook != null) return error.UnsupportedHy4DenseGroup;
        const m = try ck.get(marker_name); // the pack stores it as [1] or as a scalar
        if ((m.dtype != .i32 and m.dtype != .u32) or m.numel() != 1 or m.bytes.len != 4) return error.BadHy4DenseMarker;
        if (std.mem.readInt(u32, m.bytes[0..4], .little) != marker.value) return error.BadHy4DenseMarker;
        codebook = marker.codebook;
    }
    const cb = codebook orelse return error.BadHy4DenseMarker;
    const suh = try ck.expect(try name(&b1, "{s}.suh", .{base}), .f16, &.{k});
    const svh = try ck.expect(try name(&b1, "{s}.svh", .{base}), .f16, &.{n});
    // The siblings of a form this path does not read: a packed sign word or a bias would change the arithmetic.
    for ([_][]const u8{ "bias", "weight", "su", "sv" }) |suffix| {
        if (ck.has(try name(&b2, "{s}.{s}", .{ base, suffix }))) return error.UnsupportedHy4DenseGroup;
    }
    const parts: core.exl3_load.Parts = .{ .trellis = t, .suh = suh, .svh = svh };
    const layout: core.exl3_load.Group = .{ .prefix = base, .halves = @intCast(g.k2()), .codebook = cb, .k = k, .n = n, .suh = true, .bias = false };
    const w = try own.dense(parts, layout);
    // A width without a compiled linear would load and then fail mid-forward: refuse it here.
    if (cuda.exl3.variantIndex(w.halves, w.codebook, cuda.exl3.planOf(w).wk) == null) return error.UnsupportedHy4DenseBits;
    return .{ .n = n, .k = k, .weight = .{ .exl3 = w } };
}
