//! Hy4-flash's image tower on CUDA (plan 7.9): the 27-layer ViT and `vit.perceive`, and the row plan an image span
//! imposes on the text model (its embedding rows, its 3-axis mRoPE positions, its Engram media mask).
//!
//! The tower is the checkpoint's own `vit.*` tensors, BF16, run through the shared vision glue (`cuda.vit`): a Linear
//! is cuBLASLt, the LayerNorms, GELUs and the pixel unshuffle are `vit.cu`, and each image's frame attention is
//! `prefill_attention.cu`'s head-dim 72 kernel (every patch over every patch, no mask, no rotary). Nothing is
//! quantized. The one step with no shared kernel is the learned 128 x 128 position table, bilinearly resampled to the
//! image's patch grid with the reference's `(size + 0.1) / 128` scale: a host loop over four taps a patch, once an
//! image, into a BF16 buffer the patch embedding's rows are added to.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
pub const vc = @import("vision_config.zig");
const geom = @import("cuda_scratch.zig");

/// The text model's hidden width, which the projector outputs (`vision_config.out_hidden_size`).
pub const H: usize = geom.H;
/// The tower's widths, as the checkpoint's `vision_config` fixes them.
pub const hidden: usize = 1152;
pub const heads: usize = 16;
pub const dim: usize = 72;
pub const inner: usize = 4304;
/// Patches a merged token takes: 2 x 2.
pub const merge: usize = 2;
/// One raster patch's width: C x 16 x 16.
pub const patch_width = 3 * vc.patch * vc.patch;
/// The projector's pixel-unshuffle width: four patches side by side.
pub const merge_width = hidden * merge * merge;
/// Rows the token-wise GEMMs run at once, so their block is bounded by this and not by the patch count.
const block_rows: usize = 2048;
/// The processor's limits: LM tokens an image may take (the patch budget's ceiling, `(gh / 2) * (gw / 2 + 1)`), and
/// the images a request may hold.
const limits: vc.Limits = .{};
pub const token_limit: u32 = limits.image_tokens;
pub const image_limit: u32 = limits.max_images;

/// One image's place in a prompt: the rows it covers, its patch grid, and the bytes-and-grid hash it is identified by.
pub const Input = struct { at: usize, tokens: usize, gh: u32, gw: u32, hash: u64 };

/// A prompt's image spans, as the text model reads them: where each image's rows sit in the tower's buffer, and the
/// 3-axis position block it starts.
pub const Span = struct {
    at: usize,
    tokens: usize,
    gh: u32,
    gw: u32,
    base: usize, // the image's first row in the feature buffer
    start: i32, // the position block's first value
};

/// The identity of a prompt's images and their places in it: a kept prefix is resumed only under the same plan, because
/// an image's rows carry its own 3-axis positions. 0 is not a plan (a text-only prompt), so the text path's reuse is
/// exactly what it was.
///
/// The prompt's **length is not part of it**: an image's rows are fixed by the images and the text before them, so a
/// longer prompt — the same conversation with another turn — extends the text tail and moves no image row. Including
/// the length would make every turn of an image conversation a fresh fill, which is the case a kept prefix is for.
pub fn hashOf(inputs: []const Input) u64 {
    var h = std.hash.Wyhash.init(0x9e3779b97f4a7c15);
    for (inputs) |im| h.update(std.mem.asBytes(&[_]u64{ im.hash, im.at, im.tokens }));
    return h.final();
}

/// The 3-D positions and feature rows a prompt with images imposes, and where the text after it continues
/// (the reference's `get_rope_index` for one sequence).
pub const Plan = struct {
    a: std.mem.Allocator,
    spans: []Span,
    prompt_len: usize,
    rows: usize, // the feature rows every span takes, in prompt order
    /// Text after the prompt continues at `row + delta`: `next_position - prompt_len`, as `rope_deltas` is.
    delta: i32,
    /// The images' identity and their places in the prompt: a prefix reuse compares it with the kept prompt's.
    hash: u64,

    pub fn build(a: std.mem.Allocator, inputs: []const Input, prompt_len: usize) !Plan {
        const spans = try a.alloc(Span, inputs.len);
        errdefer a.free(spans);
        var plan: Plan = .{ .a = a, .spans = spans, .prompt_len = prompt_len, .rows = 0, .delta = 0, .hash = hashOf(inputs) };
        var cursor: usize = 0; // the first prompt row not yet placed
        var next: i32 = 0; // where the next placed row's position starts
        for (inputs, 0..) |im, i| {
            if (im.gh == 0 or im.gw == 0 or im.gh % merge != 0 or im.gw % merge != 0) return error.BadHy4VisionPlan;
            if (im.at < cursor or im.at + im.tokens > prompt_len) return error.BadHy4VisionPlan;
            if (im.tokens != vc.tokensForGrid(im.gh, im.gw)) return error.BadHy4VisionPlan;
            const start = next + @as(i32, @intCast(im.at - cursor));
            spans[i] = .{ .at = im.at, .tokens = im.tokens, .gh = im.gh, .gw = im.gw, .base = plan.rows, .start = start };
            plan.rows += im.tokens;
            next = start + @as(i32, @intCast(@max(im.gh / merge, im.gw / merge + 1)));
            cursor = im.at + im.tokens;
        }
        // The text after the prompt's last image: a row sits one past its predecessor's position.
        plan.delta = next + @as(i32, @intCast(prompt_len - cursor)) - @as(i32, @intCast(prompt_len));
        return plan;
    }

    pub fn deinit(p: *Plan) void {
        p.a.free(p.spans);
        p.* = undefined;
    }

    /// The 3-axis position of prompt row `row`, per the reference: an image's rows are `(start, start + oh, start + ow)`
    /// on its merged grid (the newline column included), text is one running position on all three axes.
    pub fn positionAt(p: Plan, row: usize) [3]i32 {
        var base: i32 = 0; // the running position of the text cursor
        var cursor: usize = 0;
        for (p.spans) |sp| {
            if (row < sp.at) {
                const v = base + @as(i32, @intCast(row - cursor));
                return .{ v, v, v };
            }
            const width: usize = sp.gw / merge + 1;
            if (row < sp.at + sp.tokens) {
                const off = row - sp.at;
                return .{ sp.start, sp.start + @as(i32, @intCast(off / width)), sp.start + @as(i32, @intCast(off % width)) };
            }
            base = sp.start + @as(i32, @intCast(@max(sp.gh / merge, width)));
            cursor = sp.at + sp.tokens;
        }
        const v = base + @as(i32, @intCast(row - cursor));
        return .{ v, v, v };
    }

    /// `count` rows of a pass from `start`: their positions, 3 an i32 row, and the feature row each takes (-1: its own
    /// token's embedding).
    pub fn fill(p: Plan, start: usize, count: usize, positions: []i32, sources: []i32) void {
        std.debug.assert(positions.len >= count * 3 and sources.len >= count);
        for (0..count) |i| {
            const row = start + i;
            positions[i * 3 ..][0..3].* = p.positionAt(row);
            sources[i] = p.featureRow(row);
        }
    }

    /// The tower row prompt row `row` takes, or -1 for a text row.
    pub fn featureRow(p: Plan, row: usize) i32 {
        for (p.spans) |sp| {
            if (row < sp.at) return -1;
            if (row < sp.at + sp.tokens) return @intCast(sp.base + (row - sp.at));
        }
        return -1;
    }

    /// Whether prompt row `row` is an image row, which the Engram branch leaves alone.
    pub fn isImage(p: Plan, row: usize) bool {
        return p.featureRow(row) >= 0;
    }
};

test "a plan's positions are the reference's, and text after an image continues where the block ended" {
    const a = std.testing.allocator;
    // One 4x6-patch image at row 2: merged grid 2 x 4 (3 + 1 newline columns), 8 rows, block 4 wide.
    var plan = try Plan.build(a, &.{.{ .at = 2, .tokens = 8, .gh = 4, .gw = 6, .hash = 7 }}, 14);
    defer plan.deinit();
    // Text before it: 2 rows at 0 and 1.
    try std.testing.expectEqual([3]i32{ 0, 0, 0 }, plan.positionAt(0));
    try std.testing.expectEqual([3]i32{ 1, 1, 1 }, plan.positionAt(1));
    // The image's block starts at 2 and runs 4 positions (max(mh = 2, mw + 1 = 4)).
    try std.testing.expectEqual([3]i32{ 2, 2, 2 }, plan.positionAt(2));
    try std.testing.expectEqual([3]i32{ 2, 2, 5 }, plan.positionAt(5));
    try std.testing.expectEqual([3]i32{ 2, 3, 2 }, plan.positionAt(6));
    try std.testing.expectEqual([3]i32{ 2, 3, 5 }, plan.positionAt(9));
    // Text after it resumes at 2 + 4 = 6, and the prompt's tail runs to row 14: a decode row is one past it.
    try std.testing.expectEqual([3]i32{ 6, 6, 6 }, plan.positionAt(10));
    try std.testing.expectEqual([3]i32{ 7, 7, 7 }, plan.positionAt(11));
    // A decode row at 14 sits at 10: the text tail ran 6, 7, 8, 9 for rows 10..13.
    try std.testing.expectEqual(@as(i32, 10 - 14), plan.delta);
    try std.testing.expectEqual([3]i32{ 10, 10, 10 }, plan.positionAt(14));
    // The feature rows follow the prompt: the image takes rows 0..8.
    try std.testing.expectEqual(@as(i32, -1), plan.featureRow(1));
    try std.testing.expectEqual(@as(i32, 0), plan.featureRow(2));
    try std.testing.expectEqual(@as(i32, 7), plan.featureRow(9));
    try std.testing.expectEqual(@as(i32, -1), plan.featureRow(10));
    try std.testing.expect(!plan.isImage(10) and plan.isImage(4));
}

test "two back-to-back images get a block each, and the second's feature rows follow the first's" {
    const a = std.testing.allocator;
    // Two 4x4-patch images side by side at row 1: each is 2 x 3 = 6 merged rows, block 3 wide.
    var plan = try Plan.build(a, &.{
        .{ .at = 1, .tokens = 6, .gh = 4, .gw = 4, .hash = 1 },
        .{ .at = 7, .tokens = 6, .gh = 4, .gw = 4, .hash = 2 },
    }, 14);
    defer plan.deinit();
    try std.testing.expectEqual([3]i32{ 1, 1, 1 }, plan.positionAt(1));
    try std.testing.expectEqual([3]i32{ 1, 2, 3 }, plan.positionAt(6));
    // The second image starts a block at 1 + 3 = 4, not where the first's rows ran out (3).
    try std.testing.expectEqual([3]i32{ 4, 4, 4 }, plan.positionAt(7));
    try std.testing.expectEqual([3]i32{ 4, 5, 6 }, plan.positionAt(12));
    try std.testing.expectEqual([3]i32{ 7, 7, 7 }, plan.positionAt(13));
    try std.testing.expectEqual(@as(i32, 8 - 14), plan.delta);
    try std.testing.expectEqual(@as(i32, 6), plan.featureRow(7));
    try std.testing.expectEqual(@as(i32, 11), plan.featureRow(12));
}

test "a kept prefix's plan identity is the images and their places, not the prompt's length" {
    const a = std.testing.allocator;
    const one = [_]Input{.{ .at = 2, .tokens = 8, .gh = 4, .gw = 6, .hash = 7 }};
    // The same conversation with one more turn: the image is at the same row with the same bytes, so the kept rows
    // are resumable — the prompt's tail is longer and no image row moves.
    var short = try Plan.build(a, &one, 14);
    defer short.deinit();
    var long = try Plan.build(a, &one, 40);
    defer long.deinit();
    try std.testing.expectEqual(short.hash, long.hash);
    // Two prompts, one image, the same rows: the same plan whatever else each carries.
    var again = try Plan.build(a, &one, 21);
    defer again.deinit();
    try std.testing.expectEqual(short.hash, again.hash);
    // Another image, or the same one further along the prompt, is another plan: its rows carry other positions.
    var moved = try Plan.build(a, &.{.{ .at = 3, .tokens = 8, .gh = 4, .gw = 6, .hash = 7 }}, 40);
    defer moved.deinit();
    try std.testing.expect(moved.hash != long.hash);
    var other = try Plan.build(a, &.{.{ .at = 2, .tokens = 8, .gh = 4, .gw = 6, .hash = 8 }}, 40);
    defer other.deinit();
    try std.testing.expect(other.hash != long.hash);
}

test "a plan refuses an image whose rows do not match its grid, and one that overlaps or overruns the prompt" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.BadHy4VisionPlan, Plan.build(a, &.{.{ .at = 0, .tokens = 5, .gh = 4, .gw = 4, .hash = 1 }}, 8));
    try std.testing.expectError(error.BadHy4VisionPlan, Plan.build(a, &.{.{ .at = 6, .tokens = 6, .gh = 4, .gw = 4, .hash = 1 }}, 8));
    try std.testing.expectError(error.BadHy4VisionPlan, Plan.build(a, &.{
        .{ .at = 0, .tokens = 6, .gh = 4, .gw = 4, .hash = 1 },
        .{ .at = 3, .tokens = 6, .gh = 4, .gw = 4, .hash = 2 },
    }, 16));
    // An odd grid is not a merged one.
    try std.testing.expectError(error.BadHy4VisionPlan, Plan.build(a, &.{.{ .at = 0, .tokens = 6, .gh = 3, .gw = 4, .hash = 1 }}, 8));
}

/// The Engram media mask a pass writes: 1 on a row the tower filled (the branch leaves its residual alone), 0 on a
/// text row. The kernel reads the mask row on every pass, so a vision-loaded model writes it on every pass — a text
/// pass after an image pass must hand the scratch all zeros (`cuda_model.passRuns`).
pub fn mediaMask(sources: []const i32, out: []u32) void {
    std.debug.assert(out.len >= sources.len);
    for (sources, out[0..sources.len]) |src, *keep| keep.* = @intFromBool(src >= 0);
}

test "a text pass after an image pass writes the media mask all zero" {
    const a = std.testing.allocator;
    // One 4x4-patch image at row 1 of a 10-row prompt: rows 1..6 read the tower, and text follows.
    var plan = try Plan.build(a, &.{.{ .at = 1, .tokens = 6, .gh = 4, .gw = 4, .hash = 3 }}, 10);
    defer plan.deinit();
    var sources: [4]i32 = undefined;
    var positions: [12]i32 = undefined;
    var media: [4]u32 = undefined;
    // The pass that holds the image: the mask is 1 on its rows alone, and a text row keeps 0.
    plan.fill(1, 4, &positions, &sources);
    mediaMask(&sources, &media);
    try std.testing.expectEqualSlices(u32, &.{ 1, 1, 1, 1 }, &media);
    plan.fill(0, 4, &positions, &sources);
    mediaMask(&sources, &media);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 1, 1 }, &media);
    // The next pass is text only (a drafting verify row, another lane's prompt, a later chunk): the same buffer, and
    // the mask the scratch still held is all zero.
    @memset(&sources, -1);
    mediaMask(&sources, &media);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 0, 0 }, &media);
}

/// One tower layer's device weights.
pub const Layer = struct {
    ln1: u64,
    ln1_bias: u64,
    ln2: u64,
    ln2_bias: u64,
    qkv: u64, // q, k and v's rows side by side: the attention kernel's own row layout
    qkv_bias: u64,
    out: u64,
    out_bias: u64,
    up: u64,
    up_bias: u64,
    down: u64,
    down_bias: u64,
};

/// The tower's per-image scratch: one image's rows at a time, and the token-wise GEMMs' block.
pub const Scratch = struct {
    patches: cuda.DeviceBuffer, // the raw patch rows, BF16
    pos: cuda.DeviceBuffer, // their interpolated position rows, then the sum
    attention: cuda.DeviceBuffer,
    h: u64, // a LayerNorm's rows
    qkv: u64,
    tmp: u64, // a fused-bias GEMM's own output, when the residual is in place
    ff: u64, // the MLP's 4304-wide rows
    acc: u64, // f32 rows: the GEMM fallback's accumulator
    merged: u64, // the unshuffled 4608-wide rows
    proj: u64, // the projector's H-wide rows
    newline: u64, // one row: `image_newline`, copied into place at the end of every merged row
};

/// The tower's weights, its per-image scratch and the host position table, read once at load.
pub const Tower = struct {
    a: std.mem.Allocator,
    d: *const cuda.Driver,
    s: cuda.Stream,
    gemms: *cuda.vit.Gemms,
    glue: *cuda.vit.Kernels,
    attention: *cuda.vit.Attention,
    norms: *const cuda.norms.Norms,
    cfg: vc.Config,
    layers: []Layer,
    patch: u64,
    patch_bias: u64,
    /// The position table, owned: the checkpoint's mappings are released layer by layer, its rows are read a image.
    table: []u16,
    newline: u64,
    perceive: struct { norm: u64, merge: u64, merge_bias: u64, out: u64, out_bias: u64 },
    scratch: Scratch,
    /// Patches one image may take: four a merged token, the bound the scratch is sized for.
    patches_max: usize,
    /// A debugging run's stage dumps (the boundary check narrows a mismatch to one stage): null in a server.
    stages: ?Stages = null,
    /// Device bytes the tower's weights take, read at load so a size report counts it.
    bytes: usize,

    /// Loads `vit.*` from `ck` for images of at most `tokens` LM tokens (`patches_max` = 4 x tokens patches).
    pub fn load(a: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, gemms: *cuda.vit.Gemms, glue: *cuda.vit.Kernels, attention: *cuda.vit.Attention, norms: *const cuda.norms.Norms, ck: *core.Checkpoint, own: *cuda.exl3_upload.Uploader, cfg: vc.Config, tokens: u32) !*Tower {
        if (cfg.hidden != hidden or cfg.heads != heads or cfg.intermediate != inner or cfg.out_hidden != H or cfg.depth == 0)
            return error.UnsupportedHy4Vision;
        if (tokens == 0 or tokens > token_limit) return error.BadHy4VisionLimits;
        const self = try a.create(Tower);
        errdefer a.destroy(self);
        const layers = try a.alloc(Layer, cfg.depth);
        errdefer a.free(layers);
        const mapped = try ck.get("vit.embeddings.position_embedding.weight");
        if (!mapped.is(.bf16, &.{ vc.num_positions, hidden })) return error.BadHy4CheckpointTensor;
        const table = try a.alloc(u16, vc.num_positions * hidden);
        errdefer a.free(table);
        for (table, 0..) |*v, i| v.* = std.mem.readInt(u16, mapped.bytes[i * 2 ..][0..2], .little);
        self.* = .{
            .a = a,
            .d = d,
            .s = s,
            .gemms = gemms,
            .glue = glue,
            .attention = attention,
            .norms = norms,
            .cfg = cfg,
            .layers = layers,
            .patch = try take(own, ck, "vit.embeddings.", "patch_embedding.weight", .bf16, &.{ hidden, 3, vc.patch, vc.patch }),
            .patch_bias = try take(own, ck, "vit.embeddings.", "patch_embedding.bias", .bf16, &.{hidden}),
            .table = table,
            .newline = try take(own, ck, "vit.perceive.", "image_newline", .bf16, &.{H}),
            .perceive = undefined,
            .scratch = undefined,
            .patches_max = 4 * @as(usize, tokens),
            .stages = null,
            .bytes = 0,
        };
        for (layers, 0..) |*layer, i| layer.* = try loadLayer(own, ck, i);
        self.perceive = .{
            .norm = try take(own, ck, "vit.perceive.", "norm.weight", .bf16, &.{hidden}),
            .merge = try take(own, ck, "vit.perceive.", "proj.0.weight", .bf16, &.{ merge_width, hidden, merge, merge }),
            .merge_bias = try take(own, ck, "vit.perceive.", "proj.0.bias", .bf16, &.{merge_width}),
            .out = try take(own, ck, "vit.perceive.", "proj.2.weight", .bf16, &.{ H, merge_width, 1, 1 }),
            .out_bias = try take(own, ck, "vit.perceive.", "proj.2.bias", .bf16, &.{H}),
        };
        self.scratch = try allocScratch(d, own, self.patches_max);
        self.scratch.newline = self.newline;
        self.bytes = own.bytes;
        return self;
    }

    pub fn deinit(self: *Tower) void {
        const a = self.a;
        var sc = self.scratch;
        sc.patches.free();
        sc.pos.free();
        sc.attention.free();
        a.free(self.table);
        a.free(self.layers);
        a.destroy(self);
    }

    pub const Stages = struct { io: std.Io, dir: std.Io.Dir };

    /// Writes `rows` x `width` BF16 rows of `buf` as FP32 under `name`, when this run dumps its stages.
    fn stage(self: *Tower, buf: u64, rows: usize, width: usize, name: []const u8) !void {
        const st = self.stages orelse return;
        const bits = try std.heap.page_allocator.alloc(u8, rows * width * 2);
        defer std.heap.page_allocator.free(bits);
        try (cuda.DeviceBuffer{ .d = self.d, .ptr = buf, .len = rows * width * 2, .borrowed = true }).downloadAsync(0, bits, self.s.handle);
        try self.s.synchronize();
        const floats = try std.heap.page_allocator.alloc(f32, rows * width);
        defer std.heap.page_allocator.free(floats);
        for (floats, 0..) |*f, i| f.* = readBf16(std.mem.readInt(u16, bits[i * 2 ..][0..2], .little));
        try st.dir.writeFile(st.io, .{ .sub_path = name, .data = std.mem.sliceAsBytes(floats) });
    }

    /// One image's rows into `out` (BF16, `(gh / 2) * (gw / 2 + 1)` of them, `H` wide): the reference's
    /// `Hy4FlashVisionTower.forward` for one frame, at the pack's own BF16.
    pub fn encode(self: *Tower, patches: []const f32, gh: u32, gw: u32, out: u64) !void {
        const count = @as(usize, gh) * gw;
        if (count == 0 or count > self.patches_max or gh % merge != 0 or gw % merge != 0) return error.BadHy4VisionGrid;
        if (patches.len != count * patch_width) return error.BadHy4VisionPatches;
        const s = self.s;
        const sc = self.scratch;
        const tmp = std.heap.page_allocator;
        // Patches up, as BF16; then the patch embedding (the 16 x 16 stride-16 conv, a Linear over raster patches) and
        // the interpolated position table added in BF16, as `emb + pos` is in torch.
        const up = try tmp.alloc(u16, patches.len);
        defer tmp.free(up);
        for (patches, up) |v, *w| w.* = bf16(v);
        try sc.patches.uploadAsync(0, std.mem.sliceAsBytes(up), s.handle);
        try cuda.vit.linear(self.gemms, self.glue, s, sc.acc, sc.patches.ptr, self.patch, self.patch_bias, sc.h, count, hidden, patch_width);
        try self.stage(sc.h, count, hidden, "patchemb.f32");
        const pos = try self.positionRows(gh, gw);
        defer tmp.free(pos);
        try sc.pos.uploadAsync(0, std.mem.sliceAsBytes(pos), s.handle);
        try self.stage(sc.pos.ptr, count, hidden, "pos.f32");
        // x lives in `sc.pos` from here: the patch rows plus their position rows, rounded once, as torch's `emb + pos`.
        try self.glue.pointwise(s, sc.h, sc.pos.ptr, sc.pos.ptr, count * hidden, .none);
        try self.stage(sc.pos.ptr, count, hidden, "embed.f32");
        // 27 x (LayerNorm, full non-causal attention, LayerNorm, GELU MLP). The attention is one frame: every patch of
        // this image over every other, which is the reference's cu_seqlens segment for it.
        for (self.layers, 0..) |layer, layer_index| {
            try self.glue.layerNorm(s, sc.pos.ptr, layer.ln1, layer.ln1_bias, sc.h, count, hidden, self.cfg.eps);
            if (layer_index == 0) try self.stage(sc.h, count, hidden, "l0-norm.f32");
            try cuda.vit.linear(self.gemms, self.glue, s, sc.acc, sc.h, layer.qkv, layer.qkv_bias, sc.qkv, count, 3 * hidden, hidden);
            try self.attention.full(s, sc.qkv, sc.attention.ptr, count, heads, dim, 1.0 / @sqrt(@as(f32, dim)));
            if (layer_index == 0) try self.stage(sc.attention.ptr, count, hidden, "l0-attn.f32");
            try cuda.vit.linearAct(self.gemms, self.glue, s, sc.acc, sc.attention.ptr, layer.out, layer.out_bias, sc.tmp, count, hidden, hidden, sc.pos.ptr, sc.pos.ptr, .none);
            try self.glue.layerNorm(s, sc.pos.ptr, layer.ln2, layer.ln2_bias, sc.h, count, hidden, self.cfg.eps);
            var at: usize = 0;
            while (at < count) : (at += block_rows) {
                const n = @min(block_rows, count - at);
                const row = sc.pos.ptr + at * hidden * 2;
                try cuda.vit.linearAct(self.gemms, self.glue, s, sc.acc, sc.h + at * hidden * 2, layer.up, layer.up_bias, sc.ff, n, inner, hidden, null, sc.ff, .gelu);
                try cuda.vit.linearAct(self.gemms, self.glue, s, sc.acc, sc.ff, layer.down, layer.down_bias, sc.tmp, n, hidden, inner, row, row, .none);
            }
            if (layer_index == 0) try self.stage(sc.pos.ptr, count, hidden, "l0.f32") else if (layer_index == 1) try self.stage(sc.pos.ptr, count, hidden, "l1.f32");
        }
        try self.perceiveRows(gh, gw, count, out);
    }

    /// The projector: RMSNorm, the 2 x 2 stride-2 merge conv (a pixel unshuffle into a Linear), exact GELU, the 1 x 1
    /// conv, then one learned newline row closing every merged row, in raster order.
    fn perceiveRows(self: *Tower, gh: u32, gw: u32, count: usize, out: u64) !void {
        const s = self.s;
        const sc = self.scratch;
        const mh: usize = gh / merge;
        const mw: usize = gw / merge;
        const merged = mh * mw;
        // The reference normalizes in float32, casts back, then scales: MLX's bf16 steps are that exactly.
        try self.norms.addRmsTail(s, sc.pos.ptr, null, self.perceive.norm, sc.h, sc.h, null, count, hidden, self.cfg.eps, .{ .round = true });
        try self.glue.unshuffle(s, sc.h, sc.qkv, gh, gw, hidden);
        try self.glue.convFlatten(s, sc.qkv, sc.merged, merged, merge * merge, 1, hidden);
        var at: usize = 0;
        while (at < merged) : (at += block_rows) {
            const n = @min(block_rows, merged - at);
            try cuda.vit.linearAct(self.gemms, self.glue, s, sc.acc, sc.merged + at * merge_width * 2, self.perceive.merge, self.perceive.merge_bias, sc.ff, n, merge_width, merge_width, null, sc.ff, .gelu);
            try cuda.vit.linear(self.gemms, self.glue, s, sc.acc, sc.ff, self.perceive.out, self.perceive.out_bias, sc.proj + at * H * 2, n, H, merge_width);
        }
        try self.stage(sc.proj, merged, H, "proj.f32");
        // Every merged row's own row goes where the prompt's placeholders are, the newline vector one past its end.
        const dst = cuda.DeviceBuffer{ .d = self.d, .ptr = out, .len = mh * (mw + 1) * H * 2, .borrowed = true };
        var oh: usize = 0;
        while (oh < mh) : (oh += 1) {
            try dst.copyFrom((oh * (mw + 1)) * H * 2, sc.proj + oh * mw * H * 2, mw * H * 2, s.handle);
            try dst.copyFrom((oh * (mw + 1) + mw) * H * 2, sc.newline, H * 2, s.handle);
        }
    }

    /// The learned 128 x 128 position table bilinearly resampled to `(gh, gw)`: one BF16 row a patch, `hidden` wide.
    pub fn positionRows(self: *Tower, gh: u32, gw: u32) ![]u16 {
        const total = @as(usize, gh) * gw;
        const tmp = std.heap.page_allocator;
        const taps = try tmp.alloc(i32, total * 4);
        defer tmp.free(taps);
        const wts = try tmp.alloc(f32, total * 4);
        defer tmp.free(wts);
        vc.positions(gh, gw, taps, wts);
        const out = try tmp.alloc(u16, total * hidden);
        errdefer tmp.free(out);
        for (0..total) |p| {
            const o = out[p * hidden ..][0..hidden];
            var acc: [hidden]f32 = undefined;
            @memset(&acc, 0);
            for (0..4) |t| {
                const w = wts[p * 4 + t];
                if (w == 0) continue;
                // `vc.positions` indexes the 128 x 128 grid; the checkpoint's row 0 is the unused cls slot, so the
                // grid's row `g` is the table's row `g + 1` (the reference embeds `arange(1, edge^2 + 1)`).
                const g: usize = @intCast(taps[p * 4 + t]);
                const row = self.table[(g + 1) * hidden ..][0..hidden];
                for (&acc, row) |*v, x| v.* += w * readBf16(x);
            }
            for (o, acc) |*v, x| v.* = bf16(x);
        }
        return out;
    }
};

fn loadLayer(own: *cuda.exl3_upload.Uploader, ck: *core.Checkpoint, i: usize) !Layer {
    var pb: [72]u8 = undefined;
    const p = try std.fmt.bufPrint(&pb, "vit.layers.{d}.", .{i});
    // q, k and v side by side: the attention kernel reads one row of [3, heads, dim] a token.
    const width = 3 * hidden;
    const bytes = hidden * hidden * 2;
    const both = try std.heap.page_allocator.alloc(u8, bytes * 3);
    defer std.heap.page_allocator.free(both);
    for ([_][]const u8{ "q_proj.weight", "k_proj.weight", "v_proj.weight" }, 0..) |suffix, k| {
        var nb: [96]u8 = undefined;
        const t = try ck.get(try std.fmt.bufPrint(&nb, "{s}self_attn.{s}", .{ p, suffix }));
        if (!t.is(.bf16, &.{ hidden, hidden })) return error.BadHy4CheckpointTensor;
        @memcpy(both[k * bytes ..][0..bytes], t.bytes);
    }
    const bias = try std.heap.page_allocator.alloc(u8, width * 2);
    defer std.heap.page_allocator.free(bias);
    for ([_][]const u8{ "q_proj.bias", "k_proj.bias", "v_proj.bias" }, 0..) |suffix, k| {
        var nb: [96]u8 = undefined;
        const t = try ck.get(try std.fmt.bufPrint(&nb, "{s}self_attn.{s}", .{ p, suffix }));
        if (!t.is(.bf16, &.{hidden})) return error.BadHy4CheckpointTensor;
        @memcpy(bias[k * hidden * 2 ..][0 .. hidden * 2], t.bytes);
    }
    return .{
        .ln1 = try take(own, ck, p, "input_layernorm.weight", .bf16, &.{hidden}),
        .ln1_bias = try take(own, ck, p, "input_layernorm.bias", .bf16, &.{hidden}),
        .ln2 = try take(own, ck, p, "post_attention_layernorm.weight", .bf16, &.{hidden}),
        .ln2_bias = try take(own, ck, p, "post_attention_layernorm.bias", .bf16, &.{hidden}),
        .qkv = try own.dev(both),
        .qkv_bias = try own.dev(bias),
        .out = try take(own, ck, p, "self_attn.o_proj.weight", .bf16, &.{ hidden, hidden }),
        .out_bias = try take(own, ck, p, "self_attn.o_proj.bias", .bf16, &.{hidden}),
        .up = try take(own, ck, p, "mlp.dense_h_to_4h.weight", .bf16, &.{ inner, hidden }),
        .up_bias = try take(own, ck, p, "mlp.dense_h_to_4h.bias", .bf16, &.{inner}),
        .down = try take(own, ck, p, "mlp.dense_4h_to_h.weight", .bf16, &.{ hidden, inner }),
        .down_bias = try take(own, ck, p, "mlp.dense_4h_to_h.bias", .bf16, &.{hidden}),
    };
}

fn take(own: *cuda.exl3_upload.Uploader, ck: *core.Checkpoint, prefix: []const u8, name: []const u8, dtype: anytype, shape: []const usize) !u64 {
    var nb: [160]u8 = undefined;
    const full = try std.fmt.bufPrint(&nb, "{s}{s}", .{ prefix, name });
    const t = try ck.get(full);
    if (!t.is(dtype, shape)) {
        std.debug.print("unexpected tensor {s}: {any} {any}\n", .{ full, t.dtype, t.shape[0..t.rank] });
        return error.BadHy4CheckpointTensor;
    }
    return own.dev(t.bytes);
}

/// The tower's per-image scratch: one image's rows at a time, and the token-wise GEMMs' block.
fn allocScratch(d: *const cuda.Driver, own: *cuda.exl3_upload.Uploader, patches_max: usize) !Scratch {
    const p = patches_max;
    const b: usize = @min(block_rows, patches_max);
    const merges = p / (merge * merge);
    var sc: Scratch = undefined;
    sc.patches = try cuda.DeviceBuffer.alloc(d, p * patch_width * 2);
    sc.pos = try cuda.DeviceBuffer.alloc(d, p * hidden * 2);
    sc.attention = try cuda.DeviceBuffer.alloc(d, p * hidden * 2);
    sc.h = try zeros(own, p * hidden * 2);
    sc.qkv = try zeros(own, p * 3 * hidden * 2);
    sc.tmp = try zeros(own, p * hidden * 2);
    sc.ff = try zeros(own, b * inner * 2);
    // The fallback's f32 accumulator: the unblocked GEMMs run a patch row (a hidden-wide output), the blocked ones a
    // 4608-wide merge row.
    sc.acc = try zeros(own, @max(p * hidden, b * merge_width) * 4);
    sc.merged = try zeros(own, merges * merge_width * 2);
    sc.proj = try zeros(own, merges * H * 2);
    sc.newline = 0; // filled by `Tower.load` once `image_newline` is up
    return sc;
}

fn zeros(own: *cuda.exl3_upload.Uploader, bytes: usize) !u64 {
    const z = try std.heap.page_allocator.alloc(u8, bytes);
    defer std.heap.page_allocator.free(z);
    @memset(z, 0);
    return own.dev(z);
}

fn readBf16(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

fn bf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if ((bits & 0x7f800000) == 0x7f800000 and (bits & 0x007fffff) != 0) return @intCast((bits >> 16) | 0x0040);
    const half = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
    return @intCast(half & 0xffff);
}

test "a bf16 word rounds to nearest even, and reads back as its own value" {
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(1.0));
    try std.testing.expectEqual(@as(u16, 0xbf80), bf16(-1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(1.0 + 0.00390625));
    try std.testing.expectEqual(@as(u16, 0x3f81), bf16(1.0 + 0.0078125));
    try std.testing.expectEqual(@as(f32, 1.5), readBf16(bf16(1.5)));
}
