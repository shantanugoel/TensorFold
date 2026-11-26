//! Hy4-flash's DFly2 drafter on CUDA: five gated-MLA layers over the target's own caches and recalled lists.
//!
//! The algorithm is [hy4-dfly2-port](../../../../../../docs/kb/hy4-dfly2-port.md)'s, the one `metal_draft.zig`
//! serves on Metal: one forward of the anchor row and five mask rows, then one correction and one head call a slot.
//! Two shares make the five layers small. Each binds a target indexed layer's cache (`dfly_target_kv_layers`)
//! instead of keeping its own, reading the target's recall list for the kept row instead of running an indexer;
//! and each writes its block's rows into that shared cache at the anchor's positions, which the verification
//! forward overwrites, so nothing the drafter stores outlives the round it was written in.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const config = @import("config.zig");
const geom = @import("cuda_scratch.zig");
const mla = @import("mla_cuda.zig");
const m = @import("mla_weights.zig");
const attn = @import("mla_attention.zig");
const rope = @import("rope.zig");

const H = geom.H;
const Q = geom.Q;
/// The block's rows: the anchor row and the checkpoint's five drafts (`num_speculative_tokens`).
pub const width = 6;
pub const drafts = width - 1;
/// A draft row's keys at most: the target's 2,048 recalled slots and the block's own six rows.
pub const key_stride = geom.sparse_stride + width;
/// Draft layers: the checkpoint's `dfly_num_hidden_layers`.
pub const layers = 5;

const Model = @import("cuda_model.zig").Model;

/// One draft layer: a target indexed layer's attention widths, without an indexer and without hyper-connections.
pub const Layer = struct {
    target_kv: usize, // the target layer whose cache this layer reads and writes
    target_index: usize, // the target layer whose recalled list it reads
    input_norm: u64,
    post_norm: u64,
    q_a: u64, // BF16 [q_lora_rank, hidden]
    q_norm: u64,
    q_b: u64, // BF16 [heads * 256, q_lora_rank]
    kv_a: u64, // BF16 [512 + 64, hidden]
    kv_norm: u64,
    key: u64, // the pack's kv_b split per head, as the target's own attention holds it
    value: u64,
    gate: u64, // BF16 [heads * value_width, hidden]
    output: u64, // BF16 [hidden, heads * value_width]
    output_norm: u64,
    sinks: u64, // the pack's BF16 sink, as F32 for the merge kernel
    mlp_gate: u64, // BF16 [inner, hidden]
    mlp_up: u64,
    mlp_down: u64, // BF16 [hidden, inner]
};

/// The drafter's own kernels: the two family ops beside the shared argmax.
pub const Kernels = struct {
    image: cuda.Module,
    fuse: cuda.Function,
    add: cuda.Function,
    argmax_image: cuda.Module,
    argmax: cuda.Function,

    pub fn load(driver: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var image = try cuda.Module.load(driver, cuda.kernels.hy4_draft);
        errdefer image.unload();
        var argmax_image = try cuda.Module.load(driver, cuda.kernels.torch_argmax);
        errdefer argmax_image.unload();
        return .{
            .image = image,
            .fuse = try image.function("tf_hy4_draft_fuse"),
            .add = try image.function("tf_hy4_draft_add"),
            .argmax_image = argmax_image,
            .argmax = try argmax_image.function("tf_argmax_rows_i32_kernel"),
        };
    }

    pub fn deinit(k: *Kernels) void {
        k.argmax_image.unload();
        k.image.unload();
        k.* = undefined;
    }

    /// Slot 0 of the block: the anchor's own embedding plus the fusion gate's share of the projection.
    fn fuseRow(k: Kernels, s: cuda.Stream, token: u64, fused: u64, gate: u64, out: u64, wide: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ token, fused, gate, out }) |v| a.add(v);
        a.add(@as(i32, @intCast(wide)));
        try cuda.launch.launch(k.fuse, .{ .grid = .{ .x = @intCast((wide + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
    }

    /// out = bf16(branch + residual) over `count` values.
    fn addRows(k: Kernels, s: cuda.Stream, branch: u64, residual: u64, out: u64, count: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ branch, residual, out }) |v| a.add(v);
        a.add(@as(i64, @intCast(count)));
        try cuda.launch.launch(k.add, .{ .grid = .{ .x = @intCast((count + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
    }

    /// Each row's first maximum over `vocab` F32 logits (torch's argmax order).
    fn argmaxRows(k: Kernels, s: cuda.Stream, logits: u64, rows: usize, vocab: usize, ids: u64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ logits, ids }) |v| a.add(v);
        for ([_]usize{ rows, vocab, vocab }) |v| a.add(@as(i64, @intCast(v)));
        a.add(@as(i32, 1)); // fp32 logits
        try cuda.launch.launch(k.argmax, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = 1024 } }, s, &a);
    }
};

/// A proposal's inputs and outputs, and the block's scratch, sized once at load.
pub const Scratch = struct {
    streams: [2]cuda.DeviceBuffer, // the block's residual, BF16 [width, hidden], rotated
    normed: u64, // BF16 [width, hidden]: a layer's input norm, and the block's final-normed rows
    norm_resid: u64, // BF16 [width, hidden]: what a norm-only call leaves as its residual (never read)
    q_raw: u64, // BF16 [width, q_lora_rank]
    q_tmp: u64,
    q_resid: u64,
    q: u64, // BF16 [width, heads, 256]
    kv: u64, // BF16 [width, 512 + 64]
    gate: u64, // BF16 [width, heads, value_width]
    absorb: u64, // BF16 [width, heads, 512 + 64]
    latent_out: u64,
    values: u64, // BF16 [width, heads, value_width]
    branch: u64, // BF16 [width, hidden]: a layer's attention or feed-forward output
    attn_scratch: u64,
    attn_bytes: usize,
    positions: cuda.DeviceBuffer, // i32 [width, 3]
    table: u64, // BF16 [width, 32, 2]
    write_slots: cuda.DeviceBuffer, // u32 [width]: the anchor's positions, which the verification rewrites
    slots: cuda.DeviceBuffer, // u32 [layers, width, key_stride]: each row's own attended set, a list a layer
    counts: cuda.DeviceBuffer, // u32 [layers, width]
    ff_gate: u64, // BF16 [width, inner]
    ff_up: u64,
    ff_inner: u64,
    down: u64, // BF16 [width, hidden]
    fused: u64, // BF16 [hidden]: mtp_fusion's projection, before the gate
    cat: u64, // BF16 [2, hidden]: [normed row, normed token] read as one 2 * hidden row
    c_gate: u64, // BF16 [hidden]
    c_up: u64,
    c_act: u64,
    c_down: u64,
    corrected: u64, // BF16 [hidden]
    previous: cuda.DeviceBuffer, // BF16 [hidden]: the token the next slot's correction reads
    logits: u64, // F32 [vocab]
    ids: cuda.DeviceBuffer, // i32 [1]
    /// The five target layers' recall lists as the target's passes wrote them, a block row each at most.
    lists: [layers]cuda.DeviceBuffer,
    captured: [layers]cuda.DeviceBuffer, // u32 [width] a layer: the counts beside `lists`
    /// The last pass's captured rows and the row its anchor sits in (a prompt chunk keeps only its last row).
    rows: usize = 0,
    anchor: usize = 0,
    /// Whether the last pass's rows attended the indexer's recall lists at all.
    selected: bool = false,
};

/// One round's interior, for the same-input differential against another engine (`draft_check.zig`'s dump mode):
/// the fused anchor row, the block's six final-normed rows, and each slot's corrected hidden and logits row.
/// A harness reads it; a served round leaves `trail` null and pays nothing.
pub const Trail = struct {
    fused: cuda.DeviceBuffer, // BF16 [H]: row 0 after the fusion
    carry: cuda.DeviceBuffer, // BF16 [H]: the target's post-final-norm row the fusion read
    cat: cuda.DeviceBuffer, // BF16 [2H]: the fusion's [normed anchor, normed carry]
    proj: cuda.DeviceBuffer, // BF16 [H]: eh_proj of that, before the gate
    rows: cuda.DeviceBuffer, // BF16 [width, H]
    corrected: cuda.DeviceBuffer, // BF16 [width - 1, H]
    cat1: cuda.DeviceBuffer, // BF16 [2H]: the first slot's correction concat
    delta: cuda.DeviceBuffer, // BF16 [H]: the first slot's down_proj output
    logits: cuda.DeviceBuffer, // F32 [width - 1, vocab]
};

pub const Draft = struct {
    layers: []Layer,
    inner: usize,
    heads: u32,
    final_norm: u64,
    // mtp_fusion: slot 0 is the anchor's raw embedding plus the gate's share of this projection.
    enorm: u64,
    hnorm: u64,
    eh: u64, // BF16 [hidden, 2 * hidden]
    fuse_gate: u64,
    // hidden_correction: h + down(silu(gate(cat)) * up(cat)) over [normed hidden, normed token].
    c_hnorm: u64,
    c_enorm: u64,
    c_gate: u64, // BF16 [hidden, 2 * hidden]
    c_up: u64,
    c_down: u64, // BF16 [hidden, hidden]
    mask: u32,
    rope: rope.Parameters,
    scratch: Scratch,
    kernels: Kernels,
    /// A round's key lists (a list a layer), built on the host as Metal's are, and one row's five captured lists.
    host_keys: []u32,
    host_counts: []u32,
    host_captured: []u32, // a layer's captured count for the source row
    host_lists: []i32,
    /// Set by a harness that wants one round's interior; null in service.
    trail: ?*Trail = null,
};

fn alloc(d: *const cuda.Driver, len: usize) !cuda.DeviceBuffer {
    var b = try cuda.DeviceBuffer.alloc(d, len);
    errdefer b.free();
    try b.fill8(0, null);
    return b;
}

fn zeros(d: *const cuda.Driver, len: usize) !u64 {
    return (try alloc(d, len)).ptr;
}

/// A BF16 tensor of the pack, exactly as `checkDraft` (weights.zig) has already admitted it.
fn take(own: *cuda.exl3_upload.Uploader, ck: *core.Checkpoint, comptime fmt: []const u8, args: anytype, shape: []const usize) !u64 {
    var nb: [192]u8 = undefined;
    const name = try std.fmt.bufPrint(&nb, fmt, args);
    const t = try ck.get(name);
    if (!t.is(.bf16, shape)) {
        std.debug.print("unexpected draft tensor {s}: {any} {any}\n", .{ name, t.dtype, t.shape[0..t.rank] });
        return error.BadHy4CheckpointTensor;
    }
    return own.dev(t.bytes);
}

/// `count` BF16 values as F32 (the merge kernel's sinks are the target attention's dtype).
fn widen(a: std.mem.Allocator, raw: []const u8, count: usize) ![]f32 {
    const out = try a.alloc(f32, count);
    for (out, 0..) |*x, i| x.* = @bitCast(@as(u32, std.mem.readInt(u16, raw[2 * i ..][0..2], .little)) << 16);
    return out;
}

/// The checkpoint's `model.dfly_layers.*` onto the device: five layers, the fusion, the correction and the head.
pub fn load(a: std.mem.Allocator, own: *cuda.exl3_upload.Uploader, d: *const cuda.Driver, ck: *core.Checkpoint, cfg: config.Model) !Draft {
    const t = cfg.text_config;
    if (t.hidden_size != H or t.q_lora_rank != Q or t.kv_lora_rank != m.latent) return error.BadHy4ModelShape;
    if (cfg.dfly_num_hidden_layers != layers or cfg.dfly_target_kv_layers.len != layers or cfg.dfly_target_index_layer.len != layers)
        return error.BadDraftGeometry;
    const inner: usize = cfg.dfly_intermediate_size orelse 3 * H;
    const heads: u32 = t.num_attention_heads;
    const hh: usize = heads;
    const channels = hh * m.value_width;
    const p = "model.dfly_layers.";
    const tmp = std.heap.page_allocator;
    const layer_weights = try a.alloc(Layer, layers);
    errdefer a.free(layer_weights);
    for (layer_weights, 0..) |*l, i| {
        var kv_b_buf: [64]u8 = undefined;
        const kv_b_name = try std.fmt.bufPrint(&kv_b_buf, p ++ "{d}.self_attn.kv_b_proj.weight", .{i});
        const kv_b = try ck.get(kv_b_name);
        if (!kv_b.is(.bf16, &.{ hh * (m.nope + m.value_width), m.latent })) return error.BadHy4CheckpointTensor;
        // The shared cache reads a draft layer through the target's own head-local split; the values are the pack's.
        const split = try m.splitBf16(tmp, kv_b.bytes, heads);
        defer tmp.free(split.key);
        defer tmp.free(split.value);
        var sink_buf: [72]u8 = undefined;
        const sink = try ck.get(try std.fmt.bufPrint(&sink_buf, p ++ "{d}.self_attn.learnable_sink_param", .{i}));
        if (!sink.is(.bf16, &.{hh})) return error.BadHy4CheckpointTensor;
        const sinks = try widen(tmp, sink.bytes, hh);
        defer tmp.free(sinks);
        l.* = .{
            .target_kv = cfg.dfly_target_kv_layers[i],
            .target_index = cfg.dfly_target_index_layer[i],
            .input_norm = try take(own, ck, p ++ "{d}.input_layernorm.weight", .{i}, &.{H}),
            .post_norm = try take(own, ck, p ++ "{d}.post_attention_layernorm.weight", .{i}, &.{H}),
            .q_a = try take(own, ck, p ++ "{d}.self_attn.q_a_proj.weight", .{i}, &.{ Q, H }),
            .q_norm = try take(own, ck, p ++ "{d}.self_attn.q_a_layernorm.weight", .{i}, &.{Q}),
            .q_b = try take(own, ck, p ++ "{d}.self_attn.q_b_proj.weight", .{i}, &.{ hh * m.query_stride, Q }),
            .kv_a = try take(own, ck, p ++ "{d}.self_attn.kv_a_proj_with_mqa.weight", .{i}, &.{ m.absorbed_stride, H }),
            .kv_norm = try take(own, ck, p ++ "{d}.self_attn.kv_a_layernorm.weight", .{i}, &.{m.latent}),
            .key = try own.dev(std.mem.sliceAsBytes(split.key)),
            .value = try own.dev(std.mem.sliceAsBytes(split.value)),
            .gate = try take(own, ck, p ++ "{d}.self_attn.linear_gate.weight", .{i}, &.{ channels, H }),
            .output = try take(own, ck, p ++ "{d}.self_attn.o_proj.weight", .{i}, &.{ H, channels }),
            .output_norm = try take(own, ck, p ++ "{d}.self_attn.post_proj_norm.weight", .{i}, &.{H}),
            .sinks = try own.dev(std.mem.sliceAsBytes(sinks)),
            .mlp_gate = try take(own, ck, p ++ "{d}.mlp.gate_proj.weight", .{i}, &.{ inner, H }),
            .mlp_up = try take(own, ck, p ++ "{d}.mlp.up_proj.weight", .{i}, &.{ inner, H }),
            .mlp_down = try take(own, ck, p ++ "{d}.mlp.down_proj.weight", .{i}, &.{ H, inner }),
        };
    }
    var sc: Scratch = undefined;
    for (&sc.streams) |*b| b.* = try alloc(d, width * H * 2);
    sc.normed = try zeros(d, width * H * 2);
    sc.norm_resid = try zeros(d, width * H * 2);
    sc.q_raw = try zeros(d, width * Q * 2);
    sc.q_tmp = try zeros(d, width * Q * 2);
    sc.q_resid = try zeros(d, width * Q * 2);
    sc.q = try zeros(d, width * hh * m.query_stride * 2);
    sc.kv = try zeros(d, width * m.absorbed_stride * 2);
    sc.gate = try zeros(d, width * channels * 2);
    sc.absorb = try zeros(d, width * hh * m.absorbed_stride * 2);
    sc.latent_out = try zeros(d, width * hh * m.latent * 2);
    sc.values = try zeros(d, width * channels * 2);
    sc.branch = try zeros(d, width * H * 2);
    const shape = try attn.Shape.init(width, heads, key_stride);
    sc.attn_bytes = shape.scratchBytes();
    sc.attn_scratch = try zeros(d, sc.attn_bytes);
    sc.positions = try alloc(d, width * 12);
    sc.table = try zeros(d, width * 64 * 2);
    sc.write_slots = try alloc(d, width * 4);
    sc.slots = try alloc(d, layers * width * key_stride * 4);
    sc.counts = try alloc(d, layers * width * 4);
    sc.ff_gate = try zeros(d, width * inner * 2);
    sc.ff_up = try zeros(d, width * inner * 2);
    sc.ff_inner = try zeros(d, width * inner * 2);
    sc.down = try zeros(d, width * H * 2);
    sc.fused = try zeros(d, H * 2);
    sc.cat = try zeros(d, 2 * H * 2);
    sc.c_gate = try zeros(d, H * 2);
    sc.c_up = try zeros(d, H * 2);
    sc.c_act = try zeros(d, H * 2);
    sc.c_down = try zeros(d, H * 2);
    sc.corrected = try zeros(d, H * 2);
    sc.previous = try alloc(d, H * 2);
    sc.logits = try zeros(d, @as(usize, t.vocab_size) * 4);
    sc.ids = try alloc(d, 4);
    for (&sc.lists) |*b| b.* = try alloc(d, width * geom.sparse_stride * 4);
    for (&sc.captured) |*b| b.* = try alloc(d, width * 4);
    const host_keys = try a.alloc(u32, layers * width * key_stride);
    errdefer a.free(host_keys);
    const host_counts = try a.alloc(u32, layers * width);
    errdefer a.free(host_counts);
    const host_lists = try a.alloc(i32, layers * geom.sparse_stride);
    errdefer a.free(host_lists);
    const host_captured = try a.alloc(u32, layers);
    errdefer a.free(host_captured);
    return .{
        .layers = layer_weights,
        .inner = inner,
        .heads = heads,
        .final_norm = try take(own, ck, p ++ "final_norm.weight", .{}, &.{H}),
        .enorm = try take(own, ck, p ++ "mtp_fusion.enorm.weight", .{}, &.{H}),
        .hnorm = try take(own, ck, p ++ "mtp_fusion.hnorm.weight", .{}, &.{H}),
        .eh = try take(own, ck, p ++ "mtp_fusion.eh_proj.weight", .{}, &.{ H, 2 * H }),
        .fuse_gate = try take(own, ck, p ++ "mtp_fusion.gate", .{}, &.{H}),
        .c_hnorm = try take(own, ck, p ++ "hidden_correction.hidden_norm.weight", .{}, &.{H}),
        .c_enorm = try take(own, ck, p ++ "hidden_correction.embed_norm.weight", .{}, &.{H}),
        .c_gate = try take(own, ck, p ++ "hidden_correction.gate_proj.weight", .{}, &.{ H, 2 * H }),
        .c_up = try take(own, ck, p ++ "hidden_correction.up_proj.weight", .{}, &.{ H, 2 * H }),
        .c_down = try take(own, ck, p ++ "hidden_correction.down_proj.weight", .{}, &.{ H, H }),
        .mask = cfg.dfly_mask_token_id,
        .rope = try t.rotary(.deepseek_sparse_attention),
        .scratch = sc,
        .kernels = try Kernels.load(d),
        .host_keys = host_keys,
        .host_counts = host_counts,
        .host_captured = host_captured,
        .host_lists = host_lists,
    };
}

pub fn free(a: std.mem.Allocator, d: *Draft) void {
    const sc = d.scratch;
    for ([_]cuda.DeviceBuffer{ sc.streams[0], sc.streams[1], sc.positions, sc.write_slots, sc.slots, sc.counts, sc.ids, sc.previous } ++ sc.lists ++ sc.captured) |b| {
        var buf = b;
        buf.free();
    }
    d.kernels.deinit();
    a.free(d.host_keys);
    a.free(d.host_counts);
    a.free(d.host_captured);
    a.free(d.host_lists);
    a.free(d.layers);
}

/// The row a proposal reads the target's hidden state and recalled lists from: the last row the target kept.
/// A prompt's own first proposal reads the last row of its final pass, which the capture names for itself.
pub fn source(d: *const Draft, kept: ?usize) usize {
    return kept orelse d.scratch.anchor;
}

/// `depth` proposals after `anchor` (the last kept pass row's position plus one); the block's own rows go into
/// the target's caches at `anchor .. anchor + width`, where the verification forward rewrites them.
pub fn propose(mm: *Model, d: *Draft, anchor: usize, bonus: u32, depth: u32, row: usize, held: []u32) !void {
    const s = mm.s;
    const k = mm.k;
    const sc = &d.scratch;
    const V: usize = mm.text.vocab_size;
    const eps = mm.text.rms_norm_eps;
    const hh: usize = d.heads;
    const channels = hh * m.value_width;
    std.debug.assert(held.len == depth and depth <= drafts);
    // Positions, write slots and the rotary table for the block's six rows.
    var positions: [width * 3]i32 = undefined;
    var writes: [width]u32 = undefined;
    for (0..width) |i| {
        const at: i32 = @intCast(anchor + i);
        positions[i * 3 ..][0..3].* = .{ at, at, at };
        writes[i] = @intCast(anchor + i);
    }
    try sc.positions.uploadAsync(0, std.mem.sliceAsBytes(&positions), s.handle);
    try sc.write_slots.uploadAsync(0, std.mem.sliceAsBytes(&writes), s.handle);
    // Each layer's rows attend the target list its own target layer kept for the source row, plus its own prefix.
    try buildKeys(mm, d, anchor, row);
    try sc.slots.uploadAsync(0, std.mem.sliceAsBytes(d.host_keys), s.handle);
    try sc.counts.uploadAsync(0, std.mem.sliceAsBytes(d.host_counts), s.handle);
    // The block's embeddings: the anchor's row, then five mask rows.
    for (0..width) |i| {
        const token = if (i == 0) bonus else d.mask;
        try sc.streams[0].copyFrom(i * H * 2, mm.embed + @as(u64, token) * H * 2, H * 2, s.handle);
    }
    // mtp_fusion on slot 0: the gate's share of eh_proj over [normed anchor, normed target hidden].
    const target_hidden = mm.scratch.head_norm + row * H * 2;
    // Every norm on this path is the target's own form. `norms.rms` is the **fp32-weight** kernel and these are
    // the pack's BF16 weights, so each row goes through `addRmsTail`'s tail instead (bf16 weights, MLX's rounded
    // bf16 steps, which is what the target's layers run) with no residual to add: the residual it writes beside
    // the row is scratch. Reading a BF16 weight as f32 does not fail — it reads the second half of the row out of
    // the neighbouring tensor and returns a norm of the right scale and the wrong direction, which is exactly the
    // 60 % the differential found in this row.
    try k.norms.addRmsTail(s, sc.streams[0].ptr, null, d.enorm, sc.norm_resid, sc.cat, null, 1, H, eps, .{ .round = true });
    try k.norms.addRmsTail(s, target_hidden, null, d.hnorm, sc.norm_resid, sc.cat + H * 2, null, 1, H, eps, .{ .round = true });
    try k.gemm.run(s, sc.cat, d.eh, sc.fused, 1, H, 2 * H, .bf16);
    if (d.trail) |t| {
        try t.cat.copyFrom(0, sc.cat, 2 * H * 2, s.handle);
        try t.proj.copyFrom(0, sc.fused, H * 2, s.handle);
    }
    try d.kernels.fuseRow(s, sc.streams[0].ptr, sc.fused, d.fuse_gate, sc.streams[0].ptr, H);
    if (d.trail) |t| try t.fused.copyFrom(0, sc.streams[0].ptr, H * 2, s.handle);
    if (d.trail) |t| try t.carry.copyFrom(0, mm.scratch.head_norm + row * H * 2, H * 2, s.handle);
    // The five layers: each writes its own K/V into the target's cache and reads the target's list.
    const shape = try attn.Shape.init(width, d.heads, key_stride);
    const res = sc.streams[0].ptr;
    const spare = sc.streams[1].ptr;
    try k.norms.addRmsTail(s, res, null, d.layers[0].input_norm, sc.norm_resid, sc.normed, null, width, H, eps, .{ .round = true });
    for (d.layers, 0..) |*l, i| {
        const target = &mm.layers[l.target_kv].attention;
        const slots = sc.slots.ptr + @as(u64, i) * width * key_stride * 4;
        const counts = sc.counts.ptr + @as(u64, i) * width * 4;
        try k.gemm.run(s, sc.normed, l.q_a, sc.q_raw, width, Q, H, .bf16);
        try k.norms.addRmsTail(s, sc.q_raw, null, l.q_norm, sc.q_tmp, sc.q_resid, null, width, Q, eps, .{ .round = true });
        try k.gemm.run(s, sc.q_resid, l.q_b, sc.q, width, hh * m.query_stride, Q, .bf16);
        try k.gemm.run(s, sc.normed, l.kv_a, sc.kv, width, m.absorbed_stride, H, .bf16);
        try k.gemm.run(s, sc.normed, l.gate, sc.gate, width, channels, H, .bf16);
        try mla.forward(k.project, k.attention, s, .{
            .query = sc.q,
            .key_weights = l.key,
            .value_weights = l.value,
            .value_logits = sc.gate,
            .positions = sc.positions.ptr,
            .rotary_table = sc.table,
            .kv = .{ .projection = sc.kv, .gain = l.kv_norm, .write_slots = sc.write_slots.ptr, .capacity = target.rows, .epsilon = eps },
            .attention = .{
                .query = sc.absorb,
                .cache = target.cache.ptr,
                .slots = slots,
                .counts = counts,
                .sinks = l.sinks,
                .scratch = sc.attn_scratch,
                .scratch_bytes = sc.attn_bytes,
                .output = sc.latent_out,
            },
            .values = sc.values,
        }, shape, d.rope);
        try k.gemm.run(s, sc.values, l.output, sc.branch, width, H, channels, .bf16);
        // The sandwich norm on the attention output, the residual, then the feed-forward's input norm.
        try k.norms.addRmsTail(s, sc.branch, res, l.post_norm, spare, sc.normed, null, width, H, eps, .{ .round = true, .branch = l.output_norm });
        try k.gemm.run(s, sc.normed, l.mlp_gate, sc.ff_gate, width, d.inner, H, .bf16);
        try k.gemm.run(s, sc.normed, l.mlp_up, sc.ff_up, width, d.inner, H, .bf16);
        try k.act.swiglu(s, sc.ff_gate, d.inner, sc.ff_up, d.inner, sc.ff_inner, null, width, d.inner);
        try k.gemm.run(s, sc.ff_inner, l.mlp_down, sc.down, width, H, d.inner, .bf16);
        // The residual, and the next layer's input norm (the block's final norm after the fifth).
        const next_norm = if (i + 1 < d.layers.len) d.layers[i + 1].input_norm else d.final_norm;
        // Writes the layer's whole output back into `res`; `spare` is free scratch again for the next attention add.
        try k.norms.addRmsTail(s, sc.down, spare, next_norm, res, sc.normed, null, width, H, eps, .{ .round = true });
    }
    if (d.trail) |t| try t.rows.copyFrom(0, sc.normed, width * H * 2, s.handle);
    // One slot at a time: each row's correction reads the token before it, then the target's own head decides.
    for (0..depth) |slot| {
        const token = if (slot == 0) bonus else held[slot - 1];
        const hidden = sc.normed + (slot + 1) * H * 2; // row 0 is the anchor's own and predicts nothing
        try sc.previous.copyFrom(0, mm.embed + @as(u64, token) * H * 2, H * 2, s.handle);
        try k.norms.addRmsTail(s, hidden, null, d.c_hnorm, sc.norm_resid, sc.cat, null, 1, H, eps, .{ .round = true });
        try k.norms.addRmsTail(s, sc.previous.ptr, null, d.c_enorm, sc.norm_resid, sc.cat + H * 2, null, 1, H, eps, .{ .round = true });
        if (d.trail) |t| if (slot == 0) try t.cat1.copyFrom(0, sc.cat, 2 * H * 2, s.handle);
        try k.gemm.run(s, sc.cat, d.c_gate, sc.c_gate, 1, H, 2 * H, .bf16);
        try k.gemm.run(s, sc.cat, d.c_up, sc.c_up, 1, H, 2 * H, .bf16);
        try k.act.swiglu(s, sc.c_gate, H, sc.c_up, H, sc.c_act, null, 1, H);
        try k.gemm.run(s, sc.c_act, d.c_down, sc.c_down, 1, H, H, .bf16);
        if (d.trail) |t| if (slot == 0) try t.delta.copyFrom(0, sc.c_down, H * 2, s.handle);
        try d.kernels.addRows(s, sc.c_down, hidden, sc.corrected, H);
        if (d.trail) |t| try t.corrected.copyFrom(@as(u64, slot) * H * 2, sc.corrected, H * 2, s.handle);
        switch (mm.lm_head.weight) {
            .bf16 => |w| try k.gemm.run(s, sc.corrected, w, sc.logits, 1, V, H, .f32),
            .exl3 => |w| try k.dense.decode(s, w, sc.corrected, .bf16, 1, sc.logits, .f32),
        }
        if (d.trail) |t| try t.logits.copyFrom(@as(u64, slot) * V * 4, sc.logits, V * 4, s.handle);
        try d.kernels.argmaxRows(s, sc.logits, 1, V, sc.ids.ptr);
        try sc.ids.downloadAsync(0, std.mem.sliceAsBytes(held[slot..][0..1]), s.handle);
        try s.synchronize();
    }
}

/// One draft row's attended set, the reference's reuse rule: the target's recalled positions below the anchor,
/// then the block's own causal prefix. A CUDA slot is its position (the indexed layers write by position), so the
/// two are the same number here, which is what `attention_slots.retained` translates on a paged Metal cache.
pub fn reuseKeys(anchor: usize, row: usize, target: []const i32, out: []u32) !usize {
    if (out.len < anchor + row + 1 -| anchor) return error.ShortHy4KeyList;
    var n: usize = 0;
    for (target) |entry| {
        if (entry < 0) continue;
        const position: usize = @intCast(entry);
        if (position >= anchor) continue;
        if (n == out.len) return error.ShortHy4KeyList;
        out[n] = @intCast(position);
        n += 1;
    }
    // The block's own rows, which the like of vLLM pads with -1 past each row's causal prefix.
    for (anchor..anchor + row + 1) |position| {
        if (n == out.len) return error.ShortHy4KeyList;
        out[n] = @intCast(position);
        n += 1;
    }
    return n;
}

/// A target below its key budget has selected nothing, so a row attends its whole causal history.
pub fn causalKeys(anchor: usize, row: usize, out: []u32) !usize {
    const at = anchor + row;
    if (out.len < at + 1) return error.ShortHy4KeyList;
    for (0..at + 1) |position| out[position] = @intCast(position);
    return at + 1;
}

/// A round's key lists on the host, as the reference's reuse kernel builds them: each draft layer's own target
/// layer's recalled list for the kept row, then that row's own causal prefix.
fn buildKeys(mm: *Model, d: *Draft, anchor: usize, row_kept: usize) !void {
    const sc = &d.scratch;
    const captured = sc.selected and row_kept < sc.rows;
    if (captured) {
        for (sc.lists, sc.captured, 0..) |list, count, i| {
            try list.downloadAsync(row_kept * geom.sparse_stride * 4, std.mem.sliceAsBytes(d.host_lists[i * geom.sparse_stride ..][0..geom.sparse_stride]), mm.s.handle);
            try count.downloadAsync(row_kept * 4, std.mem.sliceAsBytes(d.host_captured[i..][0..1]), mm.s.handle);
        }
        try mm.s.synchronize();
    }
    for (0..layers) |i| {
        // A full 2,048-entry list is the common case past the budget: it and the block's rows just fit the stride.
        const list = d.host_lists[i * geom.sparse_stride ..][0..geom.sparse_stride];
        for (0..width) |row| {
            const out = d.host_keys[i * width * key_stride + row * key_stride ..][0..key_stride];
            d.host_counts[i * width + row] = @intCast(if (captured)
                try reuseKeys(anchor, row, list[0..@min(@as(usize, d.host_captured[i]), geom.sparse_stride)], out)
            else
                try causalKeys(anchor, row, out));
        }
    }
}

test "a draft row's keys are the reference's reuse list, minus its -1 padding" {
    const reference = @import("dfly2_inputs.zig");
    const frame: reference.Frame = .{ .source_row = 2, .anchor = 13, .query_rows = 6 };
    const target = [_]i32{ 0, 8, 10, 12, 13, -1 };
    var out: [key_stride]u32 = undefined;
    var want: [16]i32 = undefined;
    for (0..width) |row| {
        const n = try reuseKeys(frame.anchor, row, &target, &out);
        try reference.reuseIndices(frame, &target, 0, @intCast(row), &want);
        var k: usize = 0;
        for (want[0 .. frame.query_rows + target.len]) |entry| if (entry >= 0) {
            try std.testing.expectEqual(@as(u32, @intCast(entry)), out[k]);
            k += 1;
        };
        try std.testing.expectEqual(k, n);
        // A row ends on its own position: nothing after it is ever in its set.
        try std.testing.expectEqual(@as(u32, @intCast(frame.anchor + row)), out[n - 1]);
    }
    // Below the key budget the target has selected nothing: the whole causal prefix, and nothing after the row.
    const dense = try causalKeys(2000, 5, &out);
    try std.testing.expectEqual(@as(usize, 2006), dense);
    try std.testing.expectEqual(@as(u32, 2005), out[dense - 1]);
    try std.testing.expectError(error.ShortHy4KeyList, causalKeys(0, width, out[0..width]));
    try std.testing.expectError(error.ShortHy4KeyList, reuseKeys(4, 0, &target, out[0..1]));
}

test "the block's list fits the stride the attention is sized for" {
    // The widest a row's set can be: every recalled slot below the anchor, then the block's own six rows.
    try std.testing.expectEqual(@as(usize, 2048 + 6), key_stride);
    try std.testing.expect(key_stride > geom.sparse_stride);
    const shape = try attn.Shape.init(width, 32, key_stride);
    try std.testing.expectEqual(@as(u32, 5), shape.chunks);
}
