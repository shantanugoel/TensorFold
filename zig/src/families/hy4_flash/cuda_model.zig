//! Hy4-flash text model on CUDA for EXL3 routed experts with BF16 elsewhere; one sequence, causal or windowed keys.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const ihc = @import("ihc.zig");
const ihc_cuda = @import("ihc_cuda.zig");
const mla = @import("mla_cuda.zig");
const m = @import("mla_weights.zig");
const attn = @import("mla_attention.zig");
const exl3 = @import("moe_exl3_cuda.zig");
const dense = @import("dense_weights.zig");
const engram = @import("engram.zig");
const engram_cuda = @import("engram_cuda.zig");
const direct = @import("bf16_rows.zig");
const engram_pass = @import("engram_pass.zig");
const geom = @import("cuda_scratch.zig");
const cuda_index = @import("cuda_index.zig");

const H = geom.H;
const S = geom.S;
const L = geom.L;
const D = geom.D;
const E = geom.E;
const K = geom.K;
const Q = geom.Q;
const sparse_stride = geom.sparse_stride;
const sliding_stride = geom.sliding_stride;
const Scratch = geom.Scratch;
/// Reads one Engram table is cut into: four tables started together is 32 of them in flight a pass.
const engram_readers = 8;

pub const Kernels = struct {
    gemm: cuda.vit.Gemms,
    norms: cuda.norms.Norms,
    project: mla.Projections,
    attention: mla.Attention,
    ihc: ihc_cuda.Kernels,
    experts: cuda.exl3_experts.Experts,
    dense: cuda.exl3.Dense,
    act: cuda.act_rows.ActRows,
    glue: cuda.moe_glue.MoeGlue,
    router: cuda.route_rows.RouteRows,
    engram: engram_cuda.Kernels,
    indexer: cuda_index.Kernels,

    /// The kernel images embedded in this build (`-Dnvcc` or `-Dprebuilt`).
    pub fn load(a: std.mem.Allocator, d: *const cuda.Driver) !Kernels {
        return .{
            .gemm = try cuda.vit.Gemms.init(a, d),
            .norms = try cuda.norms.Norms.load(d),
            .project = try mla.Projections.load(d),
            .attention = try mla.Attention.load(d),
            .ihc = try ihc_cuda.Kernels.load(d),
            .experts = try cuda.exl3_experts.Experts.load(d),
            .dense = try cuda.exl3.Dense.load(d),
            .act = try cuda.act_rows.ActRows.load(d),
            .glue = try cuda.moe_glue.MoeGlue.load(d),
            .router = try cuda.route_rows.RouteRows.load(d),
            .engram = try engram_cuda.Kernels.load(d),
            .indexer = try cuda_index.Kernels.load(d),
        };
    }
};

pub const Hc = struct { project: u64, bias: u64, scale: u64 };

pub const Attention = struct {
    sliding: bool,
    heads: u32,
    hc: Hc,
    input_norm: u64,
    q_a: u64,
    q_norm: u64,
    q_b: dense.Projection,
    kv_a: u64,
    kv_norm: u64,
    key: u64,
    value: u64,
    gate: dense.Projection,
    output: dense.Projection,
    output_norm: u64,
    sinks: u64,
    cache: cuda.DeviceBuffer, // a sliding layer's ring rows, or the context's rows behind an indexed layer
    rows: u32, // `cache`'s rows: the bound the write kernel refuses a slot past
    index: ?*cuda_index.Layer = null,
};

pub const Dense = struct { gate: dense.Projection, up: dense.Projection, down: dense.Projection };
pub const Sparse = struct {
    latent_in: u64,
    latent_out: u64,
    shared_gate: dense.Projection,
    shared_up: dense.Projection,
    shared_down: dense.Projection,
    router: u64,
    router_bias: u64,
    routed: exl3.Routed,
};

pub const Feed = struct {
    hc: Hc,
    input_norm: u64,
    output_norm: u64,
    mlp: union(enum) { dense: Dense, sparse: Sparse },
};

pub const Engram = struct {
    w: engram_cuda.Weights,
    table: direct.Table,
};

pub const Layer = struct { engram: ?Engram, attention: Attention, feed: Feed };

pub const Model = struct {
    /// The weight map, one layer's load and the per-pass scratch: `cuda_model_load.zig`, the forward's other half.
    pub const open = @import("cuda_model_load.zig").open;

    a: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    s: cuda.Stream,
    k: *Kernels,
    text: config.Text,
    own: cuda.exl3_upload.Uploader,
    layers: []Layer,
    embed: u64,
    head_hc: Hc,
    final_norm: u64,
    lm_head: dense.Projection,
    scratch: Scratch,
    index: cuda_index.Scratch,
    max_rows: usize,
    capacity: usize,
    /// The sliding layers' KV as `geom.ring.rows` slots; false is the full-length arm the ring is gated against.
    ring: bool,
    history: std.ArrayList(u32) = .empty,
    expert_scratch: exl3.Shared = .{},
    weight_bytes: usize = 0,
    /// The Engram tables in layer order, so a pass starts every table's rows at once.
    engrams: []*const direct.Table = &.{},
    /// Host nanoseconds a pass spent on the Engram rows: planning, reading and waiting.
    engram_ns: i96 = 0,

    /// What one load takes: the context's rows, the widest pass, and whether sliding layers keep a ring.
    pub const Options = struct {
        capacity: usize,
        max_rows: usize,
        ring: bool = true,
    };

    fn hcShape(self: *const Model, rows: usize, gates: u32) ihc.Shape {
        return .{ .rows = @intCast(rows), .width = H, .gates = gates, .eps = self.text.rms_norm_eps, .gate_eps = self.text.hc_eps, .post_magnitude = self.text.hc_magnitude };
    }

    /// One projection's rows: the pack's EXL3 group through the dense kernels, or its BF16 weight through cuBLAS.
    fn project(self: *Model, s: cuda.Stream, p: dense.Projection, x: u64, rows: usize, y: u64, out: cuda.cublaslt.DataType) !void {
        switch (p.weight) {
            .bf16 => |w| try self.k.gemm.run(s, x, w, y, rows, p.n, p.k, out),
            .exl3 => |w| try self.k.dense.prompt(s, w, x, .bf16, rows, y, switch (out) {
                .f32 => .f32,
                .f16 => .f16,
                .bf16 => .bf16,
            }),
        }
    }

    /// Which pass rows to hand back: `out` holds one `vocab`-wide FP32 row a pass row from `at` on.
    pub const Rows = struct { at: usize, out: []f32 };

    /// Run `tokens` (appended at the current history length) through the model; returns the last row's logits.
    pub fn pass(self: *Model, tokens: []const u32, logits_out: []f32) !void {
        return self.passRows(tokens, logits_out, null);
    }

    /// `pass`, and the rows `rows_out` names into it, as one `vocab`-wide FP32 row each.
    pub fn passRows(self: *Model, tokens: []const u32, logits_out: []f32, rows_out: ?Rows) !void {
        const rows = tokens.len;
        const start = self.history.items.len;
        if (rows == 0 or rows > self.max_rows or start + rows > self.capacity) return error.BadHy4PassRows;
        try self.history.appendSlice(self.a, tokens);
        const sc = self.scratch;
        const s = self.s;
        const k = self.k;
        const eps = self.text.rms_norm_eps;
        // Embedding into four identical streams.
        var x = sc.streams[0].ptr;
        for (tokens, 0..) |tok, r| for (0..S) |st| {
            try sc.streams[0].copyFrom((r * S + st) * H * 2, self.embed + @as(u64, tok) * H * 2, H * 2, s.handle);
        };
        // Positions (text: t = h = w), write slots and key lists for both attention kinds.
        const positions = try self.a.alloc(i32, rows * 3);
        defer self.a.free(positions);
        const writes = try self.a.alloc(u32, rows);
        defer self.a.free(writes);
        const ring_writes = try self.a.alloc(u32, rows);
        defer self.a.free(ring_writes);
        for (0..rows) |r| {
            const p: i32 = @intCast(start + r);
            positions[r * 3 ..][0..3].* = .{ p, p, p };
            writes[r] = @intCast(start + r);
            // Sliding layers read and write the ring's slots; with the ring off they keep their position's row.
            ring_writes[r] = @intCast(if (self.ring) geom.ring.slot(start + r) else start + r);
        }
        try sc.positions.uploadAsync(0, std.mem.sliceAsBytes(positions), s.handle);
        try sc.write_slots.uploadAsync(0, std.mem.sliceAsBytes(writes), s.handle);
        try sc.write_ring.uploadAsync(0, std.mem.sliceAsBytes(ring_writes), s.handle);
        // Past the budget an indexed row's list comes from the layer's indexer, so nothing is built here.
        const lists = self.capacity > sparse_stride and start + rows - 1 >= sparse_stride;
        const keys_sparse = try self.a.alloc(u32, if (lists) 0 else rows * sparse_stride);
        defer self.a.free(keys_sparse);
        const keys_sliding = try self.a.alloc(u32, rows * sliding_stride);
        defer self.a.free(keys_sliding);
        const counts_sparse = try self.a.alloc(u32, rows);
        defer self.a.free(counts_sparse);
        const counts_sliding = try self.a.alloc(u32, rows);
        defer self.a.free(counts_sliding);
        @memset(keys_sparse, 0);
        @memset(keys_sliding, 0);
        for (0..rows) |r| {
            const pos = start + r;
            if (!lists) {
                counts_sparse[r] = @intCast(pos + 1);
                for (0..pos + 1) |j| keys_sparse[r * sparse_stride + j] = @intCast(j);
            }
            const first = if (pos + 1 > sliding_stride) pos + 1 - sliding_stride else 0;
            counts_sliding[r] = @intCast(pos + 1 - first);
            for (first..pos + 1, 0..) |j, n| keys_sliding[r * sliding_stride + n] = @intCast(if (self.ring) geom.ring.slot(j) else j);
        }
        // Engram rows for the whole pass: a row hashes from the token ids alone, so all four tables start at once.
        const windows = try self.a.alloc([engram.order]u32, rows);
        defer self.a.free(windows);
        const pad = self.text.engram_pad_id;
        for (windows, 0..) |*w, r| {
            const pos = start + r;
            for (w, 0..) |*v, back| v.* = if (pos >= back) self.history.items[pos - back] else pad;
        }
        const et0 = std.Io.Clock.awake.now(self.io);
        var gathers = try engram_pass.Rows.start(self.a, self.io, self.engrams, windows, engram_cuda.input, engram_readers);
        defer gathers.close(self.io);
        self.engram_ns += et0.durationTo(std.Io.Clock.awake.now(self.io)).toNanoseconds();

        const rope_sparse = try self.text.rotary(.deepseek_sparse_attention);
        var slot: usize = 0; // index of the stream buffer x lives in
        for (self.layers) |*layer| {
            // Engram residual into the streams.
            if (layer.engram) |*e| {
                const et = std.Io.Clock.awake.now(self.io);
                const raw = try gathers.next(self.io);
                self.engram_ns += et.durationTo(std.Io.Clock.awake.now(self.io)).toNanoseconds();
                try sc.e_raw.uploadAsync(0, std.mem.sliceAsBytes(raw), s.handle);
                const out = sc.streams[(slot + 1) % 3].ptr;
                try k.engram.forward(s, &k.gemm, e.w, .{
                    .raw = sc.e_raw.ptr,
                    .normalized = sc.e_normalized,
                    .keys = sc.e_keys,
                    .value = sc.e_value,
                    .media = sc.media,
                    .state = x,
                    .output = out,
                }, @intCast(rows), @intCast(rows), H, self.text.engram_fp8_embedding);
                slot = (slot + 1) % 3;
                x = out;
            }
            // Attention half.
            const at = &layer.attention;
            try k.ihc.collapse(s, .{ .hidden = x, .projection = at.hc.project, .bias = at.hc.bias, .scale = at.hc.scale, .output = sc.collapsed, .post = sc.post }, self.hcShape(rows, 2 * S));
            try k.norms.addRmsTail(s, sc.collapsed, null, at.input_norm, sc.norm_tmp, sc.hidden, null, rows, H, eps, .{ .round = true });
            try k.gemm.run(s, sc.hidden, at.q_a, sc.q_raw, rows, Q, H, .bf16);
            try k.norms.addRmsTail(s, sc.q_raw, null, at.q_norm, sc.q_tmp, sc.q_resid, null, rows, Q, eps, .{ .round = true });
            try self.project(s, at.q_b, sc.q_resid, rows, sc.q, .bf16);
            try k.gemm.run(s, sc.hidden, at.kv_a, sc.kv, rows, m.absorbed_stride, H, .bf16);
            try self.project(s, at.gate, sc.hidden, rows, sc.gate, .bf16);
            const stride: usize = if (at.sliding) sliding_stride else sparse_stride;
            if (at.index) |ix| try cuda_index.forward(.{
                .s = s, .indexer = &k.indexer, .gemms = &k.gemm, .norms = &k.norms, .project = k.project,
                .scratch = &self.index, .layer = ix, .hidden = sc.hidden, .q_resid = sc.q_resid,
                .write_slots = sc.write_slots.ptr, .positions = sc.positions.ptr, .table = sc.table,
                .slots = sc.slots.ptr, .counts = sc.counts.ptr, .first = start, .rows = rows, .select = lists,
                .epsilon = eps, .rope = rope_sparse, .q_rank = Q, .hidden_size = H,
            });
            if (at.sliding) {
                try sc.slots.uploadAsync(0, std.mem.sliceAsBytes(keys_sliding), s.handle);
                try sc.counts.uploadAsync(0, std.mem.sliceAsBytes(counts_sliding), s.handle);
            } else if (!lists) {
                try sc.slots.uploadAsync(0, std.mem.sliceAsBytes(keys_sparse), s.handle);
                try sc.counts.uploadAsync(0, std.mem.sliceAsBytes(counts_sparse), s.handle);
            }
            const shape = try attn.Shape.init(@intCast(rows), at.heads, @intCast(stride));
            try mla.forward(k.project, k.attention, s, .{
                .query = sc.q,
                .key_weights = at.key,
                .value_weights = at.value,
                .value_logits = sc.gate,
                .positions = sc.positions.ptr,
                .rotary_table = sc.table,
                .kv = .{ .projection = sc.kv, .gain = at.kv_norm, .write_slots = if (at.sliding) sc.write_ring.ptr else sc.write_slots.ptr, .capacity = at.rows, .epsilon = eps },
                .attention = .{
                    .query = sc.absorb,
                    .cache = at.cache.ptr,
                    .slots = sc.slots.ptr,
                    .counts = sc.counts.ptr,
                    .sinks = at.sinks,
                    .scratch = sc.attn_scratch,
                    .scratch_bytes = sc.attn_scratch_bytes,
                    .output = sc.latent_out,
                },
                .values = sc.values,
            }, shape, if (at.sliding) try self.text.rotary(.sliding_attention) else rope_sparse);
            try self.project(s, at.output, sc.values, rows, sc.branch, .bf16);
            try k.norms.addRmsTail(s, sc.branch, null, at.output_norm, sc.norm_tmp, sc.branch_norm, null, rows, H, eps, .{ .round = true });
            const after_attn = sc.streams[(slot + 1) % 3].ptr;
            try k.ihc.add(s, .{ .residual = x, .branch = sc.branch_norm, .post = sc.post, .output = after_attn }, self.hcShape(rows, 2 * S));
            slot = (slot + 1) % 3;
            x = after_attn;
            // Feed-forward half.
            const f = &layer.feed;
            try k.ihc.collapse(s, .{ .hidden = x, .projection = f.hc.project, .bias = f.hc.bias, .scale = f.hc.scale, .output = sc.collapsed, .post = sc.post }, self.hcShape(rows, 2 * S));
            try k.norms.addRmsTail(s, sc.collapsed, null, f.input_norm, sc.norm_tmp, sc.hidden, null, rows, H, eps, .{ .round = true });
            switch (f.mlp) {
                .dense => |dn| {
                    try self.project(s, dn.gate, sc.hidden, rows, sc.ff_gate, .bf16);
                    try self.project(s, dn.up, sc.hidden, rows, sc.ff_up, .bf16);
                    try k.act.swiglu(s, sc.ff_gate, D, sc.ff_up, D, sc.ff_inner, null, rows, D);
                    try self.project(s, dn.down, sc.ff_inner, rows, sc.shared, .bf16);
                    try k.norms.addRmsTail(s, sc.shared, null, f.output_norm, sc.norm_tmp, sc.branch_norm, null, rows, H, eps, .{ .round = true });
                },
                .sparse => |sp| {
                    try k.gemm.run(s, sc.hidden, sp.latent_in, sc.latent, rows, L, H, .bf16);
                    try k.router.logits(s, sc.hidden, H, sp.router, sc.logits, rows, H, E);
                    try k.glue.topk(s, sc.logits, sp.router_bias, .{ .experts = E, .topk = K, .shared = false, .score = true, .normalize = true, .scale = self.text.routed_scaling_factor }, sc.picks, sc.probabilities, rows);
                    const y = try sp.routed.run(&k.experts, s, sc.latent, sc.picks, rows);
                    try k.glue.combine(s, y, .f32, sc.probabilities, sc.mix, .bf16, rows, K, L);
                    try k.gemm.run(s, sc.mix, sp.latent_out, sc.route_out, rows, H, L, .bf16);
                    try self.project(s, sp.shared_gate, sc.hidden, rows, sc.ff_gate, .bf16);
                    try self.project(s, sp.shared_up, sc.hidden, rows, sc.ff_up, .bf16);
                    try k.act.swiglu(s, sc.ff_gate, L, sc.ff_up, L, sc.ff_inner, null, rows, L);
                    try self.project(s, sp.shared_down, sc.ff_inner, rows, sc.shared, .bf16);
                    try k.norms.addRmsTail(s, sc.route_out, sc.shared, f.output_norm, sc.norm_tmp, sc.branch_norm, null, rows, H, eps, .{ .round = true });
                },
            }
            const after_ff = sc.streams[(slot + 1) % 3].ptr;
            try k.ihc.add(s, .{ .residual = x, .branch = sc.branch_norm, .post = sc.post, .output = after_ff }, self.hcShape(rows, 2 * S));
            slot = (slot + 1) % 3;
            x = after_ff;
        }
        // Hyper head, final norm and LM head: the last row alone, and every row when a scoring pass asks.
        if (rows_out) |want| {
            const V: usize = self.text.vocab_size;
            const count = want.out.len / V;
            if (want.out.len != count * V or want.at + count > rows) return error.BadHy4PassRows;
            for (0..count) |r| try self.headRow(s, x, want.at + r, want.out[r * V ..][0..V]);
        }
        try self.headRow(s, x, rows - 1, logits_out);
    }

    /// Row `row` of the pass's final streams through the hyper head, final norm and LM head, down to `out`.
    fn headRow(self: *Model, s: cuda.Stream, x: u64, row: usize, out: []f32) !void {
        const sc = self.scratch;
        const k = self.k;
        const hidden = x + row * S * H * 2;
        try k.ihc.collapse(s, .{ .hidden = hidden, .projection = self.head_hc.project, .bias = self.head_hc.bias, .scale = self.head_hc.scale, .output = sc.head_hidden, .post = 0 }, self.hcShape(1, S));
        try k.norms.addRmsTail(s, sc.head_hidden, null, self.final_norm, sc.norm_tmp, sc.head_norm, null, 1, H, self.text.rms_norm_eps, .{ .round = true });
        try self.project(s, self.lm_head, sc.head_norm, 1, sc.lm_logits, .f32);
        try s.synchronize();
        try cuda.memory.download(self.d, sc.lm_logits, std.mem.sliceAsBytes(out));
    }
};
