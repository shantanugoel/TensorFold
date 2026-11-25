//! Hy4-flash text model on CUDA for EXL3 routed experts with BF16 elsewhere; one sequence, causal or windowed keys.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const config = @import("config.zig");
const ihc = @import("ihc.zig");
const ihc_cuda = @import("ihc_cuda.zig");
const mla = @import("mla_cuda.zig");
const m = @import("mla_weights.zig");
const attn = @import("mla_attention.zig");
const rope = @import("rope.zig");
const exl3 = @import("moe_exl3_cuda.zig");
const expert_weights = @import("exl3_weights.zig");
const dense = @import("dense_weights.zig");
const load_util = @import("cuda_load_util.zig");
const engram = @import("engram.zig");
const engram_cuda = @import("engram_cuda.zig");
const direct = @import("bf16_rows.zig");
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

const Hc = struct { project: u64, bias: u64, scale: u64 };

const Attention = struct {
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
    cache: cuda.DeviceBuffer,
    index: ?*cuda_index.Layer = null,
};

const Dense = struct { gate: dense.Projection, up: dense.Projection, down: dense.Projection };
const Sparse = struct {
    latent_in: u64,
    latent_out: u64,
    shared_gate: dense.Projection,
    shared_up: dense.Projection,
    shared_down: dense.Projection,
    router: u64,
    router_bias: u64,
    routed: exl3.Routed,
};

const Feed = struct {
    hc: Hc,
    input_norm: u64,
    output_norm: u64,
    mlp: union(enum) { dense: Dense, sparse: Sparse },
};

const Engram = struct {
    w: engram_cuda.Weights,
    table: direct.Table,
};

const Layer = struct { engram: ?Engram, attention: Attention, feed: Feed };

pub const Model = struct {
    a: std.mem.Allocator,
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
    history: std.ArrayList(u32) = .empty,
    expert_scratch: exl3.Shared = .{},
    weight_bytes: usize = 0,

    fn tensor(ck: *core.Checkpoint, name: []const u8, dtype: anytype, shape: []const usize) ![]const u8 {
        const t = try ck.get(name);
        if (!t.is(dtype, shape)) {
            std.debug.print("unexpected tensor {s}: {any} {any}\n", .{ name, t.dtype, t.shape[0..t.rank] });
            return error.BadHy4CheckpointTensor;
        }
        return t.bytes;
    }

    fn up(self: *Model, ck: *core.Checkpoint, comptime fmt: []const u8, args: anytype, dtype: anytype, shape: []const usize) !u64 {
        var buf: [192]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, fmt, args);
        return self.own.dev(try tensor(ck, name, dtype, shape));
    }

    pub fn open(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, k: *Kernels, dir: []const u8, capacity: usize, max_rows: usize) !*Model {
        const cfg = try config.parse(a, try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(1 << 20)));
        const t = cfg.value.text_config;
        if (t.hidden_size != H or t.hc_mult != S or t.routed_expert_hidden_size != L or t.moe_intermediate_size != L or
            t.intermediate_size != D or t.n_routed_experts != E or t.num_experts_per_tok != K or t.q_lora_rank != Q)
            return error.BadHy4ModelShape;
        // Past the 2,048-key budget an indexed row attends recall lists, so capacity is free up to the scores' width.
        if (capacity == 0 or capacity > cuda_index.max_capacity or max_rows == 0 or max_rows > geom.row_cap) return error.BadHy4RunnerCapacity;
        const ck = try a.create(core.Checkpoint);
        ck.* = try core.Checkpoint.openModel(a, io, dir);
        const self = try a.create(Model);
        self.* = .{
            .a = a,
            .d = d,
            .s = s,
            .k = k,
            .text = t,
            // Upload tables and staging go through a freeing allocator: the arena would keep them all.
            .own = .{ .d = d, .gpa = std.heap.page_allocator, .max_buffers = 1 << 20, .slab = 256 << 20 },
            .layers = try a.alloc(Layer, t.num_hidden_layers),
            .embed = undefined,
            .head_hc = undefined,
            .final_norm = undefined,
            .lm_head = undefined,
            .scratch = undefined,
            .index = undefined,
            .max_rows = max_rows,
            .capacity = capacity,
        };
        const V: usize = t.vocab_size;
        self.embed = try self.up(ck, "model.embed_tokens.weight", .{}, .bf16, &.{ V, H });
        self.head_hc = .{
            .project = try self.up(ck, "model.hc_head.hc_head_fn", .{}, .f32, &.{ S, S * H }),
            .bias = try self.up(ck, "model.hc_head.hc_head_base", .{}, .f32, &.{S}),
            .scale = try self.up(ck, "model.hc_head.hc_head_scale", .{}, .f32, &.{1}),
        };
        self.final_norm = try self.up(ck, "model.norm.weight", .{}, .bf16, &.{H});
        self.lm_head = try dense.read(&self.own, ck, "lm_head", .{}, V, H, .bf16);
        ck.release();
        // UMA guard: device memory is not charged to the cgroup, so stop before the system runs dry.
        const floor_gib = try @import("sys").environment.read(usize, "TF_HY4_MEM_FLOOR_GIB", 10);
        for (self.layers, 0..) |*layer, i| {
            const avail = load_util.memAvailable();
            std.debug.print("layer {d:>2}: device {d:.2} GiB, MemAvailable {d:.2} GiB\n", .{ i, load_util.gib(self.own.bytes), load_util.gib(avail) });
            if (avail < floor_gib << 30) {
                std.debug.print("stopping: MemAvailable below the {d} GiB floor\n", .{floor_gib});
                return error.Hy4MemoryFloor;
            }
            layer.* = try self.loadLayer(a, io, ck, dir, i);
            // The layer is on the device: its mapped shard pages (and page cache) would otherwise stay resident.
            ck.release();
        }
        // the checkpoint's last read: `ck` is this open's own — a name never taken is a stray tensor
        try @import("loader.zig").finishCuda(ck, cfg.value);
        std.debug.print("all layers: device {d:.2} GiB, MemAvailable {d:.2} GiB\n", .{ load_util.gib(self.own.bytes), load_util.gib(load_util.memAvailable()) });
        self.weight_bytes = self.own.bytes;
        self.scratch = try geom.alloc(self.d, &self.own, self.max_rows, self.text.vocab_size);
        self.index = try cuda_index.Scratch.init(self.d, self.max_rows, self.capacity);
        return self;
    }

    fn loadLayer(self: *Model, a: std.mem.Allocator, io: std.Io, ck: *core.Checkpoint, dir: []const u8, i: usize) !Layer {
        const t = self.text;
        const sliding = t.layer_types[i] == .sliding_attention;
        const heads: u32 = if (sliding) t.swa_num_attention_heads else t.num_attention_heads;
        const hh: usize = heads;
        const p = "model.layers.{d}.";
        var kv_b_name: [96]u8 = undefined;
        const kv_b = try tensor(ck, try std.fmt.bufPrint(&kv_b_name, p ++ "self_attn.kv_b_proj.weight", .{i}), .bf16, &.{ hh * (m.nope + m.value_width), m.latent });
        const tmp = std.heap.page_allocator;
        const split = try m.splitBf16(tmp, kv_b, heads);
        defer tmp.free(split.key);
        defer tmp.free(split.value);
        var cache = try cuda.DeviceBuffer.alloc(self.d, self.capacity * m.absorbed_stride * 2);
        try cache.fill8(0, null);
        const att: Attention = .{
            .sliding = sliding,
            .heads = heads,
            .hc = .{
                .project = try self.up(ck, p ++ "hc_attn_layer.hc_pre.hc_fn", .{i}, .f32, &.{ 2 * S, S * H }),
                .bias = try self.up(ck, p ++ "hc_attn_layer.hc_pre.hc_base", .{i}, .f32, &.{2 * S}),
                .scale = try self.up(ck, p ++ "hc_attn_layer.hc_pre.hc_scale", .{i}, .f32, &.{2}),
            },
            .input_norm = try self.up(ck, p ++ "input_layernorm.weight", .{i}, .bf16, &.{H}),
            .q_a = try self.up(ck, p ++ "self_attn.q_a_proj.weight", .{i}, .bf16, &.{ Q, H }),
            .q_norm = try self.up(ck, p ++ "self_attn.q_a_layernorm.weight", .{i}, .bf16, &.{Q}),
            .q_b = try dense.read(&self.own, ck, p ++ "self_attn.q_b_proj", .{i}, hh * m.query_stride, Q, .bf16),
            .kv_a = try self.up(ck, p ++ "self_attn.kv_a_proj_with_mqa.weight", .{i}, .bf16, &.{ m.absorbed_stride, H }),
            .kv_norm = try self.up(ck, p ++ "self_attn.kv_a_layernorm.weight", .{i}, .bf16, &.{m.latent}),
            .key = try self.own.dev(std.mem.sliceAsBytes(split.key)),
            .value = try self.own.dev(std.mem.sliceAsBytes(split.value)),
            .gate = try dense.read(&self.own, ck, p ++ "self_attn.linear_gate", .{i}, hh * m.value_width, H, .bf16),
            .output = try dense.read(&self.own, ck, p ++ "self_attn.o_proj", .{i}, H, hh * m.value_width, .bf16),
            .output_norm = try self.up(ck, p ++ "self_attn.post_proj_norm.weight", .{i}, .bf16, &.{H}),
            .sinks = try self.up(ck, p ++ "self_attn.learnable_sink_param", .{i}, .f32, &.{hh}),
            .cache = cache,
            // Below the key budget nothing selects, so a short context carries no indexer state at all.
            .index = if (sliding or self.capacity <= sparse_stride) null else try cuda_index.Layer.load(a, self.d, &self.own, ck, i, t, self.capacity),
        };
        const is_dense = t.mlp_layer_types[i] == .dense;
        const feed: Feed = .{
            .hc = .{
                .project = try self.up(ck, p ++ "hc_mlp_layer.hc_pre.hc_fn", .{i}, .f32, &.{ 2 * S, S * H }),
                .bias = try self.up(ck, p ++ "hc_mlp_layer.hc_pre.hc_base", .{i}, .f32, &.{2 * S}),
                .scale = try self.up(ck, p ++ "hc_mlp_layer.hc_pre.hc_scale", .{i}, .f32, &.{2}),
            },
            .input_norm = try self.up(ck, p ++ "post_attention_layernorm.weight", .{i}, .bf16, &.{H}),
            .output_norm = if (is_dense)
                try self.up(ck, p ++ "mlp.post_fc2_norm.weight", .{i}, .bf16, &.{H})
            else
                try self.up(ck, p ++ "mlp.combine_output_norm.weight", .{i}, .bf16, &.{H}),
            .mlp = if (is_dense) .{ .dense = .{
                .gate = try dense.read(&self.own, ck, p ++ "mlp.gate_proj", .{i}, D, H, .bf16),
                .up = try dense.read(&self.own, ck, p ++ "mlp.up_proj", .{i}, D, H, .bf16),
                .down = try dense.read(&self.own, ck, p ++ "mlp.down_proj", .{i}, H, D, .bf16),
            } } else .{ .sparse = blk: {
                const experts = try std.heap.page_allocator.alloc(expert_weights.Expert, E);
                defer std.heap.page_allocator.free(experts);
                for (experts, 0..) |*e, j| e.* = try expert_weights.readExpert(ck, i, j, L, L);
                break :blk .{
                    .latent_in = try self.up(ck, p ++ "mlp.routed_expert_down_proj.weight", .{i}, .bf16, &.{ L, H }),
                    .latent_out = try self.up(ck, p ++ "mlp.routed_expert_up_proj.weight", .{i}, .bf16, &.{ H, L }),
                    .shared_gate = try dense.read(&self.own, ck, p ++ "mlp.shared_experts.gate_proj", .{i}, L, H, .bf16),
                    .shared_up = try dense.read(&self.own, ck, p ++ "mlp.shared_experts.up_proj", .{i}, L, H, .bf16),
                    .shared_down = try dense.read(&self.own, ck, p ++ "mlp.shared_experts.down_proj", .{i}, H, L, .bf16),
                    .router = try self.up(ck, p ++ "mlp.gate.weight", .{i}, .bf16, &.{ E, H }),
                    .router_bias = try self.up(ck, p ++ "mlp.gate.e_score_correction_bias", .{i}, .f32, &.{E}),
                    .routed = try exl3.Routed.init(&self.own, experts, L, L, self.max_rows, K, t.swiglu_limit, &self.expert_scratch),
                };
            } },
        };
        var eng: ?Engram = null;
        if (std.mem.indexOfScalar(u32, t.engram_layer_ids, @intCast(i + 1)) != null) {
            const e = "model.layers.{d}.engram_layer.";
            const ein: usize = engram_cuda.input;
            var key: [S]u64 = undefined;
            for (&key, 0..) |*dst, j| dst.* = try self.up(ck, e ++ "key_projs.{d}.weight", .{ i, j }, .bf16, &.{ H, ein });
            const cat = struct {
                fn run(mm: *Model, c: *core.Checkpoint, layer: usize, comptime fmt: []const u8, count: usize, size: usize) !u64 {
                    const out = try mm.a.alloc(u8, count * size * 2);
                    defer mm.a.free(out);
                    for (0..count) |j| {
                        var nb: [128]u8 = undefined;
                        @memcpy(out[j * size * 2 ..][0 .. size * 2], try tensor(c, try std.fmt.bufPrint(&nb, fmt, .{ layer, j }), .bf16, &.{size}));
                    }
                    return mm.own.dev(out);
                }
            }.run;
            const w: engram_cuda.Weights = .{
                .lookup_gamma = try cat(self, ck, i, e ++ "lookup_per_head_norms.{d}.weight", engram_cuda.columns, engram_cuda.head_width),
                .value = try self.up(ck, e ++ "value_proj.weight", .{i}, .bf16, &.{ H, ein }),
                .value_bias = try self.up(ck, e ++ "value_proj.bias", .{i}, .bf16, &.{H}),
                .value_gain = try self.up(ck, e ++ "unified_pre_gamma", .{i}, .bf16, &.{H}),
                .key = key,
                .key_bias = try cat(self, ck, i, e ++ "key_projs.{d}.bias", S, H),
                .key_gain = try cat(self, ck, i, e ++ "norm_key.{d}.weight", S, H),
                .query_gain = try cat(self, ck, i, e ++ "norm_query.{d}.weight", S, H),
                .delta_gain = try cat(self, ck, i, e ++ "unified_per_hc_scales.{d}", S, H),
            };
            var nb: [128]u8 = undefined;
            const lookup = (try ck.get(try std.fmt.bufPrint(&nb, e ++ "lookup_table", .{i}))).bytes;
            const map = try engram.TokenMap.open(lookup, t);
            const hash = try engram.Layer.open(t, @intCast(i + 1));
            var tb: [128]u8 = undefined;
            const table_name = try std.fmt.bufPrint(&tb, e ++ "multi_head_embedding.embedding.weight", .{i});
            const shard = try load_util.shardOf(a, io, dir, table_name);
            // The config's per-layer row count, so a wrong or repacked table fails here and not mid-forward.
            const rows_expected = try t.engramRows(@intCast(i + 1));
            var table = try direct.openDirect(a, shard, table_name, map, hash, rows_expected);
            errdefer table.close();
            try ck.skip(table_name);
            eng = .{ .w = w, .table = table };
        }
        return .{ .engram = eng, .attention = att, .feed = feed };
    }

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
        for (0..rows) |r| {
            const p: i32 = @intCast(start + r);
            positions[r * 3 ..][0..3].* = .{ p, p, p };
            writes[r] = @intCast(start + r);
        }
        try sc.positions.uploadAsync(0, std.mem.sliceAsBytes(positions), s.handle);
        try sc.write_slots.uploadAsync(0, std.mem.sliceAsBytes(writes), s.handle);
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
            for (first..pos + 1, 0..) |j, n| keys_sliding[r * sliding_stride + n] = @intCast(j);
        }
        // Engram windows from the committed history.
        const windows = try self.a.alloc([engram.order]u32, rows);
        defer self.a.free(windows);
        const pad = self.text.engram_pad_id;
        for (windows, 0..) |*w, r| {
            const pos = start + r;
            for (w, 0..) |*v, back| v.* = if (pos >= back) self.history.items[pos - back] else pad;
        }
        const raw_rows = try self.a.alloc(u16, rows * engram_cuda.input);
        defer self.a.free(raw_rows);

        const rope_sparse = try self.text.rotary(.deepseek_sparse_attention);
        var slot: usize = 0; // index of the stream buffer x lives in
        for (self.layers) |*layer| {
            // Engram residual into the streams.
            if (layer.engram) |*e| {
                try e.table.gather(self.a, windows, raw_rows);
                try sc.e_raw.uploadAsync(0, std.mem.sliceAsBytes(raw_rows), s.handle);
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
                .kv = .{ .projection = sc.kv, .gain = at.kv_norm, .write_slots = sc.write_slots.ptr, .capacity = @intCast(self.capacity), .epsilon = eps },
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
