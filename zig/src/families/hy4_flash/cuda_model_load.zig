//! Hy4-flash on CUDA: the weight map, one layer at a time, and the per-pass scratch the load sizes.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const config = @import("config.zig");
const mla = @import("mla_cuda.zig");
const m = @import("mla_weights.zig");
const exl3 = @import("moe_exl3_cuda.zig");
const expert_weights = @import("exl3_weights.zig");
const dense = @import("dense_weights.zig");
const load_util = @import("cuda_load_util.zig");
const engram = @import("engram.zig");
const engram_cuda = @import("engram_cuda.zig");
const direct = @import("bf16_rows.zig");
const geom = @import("cuda_scratch.zig");
const format = @import("cuda_kv8.zig");
const cuda_index = @import("cuda_index.zig");
const model = @import("cuda_model.zig");
const dfly = @import("cuda_draft.zig");

const Model = model.Model;
const Layer = model.Layer;
const Attention = model.Attention;
const Feed = model.Feed;
const Engram = model.Engram;
const Kernels = model.Kernels;
const Options = Model.Options;

const H = geom.H;
const S = geom.S;
const L = geom.L;
const D = geom.D;
const E = geom.E;
const K = geom.K;
const Q = geom.Q;
const sparse_stride = geom.sparse_stride;

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

pub fn open(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, k: *Kernels, dir: []const u8, o: Options) !*Model {
    const capacity = o.capacity;
    const max_rows = o.max_rows;
    const config_path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(config_path);
    const config_bytes = try std.Io.Dir.cwd().readFileAlloc(io, config_path, a, .limited(1 << 20));
    defer a.free(config_bytes);
    const cfg = try config.parse(a, config_bytes);
    const t = cfg.value.text_config;
    if (t.hidden_size != H or t.hc_mult != S or t.routed_expert_hidden_size != L or t.moe_intermediate_size != L or
        t.intermediate_size != D or t.n_routed_experts != E or t.num_experts_per_tok != K or t.q_lora_rank != Q)
        return error.BadHy4ModelShape;
    // Past the 2,048-key budget an indexed row attends recall lists, so capacity is free up to the scores' width.
    if (capacity == 0 or capacity > cuda_index.max_capacity or max_rows == 0 or max_rows > geom.row_cap) return error.BadHy4RunnerCapacity;
    if (o.slots == 0 or o.slots > geom.max_slots) return error.BadHy4RunnerCapacity;
    if (o.ring) try geom.ring.check(max_rows);
    const ck = try a.create(core.Checkpoint);
    ck.* = try core.Checkpoint.openModel(a, io, dir);
    // The Engram token map reads its vocabulary out of these mappings, so the model owns them and frees them last.
    errdefer {
        ck.close();
        a.destroy(ck);
    }
    const self = try a.create(Model);
    self.* = .{
        .a = a,
        .io = io,
        .d = d,
        .s = s,
        .k = k,
        .text = t,
        .cfg = cfg,
        .ck = ck,
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
        .kv8 = o.kv8,
        .slots = o.slots,
        .ring = o.ring,
        .history = try a.alloc(std.ArrayList(u32), o.slots),
    };
    for (self.history) |*h| h.* = .empty;
    const V: usize = t.vocab_size;
    self.embed = try up(self, ck, "model.embed_tokens.weight", .{}, .bf16, &.{ V, H });
    self.head_hc = .{
        .project = try up(self, ck, "model.hc_head.hc_head_fn", .{}, .f32, &.{ S, S * H }),
        .bias = try up(self, ck, "model.hc_head.hc_head_base", .{}, .f32, &.{S}),
        .scale = try up(self, ck, "model.hc_head.hc_head_scale", .{}, .f32, &.{1}),
    };
    self.final_norm = try up(self, ck, "model.norm.weight", .{}, .bf16, &.{H});
    self.lm_head = try dense.read(&self.own, ck, "lm_head", .{}, V, H, .bf16);
    ck.release();
    // UMA guard: device memory is not charged to the cgroup, so stop before the system runs dry.
    const floor_gib = try @import("sys").environment.read(usize, "TF_HY4_MEM_FLOOR_GIB", 10);
    // The Engram tables' shards come from one parse of the model index; a pack without Engram layers never reads it.
    var shards: ?load_util.Shards = if (t.engram_layer_ids.len == 0) null else try load_util.Shards.open(a, io, dir);
    defer if (shards) |*sh| sh.deinit();
    for (self.layers, 0..) |*layer, i| {
        const avail = load_util.memAvailable();
        std.debug.print("layer {d:>2}: device {d:.2} GiB, MemAvailable {d:.2} GiB\n", .{ i, load_util.gib(self.own.bytes), load_util.gib(avail) });
        if (avail < floor_gib << 30) {
            std.debug.print("stopping: MemAvailable below the {d} GiB floor\n", .{floor_gib});
            return error.Hy4MemoryFloor;
        }
        layer.* = try loadLayer(self, a, ck, if (shards) |*sh| sh else null, i);
        // The layer is on the device: its mapped shard pages (and page cache) would otherwise stay resident.
        ck.release();
    }
    // The Engram tables in layer order: a pass reads them all at once, so the order has to be the layers'.
    var engram_count: usize = 0;
    for (self.layers) |l| engram_count += @intFromBool(l.engram != null);
    const engrams = try a.alloc(*const direct.Table, engram_count);
    var seen: usize = 0;
    for (self.layers) |*l| if (l.engram) |*e| {
        engrams[seen] = &e.table;
        seen += 1;
    };
    self.engrams = engrams;
    std.debug.print("all layers: device {d:.2} GiB, MemAvailable {d:.2} GiB\n", .{ load_util.gib(self.own.bytes), load_util.gib(load_util.memAvailable()) });
    self.weight_bytes = self.own.bytes;
    // The indexer's GEMMs keep one row width whatever a pass takes, so it reads that many rows of the model's scratch.
    const index_rows = if (self.capacity > sparse_stride) geom.row_cap else self.max_rows;
    self.scratch = try geom.alloc(self.d, &self.own, @max(self.max_rows, index_rows), self.text.vocab_size);
    self.index = try cuda_index.Scratch.init(self.d, index_rows, self.capacity);
    try reserveDense(self);
    // DFly2's tower last: the pass's own scratch is allocated at the addresses a plain load gives it, so a resident
    // tower cannot move the buffers a shared round reads (it cost a 4-row pass 12 % when it came first).
    if (o.drafts > 0) {
        // The tower binds each layer to a target layer's cache and recall list. `config.zig` reads that off the
        // pack's layer types; this checks the layers the load actually built, so a ring on one of the five is caught.
        for (cfg.value.dfly_target_kv_layers, cfg.value.dfly_target_index_layer) |kv, ix| {
            if (kv >= self.layers.len or ix >= self.layers.len) return error.BadDraftGeometry;
            const att = self.layers[kv].attention;
            if (att.sliding or att.index == null or self.layers[ix].attention.index == null) return error.BadDraftGeometry;
        }
        self.draft = try dfly.load(a, &self.own, d, ck, cfg.value, self.slots);
        ck.release();
        const dr = self.draft.?;
        std.debug.print("DFly2 drafter: {d} layers of {d}, a block of {d} rows\n", .{ dr.layers.len, dr.inner, dfly.width });
    }
    // The image tower after everything else, for the same reason: a load without --vision allocates nothing more.
    if (o.vision) {
        try self.loadVision(dir, o.vision_tokens);
        ck.release();
        const tower = self.vision.?;
        std.debug.print("image tower: {d} layers, {d:.2} GiB of weights and {d:.0} MiB of scratch for {d} patches an image\n", .{ tower.layers.len, @as(f64, @floatFromInt(tower.bytes)) / (1 << 30), @as(f64, @floatFromInt(tower.scratch.patches.len + tower.scratch.pos.len + tower.scratch.attention.len)) / (1 << 20), tower.patches_max });
        self.weight_bytes = self.own.bytes;
    }
    // the checkpoint's last read: `ck` is this open's own — a name never taken is a stray tensor
    try @import("loader.zig").finishCuda(ck, cfg.value, self.draft != null, o.vision);
    // What grows with the context, so a size decision is read and not inferred: KV, indexer keys, recall scores.
    var kv_bytes: usize = 0;
    var index_bytes: usize = 0;
    var sliding: usize = 0;
    var indexers: usize = 0;
    var fp8: usize = 0;
    for (self.layers) |layer| {
        kv_bytes += @as(usize, layer.attention.rows) * format.rowBytes(layer.attention.kv8 != null);
        sliding += @intFromBool(layer.attention.sliding);
        fp8 += @intFromBool(layer.attention.kv8 != null);
        if (layer.attention.index) |ix| {
            index_bytes += ix.token_keys.len + ix.group_keys.len;
            indexers += 1;
        }
    }
    std.debug.print("capacity {d}: KV {d:.2} GiB ({d} B a token over {d} indexed layers, {d} of them E4M3, {d} sliding at {d} rows), indexer keys {d:.2} GiB ({d} B a token over {d} layers), recall scores {d:.0} MiB\n", .{
        self.capacity,
        load_util.gib(kv_bytes),
        (self.layers.len - sliding) * format.rowBytes(self.kv8),
        self.layers.len - sliding,
        fp8,
        sliding,
        if (self.ring) geom.ring.rows else self.capacity,
        load_util.gib(index_bytes),
        index_bytes / @max(self.capacity, 1),
        indexers,
        @as(f64, @floatFromInt(self.index.scoreBytes())) / (1 << 20),
    });
    return self;
}

/// The shared EXL3 scratch at its widest, inside the load's memory accounting instead of the first forward's.
fn reserveDense(self: *Model) !void {
    for (self.layers) |layer| {
        const att = [3]dense.Projection{ layer.attention.q_b, layer.attention.gate, layer.attention.output };
        const feed = switch (layer.feed.mlp) {
            .dense => |dn| [3]dense.Projection{ dn.gate, dn.up, dn.down },
            .sparse => |sp| [3]dense.Projection{ sp.shared_gate, sp.shared_up, sp.shared_down },
        };
        for (att ++ feed) |p| switch (p.weight) {
            .bf16 => {},
            .exl3 => |w| try self.k.dense.reserve(self.s, w),
        };
    }
    switch (self.lm_head.weight) {
        .bf16 => {},
        .exl3 => |w| try self.k.dense.reserve(self.s, w),
    }
    var bytes: usize = 0;
    for ([_]?cuda.DeviceBuffer{ self.k.dense.xh, self.k.dense.z, self.k.dense.counters }) |slot| bytes += if (slot) |b| b.len else 0;
    std.debug.print("EXL3 dense scratch {d:.1} MiB\n", .{@as(f64, @floatFromInt(bytes)) / (1 << 20)});
}

fn loadLayer(self: *Model, a: std.mem.Allocator, ck: *core.Checkpoint, shards: ?*const load_util.Shards, i: usize) !Layer {
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
    // Every sequence's own rows: an indexed layer's context, or a sliding layer's ring, one sequence after another.
    // `--kv-cache fp8` gives an indexed layer the E4M3 row and its two side planes; a sliding layer's ring stays BF16.
    const fp8 = self.kv8 and !sliding;
    var cache = try cuda.DeviceBuffer.alloc(self.d, self.slots * (if (sliding and self.ring) geom.ring.rows else self.capacity) * (if (fp8) @as(usize, m.latent) else m.absorbed_stride * 2));
    errdefer cache.free();
    try cache.fill8(0, null);
    var kv8_cache: ?model.Kv8 = null;
    if (fp8) {
        const rows = self.slots * self.capacity;
        var scales = try cuda.DeviceBuffer.alloc(self.d, rows * @sizeOf(f32));
        errdefer scales.free();
        try scales.fill8(0, null);
        var ropes = try cuda.DeviceBuffer.alloc(self.d, rows * m.rope * 2);
        errdefer ropes.free();
        try ropes.fill8(0, null);
        kv8_cache = .{ .scales = scales, .ropes = ropes };
    }
    const att: Attention = .{
        .sliding = sliding,
        .heads = heads,
        .hc = .{
            .project = try up(self, ck, p ++ "hc_attn_layer.hc_pre.hc_fn", .{i}, .f32, &.{ 2 * S, S * H }),
            .bias = try up(self, ck, p ++ "hc_attn_layer.hc_pre.hc_base", .{i}, .f32, &.{2 * S}),
            .scale = try up(self, ck, p ++ "hc_attn_layer.hc_pre.hc_scale", .{i}, .f32, &.{2}),
        },
        .input_norm = try up(self, ck, p ++ "input_layernorm.weight", .{i}, .bf16, &.{H}),
        .q_a = try up(self, ck, p ++ "self_attn.q_a_proj.weight", .{i}, .bf16, &.{ Q, H }),
        .q_norm = try up(self, ck, p ++ "self_attn.q_a_layernorm.weight", .{i}, .bf16, &.{Q}),
        .q_b = try dense.read(&self.own, ck, p ++ "self_attn.q_b_proj", .{i}, hh * m.query_stride, Q, .bf16),
        .kv_a = try up(self, ck, p ++ "self_attn.kv_a_proj_with_mqa.weight", .{i}, .bf16, &.{ m.absorbed_stride, H }),
        .kv_norm = try up(self, ck, p ++ "self_attn.kv_a_layernorm.weight", .{i}, .bf16, &.{m.latent}),
        .key = try self.own.dev(std.mem.sliceAsBytes(split.key)),
        .value = try self.own.dev(std.mem.sliceAsBytes(split.value)),
        .gate = try dense.read(&self.own, ck, p ++ "self_attn.linear_gate", .{i}, hh * m.value_width, H, .bf16),
        .output = try dense.read(&self.own, ck, p ++ "self_attn.o_proj", .{i}, H, hh * m.value_width, .bf16),
        .output_norm = try up(self, ck, p ++ "self_attn.post_proj_norm.weight", .{i}, .bf16, &.{H}),
        .sinks = try up(self, ck, p ++ "self_attn.learnable_sink_param", .{i}, .f32, &.{hh}),
        .cache = cache,
        .rows = @intCast(cache.len / (if (fp8) @as(usize, m.latent) else m.absorbed_stride * 2)),
        .kv8 = kv8_cache,
        // Below the key budget nothing selects, so a short context carries no indexer state at all.
        .index = if (sliding or self.capacity <= sparse_stride) null else try cuda_index.Layer.load(a, self.d, &self.own, ck, i, t, self.capacity, self.slots),
    };
    const is_dense = t.mlp_layer_types[i] == .dense;
    const feed: Feed = .{
        .hc = .{
            .project = try up(self, ck, p ++ "hc_mlp_layer.hc_pre.hc_fn", .{i}, .f32, &.{ 2 * S, S * H }),
            .bias = try up(self, ck, p ++ "hc_mlp_layer.hc_pre.hc_base", .{i}, .f32, &.{2 * S}),
            .scale = try up(self, ck, p ++ "hc_mlp_layer.hc_pre.hc_scale", .{i}, .f32, &.{2}),
        },
        .input_norm = try up(self, ck, p ++ "post_attention_layernorm.weight", .{i}, .bf16, &.{H}),
        .output_norm = if (is_dense)
            try up(self, ck, p ++ "mlp.post_fc2_norm.weight", .{i}, .bf16, &.{H})
        else
            try up(self, ck, p ++ "mlp.combine_output_norm.weight", .{i}, .bf16, &.{H}),
        .mlp = if (is_dense) .{ .dense = .{
            .gate = try dense.read(&self.own, ck, p ++ "mlp.gate_proj", .{i}, D, H, .bf16),
            .up = try dense.read(&self.own, ck, p ++ "mlp.up_proj", .{i}, D, H, .bf16),
            .down = try dense.read(&self.own, ck, p ++ "mlp.down_proj", .{i}, H, D, .bf16),
        } } else .{ .sparse = blk: {
            const experts = try std.heap.page_allocator.alloc(expert_weights.Expert, E);
            defer std.heap.page_allocator.free(experts);
            for (experts, 0..) |*e, j| e.* = try expert_weights.readExpert(ck, i, j, L, L);
            break :blk .{
                .latent_in = try up(self, ck, p ++ "mlp.routed_expert_down_proj.weight", .{i}, .bf16, &.{ L, H }),
                .latent_out = try up(self, ck, p ++ "mlp.routed_expert_up_proj.weight", .{i}, .bf16, &.{ H, L }),
                .shared_gate = try dense.read(&self.own, ck, p ++ "mlp.shared_experts.gate_proj", .{i}, L, H, .bf16),
                .shared_up = try dense.read(&self.own, ck, p ++ "mlp.shared_experts.up_proj", .{i}, L, H, .bf16),
                .shared_down = try dense.read(&self.own, ck, p ++ "mlp.shared_experts.down_proj", .{i}, H, L, .bf16),
                .router = try up(self, ck, p ++ "mlp.gate.weight", .{i}, .bf16, &.{ E, H }),
                .router_bias = try up(self, ck, p ++ "mlp.gate.e_score_correction_bias", .{i}, .f32, &.{E}),
                .routed = try exl3.Routed.init(&self.own, experts, L, L, self.max_rows, K, t.swiglu_limit, &self.expert_scratch),
            };
        } },
    };
    var eng: ?Engram = null;
    if (std.mem.indexOfScalar(u32, t.engram_layer_ids, @intCast(i + 1)) != null) {
        const e = "model.layers.{d}.engram_layer.";
        const ein: usize = engram_cuda.input;
        var key: [S]u64 = undefined;
        for (&key, 0..) |*dst, j| dst.* = try up(self, ck, e ++ "key_projs.{d}.weight", .{ i, j }, .bf16, &.{ H, ein });
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
            .value = try up(self, ck, e ++ "value_proj.weight", .{i}, .bf16, &.{ H, ein }),
            .value_bias = try up(self, ck, e ++ "value_proj.bias", .{i}, .bf16, &.{H}),
            .value_gain = try up(self, ck, e ++ "unified_pre_gamma", .{i}, .bf16, &.{H}),
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
        var sb: [416]u8 = undefined;
        const shard = try (shards orelse return error.MissingHy4EngramTable).shardOf(&sb, table_name) orelse return error.MissingHy4EngramTable;
        // The config's per-layer row count, so a wrong or repacked table fails here and not mid-forward.
        const rows_expected = try t.engramRows(@intCast(i + 1));
        var table = try direct.openDirect(a, shard, table_name, map, hash, rows_expected);
        errdefer table.close();
        try ck.skip(table_name);
        eng = .{ .w = w, .table = table };
    }
    return .{ .engram = eng, .attention = att, .feed = feed };
}

