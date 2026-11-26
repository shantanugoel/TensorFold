//! Hy4-flash's compressed sparse indexer on CUDA: token keys, pooled groups and each row's recall lists.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const config = @import("config.zig");
const mla = @import("mla_cuda.zig");
const rope = @import("rope.zig");
const geom = @import("cuda_scratch.zig");
const index = @import("index_cuda.zig");
const recall = @import("index_recall.zig");
const groups = @import("index_groups.zig");

/// The widest cache the recall scores' row width allows: recall.Shape's 65,536 groups of four slots.
pub const max_capacity = 65536 * groups.slots;

/// The indexer's kernel images: the pooling kernel and the two recall kernels.
pub const Kernels = index.Indexer;

/// One indexed layer's indexer: its weights and the two key caches the pass fills.
pub const Layer = struct {
    w: index.Weights,
    token_keys: cuda.DeviceBuffer, // [slots][capacity][128] BF16, one row a token slot of one sequence
    group_keys: cuda.DeviceBuffer, // [slots][capacity / 4][128] BF16, one row a closed group
    slots: usize, // sequences the caches hold, each sequence's own first
    capacity: usize, // slots one sequence ranks

    pub fn load(a: std.mem.Allocator, d: *const cuda.Driver, own: *cuda.exl3_upload.Uploader, ck: *core.Checkpoint, layer: usize, t: config.Text, capacity: usize, slots: usize) !*Layer {
        const width = t.compressed_indexer_head_dim;
        const heads = t.compressed_indexer_num_attention_heads;
        if (width != index.width or heads != index.heads or t.compressed_indexer_compress_ratio != groups.slots or
            t.compressed_indexer_candidate_topk != geom.sparse_stride)
            return error.UnsupportedHy4IndexGeometry;
        const self = try a.create(Layer);
        var pb: [64]u8 = undefined;
        const prefix = try std.fmt.bufPrint(&pb, "model.layers.{d}.self_attn.compressed_indexer.", .{layer});
        const h: usize = t.hidden_size;
        const q_rank: usize = t.q_lora_rank;
        self.* = .{
            .w = .{
                .token_keys = try take(own, ck, prefix, "token_key_projection.weight", .bf16, &.{ width, h }),
                .token_norm = try take(own, ck, prefix, "token_key_norm.weight", .bf16, &.{width}),
                .query = try take(own, ck, prefix, "query_projection.weight", .bf16, &.{ @as(usize, heads) * width, q_rank }),
                .head_weights = try take(own, ck, prefix, "head_weight_projection.weight", .bf16, &.{ heads, h }),
                .values = try take(own, ck, prefix, "key_value_projection.weight", .f32, &.{ width, width }),
                .gates = try take(own, ck, prefix, "key_gate_projection.weight", .f32, &.{ width, width }),
                .bias = try take(own, ck, prefix, "intra_group_ape", .f32, &.{ groups.slots, width }),
                .norm = try take(own, ck, prefix, "key_norm.weight", .f32, &.{width}),
            },
            .token_keys = try empty(d, slots * capacity * width * 2),
            .group_keys = try empty(d, slots * groups_of(capacity) * width * 2),
            .slots = slots,
            .capacity = capacity,
        };
        return self;
    }
};

/// The indexer's share of a pass: the fixed-row projections, one layer's scores and recall rows.
pub const Scratch = struct {
    gemm_rows: usize, // the GEMM row count every pass uses, so a row's keys never depend on the pass size
    shape: recall.Shape, // scores and lists are sized for `gemm_rows`; a pass's own rows come from `shapeFor`
    projected: u64,
    norm_tmp: u64,
    staged: u64,
    query: u64,
    head_weights: u64,
    recall_rows: cuda.DeviceBuffer,
    scores: cuda.DeviceBuffer, // [rows][score_groups] FP32, shared by the pass's indexed layers
    group_positions: cuda.DeviceBuffer,
    group_table: u64,

    pub fn init(d: *const cuda.Driver, max_rows: usize, capacity: usize) !Scratch {
        const score_groups: u32 = @intCast(groups_of(capacity));
        const group_rows = max_rows / groups.slots + 2;
        return .{
            .gemm_rows = max_rows,
            .shape = .{ .rows = 0, .score_groups = score_groups, .group_budget = index.budget_groups },
            .projected = try zeros(d, max_rows * index.width * 2),
            .norm_tmp = try zeros(d, max_rows * index.width * 2),
            .staged = try zeros(d, max_rows * index.width * 2),
            .query = try zeros(d, max_rows * index.heads * index.width * 2),
            .head_weights = try zeros(d, max_rows * index.heads * 2),
            .recall_rows = try cuda.DeviceBuffer.alloc(d, max_rows * @sizeOf(recall.Row)),
            .scores = try cuda.DeviceBuffer.alloc(d, max_rows * score_groups * @sizeOf(f32)),
            .group_positions = try cuda.DeviceBuffer.alloc(d, group_rows * 3 * @sizeOf(i32)),
            .group_table = try zeros(d, group_rows * 2 * rope.pairs * 2),
        };
    }

    /// Bytes the recall scores hold, so a caller can size a context against it.
    pub fn scoreBytes(self: Scratch) usize {
        return self.scores.len;
    }

    /// The score rows a recall list ranks: the groups one sequence's cache closes.
    pub fn scoreGroups(self: Scratch) u32 {
        return self.shape.score_groups;
    }

    /// The pass's own shape: the rotary, score and select kernels walk the rows a pass has, not the GEMM width.
    pub fn shapeFor(self: Scratch, rows: usize) recall.Shape {
        var shape = self.shape;
        shape.rows = @intCast(rows);
        return shape;
    }
};

/// A sequence's rows of a pass: its slot, its first position in it, its first pass row, and whether it selects.
pub const RowSet = struct { slot: usize, first: usize, rows: usize, row0: usize, select: bool };

/// One indexed layer's indexer work for one pass: every sequence's run, at one fixed GEMM width.
pub const Call = struct {
    s: cuda.Stream,
    indexer: *const index.Indexer,
    gemms: *cuda.vit.Gemms,
    norms: *const cuda.norms.Norms,
    project: mla.Projections,
    scratch: *Scratch,
    layer: *const Layer,
    hidden: u64, // the layer's input-norm rows [rows][hidden_size], at least `gemm_rows` of them
    q_resid: u64, // q_a_layernorm's rows [rows][q_lora_rank]
    positions: u64, // i32 [rows, 3], every row's own position
    table: u64, // BF16 [rows, 32, 2]
    slots: u64, // where the recall lists go: [rows][keys_a_row] i32
    counts: u64,
    sets: []const RowSet,
    rows: usize, // the pass's own rows
    epsilon: f32,
    rope: rope.Parameters,
    q_rank: usize,
    hidden_size: usize,
};

pub fn forward(c: Call) !void {
    const sc = c.scratch;
    const s = c.s;
    if (c.sets.len == 0 or c.rows == 0 or c.rows > sc.gemm_rows) return error.BadHy4IndexRows;
    for (c.sets) |set| {
        if (set.rows == 0 or set.row0 + set.rows > c.rows or set.first + set.rows > c.layer.capacity)
            return error.Hy4IndexPastCapacity;
    }
    // Token keys at fixed row widths, so a token's key never depends on how many rows share its pass.
    try c.indexer.keyRows(s, c.gemms, c.norms, c.layer.w, .{
        .hidden = c.hidden,
        .gemm_rows = sc.gemm_rows,
        .count = c.rows,
        .projected = sc.projected,
        .norm_tmp = sc.norm_tmp,
        .staged = sc.staged,
    }, c.hidden_size, c.epsilon);
    for (c.sets) |set| {
        const base = set.slot * c.layer.capacity;
        // A sequence's rows go into that sequence's own slots, one contiguous run a set.
        try c.layer.token_keys.copyFrom(@as(u64, base + set.first) * index.width * 2, sc.staged + set.row0 * index.width * 2, set.rows * index.width * 2, s.handle);
        // The groups this set closes, pooled and rotated at each group's first token, in its own arena.
        const closed = try groups.append(set.first, set.rows);
        if (closed.new_groups == 0) continue;
        const first_group = set.slot * groups_of(c.layer.capacity) + closed.first_group;
        var pos: [3 * (geom.row_cap / groups.slots + 2)]i32 = undefined;
        for (0..closed.new_groups) |g| pos[g * 3 ..][0..3].* = @splat(@intCast((closed.first_group + g) * groups.slots));
        try sc.group_positions.uploadAsync(0, std.mem.sliceAsBytes(pos[0 .. closed.new_groups * 3]), s.handle);
        try c.indexer.poolGroups(s, c.project, c.layer.w, c.layer.token_keys.ptr, c.layer.group_keys.ptr, @intCast(first_group), @intCast(closed.new_groups), sc.group_positions.ptr, sc.group_table, c.rope, c.epsilon);
    }
    var any = false;
    for (c.sets) |set| any = any or set.select;
    if (!any) return;
    // The query heads at the pass's own GEMM width, then each selecting set's own rows.
    try c.indexer.query(s, c.gemms, c.layer.w, .{ .q_resid = c.q_resid, .hidden = c.hidden, .gemm_rows = sc.gemm_rows, .query = sc.query, .head_weights = sc.head_weights }, c.q_rank, c.hidden_size);
    for (c.sets) |set| {
        if (!set.select) continue;
        const groups_per = groups_of(c.layer.capacity);
        var rows_out: [geom.row_cap]recall.Row = undefined;
        for (0..set.rows) |r| rows_out[r] = .{ .group_base = @intCast(set.slot * groups_per), .token = @intCast(set.first + r) };
        const at: u64 = @intCast(set.row0);
        try sc.recall_rows.uploadAsync(set.row0 * @sizeOf(recall.Row), std.mem.sliceAsBytes(rows_out[0..set.rows]), s.handle);
        try c.indexer.scan(s, c.project, c.layer.w, .{
            .positions = c.positions + at * 12,
            .table = c.table + at * 64 * 2,
            .query = sc.query + at * index.heads * index.width * 2,
            .head_weights = sc.head_weights + at * index.heads * 2,
            .arena = c.layer.group_keys.ptr,
            .rows = sc.recall_rows.ptr + at * @sizeOf(recall.Row),
            .scores = sc.scores.ptr + at * sc.scoreGroups() * @sizeOf(f32),
            .tokens = c.slots + at * geom.sparse_stride * 4,
            .counts = c.counts + at * 4,
        }, sc.shapeFor(set.rows), c.rope);
    }
}

/// Groups a cache of `capacity` slots closes: the last token's group included.
pub fn groups_of(capacity: usize) usize {
    return (capacity + groups.slots - 1) / groups.slots;
}

fn empty(d: *const cuda.Driver, bytes: usize) !cuda.DeviceBuffer {
    const b = try cuda.DeviceBuffer.alloc(d, bytes);
    try b.fill8(0, null);
    return b;
}

fn zeros(d: *const cuda.Driver, bytes: usize) !u64 {
    const b = try empty(d, bytes);
    return b.ptr;
}

fn take(own: *cuda.exl3_upload.Uploader, ck: *core.Checkpoint, prefix: []const u8, name: []const u8, dtype: anytype, shape: []const usize) !u64 {
    var nb: [192]u8 = undefined;
    const full = try std.fmt.bufPrint(&nb, "{s}{s}", .{ prefix, name });
    const t = try ck.get(full);
    if (!t.is(dtype, shape)) {
        std.debug.print("unexpected tensor {s}: {any} {any}\n", .{ full, t.dtype, t.shape[0..t.rank] });
        return error.BadHy4CheckpointTensor;
    }
    return own.dev(t.bytes);
}
