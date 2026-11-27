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
const exl3 = @import("moe_exl3_cuda.zig");
const dense = @import("dense_weights.zig");
const engram = @import("engram.zig");
const engram_cuda = @import("engram_cuda.zig");
const direct = @import("bf16_rows.zig");
const engram_pass = @import("engram_pass.zig");
const geom = @import("cuda_scratch.zig");
const cuda_index = @import("cuda_index.zig");
const cuda_vision = @import("cuda_vision.zig");
const dfly = @import("cuda_draft.zig");

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
    /// The vision tower's glue (LayerNorm, GELUs, pixel unshuffle) and its frame attention: loaded always, read by
    /// `--vision` only.
    vision_glue: cuda.vit.Kernels,
    pixels: cuda.vit.Attention,
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

    /// Every image and the shared EXL3 scratch, released before the context that made them.
    pub fn deinit(self: *Kernels) void {
        self.indexer.deinit();
        self.engram.deinit();
        self.router.unload();
        self.glue.unload();
        self.act.unload();
        self.dense.unload();
        self.experts.unload();
        self.ihc.deinit();
        self.attention.deinit();
        self.project.deinit();
        self.pixels.unload();
        self.vision_glue.unload();
        self.norms.unload();
        self.gemm.deinit();
        self.* = undefined;
    }

    /// The kernel images embedded in this build (`-Dnvcc` or `-Dprebuilt`).
    pub fn load(a: std.mem.Allocator, d: *const cuda.Driver) !Kernels {
        return .{
            .gemm = try cuda.vit.Gemms.init(a, d),
            .vision_glue = try cuda.vit.Kernels.load(d),
            .pixels = try cuda.vit.Attention.load(d),
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

/// An indexed layer's E4M3 cache (`--kv-cache fp8`): a scale a row, and the roped 64 columns in a BF16 side plane
/// that `cache` (the codes) no longer holds.
pub const Kv8 = struct { scales: cuda.DeviceBuffer, ropes: cuda.DeviceBuffer };

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
    /// An indexed layer's E4M3 rows: null on a BF16 cache, which is every sliding layer and the whole default arm.
    kv8: ?Kv8 = null,
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
    /// `config.json`'s own parse: `text`'s slices point into its arena, so it lives as long as the model does.
    cfg: std.json.Parsed(config.Model),
    /// The checkpoint's mappings: the Engram token map reads its vocabulary out of one, so they outlive the load.
    ck: *core.Checkpoint,
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
    /// `--kv-cache fp8`: the 12 indexed layers' KV as E4M3 rows. The sliding layers' ring stays BF16.
    kv8: bool,
    /// Sequences the caches hold, one after another: `--parallel`.
    slots: usize,
    /// The sliding layers' KV as `geom.ring.rows` slots; false is the full-length arm the ring is gated against.
    ring: bool,
    /// One token history a sequence, so a pass's Engram rows and a reuse read their own.
    history: []std.ArrayList(u32),
    expert_scratch: exl3.Shared = .{},
    weight_bytes: usize = 0,
    /// The Engram tables in layer order, so a pass starts every table's rows at once.
    engrams: []*const direct.Table = &.{},
    /// Host nanoseconds a pass spent on the Engram rows: planning, reading and waiting.
    engram_ns: i96 = 0,
    /// The pack's DFly2 drafter, when a round holds drafts: `cuda_draft.zig`, and null on the one-row path.
    draft: ?dfly.Draft = null,
    /// The image tower (`--vision`), and a sequence's prompt images: null and empty without it (`cuda_vision.zig`).
    vision: ?*cuda_vision.Tower = null,
    images: []?Images = &.{},
    /// A row's four streams after every layer, for the row-count check: does this row depend on its pass's width.
    trace: ?Trace = null,

    pub const Trace = struct {
        row: usize,
        buf: cuda.DeviceBuffer,
        /// The row's router picks and it after every layer: a near tie that flips with the pass's shape moved the row.
        picks: cuda.DeviceBuffer,
        scores: cuda.DeviceBuffer,
    };

    /// What one load takes: the context's rows, the widest pass, the ring, the drafts a round holds, the sequences.
    pub const Options = struct {
        capacity: usize,
        max_rows: usize,
        ring: bool = true,
        drafts: u32 = 0,
        slots: usize = 1,
        /// `--kv-cache fp8`: an E4M3 row for the indexed layers' KV.
        kv8: bool = false,
        /// `--vision`: load the image tower, for images of at most `vision_tokens` LM tokens.
        vision: bool = false,
        vision_tokens: u32 = 0,
    };

    /// A sequence's prompt images: the rows the tower encoded for it, and the positions and Engram mask they impose.
    pub const Images = struct {
        plan: cuda_vision.Plan,
        rows: cuda.DeviceBuffer,
        /// One row a group: the 3-D position of the token a group starts at, which the indexer rotates its keys by.
        groups: []i32,
    };

    /// The images `slot` holds, when its prompt has any.
    pub fn imagesOf(self: *Model, slot: usize) ?*Images {
        if (slot >= self.images.len) return null;
        return if (self.images[slot]) |*im| im else null;
    }

    /// Plans `slot`'s prompt images and the rows they read: the caller encodes each into `rows` at its span's `base`.
    /// The tower rows stay until the plan is replaced, so a chunked fill reads its image rows across passes.
    pub fn setImages(self: *Model, slot: usize, inputs: []const cuda_vision.Input, prompt_len: usize) !void {
        if (self.vision == null or slot >= self.images.len) return error.BadHy4VisionSlot;
        self.clearImages(slot);
        const plan = try cuda_vision.Plan.build(self.a, inputs, prompt_len);
        errdefer {
            var p = plan;
            p.deinit();
        }
        var rows = try cuda.DeviceBuffer.alloc(self.d, @max(plan.rows, 1) * H * 2);
        errdefer rows.free();
        const groups = try self.a.alloc(i32, cuda_index.groups_of(self.capacity) * 3);
        @memset(groups, 0);
        self.images[slot] = .{ .plan = plan, .rows = rows, .groups = groups };
    }

    pub fn clearImages(self: *Model, slot: usize) void {
        if (slot >= self.images.len) return;
        if (self.images[slot]) |*im| {
            im.rows.free();
            self.a.free(im.groups);
            im.plan.deinit();
            self.images[slot] = null;
        }
    }

    /// Loads the tower for images of at most `tokens` LM tokens: after the layers, so the text path's addresses are the
    /// ones a load without `--vision` gives them.
    pub fn loadVision(self: *Model, dir: []const u8, image_tokens: u32) !void {
        const cfg = (try cuda_vision.vc.Config.read(self.a, self.io, dir)) orelse return error.NoHy4VisionConfig;
        self.images = try self.a.alloc(?Images, self.slots);
        for (self.images) |*im| im.* = null;
        self.vision = try cuda_vision.Tower.load(self.a, self.d, self.s, &self.k.gemm, &self.k.vision_glue, &self.k.pixels, &self.k.norms, self.ck, &self.own, cfg, image_tokens);
    }

    /// A sequence's first slot in an indexed layer's cache, and in a sliding layer's ring.
    pub fn seqBase(self: *const Model, slot: usize) usize {
        return slot * self.capacity;
    }

    pub fn ringBase(self: *const Model, slot: usize) usize {
        return slot * (if (self.ring) geom.ring.rows else self.capacity);
    }

    /// A sequence's tokens so far, and a cut back to `n` of them (a rewind, a reuse, a failed pass).
    pub fn len(self: *const Model, slot: usize) usize {
        return self.history[slot].items.len;
    }

    pub fn cut(self: *Model, slot: usize, n: usize) void {
        self.history[slot].items.len = n;
    }

    pub fn tokens(self: *const Model, slot: usize) []const u32 {
        return self.history[slot].items;
    }

    fn hcShape(self: *const Model, rows: usize, gates: u32) ihc.Shape {
        return .{ .rows = @intCast(rows), .width = H, .gates = gates, .eps = self.text.rms_norm_eps, .gate_eps = self.text.hc_eps, .post_magnitude = self.text.hc_magnitude };
    }

    /// One projection's rows: the pack's EXL3 group through the dense kernels, or its BF16 weight through cuBLAS.
    pub fn project(self: *Model, s: cuda.Stream, p: dense.Projection, x: u64, rows: usize, y: u64, out: cuda.cublaslt.DataType) !void {
        switch (p.weight) {
            .bf16 => |w| try self.k.gemm.run(s, x, w, y, rows, p.n, p.k, out),
            .exl3 => |w| try self.k.dense.prompt(s, w, x, .bf16, rows, y, switch (out) {
                .f32 => .f32,
                .f16 => .f16,
                .bf16 => .bf16,
            }),
        }
    }

    /// Which pass rows to hand back: `out` holds one `vocab`-wide FP32 row a pass row from `at` on, `head` at a time.
    pub const Rows = struct { at: usize, out: []f32, head: ?usize = null };

    /// Run `ids` (appended at sequence 0's own history end) through the model; returns the last row's logits.
    pub fn pass(self: *Model, ids: []const u32, logits_out: []f32) !void {
        return self.passRows(ids, logits_out, null);
    }

    /// `pass`, and the rows `rows_out` names into it, as one `vocab`-wide FP32 row each.
    pub fn passRows(self: *Model, ids: []const u32, logits_out: []f32, rows_out: ?Rows) !void {
        return self.passRuns(&.{.{ .slot = 0, .tokens = ids }}, logits_out, rows_out);
    }

    /// `runs`: one pass over several sequences' rows, each run appended at its own history's end, its rows from `at`.
    /// A sequence is one run: two runs for one slot would append its tokens twice, the second at a start the first moved.
    pub fn passRuns(self: *Model, runs: []const geom.Run, logits_out: []f32, rows_out: ?Rows) !void {
        if (runs.len == 0 or runs.len > geom.max_slots) return error.BadHy4PassRows;
        var rows: usize = 0;
        var first: [geom.max_slots]usize = undefined; // each run's first pass row
        var starts: [geom.max_slots]usize = undefined; // and its first position in its own sequence
        var used: [geom.max_slots]bool = @splat(false); // one run a sequence, or its rows would append twice
        for (runs, 0..) |run, i| {
            if (run.slot >= self.slots or used[run.slot]) return error.BadHy4PassRows;
            used[run.slot] = true;
            const start = self.history[run.slot].items.len;
            if (run.tokens.len == 0 or start + run.tokens.len > self.capacity) return error.BadHy4PassRows;
            first[i] = rows;
            starts[i] = start;
            rows += run.tokens.len;
        }
        if (rows == 0 or rows > self.max_rows) return error.BadHy4PassRows;
        // Every row's sequence and its position in it, in pass order: a run's rows are its own sequence's, in order.
        const pass_a = std.heap.page_allocator; // a pass's own buffers are freed a pass; `a` keeps them a process
        const row_slot = try pass_a.alloc(u32, rows);
        defer pass_a.free(row_slot);
        const row_pos = try pass_a.alloc(u32, rows);
        defer pass_a.free(row_pos);
        const row_set = try pass_a.alloc(u32, rows);
        defer pass_a.free(row_set);
        const row_tok = try pass_a.alloc(u32, rows);
        defer pass_a.free(row_tok);
        const selecting = try pass_a.alloc(bool, runs.len);
        defer pass_a.free(selecting);
        for (runs, 0..) |run, i| {
            const last = starts[i] + run.tokens.len - 1;
            // Past the key budget an indexed row's list comes from its own sequence's indexer, so nothing is built.
            selecting[i] = self.capacity > sparse_stride and last >= sparse_stride;
            for (0..run.tokens.len) |j| {
                row_slot[first[i] + j] = @intCast(run.slot);
                row_pos[first[i] + j] = @intCast(starts[i] + j);
                row_set[first[i] + j] = @intCast(i);
                row_tok[first[i] + j] = run.tokens[j];
            }
        }
        for (runs) |run| try self.history[run.slot].appendSlice(self.a, run.tokens);
        const sc = self.scratch;
        const s = self.s;
        const k = self.k;
        const eps = self.text.rms_norm_eps;
        // A row's own image, when its sequence's prompt holds one: its tower row, its 3-D position, its Engram mask.
        const sources = try pass_a.alloc(i32, rows);
        defer pass_a.free(sources);
        const positions = try pass_a.alloc(i32, rows * 3);
        defer pass_a.free(positions);
        var any_image = false;
        for (0..rows) |r| {
            const im = self.imagesOf(row_slot[r]) orelse {
                sources[r] = -1;
                continue;
            };
            sources[r] = im.plan.featureRow(row_pos[r]);
            positions[r * 3 ..][0..3].* = im.plan.positionAt(row_pos[r]);
            any_image = any_image or sources[r] >= 0;
            // The indexer rotates a group's key by its first token's position: write the plane as those rows pass.
            if (row_pos[r] % @import("index_groups.zig").slots == 0) {
                const g = row_pos[r] / @import("index_groups.zig").slots;
                @memcpy(im.groups[g * 3 ..][0..3], positions[r * 3 ..][0..3]);
            }
        }
        // Embedding into four identical streams: an image row takes the tower's row, every other its token's.
        var x = sc.streams[0].ptr;
        for (row_tok, 0..) |tok, r| {
            const src = if (sources[r] < 0) self.embed + @as(u64, tok) * H * 2 else source: {
                const im = self.imagesOf(row_slot[r]) orelse return error.BadHy4VisionSlot;
                break :source im.rows.ptr + @as(u64, @intCast(sources[r])) * H * 2;
            };
            for (0..S) |st| try sc.streams[0].copyFrom((r * S + st) * H * 2, src, H * 2, s.handle);
        }
        // Positions (text: t = h = w), write slots and key lists for both attention kinds.
        const writes = try pass_a.alloc(u32, rows);
        defer pass_a.free(writes);
        const ring_writes = try pass_a.alloc(u32, rows);
        defer pass_a.free(ring_writes);
        for (0..rows) |r| {
            const p: i32 = @intCast(row_pos[r]);
            if (self.imagesOf(row_slot[r]) == null) positions[r * 3 ..][0..3].* = .{ p, p, p };
            // An indexed layer's cache is every sequence's in turn; a sliding layer's ring the same.
            writes[r] = @intCast(self.seqBase(row_slot[r]) + row_pos[r]);
            // Sliding layers read and write the ring's slots; with the ring off they keep their position's row.
            ring_writes[r] = @intCast(self.ringBase(row_slot[r]) + if (self.ring) geom.ring.slot(row_pos[r]) else row_pos[r]);
        }
        // The Engram branch leaves an image row's residual alone (the reference's media mask).
        if (any_image) {
            const media = try pass_a.alloc(u32, rows);
            defer pass_a.free(media);
            for (media, sources) |*keep, src| keep.* = @intFromBool(src >= 0);
            try (cuda.DeviceBuffer{ .d = self.d, .ptr = sc.media, .len = rows * 4, .borrowed = true }).uploadAsync(0, std.mem.sliceAsBytes(media), s.handle);
        }
        try sc.positions.uploadAsync(0, std.mem.sliceAsBytes(positions), s.handle);
        try sc.write_slots.uploadAsync(0, std.mem.sliceAsBytes(writes), s.handle);
        try sc.write_ring.uploadAsync(0, std.mem.sliceAsBytes(ring_writes), s.handle);
        const keys_sparse = try pass_a.alloc(u32, rows * sparse_stride);
        defer pass_a.free(keys_sparse);
        const keys_sliding = try pass_a.alloc(u32, rows * sliding_stride);
        defer pass_a.free(keys_sliding);
        const counts_sparse = try pass_a.alloc(u32, rows);
        defer pass_a.free(counts_sparse);
        const counts_sliding = try pass_a.alloc(u32, rows);
        defer pass_a.free(counts_sliding);
        @memset(keys_sparse, 0);
        @memset(keys_sliding, 0);
        for (0..rows) |r| {
            const slot = row_slot[r];
            const pos = row_pos[r];
            if (!selecting[row_set[r]]) {
                counts_sparse[r] = @intCast(pos + 1);
                for (0..pos + 1) |j| keys_sparse[r * sparse_stride + j] = @intCast(self.seqBase(slot) + j);
            }
            const from = if (pos + 1 > sliding_stride) pos + 1 - sliding_stride else 0;
            counts_sliding[r] = @intCast(pos + 1 - from);
            for (from..pos + 1, 0..) |j, n| keys_sliding[r * sliding_stride + n] = @intCast(self.ringBase(slot) + if (self.ring) geom.ring.slot(j) else j);
        }
        // Engram rows for the pass: a row hashes from its own sequence's tokens, so the four tables start at once.
        const windows = try pass_a.alloc([engram.order]u32, rows);
        defer pass_a.free(windows);
        const pad = self.text.engram_pad_id;
        for (0..rows) |r| {
            const hist = self.history[row_slot[r]].items;
            const pos = row_pos[r];
            for (&windows[r], 0..) |*v, back| v.* = if (pos >= back) hist[pos - back] else pad;
        }
        const et0 = std.Io.Clock.awake.now(self.io);
        var gathers = try engram_pass.Rows.start(pass_a, self.io, self.engrams, windows, engram_cuda.input, engram_readers);
        defer gathers.close(self.io);
        self.engram_ns += et0.durationTo(std.Io.Clock.awake.now(self.io)).toNanoseconds();

        // DFly2 reads a kept row's hidden state and its five target layers' recalled lists: a run's own last row.
        if (self.draft) |*dr| for (runs, 0..) |run, i| {
            const whole = run.tokens.len <= dfly.width;
            const at = if (whole) first[i] else first[i] + run.tokens.len - 1;
            const sq = &dr.scratch.seq[run.slot];
            sq.selected = selecting[i];
            sq.cap_at = at;
            sq.keep = if (whole) run.tokens.len else 1;
            // The head's own rows are the pass's when the caller asked for every row, else the one row it kept.
            sq.head0 = if (rows_out) |want| first[i] - want.at else 0;
            sq.anchor = if (rows_out) |want| at - want.at else 0;
        };
        var sets: [geom.max_slots]cuda_index.RowSet = undefined;
        for (runs, 0..) |run, i| sets[i] = .{ .slot = run.slot, .first = starts[i], .rows = run.tokens.len, .row0 = first[i], .select = selecting[i] };
        // A sequence with images ranks its groups by their first token's own position, not by the token's slot.
        var planes: [geom.max_slots][]const i32 = @splat(&.{});
        var any_plane = false;
        for (runs, 0..) |run, i| if (self.imagesOf(run.slot)) |im| {
            planes[i] = im.groups;
            any_plane = true;
        };
        const rope_sparse = try self.text.rotary(.deepseek_sparse_attention);
        var slot: usize = 0; // index of the stream buffer x lives in
        for (self.layers, 0..) |*layer, layer_index| {
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
            var any_dense = false;
            for (selecting) |sel| any_dense = any_dense or !sel;
            // The rows below the budget attend their own prefix; the indexer then overwrites a selecting run's rows.
            if (at.sliding) {
                try sc.slots.uploadAsync(0, std.mem.sliceAsBytes(keys_sliding), s.handle);
                try sc.counts.uploadAsync(0, std.mem.sliceAsBytes(counts_sliding), s.handle);
            } else if (any_dense) {
                try sc.slots.uploadAsync(0, std.mem.sliceAsBytes(keys_sparse), s.handle);
                try sc.counts.uploadAsync(0, std.mem.sliceAsBytes(counts_sparse), s.handle);
            }
            if (at.index) |ix| {
                try cuda_index.forward(.{
                    .s = s, .indexer = &k.indexer, .gemms = &k.gemm, .norms = &k.norms, .project = k.project,
                    .scratch = &self.index, .layer = ix, .hidden = sc.hidden, .q_resid = sc.q_resid,
                    .positions = sc.positions.ptr, .table = sc.table, .slots = sc.slots.ptr, .counts = sc.counts.ptr,
                    .sets = sets[0..runs.len], .group_planes = if (any_plane) planes[0..runs.len] else null,
                    .rows = rows, .epsilon = eps, .rope = rope_sparse, .q_rank = Q,
                    .hidden_size = H,
                });
                if (self.draft) |*dr| for (sets[0..runs.len]) |set| {
                    if (!set.select) continue;
                    try self.capture(dr, layer_index, set.slot);
                };
            }
            const shape = try attn.Shape.init(@intCast(rows), at.heads, @intCast(stride));
            const scales: u64 = if (at.kv8) |q| q.scales.ptr else 0;
            const ropes: u64 = if (at.kv8) |q| q.ropes.ptr else 0;
            try mla.forward(k.project, k.attention, s, .{
                .query = sc.q,
                .key_weights = at.key,
                .value_weights = at.value,
                .value_logits = sc.gate,
                .positions = sc.positions.ptr,
                .rotary_table = sc.table,
                .kv = .{ .projection = sc.kv, .gain = at.kv_norm, .write_slots = if (at.sliding) sc.write_ring.ptr else sc.write_slots.ptr, .capacity = at.rows, .epsilon = eps, .scales = scales, .ropes = ropes },
                .attention = .{
                    .query = sc.absorb,
                    .cache = at.cache.ptr,
                    .slots = sc.slots.ptr,
                    .counts = sc.counts.ptr,
                    .sinks = at.sinks,
                    .scratch = sc.attn_scratch,
                    .scratch_bytes = sc.attn_scratch_bytes,
                    .output = sc.latent_out,
                    .scales = scales,
                    .ropes = ropes,
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
            const feed = &layer.feed;
            try k.ihc.collapse(s, .{ .hidden = x, .projection = feed.hc.project, .bias = feed.hc.bias, .scale = feed.hc.scale, .output = sc.collapsed, .post = sc.post }, self.hcShape(rows, 2 * S));
            try k.norms.addRmsTail(s, sc.collapsed, null, feed.input_norm, sc.norm_tmp, sc.hidden, null, rows, H, eps, .{ .round = true });
            switch (feed.mlp) {
                .dense => |dn| {
                    try self.project(s, dn.gate, sc.hidden, rows, sc.ff_gate, .bf16);
                    try self.project(s, dn.up, sc.hidden, rows, sc.ff_up, .bf16);
                    try k.act.swiglu(s, sc.ff_gate, D, sc.ff_up, D, sc.ff_inner, null, rows, D);
                    try self.project(s, dn.down, sc.ff_inner, rows, sc.shared, .bf16);
                    try k.norms.addRmsTail(s, sc.shared, null, feed.output_norm, sc.norm_tmp, sc.branch_norm, null, rows, H, eps, .{ .round = true });
                },
                .sparse => |sp| {
                    try k.gemm.run(s, sc.hidden, sp.latent_in, sc.latent, rows, L, H, .bf16);
                    try k.router.logits(s, sc.hidden, H, sp.router, sc.logits, rows, H, E);
                    try k.glue.topk(s, sc.logits, sp.router_bias, .{ .experts = E, .topk = K, .shared = false, .score = true, .normalize = true, .scale = self.text.routed_scaling_factor }, sc.picks, sc.probabilities, rows);
                    if (self.trace) |t| if (t.row < rows) {
                        try t.picks.copyFrom(layer_index * K * 4, sc.picks + t.row * K * 4, K * 4, s.handle);
                        try t.scores.copyFrom(layer_index * E * 4, sc.logits + t.row * E * 4, E * 4, s.handle);
                    };
                    const y = try sp.routed.run(&k.experts, s, sc.latent, sc.picks, rows);
                    try k.glue.combine(s, y, .f32, sc.probabilities, sc.mix, .bf16, rows, K, L);
                    try k.gemm.run(s, sc.mix, sp.latent_out, sc.route_out, rows, H, L, .bf16);
                    try self.project(s, sp.shared_gate, sc.hidden, rows, sc.ff_gate, .bf16);
                    try self.project(s, sp.shared_up, sc.hidden, rows, sc.ff_up, .bf16);
                    try k.act.swiglu(s, sc.ff_gate, L, sc.ff_up, L, sc.ff_inner, null, rows, L);
                    try self.project(s, sp.shared_down, sc.ff_inner, rows, sc.shared, .bf16);
                    try k.norms.addRmsTail(s, sc.route_out, sc.shared, feed.output_norm, sc.norm_tmp, sc.branch_norm, null, rows, H, eps, .{ .round = true });
                },
            }
            const after_ff = sc.streams[(slot + 1) % 3].ptr;
            try k.ihc.add(s, .{ .residual = x, .branch = sc.branch_norm, .post = sc.post, .output = after_ff }, self.hcShape(rows, 2 * S));
            if (self.trace) |t| if (t.row < rows) {
                try t.buf.copyFrom(layer_index * S * H * 2, after_ff + t.row * S * H * 2, S * H * 2, s.handle);
            };
            slot = (slot + 1) % 3;
            x = after_ff;
        }
        // Hyper head, final norm and LM head over the rows the caller keeps, a head block at a time.
        const V: usize = self.text.vocab_size;
        var held = false; // the caller's range covers the pass's last row, so its logits come from there
        if (rows_out) |want| {
            const count = want.out.len / V;
            if (want.out.len != count * V or count == 0 or want.at + count > rows) return error.BadHy4PassRows;
            try self.headRows(s, x, want.at, count, want.out, want.head orelse geom.head_block);
            if (want.at + count == rows) {
                // The pass's last row is in range, so `logits_out` takes it too — unless that is the same storage.
                const last = want.out[(rows - 1 - want.at) * V ..][0..V];
                if (!inside(want.out, logits_out)) @memcpy(logits_out[0..V], last);
                held = true;
            }
        }
        // One row of logits runs one head row: the pass's block width here would read past the pass's own rows.
        if (!held) try self.headRows(s, x, rows - 1, 1, logits_out, @min(geom.head_block, 1));
    }

    /// Whether `part` lies wholly inside `whole`: a verify window's last row must not overwrite its first.
fn inside(whole: []const f32, part: []const f32) bool {
    const start = @intFromPtr(whole.ptr);
    const at = @intFromPtr(part.ptr);
    return at >= start and at + part.len * @sizeOf(f32) <= start + whole.len * @sizeOf(f32);
}

/// The caches, the pass's scratch, the Engram tables and the weights, released before the context is.
    pub fn free(self: *Model) void {
        if (self.draft) |*dr| dfly.free(self.a, dr);
        // The tower's weights live in the upload table, so the tower goes before it; its rows are the plans'.
        for (0..self.images.len) |i| self.clearImages(i);
        if (self.images.len > 0) self.a.free(self.images);
        if (self.vision) |v| v.deinit();
        for (self.layers) |*layer| {
            layer.attention.cache.free();
            if (layer.attention.kv8) |*q| {
                q.scales.free();
                q.ropes.free();
            }
            if (layer.attention.index) |ix| {
                ix.token_keys.free();
                ix.group_keys.free();
                self.a.destroy(ix);
            }
            if (layer.engram) |*e| e.table.close();
        }
        var sc = self.scratch;
        for (&sc.streams) |*b| b.free();
        var owned = [_]cuda.DeviceBuffer{ sc.positions, sc.slots, sc.counts, sc.write_slots, sc.write_ring, sc.e_raw, self.index.recall_rows, self.index.scores, self.index.group_positions };
        for (&owned) |*b| b.free();
        self.own.deinit();
        for (self.history) |*h| h.deinit(self.a);
        self.a.free(self.history);
        self.cfg.deinit();
        self.ck.close();
        self.a.destroy(self.ck);
        const a = self.a;
        a.free(self.engrams);
        a.free(self.layers);
        a.destroy(self);
    }

    /// A draft layer's share of a pass: its target layer's recall lists for the rows a later proposal reads.
    fn capture(self: *Model, dr: *dfly.Draft, layer: usize, slot: usize) !void {
        const keep = dr.scratch.seq[slot].keep;
        const at = dr.scratch.seq[slot].cap_at;
        for (dr.layers, 0..) |l, i| {
            if (l.target_index != layer) continue;
            const at_slot = i * dr.scratch.seq.len + slot;
            try dr.scratch.lists[at_slot].copyFrom(0, self.scratch.slots.ptr + at * sparse_stride * 4, keep * sparse_stride * 4, self.s.handle);
            try dr.scratch.captured[at_slot].copyFrom(0, self.scratch.counts.ptr + at * 4, keep * 4, self.s.handle);
        }
    }

    /// `count` pass rows from `at` through the hyper head, the final norm and the LM head, `block` rows a call.
    fn headRows(self: *Model, s: cuda.Stream, x: u64, at: usize, count: usize, out: []f32, block: usize) !void {
        const sc = self.scratch;
        const k = self.k;
        const V: usize = self.text.vocab_size;
        if (out.len != count * V or at + count > self.max_rows or block == 0 or block > geom.head_block) return error.BadHy4PassRows;
        const logits: cuda.DeviceBuffer = .{ .d = self.d, .ptr = sc.lm_logits, .len = geom.head_block * V * 4, .borrowed = true };
        var done: usize = 0;
        while (done < count) : (done += block) {
            const keep = @min(count - done, block);
            const hidden = x + (at + done) * S * H * 2;
            const shape = self.hcShape(block, S);
            try k.ihc.collapse(s, .{ .hidden = hidden, .projection = self.head_hc.project, .bias = self.head_hc.bias, .scale = self.head_hc.scale, .output = sc.head_hidden, .post = 0 }, shape);
            try k.norms.addRmsTail(s, sc.head_hidden, null, self.final_norm, sc.norm_tmp, sc.head_norm, null, block, H, self.text.rms_norm_eps, .{ .round = true });
            try self.project(s, self.lm_head, sc.head_norm, block, sc.lm_logits, .f32);
            try logits.downloadAsync(0, std.mem.sliceAsBytes(out[done * V ..][0 .. keep * V]), s.handle);
        }
        try s.synchronize();
    }
};

test "a pass takes one run a sequence: a second run for a slot is refused before any row is appended" {
    const a = std.testing.allocator;
    const hists = try a.alloc(std.ArrayList(u32), 2);
    defer a.free(hists);
    hists[0] = .empty;
    hists[1] = .empty;
    var mdl: Model = undefined;
    mdl.slots = 2;
    mdl.capacity = 4096;
    mdl.max_rows = 1024;
    mdl.history = hists;
    var logits: [1]f32 = .{0};
    const two = [_]u32{ 7, 8 };
    // One run a sequence is the shape a shared round has: each sequence's rows appended at its own end.
    try std.testing.expectError(error.BadHy4PassRows, mdl.passRuns(&.{ .{ .slot = 0, .tokens = &two }, .{ .slot = 0, .tokens = &two } }, &logits, null));
    try std.testing.expectEqual(@as(usize, 0), hists[0].items.len); // refused before the first append
    try std.testing.expectError(error.BadHy4PassRows, mdl.passRuns(&.{ .{ .slot = 2, .tokens = &two } }, &logits, null));
    try std.testing.expectError(error.BadHy4PassRows, mdl.passRuns(&[_]geom.Run{}, &logits, null));
}
