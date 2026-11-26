//! Hy4-flash behind the lane engine on CUDA: one sequence, the step 1-3 text model with the indexer and the ring.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const model = @import("cuda_model.zig");
const dfly = @import("cuda_draft.zig");
const geom = @import("cuda_scratch.zig");
const mla = @import("mla_weights.zig");
const be = lanes.backend;

/// A prompt chunk's rows: the dev binary's cap and the model's `max_rows`.
pub const prompt_rows: u32 = geom.row_cap;
/// The widest context this engine ranks: the indexer's recall scores, 65,536 groups of four slots.
pub const max_capacity: u32 = @import("cuda_index.zig").max_capacity;
pub const Kernels = model.Kernels;
pub const Model = model.Model;
/// A verify window's rows: the pending row and DFly2's five drafts, and the drafts a round proposes.
pub const max_window: u32 = dfly.width;
pub const drafts: u32 = dfly.drafts;
/// The sliding layers' ring: a rewind may read back only what the ring still holds (the ring commit's review).
const ring_rows: u32 = geom.ring.rows;
const ring_window: u32 = geom.ring.window;
/// The prompt prefix a reuse shares before its caches are kept rather than refilled from row 0.
const keep_prefix = 64;

/// A context's caches in bytes, so a size decision is read and not inferred.
pub const Bytes = struct { kv: u64 = 0, index: u64 = 0, ring: u64 = 0, scores: u64 = 0 };

pub fn cacheBytes(m: *const model.Model) Bytes {
    var b: Bytes = .{};
    for (m.layers) |layer| {
        if (layer.attention.sliding) b.ring += @as(u64, layer.attention.rows) * mla.absorbed_stride * 2 else b.kv += @as(u64, layer.attention.rows) * mla.absorbed_stride * 2;
        if (layer.attention.index) |ix| b.index += ix.token_keys.len + ix.group_keys.len;
    }
    b.scores = m.index.scoreBytes();
    return b;
}

/// What the system has free now, as the loader's own guard reads it: a GB10's device memory is the system's.
pub fn memAvailable() u64 {
    return @import("cuda_load_util.zig").memAvailable();
}

/// Per-token bytes the model's caches take at its capacity: the full-length KV, the indexer's keys and the scores.
pub fn bytesPerToken(m: *const model.Model) u64 {
    const b = cacheBytes(m);
    return (b.kv + b.index + b.scores) / @max(@as(u64, m.capacity), 1);
}

pub const Cuda = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    m: *model.Model,
    vocab: u32,
    logits: []f32, // the last pass's last row, FP32 as the model writes it: row 0 of `window`
    window: []f32, // every row of a verify window, `vocab` wide each
    words: []u16, // the row at hand as bf16 words, for the lane sampler
    /// DFly2's drafts for the next round, and how many of them the last proposal filled.
    held: [dfly.drafts]u32 = @splat(0),
    nheld: u32 = 0,
    valid_from: u32 = 0, // the oldest position whose ring slot still holds that position's own row
    kept: bool = false, // the caches hold the last released stream's tokens
    prefill_rows: u64 = 0,
    prefill_ns: i96 = 0,
    decode_rows: u64 = 0,
    decode_ns: i96 = 0,
    /// DFly2's own count: rounds that were proposed to, drafts proposed, and drafts the target then kept.
    draft_rounds: u64 = 0,
    draft_proposed: u64 = 0,
    draft_kept: u64 = 0,
    draft_ns: i96 = 0,
    /// `TF_HY4_ROUND_LOG`: every round's window, its rows' draws and what it kept, on stderr.
    log_rounds: bool = false,

    /// Loads the pack and its caches: `capacity` slots a sequence, the widest pass `prompt_rows` rows, and
    /// `drafts` DFly2 drafts a round (0: the plain one-row path, which loads no drafter at all).
    pub fn init(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, k: *model.Kernels, dir: []const u8, capacity: usize, rounds: u32) !*Cuda {
        const self = try a.create(Cuda);
        errdefer a.destroy(self);
        const m = try model.Model.open(a, io, d, s, k, dir, .{ .capacity = capacity, .max_rows = prompt_rows, .drafts = rounds });
        errdefer model.Model.free(m);
        const vocab: usize = m.text.vocab_size;
        const window = try a.alloc(f32, vocab * max_window);
        errdefer a.free(window);
        self.* = .{
            .gpa = a,
            .io = io,
            .m = m,
            .vocab = @intCast(vocab),
            .logits = window[0..vocab],
            .window = window,
            .words = try a.alloc(u16, vocab),
            .log_rounds = @import("sys").environment.read(bool, "TF_HY4_ROUND_LOG", false) catch false,
        };
        return self;
    }

    pub fn deinit(self: *Cuda) void {
        const a = self.gpa;
        a.free(self.words);
        a.free(self.window);
        model.Model.free(self.m);
        a.destroy(self);
    }

    pub fn bytes(self: *const Cuda) Bytes {
        return cacheBytes(self.m);
    }

    /// The oldest position a rewind to length `r` may read: its first row's window must still hold its own rows.
    pub fn mayRewind(self: *const Cuda, r: u32) bool {
        const first = if (r + 1 > ring_window) r + 1 - ring_window else 0;
        return first >= self.valid_from;
    }

    /// After a pass: the ring's high-water mark, so nothing before `H + 1 - ring_rows` is readable any more.
    fn bump(self: *Cuda) void {
        const h = self.m.history.items.len;
        const from = if (h > ring_rows) h - ring_rows else 0;
        self.valid_from = @max(self.valid_from, @as(u32, @intCast(from)));
    }

    /// `tokens` at the caches' end: the model's pass, then the ring's bookkeeping and the timers.
    pub fn pass(self: *Cuda, tokens: []const u32) !void {
        const start = self.m.history.items.len;
        self.logits = self.window[0..self.vocab];
        // Inside `capacity`, every write slot is inside its layer's arena: the write kernel skips a stray one silently.
        if (start + tokens.len > self.m.capacity) return error.Hy4ContextFull;
        // The model appends before it forwards: a failed pass leaves the history claiming rows nothing wrote.
        errdefer {
            self.m.history.items.len = start;
            self.kept = false;
        }
        const t0 = std.Io.Clock.awake.now(self.io);
        try self.m.pass(tokens, self.logits);
        const ns = t0.durationTo(std.Io.Clock.awake.now(self.io)).toNanoseconds();
        if (tokens.len == 1) {
            self.decode_rows += 1;
            self.decode_ns += ns;
            if (self.decode_rows % 128 == 0) std.log.debug("hy4-cuda: {d} decode rows, {d:.1} ms a token", .{ self.decode_rows, @as(f64, @floatFromInt(self.decode_ns)) / 1e6 / @as(f64, @floatFromInt(self.decode_rows)) });
        } else {
            self.prefill_rows += tokens.len;
            self.prefill_ns += ns;
        }
        self.bump();
    }

    /// The row's draw: greedy in the model's FP32 logits, else the lane sampler over the same row as bf16 words.
    fn draw(self: *Cuda, s: *lanes.Stream, position: u64, mask: []const u32) !u32 {
        if (mask.len == 0) {
            if (s.sampling) |v| {
                if (v.temperature <= 0) return self.greedy();
            } else return self.greedy();
        }
        for (self.logits, self.words) |v, *w| w.* = roundBf16(v);
        return lanes.sampling_row.draw(self.gpa, self.words, s.sampling, position, mask);
    }

    fn greedy(self: *const Cuda) u32 {
        var best: usize = 0;
        for (self.logits, 0..) |v, i| if (v > self.logits[best]) {
            best = i;
        };
        return @intCast(best);
    }

    /// One row's log probabilities over the model's FP32 logits, with the best `count` ids.
    fn row(self: *const Cuda, pick: u32, count: u8) lanes.LogprobRow {
        const top = lanes.logprob.max_top;
        var most: f64 = -std.math.inf(f64);
        for (self.logits) |v| most = @max(most, v);
        var sum: f64 = 0;
        for (self.logits) |v| sum += @exp(@as(f64, v) - most);
        const lse: f32 = @floatCast(most + @log(sum));
        var r: lanes.LogprobRow = .{ .token = pick, .logprob = self.logits[pick] - lse, .count = @min(count, top) };
        var filled: usize = 0;
        for (self.logits, 0..) |x, id| {
            if (filled == top and x <= r.logprobs[top - 1] + lse) continue;
            var at = @min(filled, top - 1);
            while (at > 0 and x > r.logprobs[at - 1] + lse) : (at -= 1) {
                r.ids[at] = r.ids[at - 1];
                r.logprobs[at] = r.logprobs[at - 1];
            }
            r.ids[at] = @intCast(id);
            r.logprobs[at] = x - lse;
            filled = @min(filled + 1, top);
        }
        return r;
    }

    /// The draw a reply starts from, and the one a window row takes: a loaded drafter widens both to its block.
    pub fn facts(self: *const Cuda) lanes.Model {
        const tower = self.m.draft != null;
        return .{
            .exact_width = if (tower) max_window else 1,
            .mtp = tower,
            .speculate = tower,
            .speculate_early = false,
            .drafts = if (tower) @intCast(dfly.drafts) else 0,
            .copy_drafts = false,
            .lone_copies = false,
            .max_streams = 1,
            .streams_exact = true,
        };
    }

    pub fn backend(self: *Cuda) be.Backend {
        // `keep_full_lone`: a window the target accepts whole is still a round's keep. The rewind is the identity
        // there, but the round's accepted count and the drafter's held state are read off `keep`, so ask for it.
        return .{ .ptr = self, .keep_full_lone = true, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .held = heldFn, .first_row = firstRowFn, .release = releaseFn } };
    }

    /// The window widths and a draft slot, timed once at load: ms a pass by rows (1..`max_window`) and ms a
    /// draft slot. The lane core's depth rule prices a round from these; without them it drafts one whenever its
    /// first draft's rate is under 0.8, whatever the block would have accepted. `serve_metal.zig`'s `warm` times the
    /// same widths. The caches end up as they started: empty, with the ring readable from row 0.
    pub const Timed = struct { rows: [max_window]f64 = @splat(0), slot_ms: f64 = 0 };

    pub fn timeWindows(self: *Cuda) !Timed {
        const dr = if (self.m.draft) |*x| x else return .{};
        var timed: Timed = .{};
        var ids: [dfly.width + 2]u32 = @splat(0);
        for (&ids, 0..) |*t, i| t.* = @intCast(1000 + i * 37);
        try self.pass(&ids);
        const at = self.m.history.items.len;
        for (1..max_window + 1) |width| {
            var times: [3]f64 = undefined;
            for (&times) |*tm| {
                self.m.history.items.len = at;
                const t0 = std.Io.Clock.awake.now(self.io);
                try self.m.passRows(ids[0..width], self.logits, .{ .at = 0, .out = self.window[0 .. width * self.vocab], .head = width });
                tm.* = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(self.io)).toNanoseconds())) / 1e6;
            }
            std.mem.sort(f64, &times, {}, std.sort.asc(f64));
            timed.rows[width - 1] = times[1];
        }
        // A draft slot's cost over a block of three: the first call also warms the tower's own path.
        var held: [dfly.drafts]u32 = @splat(0);
        self.m.history.items.len = at;
        for (0..2) |which| {
            const t0 = std.Io.Clock.awake.now(self.io);
            try dfly.propose(self.m, dr, at, ids[at - 1], 3, dfly.source(dr, null), held[0..3]);
            if (which == 1) timed.slot_ms = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(self.io)).toNanoseconds())) / 1e6 / 3;
        }
        std.log.info("hy4-cuda: window ms by rows {d:.2} {d:.2} {d:.2} {d:.2} {d:.2} {d:.2}, draft {d:.2} ms a slot", .{
            timed.rows[0], timed.rows[1], timed.rows[2], timed.rows[3], timed.rows[4], timed.rows[5], timed.slot_ms,
        });
        // The warm rows go: the first stream fills from row 0 as on a cold load.
        self.m.history.items.len = 0;
        self.valid_from = 0;
        self.nheld = 0;
        self.kept = false;
        self.decode_rows = 0;
        self.decode_ns = 0;
        self.prefill_rows = 0;
        self.prefill_ns = 0;
        return timed;
    }

    fn cast(ptr: *anyopaque) *Cuda {
        return @ptrCast(@alignCast(ptr));
    }

    /// The prompt into the caches in `prompt_rows`-row passes, resuming the kept caches' shared prefix where it may.
    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) !void {
        const self = cast(ptr);
        const ids = s.prompt();
        if (ids.len + s.max_new > self.m.capacity) return error.Hy4ContextFull;
        const t0 = std.Io.Clock.awake.now(self.io);
        const from = try self.fill(ids, s);
        const ns = t0.durationTo(std.Io.Clock.awake.now(self.io)).toNanoseconds();
        std.log.debug("hy4-cuda: prompt {d} tokens, {d} reused, {d} prefilled in {d:.1} ms", .{ ids.len, from, ids.len - from, @as(f64, @floatFromInt(ns)) / 1e6 });
    }

    /// `fill`, and the stream's own accounting: the rows it reused, its cancellation and each pass's width.
    pub fn fill(self: *Cuda, ids: []const u32, s: ?*lanes.Stream) !usize {
        if (ids.len == 0) return error.EmptyHy4Prompt;
        if (ids.len > self.m.capacity) return error.Hy4ContextFull;
        const from = self.reuseFrom(ids);
        if (s) |stream| stream.cached = @intCast(from);
        var at = from;
        while (at < ids.len) {
            if (s) |stream| {
                if (stream.isCancelled()) return error.Cancelled;
            }
            const end = @min(ids.len, at + prompt_rows);
            try self.pass(ids[at..end]);
            if (s) |stream| lanes.prefill_widths.completed(self.gpa, stream, end - at);
            at = end;
        }
        return from;
    }

    /// The row a fill starts at: the kept shared prefix, when it is long enough and the ring still holds its first row.
    pub fn reuseFrom(self: *Cuda, ids: []const u32) usize {
        var from: usize = 0;
        if (self.kept and ids.len > 0) {
            const hist = self.m.history.items;
            var same: usize = 0;
            const most = @min(hist.len, ids.len - 1);
            while (same < most and hist[same] == ids[same]) same += 1;
            if (same >= keep_prefix and self.mayRewind(@intCast(same))) from = same;
            self.kept = false;
        }
        if (self.m.history.items.len != from) {
            self.m.history.items.len = from;
            if (from == 0) self.valid_from = 0;
        }
        return from;
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) !u64 {
        const self = cast(ptr);
        if (position != self.m.history.items.len) return error.InvalidHy4Position;
        return try self.draw(s, position, &.{});
    }

    fn firstMaskedFn(ptr: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        const self = cast(ptr);
        if (position != self.m.history.items.len) return error.InvalidHy4Position;
        return try self.draw(s, position, mask);
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, f: be.Feed, position: u64) !u64 {
        const self = cast(ptr);
        if (position != self.m.history.items.len + 1) return error.InvalidHy4Position;
        const token: u32 = switch (f) {
            .value => |v| v,
            .handle => |h| @intCast(h),
        };
        try self.pass(&.{token});
        return try self.draw(s, position, &.{});
    }

    fn readFn(ptr: *anyopaque, handle: u64) !u32 {
        if (handle >= cast(ptr).vocab) return error.InvalidHy4Token;
        return @intCast(handle);
    }

    fn firstRowFn(ptr: *anyopaque, s: *lanes.Stream) !lanes.LogprobRow {
        const self = cast(ptr);
        return self.row(self.greedy(), s.logprobs orelse 0);
    }

    /// A window's rows in one pass, every row drawn at its own position; `keep` rolls the rejected rows back.
    /// With a drafter loaded the whole window is one 6-row forward whose rows all need logits; without one the
    /// window is its pending row alone and the pass hands back its last row as it always has.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        const words = (self.vocab + 31) / 32;
        if (windows.len != out.len) return error.InvalidHy4Windows;
        for (windows, out) |w, o| {
            const rows = w.rows();
            if (rows > max_window or w.parents != null or o.sampled.len != rows or o.drafts.len + 1 != rows) return error.UnsupportedHy4Window;
            const base = self.m.history.items.len;
            for (w.positions, 0..) |p, r| if (p != base + r + 1) return error.InvalidHy4Position;
            var ids: [max_window]u32 = undefined;
            ids[0] = w.pending;
            // The drafts the last proposal held, then the host's. A window wider than the drafts on hand (the
            // block did not fit the context, or a copy brought more rows) repeats its last one: any token is a
            // valid window row, and the accept rule still takes only the target's own samples.
            const held = @min(@as(usize, w.held), @as(usize, self.nheld));
            @memcpy(ids[1..][0..held], self.held[0..held]);
            for (held..w.held) |i| ids[1 + i] = if (held > 0) self.held[held - 1] else w.pending;
            @memcpy(ids[1 + w.held ..][0..w.tokens.len], w.tokens);
            self.nheld = 0;
            @memcpy(o.drafts, ids[1..rows]);
            if (w.held > 0) {
                self.draft_rounds += 1;
                self.draft_proposed += w.held;
                if (self.log_rounds) std.log.info("hy4-cuda: round {d} at row {d}: window {any}", .{ self.draft_rounds, base, ids[0..rows] });
            }
            if (self.m.draft != null) try self.verifyPass(ids[0..rows]) else try self.pass(ids[0..rows]);
            for (0..rows) |r| {
                self.logits = self.window[(@min(r, max_window - 1)) * self.vocab ..][0..self.vocab];
                const mask = if (w.masks.len >= (r + 1) * words) w.masks[r * words ..][0..words] else &.{};
                o.sampled[r] = try self.draw(w.stream, w.positions[r], mask);
                if (o.rows.len == rows) o.rows[r] = self.row(o.sampled[r], w.stream.logprobs orelse 0);
                if (self.log_rounds) std.log.info("hy4-cuda: round {d} row {d} (position {d}): drew {d} on {d:.4}", .{ self.draft_rounds, r, w.positions[r], o.sampled[r], self.logits[o.sampled[r]] });
            }
        }
        self.logits = self.window[0..self.vocab];
    }

    /// A window's rows in one forward, every row's logits kept: the drafter reads the last kept row's hidden
    /// state and its target layers' recalled lists from this pass, so it is the only pass a proposal follows.
    fn verifyPass(self: *Cuda, ids: []const u32) !void {
        const rows = ids.len;
        const start = self.m.history.items.len;
        if (start + rows > self.m.capacity) return error.Hy4ContextFull;
        self.logits = self.window[0..self.vocab];
        const t0 = std.Io.Clock.awake.now(self.io);
        try self.m.passRows(ids, self.window[0..self.vocab], .{ .at = 0, .out = self.window[0 .. rows * self.vocab], .head = rows });
        const ns = t0.durationTo(std.Io.Clock.awake.now(self.io)).toNanoseconds();
        self.decode_rows += rows;
        self.decode_ns += ns;
        self.bump();
    }

    /// Roll each stream back to the rows it kept: the target's caches hold one sequence, so this cuts its history.
    /// The row count never reaches the ring's 1,025-row rewind bound (a window is six rows), and `valid_from` only
    /// moves forward, so a rewind always reads slots this pass wrote.
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const self = cast(ptr);
        for (windows, paths) |w, path| {
            for (path, 0..) |r, i| if (r != i) return error.UnsupportedHy4Tree;
            const rows = w.rows();
            if (path.len == 0 or path.len > rows) return error.UnsupportedHy4Window;
            const base = self.m.history.items.len;
            if (base < rows) return error.InvalidHy4Position;
            self.m.history.items.len = base - rows + path.len;
            if (path.len > 1) self.draft_kept += path.len - 1;
            if (self.log_rounds) std.log.info("hy4-cuda: round {d} kept {d} of {d} rows (drafts {d} of {d}), history {d} -> {d}", .{ self.draft_rounds, path.len, rows, path.len - 1, rows - 1, base, self.m.history.items.len });
            self.nheld = 0;
        }
    }

    /// DFly2: a block of the round's anchor and five masks over the target's own caches, then one head a slot.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) !void {
        const self = cast(ptr);
        const dr = if (self.m.draft) |*x| x else return error.NoHy4DraftHead;
        for (requests) |r| {
            self.nheld = 0;
            if (r.depth > dfly.drafts) return error.UnsupportedHy4Window;
            const anchor = self.m.history.items.len;
            if (r.position != @as(u64, anchor) + 1) return error.InvalidHy4Position;
            if (r.depth == 0) continue;
            const kept: ?usize = if (r.rows) |rows| blk: {
                if (rows.len == 0) return error.UnsupportedHy4Window;
                for (rows, 0..) |x, i| if (x != i) return error.UnsupportedHy4Tree;
                break :blk rows.len - 1;
            } else null;
            const bonus: u32 = if (r.follow.len > 0) r.follow[r.follow.len - 1] else switch (r.first orelse return error.NoFirstToken) {
                .value => |v| v,
                .handle => |h| @intCast(h),
            };
            // The block's rows must fit the context, and a target that has selected nothing gives the dense
            // prefix, which is wider than a draft row's stride once the context passes 2,048 positions.
            const dense = !dr.scratch.selected;
            if (anchor + dfly.width > self.m.capacity or (dense and anchor + dfly.width > dfly.key_stride)) {
                // The block would write past the context: hold the anchor's own token so the planned window is
                // still one the target can verify, and let drafting resume once the reply is done.
                @memset(self.held[0..r.depth], bonus);
                self.nheld = r.depth;
                continue;
            }
            const t0 = std.Io.Clock.awake.now(self.io);
            try dfly.propose(self.m, dr, anchor, bonus, r.depth, dfly.source(dr, kept), self.held[0..r.depth]);
            if (self.draft_rounds <= 3) std.log.info("hy4-cuda: DFly2 block at {d} from row {d}: bonus {d}, proposals {any}", .{ anchor, dfly.source(dr, kept), bonus, self.held[0..r.depth] });
            self.draft_ns += t0.durationTo(std.Io.Clock.awake.now(self.io)).toNanoseconds();
            self.nheld = r.depth;
            if (self.draft_rounds % 16 == 0) std.log.info("hy4-cuda: DFly2 {d:.2} accepted a draft over {d} steps ({d} proposed, {d} kept), {d:.1} ms a proposal", .{
                if (self.draft_rounds == 0) 0 else @as(f64, @floatFromInt(self.draft_kept)) / @as(f64, @floatFromInt(self.draft_rounds)),
                self.draft_rounds,
                self.draft_proposed,
                self.draft_kept,
                @as(f64, @floatFromInt(self.draft_ns)) / 1e6 / @as(f64, @floatFromInt(self.draft_proposed)),
            });
        }
        self.logits = self.window[0..self.vocab];
    }

    /// The drafts a proposal holds for the stream, in window order (a grammar walks them).
    fn heldFn(ptr: *anyopaque, _: *lanes.Stream, out: []u32) !void {
        const self = cast(ptr);
        if (out.len > self.nheld) return error.Hy4DraftsNotHeld;
        @memcpy(out, self.held[0..out.len]);
    }

    /// The caches stay for the next prompt: a shared prefix resumes there instead of filling from row 0.
    fn releaseFn(ptr: *anyopaque, _: *lanes.Stream) void {
        const self = cast(ptr);
        self.kept = true;
        self.nheld = 0;
    }
};

/// A FP32 logit as the bf16 word a bf16 head would write (round to nearest even, NaN kept).
fn roundBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if ((bits & 0x7f800000) == 0x7f800000 and (bits & 0x007fffff) != 0) return @intCast((bits >> 16) | 0x0040);
    const half = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
    return @intCast(half & 0xffff);
}

test "a logit rounds to the bf16 word the head would write, ties to even" {
    try std.testing.expectEqual(@as(u16, 0x3f80), roundBf16(1.0));
    try std.testing.expectEqual(@as(u16, 0xbf80), roundBf16(-1.0));
    try std.testing.expectEqual(@as(u16, 0), roundBf16(0.0));
    // 1 + 2^-8 sits exactly between two bf16 words: the even one is 1.0's.
    try std.testing.expectEqual(@as(u16, 0x3f80), roundBf16(1.0 + 0.00390625));
    try std.testing.expectEqual(@as(u16, 0x3f81), roundBf16(1.0 + 0.0078125));
}

test "a context's caches split into the KV that grows, the ring that does not and the indexer's keys" {
    // The shape v3a loads with, from the family's own geometry: 30 sliding layers, 12 indexed ones.
    var buf: [1024]u8 align(8) = undefined; // a stand-in for the indexer's device-side layers: only the keys' sizes are read
    var indexers: [12]@import("cuda_index.zig").Layer = undefined;
    for (&indexers) |*ix| {
        ix.* = undefined;
        ix.token_keys = .{ .d = undefined, .ptr = @intFromPtr(&buf), .len = 262_144 * 128 * 2, .owner = null, .borrowed = true };
        ix.group_keys = .{ .d = undefined, .ptr = @intFromPtr(&buf), .len = 65_536 * 128 * 2, .owner = null, .borrowed = true };
    }
    var layers: [42]model.Layer = undefined;
    for (&layers, 0..) |*layer, i| {
        const sliding = i < 30;
        var att: model.Attention = undefined;
        att.sliding = sliding;
        att.rows = if (sliding) ring_rows else 262_144;
        att.index = if (sliding) null else &indexers[i - 30];
        layer.* = .{ .attention = att, .engram = null, .feed = undefined };
    }
    // 12 indexed layers: the recall scores' width is 65,536 groups, so 1,024 scores a token.
    const scores: usize = 1024 * 65_536 * 4;
    var m: model.Model = undefined;
    m.layers = &layers;
    m.capacity = 262_144;
    m.index.scores = .{ .d = undefined, .ptr = 0, .len = scores, .owner = null, .borrowed = false };
    const b = cacheBytes(&m);
    try std.testing.expectEqual(@as(u64, 12 * 262_144 * 1152), b.kv);
    try std.testing.expectEqual(@as(u64, 12 * 83_886_080), b.index);
    // The ring is flat: 30 layers of 1,536 slots of 576 bf16 words.
    const ring: u64 = 30 * ring_rows * 576 * 2;
    try std.testing.expectEqual(ring, b.ring);
    try std.testing.expectEqual(@as(u64, 53_084_160), ring);
    try std.testing.expect(ring_window == 512 and ring_rows == 3 * ring_window);
    // A size decision's rate counts everything that grows: 13,824 B of KV, 3,840 B of indexer keys, 1,024 B of scores.
    try std.testing.expectEqual(@as(u64, 18_688), bytesPerToken(&m));
}

test "a rewind is allowed exactly while the ring still holds the rows its first window reads" {
    var c: Cuda = undefined;
    c.valid_from = 0;
    try std.testing.expect(c.mayRewind(0));
    try std.testing.expect(c.mayRewind(ring_rows));
    c.valid_from = 512; // the state after a 2,048-token fill
    try std.testing.expect(c.mayRewind(1023)); // first row reads from 512: still its own
    try std.testing.expect(!c.mayRewind(1022)); // first row would read 511, overwritten by then
    try std.testing.expect(!c.mayRewind(0));
}

test "a rewind drops 1,025 rows and no more: a row reads back 511 of its own" {
    var c: Cuda = undefined;
    const h: u32 = 8_192; // a fill past four ring wraps
    c.valid_from = h - ring_rows;
    // The first row a rewind to `r` writes reads positions r-511..r, so r-511 must still be its own row.
    try std.testing.expectEqual(@as(u32, h - ring_rows), h - 1025 - 511);
    try std.testing.expect(c.mayRewind(h - 1024));
    try std.testing.expect(c.mayRewind(h - 1025));
    try std.testing.expect(!c.mayRewind(h - 1026));
}

test "a prompt resumes at its shared prefix only above the keep floor and inside the ring's bound" {
    var hist: [4_096]u32 = undefined;
    var prompt: [4_096]u32 = undefined;
    for (&hist, 0..) |*t, i| t.* = @intCast(i % 1000 + 1);
    @memcpy(&prompt, &hist);
    var m: model.Model = undefined;
    m.capacity = 262_144;
    var c: Cuda = undefined;
    c.m = &m;

    // A 2,048-row history, the caches kept, and a prompt sharing 1,024 rows of it: the fill resumes at 1,024.
    prompt[1024] = 5;
    m.history = .{ .items = hist[0..2048], .capacity = hist.len, .pointer_stability = .{} };
    c.valid_from = 2048 - ring_rows;
    c.kept = true;
    try std.testing.expectEqual(@as(usize, 1024), c.reuseFrom(&prompt));
    try std.testing.expect(!c.kept); // one fill decides, and the caches are no longer kept
    try std.testing.expectEqual(@as(usize, 1024), m.history.items.len); // the history is cut to the resume

    // A 63-row prefix is below the floor: a shorter fill is not worth resuming, so the next prompt starts at row 0.
    m.history = .{ .items = hist[0..63], .capacity = hist.len, .pointer_stability = .{} };
    c.valid_from = 0;
    c.kept = true;
    try std.testing.expectEqual(@as(usize, 0), c.reuseFrom(&prompt));
    try std.testing.expectEqual(@as(usize, 0), m.history.items.len);

    // A prefix the ring no longer holds: 300 rows into a 2,403-row fill, its first new row reads an overwritten slot.
    prompt[300] = 6;
    m.history = .{ .items = hist[0..2403], .capacity = hist.len, .pointer_stability = .{} };
    c.valid_from = 2403 - ring_rows;
    c.kept = true;
    try std.testing.expectEqual(@as(usize, 0), c.reuseFrom(&prompt));
}
