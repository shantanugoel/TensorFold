//! Hy4-flash behind the lane engine on CUDA: one sequence, the step 1-3 text model (dense EXL3 groups, the indexer,
//! the sliding layers' ring). A prompt fills in `prompt_rows`-row passes; a reply is one row a pass.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const model = @import("cuda_model.zig");
const geom = @import("cuda_scratch.zig");
const mla = @import("mla_weights.zig");
const be = lanes.backend;

/// A prompt chunk's rows: the dev binary's cap and the model's `max_rows`.
pub const prompt_rows: u32 = geom.row_cap;
/// The widest context this engine ranks: the indexer's recall scores, 65,536 groups of four slots.
pub const max_capacity: u32 = @import("cuda_index.zig").max_capacity;
pub const Kernels = model.Kernels;
pub const Model = model.Model;
/// A verify window's rows: no draft head yet, so its pending row alone.
pub const max_window: u32 = 1;
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

/// Per-token cache bytes a context of `capacity` slots takes at `layers`, for a size decision taken before a load.
pub fn bytesPerToken(indexed: usize) u64 {
    return indexed * (mla.absorbed_stride * 2 + 320);
}

pub const Cuda = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    m: *model.Model,
    vocab: u32,
    logits: []f32, // the last pass's last row, FP32 as the model writes it
    words: []u16, // the same row as bf16 words, for the lane sampler
    valid_from: u32 = 0, // the oldest position whose ring slot still holds that position's own row
    kept: bool = false, // the caches hold the last released stream's tokens
    prefill_rows: u64 = 0,
    prefill_ns: i96 = 0,
    decode_rows: u64 = 0,
    decode_ns: i96 = 0,

    /// Loads the pack and its caches: `capacity` slots a sequence, the widest pass `prompt_rows` rows.
    pub fn init(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, k: *model.Kernels, dir: []const u8, capacity: usize) !*Cuda {
        const self = try a.create(Cuda);
        errdefer a.destroy(self);
        const m = try model.Model.open(a, io, d, s, k, dir, .{ .capacity = capacity, .max_rows = prompt_rows });
        errdefer model.Model.free(m);
        const vocab: usize = m.text.vocab_size;
        const logits = try a.alloc(f32, vocab);
        errdefer a.free(logits);
        self.* = .{
            .gpa = a,
            .io = io,
            .m = m,
            .vocab = @intCast(vocab),
            .logits = logits,
            .words = try a.alloc(u16, vocab),
        };
        return self;
    }

    pub fn deinit(self: *Cuda) void {
        const a = self.gpa;
        a.free(self.words);
        a.free(self.logits);
        model.Model.free(self.m);
        a.destroy(self);
    }

    pub fn bytes(self: *const Cuda) Bytes {
        return cacheBytes(self.m);
    }

    /// The slot position `pos` is written to and read from: its own, or its place in the ring. `Model.pass` refuses a
    /// pass past `capacity`, so a slot is inside its layer's arena — the write kernel would skip it silently otherwise.
    pub fn writeSlot(self: *const Cuda, pos: usize) usize {
        return if (self.m.ring) geom.ring.slot(pos) else pos;
    }

    /// The oldest position a rewind to length `r` may read: its first row's window must still hold its own rows.
    fn mayRewind(self: *const Cuda, r: u32) bool {
        const first = if (r + 1 > ring_window) r + 1 - ring_window else 0;
        return first >= self.valid_from;
    }

    /// After a pass: the ring's high-water mark. Slots hold the last `ring_rows` positions written, so nothing before
    /// `H + 1 - ring_rows` is readable any more.
    fn bump(self: *Cuda) void {
        const h = self.m.history.items.len;
        const from = if (h > ring_rows) h - ring_rows else 0;
        self.valid_from = @max(self.valid_from, @as(u32, @intCast(from)));
    }

    /// `tokens` at the caches' end: the model's pass, then the ring's bookkeeping and the timers.
    fn pass(self: *Cuda, tokens: []const u32) !void {
        const start = self.m.history.items.len;
        if (start + tokens.len > self.m.capacity) return error.Hy4ContextFull;
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

    /// The row's draw: greedy in the model's own FP32 logits (the dev binary's argmax), else the lane sampler over the
    /// same row rounded to bf16 the way a bf16 head would write it.
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

    /// The draw a reply starts from, and the one a window row takes.
    pub fn facts() lanes.Model {
        return .{ .exact_width = max_window, .mtp = false, .speculate = false, .drafts = 0, .copy_drafts = false, .lone_copies = false, .max_streams = 1, .streams_exact = true };
    }

    pub fn backend(self: *Cuda) be.Backend {
        return .{ .ptr = self, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .first_row = firstRowFn, .release = releaseFn } };
    }

    fn cast(ptr: *anyopaque) *Cuda {
        return @ptrCast(@alignCast(ptr));
    }

    /// The prompt into the caches, in `prompt_rows`-row passes. A prompt that shares the kept caches' prefix resumes
    /// there when the ring still holds the rows its first new row reads; otherwise it fills from row 0.
    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) !void {
        const self = cast(ptr);
        const ids = s.prompt();
        if (ids.len == 0) return error.EmptyHy4Prompt;
        if (ids.len + s.max_new > self.m.capacity) return error.Hy4ContextFull;
        var from: usize = 0;
        if (self.kept) {
            const hist = self.m.history.items;
            var same: usize = 0;
            const most = @min(hist.len, ids.len - 1);
            while (same < most and hist[same] == ids[same]) same += 1;
            if (same >= keep_prefix and same <= self.m.capacity and self.mayRewind(@intCast(same))) from = same;
            self.kept = false;
        }
        if (self.m.history.items.len != from) {
            self.m.history.items.len = from;
            if (from == 0) self.valid_from = 0;
        }
        s.cached = @intCast(from);
        var at = from;
        while (at < ids.len) {
            if (s.isCancelled()) return error.Cancelled;
            const end = @min(ids.len, at + prompt_rows);
            try self.pass(ids[at..end]);
            lanes.prefill_widths.completed(self.gpa, s, end - at);
            at = end;
        }
        std.log.debug("hy4-cuda: prompt {d} tokens, {d} reused, {d:.1} ms ({d:.1} tok/s)", .{ ids.len, from, @as(f64, @floatFromInt(self.prefill_ns)) / 1e6, @as(f64, @floatFromInt(self.prefill_rows)) * 1e9 / @as(f64, @floatFromInt(@max(self.prefill_ns, 1))) });
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

    /// The window's pending row in one pass, drawn at its own position; `keep` rolls a path's rejected rows back.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(ptr);
        const words = (self.vocab + 31) / 32;
        if (windows.len != out.len) return error.InvalidHy4Windows;
        for (windows, out) |w, o| {
            const rows = w.rows();
            if (rows > max_window or w.parents != null or w.held != 0 or o.sampled.len != rows or o.drafts.len + 1 != rows) return error.UnsupportedHy4Window;
            const base = self.m.history.items.len;
            for (w.positions, 0..) |p, r| if (p != base + r + 1) return error.InvalidHy4Position;
            var ids: [max_window]u32 = undefined;
            ids[0] = w.pending;
            @memcpy(ids[1..][0..w.tokens.len], w.tokens);
            try self.pass(ids[0..rows]);
            for (0..rows) |r| {
                const mask = if (w.masks.len >= (r + 1) * words) w.masks[r * words ..][0..words] else &.{};
                o.sampled[r] = try self.draw(w.stream, w.positions[r], mask);
                if (o.rows.len == rows) o.rows[r] = self.row(o.sampled[r], w.stream.logprobs orelse 0);
            }
        }
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        _ = ptr;
        for (windows, paths) |w, path| {
            const rows = w.rows();
            if (rows != max_window or path.len != rows) return error.UnsupportedHy4Tree;
        }
    }

    fn draftFn(_: *anyopaque, _: []const be.DraftRequest) !void {
        return error.NoHy4DraftHead;
    }

    /// The caches stay for the next prompt: a shared prefix resumes there instead of filling from row 0.
    fn releaseFn(ptr: *anyopaque, _: *lanes.Stream) void {
        cast(ptr).kept = true;
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
    const indexed: usize = 12;
    const per_token = bytesPerToken(indexed);
    try std.testing.expectEqual(@as(u64, 12 * (1152 + 320)), per_token);
    const at_262144 = per_token * 262144;
    try std.testing.expect(at_262144 > 4 << 30 and at_262144 < 5 << 30);
    // The ring is flat: 30 layers of 1,536 slots of 576 bf16 words.
    const ring: u64 = 30 * ring_rows * 576 * 2;
    try std.testing.expectEqual(@as(u64, 53_084_160), ring);
    try std.testing.expect(ring_window == 512 and ring_rows == 3 * ring_window);
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
