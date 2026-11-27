//! Hy4-flash behind the lane engine on CUDA: up to `--parallel` sequences of the step 1-3 text model.
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
/// Sequences one server holds at once: `--parallel`, and the per-slot arrays' width.
pub const max_slots: usize = geom.max_slots;
/// A verify window's rows: the pending row and DFly2's five drafts, and the drafts a round proposes.
pub const max_window: u32 = dfly.width;
pub const drafts: u32 = dfly.drafts;
/// The sliding layers' ring: a rewind may read back only what the ring still holds.
const ring_rows: u32 = geom.ring.rows;
const ring_window: u32 = geom.ring.window;
/// The prompt prefix a reuse shares before its caches are kept rather than refilled from row 0.
const keep_prefix = 64;
/// A shared round's rows at most: every sequence's widest window, and never below the engine's own default.
pub fn batchRows(slots: usize) usize {
    return @max(32, slots * @max(@as(usize, max_window), 1));
}

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

/// The positions a request may fill with `drafts` a round: the caches' rows less the room a round's window needs.
pub fn serveCap(capacity: usize, rounds: u32) usize {
    return if (capacity > @as(usize, rounds)) capacity - @as(usize, rounds) else capacity;
}

/// Whether a context can be split into sequences: the indexer pools four slots a group and a recall list ranks whole
/// groups, so past one sequence the cache must close on a group (the server rounds its context to 512 anyway).
pub fn capacityFits(capacity: usize, slots: usize) bool {
    return slots <= 1 or capacity % @import("index_groups.zig").slots == 0;
}

/// Per-token bytes the caches take at this capacity over every sequence: the KV, the indexer's keys and the scores.
pub fn bytesPerToken(m: *const model.Model) u64 {
    const b = cacheBytes(m);
    return (b.kv + b.index + b.scores) / @max(@as(u64, m.capacity), 1);
}

/// One sequence's place in the server: the stream that holds it, its ring's mark, and its drafts.
const Seq = struct {
    stream: ?*lanes.Stream = null,
    valid_from: u32 = 0, // the oldest position whose ring slot still holds that position's own row
    kept: bool = false, // the caches hold the last released stream's tokens
    from: usize = 0, // the row the kept shared prefix resumes at
    at: usize = 0, // prompt tokens filled so far
    held: [dfly.drafts]u32 = @splat(0),
    nheld: u32 = 0,
    /// The drafts the last block proposed for this sequence: 0 where the block was skipped, so a held row that is
    /// only a repeat is neither proposed nor kept in the counters.
    proposed: u32 = 0,
};

pub const Cuda = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    m: *model.Model,
    vocab: u32,
    /// The drafts a round proposes at most: `--mtp-drafts`, 0 without a drafter.
    drafts: u32,
    /// The positions a request may fill: the caches' rows less the drafts a round's window writes past the last token.
    serve_cap: usize,
    /// `TF_HY4_DRAFT_LANES`: a shared round with this many busy lanes or more drafts none (2: only a lone stream).
    draft_lanes: u32,
    logits: []f32, // the row at hand, FP32 as the model writes it
    window: []f32, // every row of a round's windows, `vocab` wide each
    words: []u16, // the row at hand as bf16 words, for the lane sampler
    seq: [max_slots]Seq = @splat(Seq{}),
    slots: usize,
    batch_rows: usize,
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

    /// Loads the pack: `slots` sequences' caches of `capacity` positions, `prompt_rows` rows a pass, `drafts` a round.
    pub fn init(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, k: *model.Kernels, dir: []const u8, capacity: usize, rounds: u32, slots: usize) !*Cuda {
        // A sequence's cache must close on the indexer's four-slot group, or its pooled keys and recall lists overlap.
        if (!capacityFits(capacity, slots)) return error.Hy4UngroupedCapacity;
        const self = try a.create(Cuda);
        errdefer a.destroy(self);
        const m = try model.Model.open(a, io, d, s, k, dir, .{ .capacity = capacity, .max_rows = prompt_rows, .drafts = rounds, .slots = slots });
        errdefer model.Model.free(m);
        const vocab: usize = m.text.vocab_size;
        // Every row a round can hold: the engine's batch width, which is this many sequences' windows at least.
        const rows: usize = batchRows(slots);
        const window = try a.alloc(f32, vocab * rows);
        errdefer a.free(window);
        self.* = .{
            .gpa = a,
            .io = io,
            .m = m,
            .vocab = @intCast(vocab),
            .drafts = rounds,
            // A round verifies `1 + drafts` rows at the history's end, so a request stops that many rows short of the
            // caches: without the reserve, a reply that fills the context dies on its last window (Hy4ContextFull).
            .serve_cap = serveCap(capacity, rounds),
            .draft_lanes = @import("sys").environment.read(u32, "TF_HY4_DRAFT_LANES", 2) catch 2,
            .logits = window[0..vocab],
            .window = window,
            .words = try a.alloc(u16, vocab),
            .slots = slots,
            .batch_rows = rows,
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

    /// The context a request may fill: the caches' rows less the drafts a round's window writes past its last token.
    pub fn served(self: *const Cuda) u32 {
        return @intCast(self.serve_cap);
    }

    /// The slot the stream holds; null before the first prompt chunk handed it one.
    pub fn slotOf(self: *const Cuda, s: *const lanes.Stream) ?usize {
        for (self.seq[0..self.slots], 0..) |q, i| if (q.stream == s) return i;
        return null;
    }

    /// Whether the slot still holds this stream's own prompt. A released stream's address goes back to the pool and the
    /// next request may be given it, so `slotOf`'s address is evidence only while the history it filled is the prompt's.
    fn owns(self: *const Cuda, slot: usize, ids: []const u32) bool {
        const q = self.seq[slot];
        const hist = self.m.tokens(slot);
        if (q.at > hist.len or q.at > ids.len) return false;
        if (q.at == 0) return hist.len == 0; // nothing filled yet: a slot with history is an older request's
        return std.mem.eql(u32, hist[0..q.at], ids[0..q.at]);
    }

    /// A slot whose old request is gone: its facts go, its caches stay kept so the new prompt may resume on them.
    fn reclaim(self: *Cuda, slot: usize) void {
        const from = self.seq[slot].valid_from;
        self.seq[slot] = .{ .kept = true, .valid_from = from };
    }

    /// A free slot for `s`, or NoFreeSlot: what it held stays, a kept prefix and the ring's high-water mark.
    fn openSlot(self: *Cuda, s: *lanes.Stream) !usize {
        for (self.seq[0..self.slots], 0..) |*q, i| {
            if (q.stream != null) continue;
            const kept = q.kept;
            const from = q.valid_from;
            q.* = .{ .stream = s, .kept = kept, .valid_from = from };
            return i;
        }
        return error.NoFreeSlot;
    }

    /// A free slot for `s`, and the row its fill starts at: the kept prefix its caches still hold.
    fn openFill(self: *Cuda, s: *lanes.Stream, ids: []const u32) !usize {
        const i = try self.openSlot(s);
        const q = &self.seq[i];
        q.from = self.reuseFrom(i, ids);
        s.cached = @intCast(q.from);
        q.at = q.from;
        s.fill_at.store(@intCast(q.at), .monotonic);
        return i;
    }

    pub fn freeSlots(self: *const Cuda) usize {
        var n: usize = 0;
        for (self.seq[0..self.slots]) |q| n += @intFromBool(q.stream == null);
        return n;
    }

    /// The oldest position a rewind to length `r` may read: its first row's window must still hold its own rows.
    pub fn mayRewind(self: *const Cuda, slot: usize, r: u32) bool {
        const first = if (r + 1 > ring_window) r + 1 - ring_window else 0;
        return first >= self.seq[slot].valid_from;
    }

    /// After a pass: the ring's high-water mark, so nothing before `H + 1 - ring_rows` is readable any more.
    pub fn bump(self: *Cuda, slot: usize) void {
        const h = self.m.len(slot);
        const from = if (h > ring_rows) h - ring_rows else 0;
        self.seq[slot].valid_from = @max(self.seq[slot].valid_from, @as(u32, @intCast(from)));
    }

    /// `tokens` at the sequence's caches' end: the model's pass, then the ring's bookkeeping and the timers.
    pub fn pass(self: *Cuda, slot: usize, tokens: []const u32) !void {
        const start = self.m.len(slot);
        self.logits = self.window[0..self.vocab];
        // Inside `capacity` every write slot is inside its sequence's arena: a stray one is skipped silently.
        if (start + tokens.len > self.serve_cap) return error.Hy4ContextFull;
        // The model appends before it forwards: a failed pass leaves the history claiming rows nothing wrote.
        errdefer {
            self.m.cut(slot, start);
            self.seq[slot].kept = false;
        }
        const t0 = std.Io.Clock.awake.now(self.io);
        try self.m.passRuns(&.{.{ .slot = slot, .tokens = tokens }}, self.logits, null);
        const ns = t0.durationTo(std.Io.Clock.awake.now(self.io)).toNanoseconds();
        if (tokens.len == 1) {
            self.decode_rows += 1;
            self.decode_ns += ns;
            if (self.decode_rows % 128 == 0) std.log.debug("hy4-cuda: {d} decode rows, {d:.1} ms a token", .{ self.decode_rows, @as(f64, @floatFromInt(self.decode_ns)) / 1e6 / @as(f64, @floatFromInt(self.decode_rows)) });
        } else {
            self.prefill_rows += tokens.len;
            self.prefill_ns += ns;
        }
        self.bump(slot);
    }

    /// The row's draw: greedy in the model's FP32 logits, else the lane sampler over the same row as bf16 words.
    pub fn draw(self: *Cuda, s: *lanes.Stream, position: u64, mask: []const u32) !u32 {
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
    pub fn row(self: *const Cuda, pick: u32, count: u8) lanes.LogprobRow {
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
        // `streams_exact` is the engine's "may several streams share a forward": each sequence's rows are its own.
        return .{
            .exact_width = if (tower) max_window else 1,
            .mtp = tower,
            .speculate = tower,
            .speculate_early = false,
            .drafts = if (tower) self.drafts else 0,
            .copy_drafts = false,
            .lone_copies = false,
            .hidden_rows = self.slots > 1,
            .plain_share = self.slots > 1,
            .streams_exact = true,
            .batch_rows = @intCast(self.batch_rows),
            .max_streams = @intCast(self.slots),
            // Drafting at several busy lanes costs more than it saves here: a shared round runs every sequence's
            // block one after another (5 layers of six rows and a synced head a slot each), so four streams drafting
            // read 7.3 tok/s a lane against 14.5 plain. A lone stream keeps its round whole and its gain.
            .plain_streams = self.draft_lanes,
        };
    }

    pub fn backend(self: *Cuda) be.Backend {
        // `keep_full_lone`: a window accepted whole is still a round's keep — the round's count is read off it.
        const round = @import("serve_cuda_round.zig");
        return .{ .ptr = self, .keep_full_lone = true, .vtable = &.{ .prefill = prefillFn, .prefill_step = prefillStepFn, .cancel_prefill = cancelPrefillFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = round.verifyFn, .keep = round.keepFn, .draft = round.draftFn, .held = round.heldFn, .first_row = firstRowFn, .release = releaseFn, .reports_fill = true } };
    }

    /// The window widths and a draft block's price, timed once at load at depth 1 and at `drafts`: a block costs
    /// what its first slot costs plus a little for each more, so the rule prices a round as `fixed + per * drafts`.
    pub const Timed = struct {
        rows: [max_window]f64 = @splat(0), // a pass by rows
        block: [2]f64 = .{ 0, 0 }, // the block at depth 1, and at `drafts`
        fixed: f64 = 0, // the block's charge at any positive depth
        per: f64 = 0, // and what each draft past the first adds
    };

    /// The block's fixed charge and its slope per draft, from the two depths timed (drafts 1: all fixed).
    pub fn blockPrice(shallow: f64, deep: f64, slots: u32) Timed {
        const fixed = @max(shallow, 0);
        if (slots < 2 or deep <= shallow) return .{ .fixed = fixed, .per = 0, .block = .{ shallow, deep } };
        const per = (deep - shallow) / @as(f64, @floatFromInt(slots - 1));
        return .{ .fixed = @max(fixed - per, 0), .per = per, .block = .{ shallow, deep } };
    }

    pub fn timeWindows(self: *Cuda) !Timed {
        const dr = if (self.m.draft) |*x| x else return .{};
        var timed: Timed = .{};
        var ids: [dfly.width + 2]u32 = @splat(0);
        for (&ids, 0..) |*t, i| t.* = @intCast(1000 + i * 37);
        try self.pass(0, &ids);
        const at = self.m.len(0);
        for (1..max_window + 1) |width| {
            var times: [3]f64 = undefined;
            for (&times) |*tm| {
                self.m.cut(0, at);
                const t0 = std.Io.Clock.awake.now(self.io);
                try self.m.passRuns(&.{.{ .slot = 0, .tokens = ids[0..width] }}, self.logits, .{ .at = 0, .out = self.window[0 .. width * self.vocab], .head = width });
                tm.* = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(self.io)).toNanoseconds())) / 1e6;
            }
            std.mem.sort(f64, &times, {}, std.sort.asc(f64));
            timed.rows[width - 1] = times[1];
        }
        // A block at the shallowest and deepest depth a round asks for: one and `drafts` slots, each timed twice.
        var held: [dfly.drafts]u32 = @splat(0);
        var block: [dfly.drafts + 1]f64 = @splat(0); // by depth: 0 is never asked for
        for ([_]u32{ 1, self.drafts }) |depth| {
            const from = dfly.source(dr, 0, null);
            self.m.cut(0, at);
            for (0..2) |which| {
                const t0 = std.Io.Clock.awake.now(self.io);
                try dfly.propose(self.m, dr, 0, at, ids[at - 1], depth, from, held[0..depth]);
                if (which == 1) block[depth] = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(self.io)).toNanoseconds())) / 1e6;
            }
        }
        // The block's own price: its first slot's cost, plus what each draft past it adds.
        const shallow = block[1];
        const deep = block[@min(self.drafts, dfly.drafts)];
        const price = blockPrice(shallow, deep, self.drafts);
        timed.block = .{ shallow, deep };
        timed.fixed = price.fixed;
        timed.per = price.per;
        std.log.info("hy4-cuda: window ms by rows {d:.2} {d:.2} {d:.2} {d:.2} {d:.2} {d:.2}, a draft block {d:.2} ms at one slot and {d:.2} at {d}: {d:.2} ms fixed and {d:.2} a draft", .{
            timed.rows[0], timed.rows[1], timed.rows[2], timed.rows[3], timed.rows[4], timed.rows[5],
            shallow,    deep,    self.drafts,    timed.fixed,     timed.per,
        });
        // The warm rows go: the first stream fills from row 0 as on a cold load, in every sequence.
        for (self.seq[0..self.slots]) |*q| q.* = .{};
        for (0..self.slots) |i| {
            self.m.cut(i, 0);
            dr.scratch.seq[i] = .{};
        }
        self.decode_rows = 0;
        self.decode_ns = 0;
        self.prefill_rows = 0;
        self.prefill_ns = 0;
        return timed;
    }

    fn cast(ptr: *anyopaque) *Cuda {
        return @ptrCast(@alignCast(ptr));
    }

    /// The prompt into the sequence's caches in `prompt_rows`-row passes, resuming the kept shared prefix.
    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) !void {
        const self = cast(ptr);
        const ids = s.prompt();
        if (ids.len + s.max_new > self.serve_cap) return error.Hy4ContextFull;
        const t0 = std.Io.Clock.awake.now(self.io);
        while (!try self.prefillStep(s, 0)) {}
        const from = self.seq[self.slotOf(s) orelse return error.NoFreeSlot].from;
        const ns = t0.durationTo(std.Io.Clock.awake.now(self.io)).toNanoseconds();
        std.log.debug("hy4-cuda: prompt {d} tokens, {d} reused, {d} prefilled in {d:.1} ms", .{ ids.len, from, ids.len - from, @as(f64, @floatFromInt(ns)) / 1e6 });
    }

    /// `prompt_rows`-row passes of `ids` at sequence 0's caches, resuming the kept shared prefix where it may: a
    /// tool's whole fill, where a served prompt goes through `prefillStep` a chunk a call.
    pub fn fillIds(self: *Cuda, ids: []const u32) !usize {
        if (ids.len == 0) return error.EmptyHy4Prompt;
        if (ids.len > self.serve_cap) return error.Hy4ContextFull;
        const from = self.reuseFrom(0, ids);
        var at = from;
        while (at < ids.len) {
            const end = @min(ids.len, at + prompt_rows);
            try self.pass(0, ids[at..end]);
            at = end;
        }
        return from;
    }

    /// One chunk of a prompt: at most `max_rows` rows (0: this engine's own cut), true once the prompt is in.
    fn prefillStepFn(ptr: *anyopaque, s: *lanes.Stream, max_rows: u32) !bool {
        return cast(ptr).prefillStep(s, max_rows);
    }

    pub fn prefillStep(self: *Cuda, s: *lanes.Stream, max_rows: u32) !bool {
        const ids = s.prompt();
        if (ids.len == 0) return error.EmptyHy4Prompt;
        if (ids.len + s.max_new > self.serve_cap) return error.Hy4ContextFull;
        // The pool hands a released stream's address to the next request: a slot is this stream's while its own prompt
        // is still what the slot filled, and otherwise it goes back to the free list with its caches kept.
        const held = self.slotOf(s);
        if (held) |i| if (!self.owns(i, ids)) self.reclaim(i);
        const slot = if (held) |i| if (self.seq[i].stream == s) i else try self.openFill(s, ids) else try self.openFill(s, ids);
        const q = &self.seq[slot];
        if (q.at >= ids.len) return true;
        if (s.isCancelled()) {
            s.keepInterruptedPrefill(@intCast(q.at));
            return error.Cancelled;
        }
        const cap: usize = if (max_rows == 0) prompt_rows else @min(max_rows, prompt_rows);
        const end = @min(ids.len, q.at + cap);
        try self.pass(slot, ids[q.at..end]);
        s.fill_at.store(@intCast(end), .monotonic);
        lanes.prefill_widths.completed(self.gpa, s, end - q.at);
        q.at = end;
        return end == ids.len;
    }

    /// A cancelled prompt's rows are kept: the next prompt sharing them resumes there (no prompt work runs here).
    fn cancelPrefillFn(ptr: *anyopaque, s: *lanes.Stream) !void {
        const self = cast(ptr);
        const slot = self.slotOf(s) orelse return;
        if (!self.owns(slot, s.prompt())) return; // the slot is another request's now
        self.seq[slot].kept = true;
        s.keepInterruptedPrefill(@intCast(self.seq[slot].at));
    }

    /// The row a fill starts at: the kept shared prefix, when it is long enough and the ring still holds its first row.
    pub fn reuseFrom(self: *Cuda, slot: usize, ids: []const u32) usize {
        var from: usize = 0;
        if (self.seq[slot].kept and ids.len > 0) {
            const hist = self.m.tokens(slot);
            var same: usize = 0;
            const most = @min(hist.len, ids.len - 1);
            while (same < most and hist[same] == ids[same]) same += 1;
            if (same >= keep_prefix and self.mayRewind(slot, @intCast(same))) from = same;
            self.seq[slot].kept = false;
        }
        if (self.m.len(slot) != from) {
            self.m.cut(slot, from);
            if (from == 0) self.seq[slot].valid_from = 0;
        }
        return from;
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) !u64 {
        const self = cast(ptr);
        const slot = self.slotOf(s) orelse return error.UnknownHy4Stream;
        if (position != self.m.len(slot)) return error.InvalidHy4Position;
        return try self.draw(s, position, &.{});
    }

    fn firstMaskedFn(ptr: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        const self = cast(ptr);
        const slot = self.slotOf(s) orelse return error.UnknownHy4Stream;
        if (position != self.m.len(slot)) return error.InvalidHy4Position;
        return try self.draw(s, position, mask);
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, f: be.Feed, position: u64) !u64 {
        const self = cast(ptr);
        const slot = self.slotOf(s) orelse return error.UnknownHy4Stream;
        if (position != self.m.len(slot) + 1) return error.InvalidHy4Position;
        const token: u32 = switch (f) {
            .value => |v| v,
            .handle => |h| @intCast(h),
        };
        try self.pass(slot, &.{token});
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

    /// The caches stay for the next prompt: a shared prefix resumes there instead of filling from row 0.
    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = cast(ptr);
        const slot = self.slotOf(s) orelse return;
        if (!self.owns(slot, s.prompt())) return; // the address came back with another request: not this slot's
        self.seq[slot] = .{ .kept = true, .valid_from = self.seq[slot].valid_from };
    }
};

/// A FP32 logit as the bf16 word a bf16 head would write (round to nearest even, NaN kept).
fn roundBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if ((bits & 0x7f800000) == 0x7f800000 and (bits & 0x007fffff) != 0) return @intCast((bits >> 16) | 0x0040);
    const half = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
    return @intCast(half & 0xffff);
}

test "a draft block prices as a fixed charge and a slope per draft" {
    // Depth 1 and depth 5 on a GB10: 9.46 and 17.61 ms. A round pays the first slot whole and a little for more.
    const p = Cuda.blockPrice(9.46, 17.61, 5);
    try std.testing.expectApproxEqAbs(@as(f64, 7.4225), p.fixed, 1e-3);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0375), p.per, 1e-3);
    // A flat block (the same cost at any depth) is all fixed charge.
    const flat = Cuda.blockPrice(9.46, 9.46, 5);
    try std.testing.expectEqual(@as(f64, 9.46), flat.fixed);
    try std.testing.expectEqual(@as(f64, 0), flat.per);
    // One draft is all fixed, and a cheaper deep block does not price the shallow one below its own cost.
    const one = Cuda.blockPrice(9.46, 0, 1);
    try std.testing.expectEqual(@as(f64, 9.46), one.fixed);
    const odd = Cuda.blockPrice(12, 11, 5);
    try std.testing.expectEqual(@as(f64, 12), odd.fixed);
}

test "a context split between sequences closes on the indexer's four-slot group" {
    // One sequence ranks its own slots, so any context it can hold is addressable.
    try std.testing.expect(capacityFits(4097, 1));
    try std.testing.expect(capacityFits(262_144, 8));
    // Several sequences pool groups inside each one's arena: a context short of a group leaves a half group at its end.
    try std.testing.expect(!capacityFits(4097, 2));
    try std.testing.expect(!capacityFits(262_142, 4));
}

test "a request fills the caches less the room a round's window writes past its last token" {
    // A reply that fills the context ends with a window of `1 + drafts` rows: without the reserve the last one
    // runs past the caches and the request dies on Hy4ContextFull.
    try std.testing.expectEqual(@as(usize, 262_139), serveCap(262_144, 5));
    try std.testing.expectEqual(@as(usize, 262_142), serveCap(262_144, 2));
    try std.testing.expectEqual(@as(usize, 262_144), serveCap(262_144, 0));
    // A context smaller than the drafts keeps its rows rather than wrapping.
    try std.testing.expectEqual(@as(usize, 4), serveCap(4, 5));
}

test "a round drafts only at one busy lane, and never more drafts than `--mtp-drafts` names" {
    var m: model.Model = undefined;
    m.draft = null; // no tower: nothing to propose, so no window to widen
    var c: Cuda = undefined;
    c.m = &m;
    c.slots = 4;
    c.batch_rows = 24;
    c.drafts = 0;
    c.draft_lanes = 2;
    const f = c.facts();
    try std.testing.expectEqual(@as(u32, 0), f.drafts);
    try std.testing.expect(!f.mtp and !f.speculate);
    // A shared round of two streams or more drafts none: its block runs a sequence at a time, and measured that
    // costs more than the drafts it lands (4 lanes: 7.3 tok/s a lane against 14.5 plain).
    try std.testing.expectEqual(@as(u32, 2), f.plain_streams);
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

test "a slot belongs to a stream only while the history it filled is that stream's prompt" {
    var hist: [8]u32 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const hists = try std.testing.allocator.alloc(std.ArrayList(u32), 2);
    defer std.testing.allocator.free(hists);
    hists[0] = .{ .items = hist[0..4], .capacity = hist.len, .pointer_stability = .{} };
    hists[1] = .{ .items = hist[0..0], .capacity = hist.len, .pointer_stability = .{} };
    var m: model.Model = undefined;
    m.capacity = 262_144;
    m.slots = 2;
    m.history = hists;
    var s: lanes.Stream = undefined;
    var c: Cuda = undefined;
    c.m = &m;
    c.slots = 2;
    c.seq = @splat(Seq{});
    c.seq[0] = .{ .stream = &s, .at = 4 };
    // the prompt the slot filled: its stream's, whatever comes after the filled rows
    try std.testing.expect(c.owns(0, &.{ 1, 2, 3, 4, 9, 9 }));
    // another request's prompt behind a reused address: not this slot's, and the slot is reclaimed
    try std.testing.expect(!c.owns(0, &.{ 1, 2, 9, 4 }));
    try std.testing.expect(!c.owns(0, &.{ 5 }));
    // a slot that has filled nothing owns nothing but an empty history
    c.seq[1] = .{ .stream = &s, .at = 0 };
    try std.testing.expect(c.owns(1, &.{ 1, 2 }));
    hists[1] = .{ .items = hist[0..2], .capacity = hist.len, .pointer_stability = .{} };
    try std.testing.expect(!c.owns(1, &.{ 1, 2 }));
    c.reclaim(1);
    try std.testing.expect(c.seq[1].stream == null and c.seq[1].kept);
}

test "a rewind is allowed exactly while the ring still holds the rows its first window reads" {
    var c: Cuda = undefined;
    c.seq = @splat(Seq{});
    c.seq[0] = .{ .valid_from = 0 };
    try std.testing.expect(c.mayRewind(0, 0));
    try std.testing.expect(c.mayRewind(0, ring_rows));
    c.seq[0].valid_from = 512; // the state after a 2,048-token fill
    try std.testing.expect(c.mayRewind(0, 1023)); // first row reads from 512: still its own
    try std.testing.expect(!c.mayRewind(0, 1022)); // first row would read 511, overwritten by then
    try std.testing.expect(!c.mayRewind(0, 0));
}

test "a rewind drops 1,025 rows and no more: a row reads back 511 of its own" {
    var c: Cuda = undefined;
    const h: u32 = 8_192; // a fill past four ring wraps
    c.seq = @splat(Seq{});
    c.seq[1] = .{ .valid_from = h - ring_rows };
    // The first row a rewind to `r` writes reads positions r-511..r, so r-511 must still be its own row.
    try std.testing.expectEqual(@as(u32, h - ring_rows), h - 1025 - 511);
    try std.testing.expect(c.mayRewind(1, h - 1024));
    try std.testing.expect(c.mayRewind(1, h - 1025));
    try std.testing.expect(!c.mayRewind(1, h - 1026));
    // Two sequences' marks are their own: the second's does not move the first's.
    try std.testing.expect(c.mayRewind(0, 0));
}

test "a prompt resumes at its shared prefix only above the keep floor and inside the ring's bound" {
    var hist: [4_096]u32 = undefined;
    var prompt: [4_096]u32 = undefined;
    for (&hist, 0..) |*t, i| t.* = @intCast(i % 1000 + 1);
    @memcpy(&prompt, &hist);
    const hists = try std.heap.page_allocator.alloc(std.ArrayList(u32), 2);
    defer std.heap.page_allocator.free(hists);
    hists[0] = .{ .items = hist[0..0], .capacity = hist.len, .pointer_stability = .{} };
    hists[1] = .{ .items = hist[0..0], .capacity = hist.len, .pointer_stability = .{} };
    var m: model.Model = undefined;
    m.capacity = 262_144;
    m.slots = 2;
    m.history = hists;
    var c: Cuda = undefined;
    c.m = &m;
    c.slots = 2;
    c.seq = @splat(Seq{});

    // A 2,048-row history, the caches kept, and a prompt sharing 1,024 rows of it: the fill resumes at 1,024.
    prompt[1024] = 5;
    m.history[0] = .{ .items = hist[0..2048], .capacity = hist.len, .pointer_stability = .{} };
    c.seq[0] = .{ .kept = true, .valid_from = 2048 - ring_rows };
    try std.testing.expectEqual(@as(usize, 1024), c.reuseFrom(0, &prompt));
    try std.testing.expect(!c.seq[0].kept); // one fill decides, and the caches are no longer kept
    try std.testing.expectEqual(@as(usize, 1024), m.len(0)); // the history is cut to the resume
    // The other sequence's history is untouched by the first sequence's resume.
    try std.testing.expectEqual(@as(usize, 0), m.len(1));

    // A 63-row prefix is below the floor: a shorter fill is not worth resuming, so the next prompt starts at row 0.
    m.history[0] = .{ .items = hist[0..63], .capacity = hist.len, .pointer_stability = .{} };
    c.seq[0] = .{ .kept = true, .valid_from = 0 };
    try std.testing.expectEqual(@as(usize, 0), c.reuseFrom(0, &prompt));
    try std.testing.expectEqual(@as(usize, 0), m.len(0));

    // A prefix the ring no longer holds: 300 rows into a 2,403-row fill, its first new row reads an overwritten slot.
    prompt[300] = 6;
    m.history[0] = .{ .items = hist[0..2403], .capacity = hist.len, .pointer_stability = .{} };
    c.seq[0] = .{ .kept = true, .valid_from = 2403 - ring_rows };
    try std.testing.expectEqual(@as(usize, 0), c.reuseFrom(0, &prompt));
}
