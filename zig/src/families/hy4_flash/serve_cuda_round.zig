//! Hy4-flash's round on CUDA: a round's windows in one forward, each sequence's rewind, and its DFly2 block.
const std = @import("std");
const lanes = @import("lanes");
const geom = @import("cuda_scratch.zig");
const dfly = @import("cuda_draft.zig");
const be = lanes.backend;
const serve = @import("serve_cuda.zig");
const Cuda = serve.Cuda;
const max_slots = serve.max_slots;
const max_window = serve.max_window;

fn cast(ptr: *anyopaque) *Cuda {
    return @ptrCast(@alignCast(ptr));
}

/// A round's windows in one pass, every row drawn at its own position; `keep` rolls each sequence back.
///
/// A window that cannot run ends its own stream rather than the round: one reply at the caches' end (or one the
/// backend cannot build a key list for) must not fail the other sequences sharing the pass. A window that only runs
/// past the caches is cut to the rows that fit, and its chain is stopped after them, so its reply keeps the rows it
/// really has and the next round drafts again from there.
pub fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
    const self = cast(ptr);
    const words = (self.vocab + 31) / 32;
    if (windows.len != out.len or windows.len == 0 or windows.len > self.slots) return error.InvalidHy4Windows;
    var ids: [max_slots][max_window]u32 = undefined;
    var runs: [max_slots]geom.Run = undefined;
    var row0: [max_slots]usize = undefined;
    var ran: [max_slots]usize = @splat(0); // each window's rows in the pass; 0: it does not run (its stream ends)
    var total: usize = 0;
    var n: usize = 0;
    for (windows, out, 0..) |w, o, i| {
        const rows = w.rows();
        if (rows > max_window or w.parents != null or o.sampled.len != rows or o.drafts.len + 1 != rows) return error.UnsupportedHy4Window;
        const slot = self.slotOf(w.stream) orelse return error.UnknownHy4Stream;
        const q = &self.seq[slot];
        const base = self.m.len(slot);
        for (w.positions, 0..) |p, r| if (p != base + r + 1) return error.InvalidHy4Position;
        // A window past the caches is cut to the rows that fit; its chain stops on the last row's own draft, which is
        // one this backend filled. Without such a draft (the host's tokens run to the window's end) there is no row to
        // stop on, and the reply ends instead: committing rows nothing wrote would be worse than a short reply.
        const room = self.m.capacity - base;
        if (room == 0 or (rows > room and room - 1 >= w.held)) {
            self.failStream(w.stream, "the reply's next window runs past the context's caches");
            @memset(o.sampled, std.math.maxInt(u32)); // a row that did not run draws nothing, and no draft matches it
            @memset(o.drafts, 0);
            q.nheld = 0;
            q.proposed = 0;
            q.ran = 0;
            continue;
        }
        const rows_ran = @min(rows, room);
        ids[i][0] = w.pending;
        // The drafts the last proposal held, then the host's; a wider window repeats its last.
        const held = @min(@as(usize, w.held), @as(usize, q.nheld));
        @memcpy(ids[i][1..][0..held], q.held[0..held]);
        for (held..w.held) |k| ids[i][1 + k] = if (held > 0) q.held[held - 1] else w.pending;
        @memcpy(ids[i][1 + w.held ..][0..w.tokens.len], w.tokens);
        // A row the block skipped (near the context's end) or that no proposal made is not a draft of this round.
        const real = @min(@as(usize, w.held), @as(usize, q.proposed));
        q.nheld = 0;
        q.proposed = @intCast(real); // `keep` reads the round's own proposals after this pass, and clears them
        q.ran = @intCast(rows_ran); // and the rows this pass really appended, which a shortened window cuts by
        @memcpy(o.drafts, ids[i][1..rows]);
        if (real > 0) {
            self.draft_rounds += 1;
            self.draft_proposed += real;
            if (self.log_rounds) std.log.info("hy4-cuda: round {d} at row {d}: window {any}", .{ self.draft_rounds, base, ids[i][0..rows] });
        }
        if (rows_ran < rows) std.log.info("hy4-cuda: a window at row {d} runs {d} of its {d} rows: the caches end at {d}", .{ base, rows_ran, rows, self.m.capacity });
        runs[n] = .{ .slot = slot, .tokens = ids[i][0..rows_ran] };
        row0[i] = total;
        ran[i] = rows_ran;
        total += rows_ran;
        n += 1;
    }
    if (total > self.window.len / self.vocab) return error.UnsupportedHy4Window;
    if (n > 0) try verifyPass(self, runs[0..n], total);
    for (windows, out, 0..) |w, o, i| {
        for (0..ran[i]) |r| {
            self.logits = self.window[(row0[i] + r) * self.vocab ..][0..self.vocab];
            const mask = if (w.masks.len >= (r + 1) * words) w.masks[r * words ..][0..words] else &.{};
            o.sampled[r] = try self.draw(w.stream, w.positions[r], mask);
            if (o.rows.len == w.rows()) o.rows[r] = self.row(o.sampled[r], w.stream.logprobs orelse 0);
            if (self.log_rounds) std.log.info("hy4-cuda: round {d} row {d} (position {d}): drew {d} on {d:.4}", .{ self.draft_rounds, r, w.positions[r], o.sampled[r], self.logits[o.sampled[r]] });
        }
        // The chain stops at the last row that ran: row `ran` was not written, and its own draft must not match.
        if (ran[i] > 0 and ran[i] < w.rows()) o.drafts[ran[i] - 1] = o.sampled[ran[i] - 1] ^ 1;
    }
    self.logits = self.window[0..self.vocab];
}

/// A round's rows in one forward, every row's logits kept: the drafter reads its captured lists from this pass.
pub fn verifyPass(self: *Cuda, runs: []const geom.Run, rows: usize) !void {
    var starts: [max_slots]usize = undefined;
    for (runs, 0..) |run, i| {
        starts[i] = self.m.len(run.slot);
        if (starts[i] + run.tokens.len > self.m.capacity) return error.Hy4ContextFull;
    }
    self.logits = self.window[0..self.vocab];
    const t0 = std.Io.Clock.awake.now(self.io);
    errdefer for (runs, 0..) |run, i| {
        self.m.cut(run.slot, starts[i]);
        self.seq[run.slot].kept = false;
        // A pass that failed partway may have written ring slots up to its own last row: the mark moves with it.
        self.bumpRows(run.slot, starts[i] + run.tokens.len);
    };
    try self.m.passRuns(runs, self.logits, .{ .at = 0, .out = self.window[0 .. rows * self.vocab], .head = rows });
    const ns = t0.durationTo(std.Io.Clock.awake.now(self.io)).toNanoseconds();
    self.decode_rows += rows;
    self.decode_ns += ns;
    // `TF_HY4_ROUND_LOG`: the pass's own cost by rows, so a round's wall time can be read against it.
    if (self.log_rounds and self.decode_rows % 128 < rows) std.log.info("hy4-cuda: pass {d} rows over {d} streams in {d:.2} ms ({d:.2} ms a row, {d} rows so far)", .{ rows, runs.len, @as(f64, @floatFromInt(ns)) / 1e6, @as(f64, @floatFromInt(ns)) / 1e6 / @as(f64, @floatFromInt(@max(rows, 1))), self.decode_rows });
    for (runs) |run| self.bump(run.slot);
}

/// Roll each stream back to the rows it kept: each sequence's caches hold its own, so this cuts its history.
pub fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
    const self = cast(ptr);
    for (windows, paths) |w, path| {
        for (path, 0..) |r, i| if (r != i) return error.UnsupportedHy4Tree;
        const slot = self.slotOf(w.stream) orelse return error.UnknownHy4Stream;
        const q = &self.seq[slot];
        // A window this round cut to the caches' end appended fewer rows than the window names, and one it ended wrote
        // none: `q.ran` is what the pass added, so the cut is off that and not off the window's own width.
        const rows = @as(usize, q.ran);
        const base = self.m.len(slot);
        q.ran = 0;
        if (rows == 0) {
            q.nheld = 0;
            q.proposed = 0;
            continue;
        }
        if (path.len == 0 or path.len > rows or base < rows) return error.InvalidHy4Position;
        self.m.cut(slot, base - rows + path.len);
        // Rows a verified draft kept: a repeated row (the block was skipped) or a host token is not one.
        const kept = @min(@as(usize, path.len - 1), @as(usize, q.proposed));
        self.draft_kept += kept;
        if (self.log_rounds) std.log.info("hy4-cuda: round {d} kept {d} of {d} rows (drafts {d} of {d}), history {d} -> {d}", .{ self.draft_rounds, path.len, rows, kept, q.proposed, base, self.m.len(slot) });
        q.nheld = 0;
        q.proposed = 0;
    }
}

/// DFly2: a block of the round's anchor and five masks over the target's own caches, then one head a slot.
/// A stream whose block cannot run (a key list past the stride, a shape this backend does not draft) ends alone:
/// the other sequences' blocks still run, and the reply keeps the tokens its rounds already landed.
pub fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) !void {
    const self = cast(ptr);
    const dr = if (self.m.draft) |*x| x else return error.NoHy4DraftHead;
    for (requests) |r| {
        const slot = self.slotOf(r.stream) orelse return error.UnknownHy4Stream;
        self.seq[slot].nheld = 0;
        self.seq[slot].proposed = 0;
        if (r.stream.finished) continue; // a window ended it this round
        if (r.depth > self.drafts) return error.UnsupportedHy4Window;
        // The anchor is the cache length after this round's keep: a shared round drafts before it cuts the window.
        if (r.position == 0) return error.InvalidHy4Position;
        const anchor: usize = @intCast(r.position - 1);
        if (r.depth == 0) continue;
        const kept: ?usize = if (r.rows) |rows| blk: {
            if (rows.len == 0) return error.UnsupportedHy4Window;
            var chain = true;
            for (rows, 0..) |x, i| chain = chain and x == i;
            if (!chain) {
                self.failStream(r.stream, "a tree is not a window this engine drafts");
                continue;
            }
            break :blk rows.len - 1;
        } else null;
        const bonus: u32 = if (r.follow.len > 0) r.follow[r.follow.len - 1] else switch (r.first orelse return error.NoFirstToken) {
            .value => |v| v,
            .handle => |h| @intCast(h),
        };
        // The block must fit the context; below the key budget a row's list is the dense prefix, wider than a stride.
        const dense = !dr.scratch.seq[slot].selected;
        if (anchor + dfly.width > self.m.capacity or (dense and anchor + dfly.width > dfly.key_stride)) {
            // The block would write past the context: hold the anchor's own token, so the planned window verifies.
            @memset(self.seq[slot].held[0..r.depth], bonus);
            self.seq[slot].nheld = r.depth;
            continue;
        }
        const from = dfly.source(dr, slot, kept);
        const t0 = std.Io.Clock.awake.now(self.io);
        dfly.propose(self.m, dr, slot, anchor, bonus, r.depth, from, self.seq[slot].held[0..r.depth]) catch |e| switch (e) {
            // A list that does not fit its row's stride is this stream's own data: end it, keep the others' blocks.
            error.ShortHy4KeyList => {
                self.failStream(r.stream, "a draft row's key list is wider than its stride at this anchor");
                continue;
            },
            else => return e,
        };
        if (self.draft_rounds <= 3) std.log.info("hy4-cuda: DFly2 block at {d} from row {d} in slot {d}: bonus {d}, proposals {any}", .{ anchor, from.head, slot, bonus, self.seq[slot].held[0..r.depth] });
        self.draft_ns += t0.durationTo(std.Io.Clock.awake.now(self.io)).toNanoseconds();
        self.seq[slot].nheld = r.depth;
        self.seq[slot].proposed = r.depth;
        if (self.draft_rounds > 0 and self.draft_rounds % 16 == 0) std.log.info("hy4-cuda: DFly2 {d:.2} drafts kept a round over {d} rounds ({d} proposed, {d} kept), {d:.1} ms a proposal", .{
            @as(f64, @floatFromInt(self.draft_kept)) / @as(f64, @floatFromInt(self.draft_rounds)),
            self.draft_rounds,
            self.draft_proposed,
            self.draft_kept,
            @as(f64, @floatFromInt(self.draft_ns)) / 1e6 / @as(f64, @floatFromInt(@max(self.draft_proposed, 1))),
        });
    }
    self.logits = self.window[0..self.vocab];
}

/// The drafts a proposal holds for the stream, in window order (a grammar walks them).
pub fn heldFn(ptr: *anyopaque, s: *lanes.Stream, out: []u32) !void {
    const self = cast(ptr);
    const slot = self.slotOf(s) orelse return error.UnknownHy4Stream;
    if (out.len > self.seq[slot].nheld) return error.Hy4DraftsNotHeld;
    @memcpy(out, self.seq[slot].held[0..out.len]);
}

