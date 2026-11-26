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
pub fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
    const self = cast(ptr);
    const words = (self.vocab + 31) / 32;
    if (windows.len != out.len or windows.len == 0 or windows.len > self.slots) return error.InvalidHy4Windows;
    var ids: [max_slots][max_window]u32 = undefined;
    var runs: [max_slots]geom.Run = undefined;
    var row0: [max_slots]usize = undefined;
    var total: usize = 0;
    for (windows, out, 0..) |w, o, i| {
        const rows = w.rows();
        if (rows > max_window or w.parents != null or o.sampled.len != rows or o.drafts.len + 1 != rows) return error.UnsupportedHy4Window;
        const slot = self.slotOf(w.stream) orelse return error.UnknownHy4Stream;
        const q = &self.seq[slot];
        const base = self.m.len(slot);
        for (w.positions, 0..) |p, r| if (p != base + r + 1) return error.InvalidHy4Position;
        ids[i][0] = w.pending;
        // The drafts the last proposal held, then the host's; a wider window repeats its last.
        const held = @min(@as(usize, w.held), @as(usize, q.nheld));
        @memcpy(ids[i][1..][0..held], q.held[0..held]);
        for (held..w.held) |k| ids[i][1 + k] = if (held > 0) q.held[held - 1] else w.pending;
        @memcpy(ids[i][1 + w.held ..][0..w.tokens.len], w.tokens);
        q.nheld = 0;
        @memcpy(o.drafts, ids[i][1..rows]);
        if (w.held > 0) {
            self.draft_rounds += 1;
            self.draft_proposed += w.held;
            if (self.log_rounds) std.log.info("hy4-cuda: round {d} at row {d}: window {any}", .{ self.draft_rounds, base, ids[i][0..rows] });
        }
        runs[i] = .{ .slot = slot, .tokens = ids[i][0..rows] };
        row0[i] = total;
        total += rows;
    }
    if (total > self.window.len / self.vocab) return error.UnsupportedHy4Window;
    try verifyPass(self, runs[0..windows.len], total);
    for (windows, out, 0..) |w, o, i| {
        for (0..w.rows()) |r| {
            self.logits = self.window[(row0[i] + r) * self.vocab ..][0..self.vocab];
            const mask = if (w.masks.len >= (r + 1) * words) w.masks[r * words ..][0..words] else &.{};
            o.sampled[r] = try self.draw(w.stream, w.positions[r], mask);
            if (o.rows.len == w.rows()) o.rows[r] = self.row(o.sampled[r], w.stream.logprobs orelse 0);
            if (self.log_rounds) std.log.info("hy4-cuda: round {d} row {d} (position {d}): drew {d} on {d:.4}", .{ self.draft_rounds, r, w.positions[r], o.sampled[r], self.logits[o.sampled[r]] });
        }
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
    };
    try self.m.passRuns(runs, self.logits, .{ .at = 0, .out = self.window[0 .. rows * self.vocab], .head = rows });
    const ns = t0.durationTo(std.Io.Clock.awake.now(self.io)).toNanoseconds();
    self.decode_rows += rows;
    self.decode_ns += ns;
    for (runs) |run| self.bump(run.slot);
}

/// Roll each stream back to the rows it kept: each sequence's caches hold its own, so this cuts its history.
pub fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
    const self = cast(ptr);
    for (windows, paths) |w, path| {
        for (path, 0..) |r, i| if (r != i) return error.UnsupportedHy4Tree;
        const rows = w.rows();
        if (path.len == 0 or path.len > rows) return error.UnsupportedHy4Window;
        const slot = self.slotOf(w.stream) orelse return error.UnknownHy4Stream;
        const base = self.m.len(slot);
        if (base < rows) return error.InvalidHy4Position;
        self.m.cut(slot, base - rows + path.len);
        if (path.len > 1) self.draft_kept += path.len - 1;
        if (self.log_rounds) std.log.info("hy4-cuda: round {d} kept {d} of {d} rows (drafts {d} of {d}), history {d} -> {d}", .{ self.draft_rounds, path.len, rows, path.len - 1, rows - 1, base, self.m.len(slot) });
        self.seq[slot].nheld = 0;
    }
}

/// DFly2: a block of the round's anchor and five masks over the target's own caches, then one head a slot.
pub fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) !void {
    const self = cast(ptr);
    const dr = if (self.m.draft) |*x| x else return error.NoHy4DraftHead;
    for (requests) |r| {
        const slot = self.slotOf(r.stream) orelse return error.UnknownHy4Stream;
        self.seq[slot].nheld = 0;
        if (r.depth > dfly.drafts) return error.UnsupportedHy4Window;
        const anchor = self.m.len(slot);
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
        // The block must fit the context; below the key budget a row's list is the dense prefix, wider than a stride.
        const dense = !dr.scratch.seq[slot].selected;
        if (anchor + dfly.width > self.m.capacity or (dense and anchor + dfly.width > dfly.key_stride)) {
            // The block would write past the context: hold the anchor's own token, so the planned window verifies.
            @memset(self.seq[slot].held[0..r.depth], bonus);
            self.seq[slot].nheld = r.depth;
            continue;
        }
        const head = dfly.source(dr, slot, kept);
        const t0 = std.Io.Clock.awake.now(self.io);
        try dfly.propose(self.m, dr, slot, anchor, bonus, r.depth, head, self.seq[slot].held[0..r.depth]);
        if (self.draft_rounds <= 3) std.log.info("hy4-cuda: DFly2 block at {d} from row {d} in slot {d}: bonus {d}, proposals {any}", .{ anchor, head, slot, bonus, self.seq[slot].held[0..r.depth] });
        self.draft_ns += t0.durationTo(std.Io.Clock.awake.now(self.io)).toNanoseconds();
        self.seq[slot].nheld = r.depth;
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
pub fn heldFn(ptr: *anyopaque, s: *lanes.Stream, out: []u32) !void {
    const self = cast(ptr);
    const slot = self.slotOf(s) orelse return error.UnknownHy4Stream;
    if (out.len > self.seq[slot].nheld) return error.Hy4DraftsNotHeld;
    @memcpy(out, self.seq[slot].held[0..out.len]);
}

