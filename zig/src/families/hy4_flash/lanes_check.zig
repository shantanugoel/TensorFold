//! The batching review's isolation checks on a device: one sequence's traffic must leave another's caches, keys and
//! recall lists byte for byte, a reused (or cancelled) fill must read what a cold one does, and a window past the
//! caches must end only its own stream. One stage a run; every stage loads the pack once.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const geom = @import("cuda_scratch.zig");
const mla = @import("mla_weights.zig");
const index = @import("cuda_index.zig");
const family = @import("serve_cuda.zig");
const be = lanes.backend;

const usage = "usage: tf-hy4-cuda-lanes MODEL_DIR CAPACITY STAGE\n" ++
    "  CAPACITY   the context the model keeps, in slots (whole pages; `capacity` wants 8 to 128)\n" ++
    "  STAGE      neighbour  slot 0's KV, ring, indexer keys and groups against slot 1's traffic past 2,048 rows,\n" ++
    "                        started at a different offset mod 4 and resumed from a kept prefix\n" ++
    "             recall     the same history in slots 0 and 1 in one pass: slot 1's recall lists are slot 0's + capacity\n" ++
    "             reuse      a released fill and a cancelled one, each resumed by a prompt sharing 100 rows, against\n" ++
    "                        the same prompt on a cold slot\n" ++
    "             capacity   a window that runs past the caches ends its stream; the round's other window is unchanged\n";

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 4) {
        std.debug.print("{s}", .{usage});
        return error.BadArgs;
    }
    const dir = args[1];
    const capacity = try std.fmt.parseInt(usize, args[2], 10);
    const stage = args[3];
    const slots: usize = if (std.mem.eql(u8, stage, "reuse")) 3 else 2;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    if (try ctx.capability() != 121) return error.NeedGb10;
    var s = try cuda.Stream.init(&driver, true);
    defer s.deinit();
    var kernels = try family.Kernels.load(a, &driver);
    defer kernels.deinit();
    const c = try family.Cuda.init(a, io, &driver, s, &kernels, dir, capacity, 0, slots, 0, false);
    defer c.deinit();
    const b = c.backend();
    std.debug.print("loaded: {d}-slot context x {d} sequences, {d} layers\n", .{ capacity, slots, c.m.layers.len });

    if (std.mem.eql(u8, stage, "neighbour")) return neighbour(c, a);
    if (std.mem.eql(u8, stage, "recall")) return recall(c, a);
    if (std.mem.eql(u8, stage, "reuse")) return reuse(c, a, b);
    if (std.mem.eql(u8, stage, "capacity")) return capacityStage(c, a, b);
    std.debug.print("{s}", .{usage});
    return error.BadArgs;
}

/// Deterministic tokens in one vocabulary range: `n` of them, a seed's own order.
fn tokens(a: std.mem.Allocator, n: usize, seed: u64) ![]u32 {
    const ids = try a.alloc(u32, n);
    var state = seed *% 6364136223846793005 +% 1442695040888963407;
    for (ids) |*t| {
        state = state *% 6364136223846793005 +% 1442695040888963407;
        t.* = @intCast((state >> 33) % 9000 + 1000);
    }
    return ids;
}

/// One sequence's rows of every layer's cache (the KV of an indexed layer, the ring of a sliding one) and of the
/// indexers' two key arenas, as bytes: a neighbour's pass must not move any of them.
fn readSlot(c: *family.Cuda, a: std.mem.Allocator, slot: usize) ![][3][]u8 {
    const parts = try a.alloc([3][]u8, c.m.layers.len);
    for (c.m.layers, parts) |layer, *d| {
        const rows: usize = if (layer.attention.sliding) geom.ring.rows else c.m.capacity;
        const cache = try a.alloc(u8, rows * mla.absorbed_stride * 2);
        try layer.attention.cache.download(slot * cache.len, cache);
        var keys: [2][]u8 = .{ &.{}, &.{} };
        if (layer.attention.index) |ix| {
            const tk = try a.alloc(u8, c.m.capacity * 128 * 2);
            try ix.token_keys.download(slot * tk.len, tk);
            const gk = try a.alloc(u8, index.groups_of(c.m.capacity) * 128 * 2);
            try ix.group_keys.download(slot * gk.len, gk);
            keys = .{ tk, gk };
        }
        d.* = .{ cache, keys[0], keys[1] };
    }
    return parts;
}

/// How many of `before`'s parts `after` no longer matches: a cache, its token keys or its group keys.
fn movedParts(before: [][3][]u8, after: [][3][]u8) usize {
    var differ: usize = 0;
    for (before, after) |x, y| for (x, y) |p, q| {
        if (!std.mem.eql(u8, p, q)) differ += 1;
    };
    return differ;
}

/// Fill one sequence from row 0 in the engine's own chunk width.
fn fill(c: *family.Cuda, slot: usize, ids: []const u32) !void {
    var at: usize = 0;
    while (at < ids.len) : (at += family.prompt_rows) try c.pass(slot, ids[at..@min(ids.len, at + family.prompt_rows)]);
}

/// (a) slot 0's arenas against slot 1's traffic: past 2,048 rows of its own history, its group phase (start mod 4)
/// different, and a resumed kept prefix in the middle. A stray write into another sequence's arena is silent in the
/// write kernel, so this is the read that would notice it.
fn neighbour(c: *family.Cuda, a: std.mem.Allocator) !void {
    const ids_a = try tokens(a, 2300, 7);
    const ids_b = try tokens(a, 2400, 11);
    try fill(c, 0, ids_a);
    std.debug.print("slot 0: {d} rows\n", .{c.m.len(0)});
    const before = try readSlot(c, a, 0);

    // Slot 1's first pass starts at row 3 and every later one at 3 mod 4, where slot 0's start at 0; it resumes its
    // kept prefix at 515 and runs past 2,048 rows of its own history.
    try c.pass(1, ids_b[0..3]);
    try c.pass(1, ids_b[3..515]);
    c.seq[1].kept = true;
    const from = c.reuseFrom(1, ids_b, 0);
    if (from < 64) {
        std.debug.print("slot 1: the reuse did not engage ({d} rows)\n", .{from});
        return error.ReuseDidNotEngage;
    }
    var at: usize = from;
    while (at < ids_b.len) : (at += family.prompt_rows) try c.pass(1, ids_b[at..@min(ids_b.len, at + family.prompt_rows)]);
    std.debug.print("slot 1: {d} rows, resumed at {d} (its first pass began at row 3)\n", .{ c.m.len(1), from });

    const after = try readSlot(c, a, 0);
    const differ = movedParts(before, after);
    std.debug.print("{s}: slot 0's {d} cache and key parts differ in {d}\n", .{ if (differ == 0) "PASS" else "FAIL", before.len * 3, differ });
    if (differ != 0) return error.NeighbourMovedSlot0;
}

/// (c) the same history in slots 0 and 1, one pass: each sequence's recall list names its own arena, so slot 1's
/// entries are slot 0's plus its own base (`slot * capacity`), and none names slot 0's rows.
fn recall(c: *family.Cuda, a: std.mem.Allocator) !void {
    const ids = try tokens(a, 2300, 13);
    try fill(c, 0, ids);
    try fill(c, 1, ids);
    const one = [_]u32{ids[0]};
    // `logits_out` is where the pass's last row lands: a full vocab row, and inside `out`'s span so it is not copied.
    const runs = [_]geom.Run{ .{ .slot = 0, .tokens = &one }, .{ .slot = 1, .tokens = &one } };
    try c.m.passRuns(&runs, c.logits, .{ .at = 0, .out = c.window[0 .. 2 * c.vocab], .head = 2 });
    const lists = try a.alloc(u32, 2 * geom.sparse_stride);
    const counts = try a.alloc(u32, 2);
    try c.m.scratch.slots.download(0, std.mem.sliceAsBytes(lists));
    try c.m.scratch.counts.download(0, std.mem.sliceAsBytes(counts));
    const first = lists[0..geom.sparse_stride];
    const second = lists[geom.sparse_stride..];
    const base: u32 = @intCast(c.m.capacity);
    var off: usize = 0;
    var outside: usize = 0;
    for (0..@min(@as(usize, counts[0]), geom.sparse_stride)) |i| {
        if (second[i] != first[i] + base) off += 1;
        if (second[i] < base) outside += 1; // a slot of slot 0's arena: another sequence's rows
    }
    const same_count = counts[1] == counts[0] and counts[0] > 0;
    std.debug.print("lists: {d} and {d} entries; slot 1's differ from slot 0's + {d} in {d}, {d} outside its arena\n", .{ counts[0], counts[1], base, off, outside });
    std.debug.print("{s}: the two sequences rank their own arenas\n", .{if (same_count and off == 0 and outside == 0) "PASS" else "FAIL"});
    if (!same_count or off != 0 or outside != 0) return error.RecallListsCrossSlots;
}

fn stream(a: std.mem.Allocator, id: []const u8, ids: []const u32) !lanes.Stream {
    return lanes.Stream.init(a, .{ .id = id, .prompt = ids, .max_new = 1, .drafts = false });
}

/// A whole prompt through the server's own path, chunk by chunk.
fn prefill(b: be.Backend, s: *lanes.Stream) !void {
    const step = b.vtable.prefill_step orelse return error.NoPrefillStep;
    while (!try step(b.ptr, s, 0)) {}
}

/// (d) prompt B after prompt A filled a slot past the ring's wrap, and after A was cancelled mid-fill: B's logits are
/// the cold slot's in both cases. The wrap arm's 100-row prefix is correctly refused (the ring overwrote those slots),
/// so it proves B's logits do not depend on the slot's stale facts; the cancelled arm, filled a chunk and then cut,
/// resumes at the shared prefix and proves a resumed fill reads the rows a fresh one wrote.
fn reuse(c: *family.Cuda, a: std.mem.Allocator, b: be.Backend) !void {
    const ids_a = try tokens(a, 2400, 17);
    const ids_b = try tokens(a, 600, 19);
    @memcpy(ids_b[0..100], ids_a[0..100]); // the prefix B shares with A
    // A filled whole and released, as a finished request leaves its slot; B then shares 100 rows of it.
    var sa = try stream(a, "A", ids_a);
    try prefill(b, &sa);
    const long_slot = c.slotOf(&sa).?;
    b.release(&sa);
    std.debug.print("released: slot {d} holds {d} rows, and its 100-row prefix is past the ring's mark\n", .{ long_slot, c.m.len(long_slot) });
    var sb = try stream(a, "B", ids_b);
    try prefill(b, &sb);
    const after_release = try a.dupe(f32, c.logits);
    std.debug.print("after a released fill (slot {d}): {d} rows reused\n", .{ c.slotOf(&sb).?, sb.cached });

    // A cancelled after one chunk: under the ring's 1,536 rows, so its first rows are still readable, and B resumes
    // inside them. The host releases a cancelled stream's slot with its prefix kept.
    var sc = try stream(a, "A2", ids_a);
    const step = b.vtable.prefill_step orelse return error.NoPrefillStep;
    _ = try step(b.ptr, &sc, 0);
    try (b.vtable.cancel_prefill orelse return error.NoCancelPrefill)(b.ptr, &sc);
    const cut_slot = c.slotOf(&sc).?;
    b.release(&sc);
    std.debug.print("cancelled mid-fill: slot {d} holds {d} rows, kept\n", .{ cut_slot, c.m.len(cut_slot) });
    var sd = try stream(a, "B2", ids_b);
    try prefill(b, &sd);
    const after_cancel = try a.dupe(f32, c.logits);
    std.debug.print("after a cancelled fill (slot {d}): {d} rows reused\n", .{ c.slotOf(&sd).?, sd.cached });

    // The cold slot last, so nothing before it could have warmed it: a fresh fill of the same prompt.
    var cold = try stream(a, "cold", ids_b);
    try prefill(b, &cold);
    const want = try a.dupe(f32, c.logits);
    std.debug.print("cold slot {d}: {d} rows, {d} reused\n", .{ c.slotOf(&cold).?, c.m.len(c.slotOf(&cold).?), cold.cached });

    const same_released = std.mem.eql(f32, want, after_release);
    const same_cancelled = std.mem.eql(f32, want, after_cancel);
    const engaged = sd.cached >= 100;
    std.debug.print("{s}: released {s} ({d} rows reused), cancelled {s} ({d} rows reused)\n", .{
        if (same_released and same_cancelled and engaged) "PASS" else "FAIL",
        if (same_released) "identical" else "differs",
        sb.cached,
        if (same_cancelled) "identical" else "differs",
        sd.cached,
    });
    if (!same_released or !same_cancelled or !engaged) return error.ReuseMovedTheLogits;
}

/// (e) two windows in one round, the first at the caches' end: it ends its own stream, and the second draws the rows it
/// draws alone. A context of 8 to 128 slots keeps the overrun exact and a pass trivial.
fn capacityStage(c: *family.Cuda, a: std.mem.Allocator, b: be.Backend) !void {
    const cap = c.m.capacity;
    if (cap > 128 or cap < 8) {
        std.debug.print("capacity: use a context of 8 to 128 slots for this stage ({d} asked)\n", .{cap});
        return error.BadCapacity;
    }
    const ids = try tokens(a, 256, 23);
    const half = cap / 2;
    try c.pass(0, ids[0 .. cap - 1]);
    try c.pass(1, ids[0..half]);
    var near = try stream(a, "near", ids[0..cap]);
    var mid = try stream(a, "mid", ids[0 .. half + 2]);
    c.seq[0] = .{ .stream = &near, .at = @intCast(cap - 1) };
    c.seq[1] = .{ .stream = &mid, .at = @intCast(half) };

    const base1 = c.m.len(1);
    var pos0 = [2]u64{ c.m.len(0) + 1, c.m.len(0) + 2 };
    var pos1 = [2]u64{ base1 + 1, base1 + 2 };
    const draft = [_]u32{ids[3]};
    var w = [2]be.Window{
        .{ .stream = &near, .pending = ids[1], .held = 0, .tokens = &draft, .parents = null, .positions = &pos0 },
        .{ .stream = &mid, .pending = ids[2], .held = 0, .tokens = &draft, .parents = null, .positions = &pos1 },
    };
    var sampled0: [2]u32 = undefined;
    var sampled1: [2]u32 = undefined;
    var drafts0: [1]u32 = undefined;
    var drafts1: [1]u32 = undefined;
    var out = [2]be.Verified{ .{ .sampled = &sampled0, .drafts = &drafts0 }, .{ .sampled = &sampled1, .drafts = &drafts1 } };
    try b.verify(&w, &out);
    const shared = try a.dupe(u32, sampled1[0..2]);
    std.debug.print("the round: `near` {s}, `mid` {s}\n", .{ if (near.finished) "ended" else "ran", if (mid.finished) "ended" else "ran" });

    // The same window on the same rows, alone: the ended stream's neighbour keeps its draws.
    c.m.cut(1, base1);
    c.seq[1] = .{ .stream = &mid, .at = @intCast(base1) };
    var alone: [2]u32 = undefined;
    var alone_drafts: [1]u32 = undefined;
    var only = [_]be.Window{.{ .stream = &mid, .pending = ids[2], .held = 0, .tokens = &draft, .parents = null, .positions = &pos1 }};
    var only_out = [1]be.Verified{.{ .sampled = &alone, .drafts = &alone_drafts }};
    try b.verify(&only, &only_out);

    const ended = near.finished and near.reason == .@"error" and near.problem.len > 0;
    const untouched = !mid.finished;
    const same = std.mem.eql(u32, shared, &alone);
    std.debug.print("{s}: `near` ended {s}, `mid` drew {d} of {d} as it does alone\n", .{
        if (ended and untouched and same) "PASS" else "FAIL",
        if (ended) "with its own words" else "not",
        @as(usize, @intFromBool(same)),
        @as(usize, 2),
    });
    if (!ended or !untouched or !same) return error.CapacityLeakedToTheRound;
}
