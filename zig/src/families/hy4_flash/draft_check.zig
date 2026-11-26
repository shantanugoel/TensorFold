//! The DFly2 verify window on a device: one row against the same row inside the block, and the round after a
//! proposal. A host check cannot see this: it needs the real head, the real caches and the drafter's own writes.
const std = @import("std");
const cuda = @import("cuda");
const model = @import("cuda_model.zig");
const dfly = @import("cuda_draft.zig");
const family = @import("serve_cuda.zig");

const usage = "usage: tf-hy4-cuda-draft MODEL_DIR PROMPT.u32 CAPACITY [ROWS]\n" ++
    "  PROMPT.u32  little-endian u32 token ids\n" ++
    "  CAPACITY    the context the model keeps, in slots (past 2,048 the indexer runs too)\n" ++
    "  ROWS        the verify window's rows (default 6)\n";

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 4) {
        std.debug.print("{s}", .{usage});
        return error.BadArgs;
    }
    const dir = args[1];
    const prompt = try readIds(a, io, args[2]);
    const capacity = try std.fmt.parseInt(usize, args[3], 10);
    const rows: usize = if (args.len > 4) try std.fmt.parseInt(usize, args[4], 10) else dfly.width;
    if (prompt.len < 8 or rows < 2 or rows > dfly.width) return error.BadArgs;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    if (try ctx.capability() != 121) return error.NeedGb10;
    var s = try cuda.Stream.init(&driver, true);
    defer s.deinit();
    var kernels = try family.Kernels.load(a, &driver);
    defer kernels.deinit();
    const c = try family.Cuda.init(a, io, &driver, s, &kernels, dir, capacity, dfly.drafts);
    defer c.deinit();
    const m = c.m;
    const V: usize = m.text.vocab_size;
    const dr = &m.draft.?;
    std.debug.print("loaded: {d}-slot context, {d}-row window, prompt {d} tokens\n", .{ capacity, rows, prompt.len });

    // The prompt, then the anchor's own row as the plain path draws it.
    try c.pass(prompt);
    const one = try a.alloc(f32, V);
    const block = try a.alloc(f32, rows * V);
    const wide = try a.alloc(f32, rows * V);
    const anchor = m.history.items.len;
    const bonus = argmax(c.logits);
    std.debug.print("anchor {d}, plain draw {d}\n", .{ anchor, bonus });

    // (1) the anchor alone, the plain one-row path with the LM head's own 128-row block.
    const ids1 = [1]u32{bonus};
    try m.pass(&ids1, one);
    m.history.items.len = anchor;
    std.debug.print("one row: {d}\n", .{argmax(one)});

    // (2) the same anchor as the block's first row, the verify path's head over `rows` rows.
    var ids: [dfly.width]u32 = @splat(dr.mask);
    ids[0] = bonus;
    try m.passRows(ids[0..rows], c.logits, .{ .at = 0, .out = block[0 .. rows * V], .head = rows });
    m.history.items.len = anchor;
    std.debug.print("block row 0: {d}, max|one - block row 0| {d:.6}, top-2 one {d} {d:.4}, block {d} {d:.4}\n", .{
        argmax(block[0..V]),
        maxDiff(one, block[0..V]),
        top2(one)[0],
        top2(one)[1],
        top2(block[0..V])[0],
        top2(block[0..V])[1],
    });
    for (0..rows) |r| std.debug.print("  block row {d}: {d}\n", .{ r, argmax(block[r * V ..][0..V]) });

    // (2b) the same block with the LM head at its own 128-row block: is the row count the head's, or the pass's?
    try m.passRows(ids[0..rows], c.logits, .{ .at = 0, .out = wide[0 .. rows * V], .head = null });
    m.history.items.len = anchor;
    var widest: f32 = 0;
    for (0..rows) |r| widest = @max(widest, maxDiff(block[r * V ..][0..V], wide[r * V ..][0..V]));
    std.debug.print("head at 128 rows: row 0 {d}, max|head {d} - head 128| {d:.6}\n", .{ argmax(wide[0..V]), rows, widest });

    // (3) a proposal, then the same block again: nothing the drafter wrote may move the target's own rows.
    var held: [dfly.drafts]u32 = @splat(0);
    try dfly.propose(m, dr, anchor, bonus, @intCast(rows - 1), dfly.source(dr, null), held[0 .. rows - 1]);
    std.debug.print("proposals {any}\n", .{held[0 .. rows - 1]});
    for (0..rows - 1) |i| ids[1 + i] = held[i];
    try m.passRows(ids[0..rows], c.logits, .{ .at = 0, .out = block[0 .. rows * V], .head = rows });
    m.history.items.len = anchor;
    std.debug.print("after a proposal, row 0: {d}, max|one - row 0| {d:.6}\n", .{ argmax(block[0..V]), maxDiff(one, block[0..V]) });
    for (0..rows) |r| std.debug.print("  verified row {d}: {d} (proposed {d})\n", .{ r, argmax(block[r * V ..][0..V]), if (r == 0) bonus else held[r - 1] });

    // (4) eight rounds as the server runs them: propose, verify the window, keep the accepted prefix. Then the
    // same eight steps one row at a time from the same prompt: the two streams are the D1 comparison in miniature.
    const steps = 8;
    var plain: [steps]u32 = undefined;
    var drafted: [steps]u32 = undefined;
    var accepted: usize = 0;
    var proposed: usize = 0;
    for (0..2) |arm| {
        m.history.items.len = prompt.len;
        c.valid_from = 0;
        if (arm == 0) try c.pass(prompt); // the logits the first draw comes from
        var pending = argmax(c.logits);
        var prev_kept: usize = 1; // the last kept row of the pass before this round
        for (0..steps) |step| {
            const at = m.history.items.len;
            const draw: u32 = if (arm == 0) blk: {
                const only = [1]u32{pending};
                try m.pass(&only, one);
                m.history.items.len = at;
                break :blk argmax(one);
            } else blk: {
                const depth: u32 = @intCast(rows - 1);
                var hold: [dfly.drafts]u32 = @splat(0);
                try dfly.propose(m, dr, at, pending, depth, if (step == 0) dfly.source(dr, null) else prev_kept - 1, hold[0..depth]);
                ids[0] = pending;
                for (0..depth) |i| ids[1 + i] = hold[i];
                try m.passRows(ids[0..rows], c.logits, .{ .at = 0, .out = block[0 .. rows * V], .head = rows });
                var kept: usize = 1;
                for (0..depth) |i| {
                    proposed += 1;
                    if (argmax(block[i * V ..][0..V]) != hold[i]) break;
                    kept += 1;
                    accepted += 1;
                }
                m.history.items.len = at + kept;
                prev_kept = kept;
                break :blk argmax(block[(kept - 1) * V ..][0..V]);
            };
            if (arm == 0) plain[step] = draw else drafted[step] = draw;
            pending = draw;
        }
        if (arm == 0) {
            std.debug.print("plain 8: {any}\n", .{plain});
            m.history.items.len = prompt.len;
        } else std.debug.print("drafted 8: {any} ({d}/{d} drafts accepted)\n", .{ drafted, accepted, proposed });
    }
    var agree: usize = 0;
    for (plain, drafted) |x, y| agree += @intFromBool(x == y);
    std.debug.print("the two streams agree on {d}/{d} tokens\n", .{ agree, steps });
}

fn argmax(l: []const f32) u32 {
    var best: usize = 0;
    for (l, 0..) |v, i| if (v > l[best]) {
        best = i;
    };
    return @intCast(best);
}

fn maxDiff(x: []const f32, y: []const f32) f32 {
    var most: f32 = 0;
    for (x, y) |p, q| most = @max(most, @abs(p - q));
    return most;
}

fn top2(l: []const f32) [2]f32 {
    var first: f32 = -std.math.inf(f32);
    var second: f32 = -std.math.inf(f32);
    for (l) |v| {
        if (v > first) {
            second = first;
            first = v;
        } else if (v > second) second = v;
    }
    return .{ first, second };
}

fn readIds(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]u32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20));
    if (bytes.len % 4 != 0) return error.BadPrompt;
    const out = try a.alloc(u32, bytes.len / 4);
    for (out, 0..) |*id, i| id.* = std.mem.readInt(u32, bytes[4 * i ..][0..4], .little);
    return out;
}
