//! The DFly2 verify window on a device: one row against the same row inside the block, and the round after a
//! proposal. A host check cannot see this: it needs the real head, the real caches and the drafter's own writes.
const std = @import("std");
const cuda = @import("cuda");
const model = @import("cuda_model.zig");
const dfly = @import("cuda_draft.zig");
const family = @import("serve_cuda.zig");

/// Most blocks a width's KL distribution keeps, for the median and p90 (`POSITIONS` is read up to this).
const most_blocks = 512;

const usage = "usage: tf-hy4-cuda-draft MODEL_DIR PROMPT.u32 CAPACITY [ROWS] [POSITIONS] [TRACE]\n" ++
    "       tf-hy4-cuda-draft MODEL_DIR PROMPT.u32 CAPACITY dump DIR CONT.u32 N\n" ++
    "  PROMPT.u32  little-endian u32 token ids\n" ++
    "  CAPACITY    the context the model keeps, in slots (past 2,048 the indexer runs too)\n" ++
    "  ROWS        the verify window's rows (default 6)\n" ++
    "  POSITIONS   instead of the round checks, measure the row count: that many rows a window width,\n" ++
    "              every row against the same row fed one at a time (KL, top-1, max |delta logit|)\n" ++
    "  TRACE       1: also compare one row's hidden state after every layer, two ways, to place a difference\n" ++
    "  dump        the same-input differential: for t = 0..N-1 a round from PROMPT + CONT's first t tokens,\n" ++
    "              dumping the fused anchor row, the block's final-normed rows, each slot's corrected hidden\n" ++
    "              and logits row, and the verify pass's own rows, to DIR/round-{t}.bin\n";

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
    const dumps: bool = args.len > 4 and std.mem.eql(u8, args[4], "dump");
    const rows: usize = if (!dumps and args.len > 4) try std.fmt.parseInt(usize, args[4], 10) else dfly.width;
    const positions: usize = if (!dumps and args.len > 5) try std.fmt.parseInt(usize, args[5], 10) else 0;
    const trace: bool = if (!dumps and args.len > 6) (try std.fmt.parseInt(u32, args[6], 10)) != 0 else false;
    if (dumps and args.len < 8) return error.BadArgs;
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
    const c = try family.Cuda.init(a, io, &driver, s, &kernels, dir, capacity, dfly.drafts, 1);
    defer c.deinit();
    const m = c.m;
    const V: usize = m.text.vocab_size;
    const dr = &m.draft.?;
    std.debug.print("loaded: {d}-slot context, {d}-row window, prompt {d} tokens\n", .{ capacity, rows, prompt.len });

    if (dumps) {
        try dumpRounds(a, io, c, prompt, args[5], args[6], try std.fmt.parseInt(usize, args[7], 10));
        return;
    }

    // The prompt, then the anchor's own row as the plain path draws it.
    _ = try c.fillIds(prompt); // a prompt longer than one pass fills in `prompt_rows` chunks
    const one = try a.alloc(f32, V);
    const block = try a.alloc(f32, rows * V);
    const wide = try a.alloc(f32, rows * V);
    const anchor = m.history[0].items.len;
    const bonus = argmax(c.logits);
    std.debug.print("anchor {d}, plain draw {d}\n", .{ anchor, bonus });

    // The row count on its own: nothing else in this program runs, so a difference is the pass's and not a
    // round's. Every width sees the same greedy token stream, so the arms differ only in how many rows go
    // through one forward.
    if (positions > 0) {
        try measureRows(a, c, prompt.len, positions, argmax(c.logits), trace);
        return;
    }

    // (1) the anchor alone, the plain one-row path with the LM head's own 128-row block.
    const ids1 = [1]u32{bonus};
    try m.pass(&ids1, one);
    m.history[0].items.len = anchor;
    std.debug.print("one row: {d}\n", .{argmax(one)});

    // (2) the same anchor as the block's first row, the verify path's head over `rows` rows.
    var ids: [dfly.width]u32 = @splat(dr.mask);
    ids[0] = bonus;
    try m.passRows(ids[0..rows], c.logits, .{ .at = 0, .out = block[0 .. rows * V], .head = rows });
    m.history[0].items.len = anchor;
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
    m.history[0].items.len = anchor;
    var widest: f32 = 0;
    for (0..rows) |r| widest = @max(widest, maxDiff(block[r * V ..][0..V], wide[r * V ..][0..V]));
    std.debug.print("head at 128 rows: row 0 {d}, max|head {d} - head 128| {d:.6}\n", .{ argmax(wide[0..V]), rows, widest });

    // (3) a proposal, then the same block again: nothing the drafter wrote may move the target's own rows.
    var held: [dfly.drafts]u32 = @splat(0);
    try dfly.propose(m, dr, 0, anchor, bonus, @intCast(rows - 1), dfly.source(dr, 0, null), held[0 .. rows - 1]);
    std.debug.print("proposals {any}\n", .{held[0 .. rows - 1]});
    for (0..rows - 1) |i| ids[1 + i] = held[i];
    try m.passRows(ids[0..rows], c.logits, .{ .at = 0, .out = block[0 .. rows * V], .head = rows });
    m.history[0].items.len = anchor;
    std.debug.print("after a proposal, row 0: {d}, max|one - row 0| {d:.6}\n", .{ argmax(block[0..V]), maxDiff(one, block[0..V]) });
    for (0..rows) |r| std.debug.print("  verified row {d}: {d} (proposed {d})\n", .{ r, argmax(block[r * V ..][0..V]), if (r == 0) bonus else held[r - 1] });

    // (4) eight rounds as the server runs them: propose, verify the window, keep the accepted prefix. Then the
    // same eight steps one row at a time from the same prompt: the two streams are the D1 comparison in miniature.
    const steps = 32;
    var plain: [steps]u32 = undefined;
    var drafted: [steps]u32 = undefined;
    var accepted: usize = 0;
    var proposed: usize = 0;
    var slot_proposed: [dfly.drafts]usize = @splat(0);
    var slot_kept: [dfly.drafts]usize = @splat(0);
    var rounds: usize = 0;
    for (0..2) |arm| {
        m.history[0].items.len = prompt.len;
        c.seq[0].valid_from = 0;
        if (arm == 0) _ = try c.fillIds(prompt); // the logits the first draw comes from
        var pending = argmax(c.logits);
        var prev_kept: usize = 1; // the last kept row of the pass before this round
        for (0..steps) |step| {
            const at = m.history[0].items.len;
            const draw: u32 = if (arm == 0) blk: {
                const only = [1]u32{pending};
                try m.pass(&only, one);
                m.history[0].items.len = at;
                break :blk argmax(one);
            } else blk: {
                const depth: u32 = @intCast(rows - 1);
                var hold: [dfly.drafts]u32 = @splat(0);
                try dfly.propose(m, dr, 0, at, pending, depth, if (step == 0) dfly.source(dr, 0, null) else prev_kept - 1, hold[0..depth]);
                ids[0] = pending;
                for (0..depth) |i| ids[1 + i] = hold[i];
                try m.passRows(ids[0..rows], c.logits, .{ .at = 0, .out = block[0 .. rows * V], .head = rows });
                var kept: usize = 1;
                rounds += 1;
                for (0..depth) |i| {
                    proposed += 1;
                    slot_proposed[i] += 1;
                    if (argmax(block[i * V ..][0..V]) != hold[i]) break;
                    slot_kept[i] += 1;
                    kept += 1;
                    accepted += 1;
                }
                m.history[0].items.len = at + kept;
                prev_kept = kept;
                break :blk argmax(block[(kept - 1) * V ..][0..V]);
            };
            if (arm == 0) plain[step] = draw else drafted[step] = draw;
            pending = draw;
        }
        if (arm == 0) {
            std.debug.print("plain {d}: {any}\n", .{ steps, plain });
            m.history[0].items.len = prompt.len;
        } else {
            std.debug.print("drafted {d}: {any}\n", .{ steps, drafted });
            std.debug.print("{d}/{d} drafts accepted over {d} rounds = {d:.2} a step; by slot:\n", .{ accepted, proposed, rounds, @as(f64, @floatFromInt(accepted)) / @as(f64, @floatFromInt(@max(rounds, 1))) });
            for (0..dfly.drafts) |i| if (slot_proposed[i] > 0)
                std.debug.print("  slot {d}: {d} of {d} kept ({d:.2})\n", .{ i + 1, slot_kept[i], slot_proposed[i], @as(f64, @floatFromInt(slot_kept[i])) / @as(f64, @floatFromInt(slot_proposed[i])) });
        }
    }
    var agree: usize = 0;
    for (plain, drafted) |x, y| agree += @intFromBool(x == y);
    std.debug.print("the two streams agree on {d}/{d} tokens\n", .{ agree, steps });
}

/// A pass's cost by head block and row count: the plain step (`head` null: the head's own 128-row block) against
/// the same row with the head at 1, and a window of `width` rows against the same rows one at a time.
fn timeShapes(a: std.mem.Allocator, c: *family.Cuda, from: usize) !void {
    const m = c.m;
    const V: usize = m.text.vocab_size;
    const ids = try a.alloc(u32, dfly.width);
    for (ids, 0..) |*t, i| t.* = @intCast(1000 + i * 37);
    const out = try a.alloc(f32, dfly.width * V);
    std.debug.print("a pass by shape (median of 7):\n", .{});
    const shapes = [_]struct { name: []const u8, rows: usize, head: ?usize }{
        .{ .name = "1 row, head's own 128-row block", .rows = 1, .head = null },
        .{ .name = "1 row, head at 1", .rows = 1, .head = 1 },
        .{ .name = "2 rows, head at 2", .rows = 2, .head = 2 },
        .{ .name = "6 rows, head at 6", .rows = 6, .head = 6 },
        .{ .name = "6 rows, head at 128", .rows = 6, .head = null },
    };
    for (shapes) |sh| {
        var times: [7]f64 = undefined;
        for (&times) |*tm| {
            m.history[0].items.len = from;
            const t0 = std.Io.Clock.awake.now(c.io);
            try m.passRows(ids[0..sh.rows], c.logits, .{ .at = 0, .out = out[0 .. sh.rows * V], .head = sh.head });
            tm.* = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(c.io)).toNanoseconds())) / 1e6;
        }
        std.mem.sort(f64, &times, {}, std.sort.asc(f64));
        std.debug.print("  {s:<38} {d:8.2} ms ({d:6.2} ms a row)\n", .{ sh.name, times[3], times[3] / @as(f64, @floatFromInt(sh.rows)) });
    }
    m.history[0].items.len = from;
}

/// One row's four streams after every layer, the same row fed alone and inside a six-row window: the first layer
/// whose output moves is the kernel that depends on the pass's row count, and the trunk's rest cannot be blamed.
fn traceLayers(a: std.mem.Allocator, c: *family.Cuda, from: usize, first: u32) !void {
    const geom = @import("cuda_scratch.zig");
    const m = c.m;
    const V: usize = m.text.vocab_size;
    const row_bytes = geom.S * geom.H * 2; // four streams of BF16 hidden
    const layers = m.layers.len;
    const geom_k = @import("cuda_scratch.zig").K;
    const geom_e = @import("cuda_scratch.zig").E;
    var buf = try cuda.DeviceBuffer.alloc(m.d, layers * row_bytes);
    defer buf.free();
    var picks = try cuda.DeviceBuffer.alloc(m.d, layers * geom_k * 4);
    defer picks.free();
    var scores = try cuda.DeviceBuffer.alloc(m.d, layers * geom_e * 4);
    defer scores.free();
    const alone_picks = try a.alloc(u8, layers * geom_k * 4);
    const window_picks = try a.alloc(u8, layers * geom_k * 4);
    const alone_scores = try a.alloc(u8, layers * geom_e * 4);
    const window_scores = try a.alloc(u8, layers * geom_e * 4);
    const alone = try a.alloc(u8, layers * row_bytes);
    const window = try a.alloc(u8, layers * row_bytes);
    var ids: [dfly.width]u32 = @splat(0);
    for (&ids, 0..) |*t, i| t.* = @intCast(1000 + i * 37);
    ids[0] = first;
    const out = try a.alloc(f32, dfly.width * V);
    m.history[0].items.len = from;
    m.trace = .{ .row = 0, .buf = buf, .picks = picks, .scores = scores };
    defer m.trace = null;
    try c.pass(0, ids[0..1]);
    try buf.downloadAsync(0, alone, m.s.handle);
    try picks.downloadAsync(0, alone_picks, m.s.handle);
    try scores.downloadAsync(0, alone_scores, m.s.handle);
    try m.s.synchronize();
    m.history[0].items.len = from;
    try m.passRows(ids[0..], c.logits, .{ .at = 0, .out = out[0 .. dfly.width * V], .head = dfly.width });
    try buf.downloadAsync(0, window, m.s.handle);
    try picks.downloadAsync(0, window_picks, m.s.handle);
    try scores.downloadAsync(0, window_scores, m.s.handle);
    try m.s.synchronize();
    m.history[0].items.len = from;
    std.debug.print("row 0's hidden after every layer, one row alone against the same row in a {d}-row pass:\n", .{dfly.width});
    std.debug.print("  layer   relative RMS   max |delta|   (row alone's RMS)\n", .{});
    for (0..layers) |l| {
        const x = alone[l * row_bytes ..][0..row_bytes];
        const y = window[l * row_bytes ..][0..row_bytes];
        var diff: f64 = 0;
        var base: f64 = 0;
        var most: f32 = 0;
        var i: usize = 0;
        while (i < geom.S * geom.H) : (i += 1) {
            const u = widenHalf(x, i);
            const v = widenHalf(y, i);
            const d = u - v;
            diff += @as(f64, d) * d;
            base += @as(f64, v) * v;
            most = @max(most, @abs(d));
        }
        std.debug.print("  {d:>5} {d:13.5} {d:12.5} {d:16.5}\n", .{ l, @sqrt(diff / base), most, @sqrt(base / @as(f64, geom.S * geom.H)) });
    }
    // The routed half is the one discrete decision a row makes: an expert set that differs is a different row.
    std.debug.print("row 0's router picks, one row alone against the same row in a {d}-row pass:\n", .{dfly.width});
    var differs: usize = 0;
    for (0..layers) |l| {
        var apart: usize = 0;
        var gap_alone: f32 = 0;
        var gap_window: f32 = 0;
        for (0..geom_k) |k| {
            const u = std.mem.readInt(u32, alone_picks[(l * geom_k + k) * 4 ..][0..4], .little);
            const v = std.mem.readInt(u32, window_picks[(l * geom_k + k) * 4 ..][0..4], .little);
            if (u != v) apart += 1;
        }
        if (apart == 0) continue;
        differs += 1;
        // How close the two arms' own scores were: a tie is a rounding difference, a gap is a wrong read.
        for (0..geom_k) |k| {
            const u = std.mem.readInt(u32, alone_picks[(l * geom_k + k) * 4 ..][0..4], .little);
            const v = std.mem.readInt(u32, window_picks[(l * geom_k + k) * 4 ..][0..4], .little);
            const su = readF32(alone_scores, l * geom_e + u);
            const sv = readF32(window_scores, l * geom_e + v);
            gap_alone = @max(gap_alone, @abs(su - readF32(alone_scores, l * geom_e + v)));
            gap_window = @max(gap_window, @abs(sv - readF32(window_scores, l * geom_e + u)));
        }
        std.debug.print("  layer {d:>2}: {d} of {d} picks differ; |score(chosen) - score(other)| {d:.6} alone, {d:.6} in the window\n", .{ l, apart, geom_k, gap_alone, gap_window });
    }
    std.debug.print("  {d} of {d} routed layers pick a different expert set at row 0\n", .{ differs, layers });
}

/// The same-input differential: `count` rounds, round t from `prompt` + the continuation's first t tokens, each
/// proposed and then verified, with everything a comparison against another engine needs written to
/// `dir/round-{t}.bin`: a magic word, seven header words, then the arrays in the order the header names them.
///
/// The prefix is the plain greedy continuation, so both engines see the same ids and the same target state and what
/// differs between them is the tower. A round's own draw is dumped beside the teacher's token: a round the engine
/// drew something else at is one the other engine reached from a different anchor, and the reader can drop it.
fn dumpRounds(a: std.mem.Allocator, io: std.Io, c: *family.Cuda, prompt: []const u32, dir: []const u8,
              cont_path: []const u8, count: usize) !void {
    const m = c.m;
    const V: usize = m.text.vocab_size;
    const dr = &m.draft.?;
    const cont = try readIds(a, io, cont_path);
    if (cont.len < count + dfly.drafts) return error.BadPrompt;
    const H = @import("cuda_scratch.zig").H;
    var trail: dfly.Trail = .{
        .fused = try cuda.DeviceBuffer.alloc(m.d, H * 2),
        .carry = try cuda.DeviceBuffer.alloc(m.d, H * 2),
        .cat = try cuda.DeviceBuffer.alloc(m.d, 2 * H * 2),
        .proj = try cuda.DeviceBuffer.alloc(m.d, H * 2),
        .cat1 = try cuda.DeviceBuffer.alloc(m.d, 2 * H * 2),
        .delta = try cuda.DeviceBuffer.alloc(m.d, H * 2),
        .rows = try cuda.DeviceBuffer.alloc(m.d, dfly.width * H * 2),
        .corrected = try cuda.DeviceBuffer.alloc(m.d, dfly.drafts * H * 2),
        .logits = try cuda.DeviceBuffer.alloc(m.d, dfly.drafts * V * 4),
    };
    defer {
        trail.fused.free();
        trail.carry.free();
        trail.cat.free();
        trail.proj.free();
        trail.cat1.free();
        trail.delta.free();
        trail.rows.free();
        trail.corrected.free();
        trail.logits.free();
    }
    dr.trail = &trail;
    defer dr.trail = null;
    const fused = try a.alloc(u8, H * 2);
    const carry = try a.alloc(u8, H * 2);
    const cat = try a.alloc(u8, 2 * H * 2);
    const proj = try a.alloc(u8, H * 2);
    const cat1 = try a.alloc(u8, 2 * H * 2);
    const delta = try a.alloc(u8, H * 2);
    const rows_host = try a.alloc(u8, dfly.width * H * 2);
    const corrected = try a.alloc(u8, dfly.drafts * H * 2);
    const logits_host = try a.alloc(u8, dfly.drafts * V * 4);
    const verify = try a.alloc(f32, dfly.width * V);
    const block = try a.alloc(f32, dfly.width * V);
    var bytes: std.ArrayList(u8) = .empty;
    var ids: [dfly.width]u32 = @splat(dr.mask);
    var held: [dfly.drafts]u32 = @splat(0);
    const prefix = try a.alloc(u32, prompt.len + count + 1);
    var accepted: usize = 0;
    var slot_ok: [dfly.drafts]usize = @splat(0);
    for (0..count) |t| {
        @memcpy(prefix[0..prompt.len], prompt);
        @memcpy(prefix[prompt.len..][0..t], cont[0..t]);
        // The other engine's shape, exactly: its Generator prefills the prefix without its last token and decodes
        // that one as a single row, so the carry this round fuses comes from a 1-row pass here too. A prefill
        // chunk's last row instead is a *different* row at BF16 (the pass shape is a wall in its own right, see
        // the note's "What 1.43 was"), and that difference would sit on top of everything this dump measures.
        const last = prompt.len + t - 1; // the prefix without its own last token; t = 0 is the prompt itself
        _ = try c.fillIds(prefix[0..last]);
        const only = [1]u32{prefix[last]};
        try m.pass(&only, c.logits);
        const anchor = m.history[0].items.len;
        const drawn = argmax(c.logits);
        const bonus = cont[t]; // teacher: the continuation's own token, whether or not this engine drew it
        try dfly.propose(m, dr, 0, anchor, bonus, dfly.drafts, dfly.source(dr, 0, null), &held);
        ids[0] = bonus;
        for (0..dfly.drafts) |i| ids[1 + i] = held[i];
        try m.passRows(ids[0..], c.logits, .{ .at = 0, .out = block[0 .. dfly.width * V], .head = dfly.width });
        @memcpy(verify, block);
        m.history[0].items.len = anchor;
        try trail.fused.downloadAsync(0, fused, m.s.handle);
        try trail.carry.downloadAsync(0, carry, m.s.handle);
        try trail.cat.downloadAsync(0, cat, m.s.handle);
        try trail.proj.downloadAsync(0, proj, m.s.handle);
        try trail.cat1.downloadAsync(0, cat1, m.s.handle);
        try trail.delta.downloadAsync(0, delta, m.s.handle);
        try trail.rows.downloadAsync(0, rows_host, m.s.handle);
        try trail.corrected.downloadAsync(0, corrected, m.s.handle);
        try trail.logits.downloadAsync(0, logits_host, m.s.handle);
        try m.s.synchronize();
        // Proposals against the continuation's own next tokens: the number the other engine's rounds give.
        var line: [dfly.drafts]u8 = @splat('.');
        for (0..dfly.drafts) |i| {
            if (held[i] == cont[t + 1 + i]) {
                line[i] = '#';
                slot_ok[i] += 1;
            }
        }
        accepted += std.mem.count(u8, &line, "#");
        std.debug.print("round {d}: anchor {d} bonus {d} (this engine drew {d}), proposals {any}, continuation {any} {s}, verify rows {any}\n", .{
            t, anchor, bonus, drawn, held, cont[t + 1 ..][0..dfly.drafts], line, block0(block, V, dfly.width),
        });
        bytes.clearRetainingCapacity();
        for ([_]u32{ 0x344c4648, @intCast(anchor), bonus, drawn, dfly.width, dr.mask }) |word| try appendWord(&bytes, a, word);
        for (held) |h| try appendWord(&bytes, a, h);
        try bytes.appendSlice(a, fused);
        try bytes.appendSlice(a, carry);
        try bytes.appendSlice(a, cat);
        try bytes.appendSlice(a, proj);
        try bytes.appendSlice(a, cat1);
        try bytes.appendSlice(a, delta);
        try bytes.appendSlice(a, rows_host);
        try bytes.appendSlice(a, corrected);
        try bytes.appendSlice(a, logits_host);
        try bytes.appendSlice(a, std.mem.sliceAsBytes(verify));
        var name: [96]u8 = undefined;
        const path = try std.fmt.bufPrint(&name, "{s}/round-{d}.bin", .{ dir, t });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes.items });
    }
    std.debug.print("teacher rounds: {d} of {d} proposals matched over {d} rounds ({d:.2} a round); by slot", .{
        accepted, count * dfly.drafts, count, @as(f64, @floatFromInt(accepted)) / @as(f64, @floatFromInt(@max(count, 1))),
    });
    for (slot_ok, 0..) |n, i| std.debug.print(" {d}: {d:.2}", .{ i + 1, @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(@max(count, 1))) });
    std.debug.print("\n", .{});
}

fn appendWord(list: *std.ArrayList(u8), a: std.mem.Allocator, word: u32) !void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, word, .little);
    try list.appendSlice(a, &buf);
}

/// Every row's argmax of a `rows`-row block of logits.
fn block0(logits: []const f32, V: usize, rows: usize) [dfly.width]u32 {
    var out: [dfly.width]u32 = undefined;
    for (0..rows) |r| out[r] = argmax(logits[r * V ..][0..V]);
    return out;
}

fn readF32(bytes: []const u8, i: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little));
}

fn widenHalf(bytes: []const u8, i: usize) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)) << 16);
}

/// The same rows two ways: `positions` rows of one greedy stream fed a row at a time, then the same tokens in
/// one window of `width` rows, row against row. Per row: KL(window || single), top-1 agreement and |delta|.
/// A window that reshapes the trunk shows up here and nowhere else; the harness's rounds hide it in acceptance.
fn measureRows(a: std.mem.Allocator, c: *family.Cuda, from: usize, positions: usize, first: u32, trace: bool) !void {
    const m = c.m;
    const V: usize = m.text.vocab_size;
    const single = try a.alloc(f32, dfly.width * V);
    const block = try a.alloc(f32, dfly.width * V);
    const ids = try a.alloc(u32, dfly.width);
    // What a row costs by shape. The plain path's head runs `geom.head_block` rows whatever the pass holds;
    // a window runs its own width. Same logits either way (the head at 6 rows is bit-identical at 128).
    try timeShapes(a, c, from);
    if (trace) try traceLayers(a, c, from, first);
    std.debug.print("row count: {d} rows a width from position {d}, KL(window || one at a time)\n", .{ positions, from });
    for ([_]usize{ 6, 4, 2 }) |width| {
        if (width > dfly.width) continue;
        m.history[0].items.len = from;
        // The stream's first token is the prompt's own draw; each arm re-derives the rest from its own rows, so
        // the tokens are the one-row arm's and both arms see them.
        var next: u32 = first;
        var kl_all: [dfly.width][most_blocks]f64 = @splat(@splat(0));
        var kl_most: [dfly.width]f64 = @splat(0);
        var worst: [dfly.width]f32 = @splat(0);
        var apart: [dfly.width]f32 = @splat(0);
        var miss: [dfly.width]usize = @splat(0);
        var control: f32 = 0; // the same one-row pass twice: what this box's own arithmetic does
        var blocks: usize = 0;
        while (blocks * width < positions) : (blocks += 1) {
            const at = m.history[0].items.len;
            if (at + width > m.capacity) break;
            for (0..width) |r| {
                ids[r] = next;
                const only = [1]u32{next};
                try c.pass(0, &only);
                @memcpy(single[r * V ..][0..V], c.logits);
                next = argmax(c.logits);
            }
            m.history[0].items.len = at;
            try m.passRows(ids[0..width], c.logits, .{ .at = 0, .out = block[0 .. width * V], .head = width });
            m.history[0].items.len = at + width;
            if (blocks == 0) { // a control: row 0 fed alone twice, which must be the same numbers
                m.history[0].items.len = at;
                const again = try a.alloc(f32, V);
                defer a.free(again);
                const only = [1]u32{ids[0]};
                try m.passRows(&only, c.logits, .{ .at = 0, .out = again, .head = 1 });
                control = maxDiff(single[0..V], again);
                m.history[0].items.len = at + width;
            }
            for (0..width) |r| {
                const kl = divergence(block[r * V ..][0..V], single[r * V ..][0..V]);
                if (blocks < most_blocks) kl_all[r][blocks] = kl;
                kl_most[r] = @max(kl_most[r], kl);
                var most: f32 = 0;
                var total: f32 = 0;
                for (block[r * V ..][0..V], single[r * V ..][0..V]) |x, y| {
                    most = @max(most, @abs(x - y));
                    total += @abs(x - y);
                }
                worst[r] = @max(worst[r], most);
                apart[r] = @max(apart[r], total / @as(f32, @floatFromInt(V)));
                miss[r] += @intFromBool(argmax(block[r * V ..][0..V]) != argmax(single[r * V ..][0..V]));
            }
        }
        if (blocks == 0) continue;
        std.debug.print("  width {d}: {d} blocks ({d} rows), the same row alone twice: max |d| {d:.2}\n", .{ width, blocks, blocks * width, control });
        std.debug.print("    row  median KL    p90 KL   max KL   over 0.05   top-1 miss   max |d|   mean |d|\n", .{});
        for (0..width) |r| {
            const n = @min(blocks, most_blocks);
            std.mem.sort(f64, kl_all[r][0..n], {}, std.sort.asc(f64));
            const median = if (n == 0) 0 else kl_all[r][n / 2];
            const p90 = if (n == 0) 0 else kl_all[r][@min(n - 1, (n * 9) / 10)];
            var over: usize = 0;
            for (kl_all[r][0..n]) |v| over += @intFromBool(v > 0.05);
            std.debug.print("  {d:>5} {d:9.5} {d:9.5} {d:8.5} {d:9} of {d:<3} {d:7} of {d:<3} {d:8.4} {d:9.5}\n", .{
                r, median, p90, kl_most[r], over, n, miss[r], n, worst[r], apart[r],
            });
        }
    }
}

/// KL(a || b) over one row of FP32 logits, nats.
fn divergence(a: []const f32, b: []const f32) f64 {
    var la: f64 = -std.math.inf(f64);
    var lb: f64 = -std.math.inf(f64);
    for (a) |v| la = @max(la, v);
    for (b) |v| lb = @max(lb, v);
    var za: f64 = 0;
    var zb: f64 = 0;
    for (a, b) |x, y| {
        za += @exp(@as(f64, x) - la);
        zb += @exp(@as(f64, y) - lb);
    }
    const na = la + @log(za);
    const nb = lb + @log(zb);
    var kl: f64 = 0;
    for (a, b) |x, y| {
        const lg = @as(f64, x) - na;
        kl += @exp(lg) * (lg - (@as(f64, y) - nb));
    }
    return @max(kl, 0);
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
