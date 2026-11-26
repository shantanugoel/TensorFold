//! Prefix reuse against a fresh fill, on a device: both arms' logits and greedy tokens must be byte-identical.
const std = @import("std");
const cuda = @import("cuda");
const model = @import("cuda_model.zig");
const family = @import("serve_cuda.zig");

const usage = "usage: tf-hy4-cuda-reuse MODEL_DIR PROMPT.u32 CAPACITY REUSE_FROM [TOKENS]\n" ++
    "  CAPACITY      the context the model keeps, in slots (past 2,048 the indexer runs too)\n" ++
    "  REUSE_FROM    the row the second arm's prompt resumes at; not a multiple of four, and past the ring's wrap\n" ++
    "  TOKENS        greedy tokens both arms continue with (default 8)\n";

/// One arm's rows: the logits after the fill and after each of `TOKENS` passes, plus the tokens drawn from them.
const Arm = struct {
    logits: []f32, // (TOKENS + 1) rows of `vocab`
    tokens: []u32,
    vocab: usize,

    fn row(self: *const Arm, i: usize) []const f32 {
        return self.logits[i * self.vocab ..][0..self.vocab];
    }
};

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 5) {
        std.debug.print("{s}", .{usage});
        return error.BadArgs;
    }
    const dir = args[1];
    const prompt_ids = try readIds(a, io, args[2]);
    const capacity = try std.fmt.parseInt(usize, args[3], 10);
    const from = try std.fmt.parseInt(usize, args[4], 10);
    const tokens: usize = if (args.len > 5) try std.fmt.parseInt(usize, args[5], 10) else 8;
    if (from == 0 or from >= prompt_ids.len or from % 4 == 0) return error.BadReuseRow;
    if (capacity <= prompt_ids.len + tokens) return error.BadArgs;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    if (try ctx.capability() != 121) return error.NeedGb10;
    var s = try cuda.Stream.init(&driver, true);
    defer s.deinit();
    var kernels = try family.Kernels.load(a, &driver);
    defer kernels.deinit();
    const c = try family.Cuda.init(a, io, &driver, s, &kernels, dir, capacity);
    defer c.deinit();
    std.debug.print("loaded: {d}-slot context, reuse from row {d}, {d} tokens\n", .{ capacity, from, tokens });

    // (a) the fresh fill, then (b) the same prompt resumed from row `from`: both on one load, so only the fill differs.
    const fresh = try arm(c, a, prompt_ids, 0, tokens, "fresh");
    const reused = try arm(c, a, prompt_ids, from, tokens, "reuse");

    var bad: usize = 0;
    for (0..tokens + 1) |i| {
        if (!std.mem.eql(f32, fresh.row(i), reused.row(i))) {
            if (bad == 0) std.debug.print("first differing row: {d}\n", .{i});
            bad += 1;
        }
    }
    var tokens_differ = false;
    for (fresh.tokens, reused.tokens) |x, y| tokens_differ = tokens_differ or x != y;
    std.debug.print("{s}: logits rows differing {d}/{d}, greedy tokens {s}\n", .{
        if (bad == 0 and !tokens_differ) "PASS" else "FAIL", bad, tokens + 1, if (tokens_differ) "differ" else "identical",
    });
    if (bad != 0 or tokens_differ) return error.ReuseChangedTheLogits;
}

/// The prompt filled with a resume at `reuse` (0: from row 0), then `tokens` greedy passes; the reuse is reported.
fn arm(c: *family.Cuda, a: std.mem.Allocator, ids: []const u32, reuse: usize, tokens: usize, comptime name: []const u8) !Arm {
    reset(c);
    if (reuse != 0) {
        // The kept caches hold the shared prefix, as a released stream leaves them.
        const kept = try c.fill(ids[0..reuse], null);
        if (kept != 0) return error.BadReuseRow;
        c.kept = true;
    }
    const from = try c.fill(ids, null);
    if (reuse != 0 and from != reuse) {
        std.debug.print("{s}: the fill reused {d} rows, not {d}\n", .{ name, from, reuse });
        return error.ReuseDidNotEngage;
    }
    const vocab: usize = c.vocab;
    const rows = try a.alloc(f32, vocab * (tokens + 1));
    const drawn = try a.alloc(u32, tokens);
    @memcpy(rows[0..vocab], c.logits);
    std.debug.print("{s}: {d} rows reused, {d} prefilled\n", .{ name, from, ids.len - from });
    var token = argmax(c.logits);
    for (0..tokens) |i| {
        try c.pass(&.{token});
        @memcpy(rows[(i + 1) * vocab ..][0..vocab], c.logits);
        token = argmax(c.logits);
        drawn[i] = token;
    }
    return .{ .logits = rows, .tokens = drawn, .vocab = vocab };
}

/// An empty context: no history, no kept prefix, and every layer's cache cleared so neither arm reads the other's rows.
fn reset(c: *family.Cuda) void {
    c.m.cut(0, 0);
    c.valid_from = 0;
    c.kept = false;
    for (c.m.layers) |layer| layer.attention.cache.fill8(0, null) catch |e| std.debug.panic("clearing a layer's cache: {s}", .{@errorName(e)});
}

fn argmax(logits: []const f32) u32 {
    var best: usize = 0;
    for (logits, 0..) |v, i| if (v > logits[best]) {
        best = i;
    };
    return @intCast(best);
}

fn readIds(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]u32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 22));
    if (bytes.len < 4 or bytes.len % 4 != 0) return error.BadPrompt;
    const ids = try a.alloc(u32, bytes.len / 4);
    @memcpy(std.mem.sliceAsBytes(ids), bytes);
    return ids;
}
