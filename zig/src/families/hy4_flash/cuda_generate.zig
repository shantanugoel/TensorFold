//! Greedy generation and teacher-forced scoring with the CUDA Hy4-flash text model.
const std = @import("std");
const cuda = @import("cuda");
const model = @import("cuda_model.zig");

const usage = "usage: tf-hy4-cuda-generate MODEL_DIR PROMPT_IDS.u32 NEW_TOKENS OUT_DIR [CHUNK] [--score] [--logits-last N] [--no-ring] [--kv-cache fp8]\n" ++
    "  CHUNK            rows a pass takes, at most 1024 (default 1024)\n" ++
    "  --score          also write logits.f32 (every row but the last), scores.f32 and report.json\n" ++
    "  --logits-last N  with --score: keep only the last N of those rows, as exl3_reference.py does\n" ++
    "  --no-ring        keep the sliding layers' full-length KV: the arm the ring is gated against\n" ++
    "  --kv-cache fp8   the indexed layers' KV as E4M3 rows (the plan 7.7 arm; the ring stays BF16)\n";

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 5) {
        std.debug.print("{s}", .{usage});
        return error.BadArgs;
    }
    var chunk: usize = 1024;
    var score = false;
    var last: usize = 0;
    var ring = true;
    var kv8 = false;
    var i: usize = 5;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--score")) {
            score = true;
        } else if (std.mem.eql(u8, args[i], "--no-ring")) {
            ring = false;
        } else if (std.mem.eql(u8, args[i], "--kv-cache")) {
            if (i + 1 == args.len or !std.mem.eql(u8, args[i + 1], "fp8")) return error.BadArgs;
            i += 1;
            kv8 = true;
        } else if (std.mem.eql(u8, args[i], "--logits-last")) {
            if (i + 1 == args.len) return error.BadArgs;
            i += 1;
            last = try std.fmt.parseInt(usize, args[i], 10);
        } else chunk = try std.fmt.parseInt(usize, args[i], 10);
    }
    if (chunk == 0 or chunk > 1024) return error.BadArgs;
    const prompt_bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[2], a, .limited(1 << 20));
    if (prompt_bytes.len < 4) {
        std.debug.print("prompt {s} is empty: it holds no token ids\n", .{args[2]});
        return error.EmptyPrompt;
    }
    const prompt = try a.alloc(u32, prompt_bytes.len / 4);
    @memcpy(std.mem.sliceAsBytes(prompt), prompt_bytes[0 .. prompt.len * 4]);
    if (score and prompt.len < 2) {
        std.debug.print("--score needs two prompt tokens at least: a single token has no next token to score\n", .{});
        return error.ShortPrompt;
    }
    const new_tokens = try std.fmt.parseInt(usize, args[3], 10);
    try std.Io.Dir.cwd().createDirPath(io, args[4]);
    const out = try std.Io.Dir.cwd().openDir(io, args[4], .{});
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    if (try ctx.capability() != 121) return error.NeedGb10;
    var s = try cuda.Stream.init(&driver, true);
    defer s.deinit();
    var kernels = try model.Kernels.load(a, &driver);
    const t0 = std.Io.Clock.awake.now(io);
    const capacity = prompt.len + new_tokens;
    const max_rows = @min(chunk, prompt.len);
    const hy4 = try model.Model.open(a, io, &driver, s, &kernels, args[1], .{ .capacity = capacity, .max_rows = max_rows, .ring = ring, .kv8 = kv8 });
    const load_ns = t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
    std.debug.print("loaded {d} layers, {d:.2} GiB device weights in {d:.1} s\n", .{ hy4.layers.len, @as(f64, @floatFromInt(hy4.weight_bytes)) / (1 << 30), @as(f64, @floatFromInt(load_ns)) / 1e9 });
    const V: usize = hy4.text.vocab_size;
    const logits = try a.alloc(f32, V);
    // Prompt rows in `chunk`-row passes: a causal model's seam is exact, and the cap is 1,024 rows a pass.
    var pass_logits: ?[]f32 = null;
    var tail: []f32 = &.{};
    var scores: std.ArrayList(f32) = .empty;
    const scored = if (prompt.len == 0) 0 else prompt.len - 1; // row i's logits predict token i + 1
    const keep = if (last == 0 or last >= scored) scored else last;
    if (score) {
        try out.writeFile(io, .{ .sub_path = "prompt.u32", .data = std.mem.sliceAsBytes(prompt) });
        pass_logits = try a.alloc(f32, @max(max_rows, 1) * V);
        tail = try a.alloc(f32, @max(keep, 1) * V);
        try scores.ensureTotalCapacity(a, scored);
    }
    const p0 = std.Io.Clock.awake.now(io);
    var model_ns: i96 = 0; // the passes alone, so the host's scoring is not read as the model's time
    var score_ns: i96 = 0; // the log-softmax and the tail's copy, which are this binary's bookkeeping
    var row: usize = 0;
    while (row < prompt.len) : (row += chunk) {
        const n = @min(chunk, prompt.len - row);
        const m0 = std.Io.Clock.awake.now(io);
        if (pass_logits) |all| {
            try hy4.passRows(prompt[row..][0..n], logits, .{ .at = 0, .out = all[0 .. n * V] });
        } else {
            try hy4.pass(prompt[row..][0..n], logits);
        }
        model_ns += m0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
        if (pass_logits) |all| {
            const s0 = std.Io.Clock.awake.now(io);
            // Every prompt row but the last one predicts a token, so the last pass's tail row is out.
            try appendScores(&scores, all[0 .. n * V], prompt[row..][0..n], @min(n, prompt.len - 1 - row), V);
            // logits.f32 holds the window's last `keep` rows, so a long window's every-row logits never fit.
            const lo = @max(row, scored - keep);
            const hi = @min(row + n, scored);
            if (lo < hi) @memcpy(tail[(lo - (scored - keep)) * V ..][0 .. (hi - lo) * V], all[(lo - row) * V ..][0 .. (hi - lo) * V]);
            score_ns += s0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
        }
    }
    const prefill_ns = p0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
    try out.writeFile(io, .{ .sub_path = "prefill-logits.f32", .data = std.mem.sliceAsBytes(logits) });
    if (pass_logits != null) try writeScores(io, a, out, tail[0 .. keep * V], scores.items, prompt.len, keep, chunk, ring, .{ .all = prefill_ns, .model = model_ns, .score = score_ns });
    const d0 = std.Io.Clock.awake.now(io);
    var copy: std.ArrayList(u32) = .empty;
    for (0..new_tokens) |step| {
        var best: usize = 0;
        for (logits, 0..) |v, idx| if (v > logits[best]) {
            best = idx;
        };
        try copy.append(a, @intCast(best));
        if (step + 1 == new_tokens) break;
        const one = [1]u32{@intCast(best)};
        try hy4.pass(&one, logits);
    }
    const decode_ns = d0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
    try out.writeFile(io, .{ .sub_path = "generated.u32", .data = std.mem.sliceAsBytes(copy.items) });
    std.debug.print("prompt {d} tokens: prefill {d:.1} ms ({d:.1} tok/s) [model {d:.1} ms, {d:.1} tok/s; score {d:.1} ms]; decode {d} tokens {d:.1} ms/token\nengram rows {d:.1} ms of the prefill\ngenerated={any}\n", .{
        prompt.len,
        @as(f64, @floatFromInt(prefill_ns)) / 1e6,
        @as(f64, @floatFromInt(prompt.len)) * 1e9 / @as(f64, @floatFromInt(prefill_ns)),
        @as(f64, @floatFromInt(model_ns)) / 1e6,
        @as(f64, @floatFromInt(prompt.len)) * 1e9 / @as(f64, @floatFromInt(model_ns)),
        @as(f64, @floatFromInt(score_ns)) / 1e6,
        new_tokens,
        @as(f64, @floatFromInt(decode_ns)) / 1e6 / @as(f64, @floatFromInt(@max(new_tokens, 2) - 1)),
        @as(f64, @floatFromInt(hy4.engram_ns)) / 1e6,
        copy.items,
    });
}

/// The first `count` rows' next-token log-probability: row `i` of the pass predicts `tokens[i + 1]`.
fn appendScores(scores: *std.ArrayList(f32), rows: []const f32, tokens: []const u32, count: usize, voc: usize) !void {
    for (0..count) |i| {
        const next = tokens[i + 1];
        const row = rows[i * voc ..][0..voc];
        var hi: f32 = row[0];
        for (row) |v| hi = @max(hi, v);
        var sum: f64 = 0;
        for (row) |v| sum += @exp(v - hi);
        if (next >= voc) return error.BadHy4PassRows;
        scores.appendAssumeCapacity(row[next] - (hi + @as(f32, @floatCast(@log(sum)))));
    }
}

/// `exl3_reference.py --score --save-logits`'s layout: logits.f32 for the kept rows, every scored row's log-prob.
const Phases = struct { all: i96, model: i96, score: i96 };

fn writeScores(io: std.Io, a: std.mem.Allocator, out: std.Io.Dir, tail: []const f32, scores: []const f32, prompt_tokens: usize, keep: usize, chunk: usize, ring: bool, ns: Phases) !void {
    try out.writeFile(io, .{ .sub_path = "logits.f32", .data = std.mem.sliceAsBytes(tail) });
    try out.writeFile(io, .{ .sub_path = "scores.f32", .data = std.mem.sliceAsBytes(scores) });
    var nll: f64 = 0;
    for (scores) |s| nll -= @as(f64, s);
    const rows = scores.len;
    const per = @as(f64, @floatFromInt(@max(rows, 1))); // a one-token prompt scores nothing
    const report = try std.fmt.allocPrint(a, "{{\"prompt_tokens\": {d}, \"scored\": {d}, \"logits_rows\": {d}, \"nll\": {d}, \"ppl\": {d}, \"chunk\": {d}, \"ring\": {s}, \"prefill_s\": {d}, \"model_s\": {d}, \"score_s\": {d}}}\n", .{
        prompt_tokens, rows, keep, nll / per, @exp(nll / per), chunk, if (ring) "true" else "false",
        @as(f64, @floatFromInt(ns.all)) / 1e9, @as(f64, @floatFromInt(ns.model)) / 1e9, @as(f64, @floatFromInt(ns.score)) / 1e9,
    });
    try out.writeFile(io, .{ .sub_path = "report.json", .data = report });
    std.debug.print("scored {d} rows, kept {d}: nll {d:.4} ppl {d:.4}\n", .{ rows, keep, nll / per, @exp(nll / per) });
}
