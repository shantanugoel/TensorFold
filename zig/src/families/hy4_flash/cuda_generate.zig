//! Greedy generation and teacher-forced scoring with the CUDA Hy4-flash text model.
const std = @import("std");
const cuda = @import("cuda");
const model = @import("cuda_model.zig");

const usage = "usage: tf-hy4-cuda-generate MODEL_DIR PROMPT_IDS.u32 NEW_TOKENS OUT_DIR [CHUNK] [--score]\n" ++
    "  CHUNK     rows a pass takes, at most 1024 (default 1024)\n" ++
    "  --score   also write logits.f32 (every row but the last), scores.f32 and report.json\n";

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 5 or args.len > 7) {
        std.debug.print("{s}", .{usage});
        return error.BadArgs;
    }
    var chunk: usize = 1024;
    var score = false;
    for (args[5..]) |arg| {
        if (std.mem.eql(u8, arg, "--score")) score = true else chunk = try std.fmt.parseInt(usize, arg, 10);
    }
    if (chunk == 0 or chunk > 1024) return error.BadArgs;
    const prompt_bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[2], a, .limited(1 << 20));
    const prompt = try a.alloc(u32, prompt_bytes.len / 4);
    @memcpy(std.mem.sliceAsBytes(prompt), prompt_bytes[0 .. prompt.len * 4]);
    const new_tokens = try std.fmt.parseInt(usize, args[3], 10);
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
    const hy4 = try model.Model.open(a, io, &driver, s, &kernels, args[1], capacity, @max(max_rows, 1));
    const load_ns = t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
    std.debug.print("loaded {d} layers, {d:.2} GiB device weights in {d:.1} s\n", .{ hy4.layers.len, @as(f64, @floatFromInt(hy4.weight_bytes)) / (1 << 30), @as(f64, @floatFromInt(load_ns)) / 1e9 });
    const V: usize = hy4.text.vocab_size;
    const logits = try a.alloc(f32, V);
    // Prompt rows in `chunk`-row passes: a causal model's seam is exact, and the cap is 1,024 rows a pass.
    var pass_logits: ?[]f32 = null;
    if (score) {
        try out.writeFile(io, .{ .sub_path = "prompt.u32", .data = std.mem.sliceAsBytes(prompt) });
        pass_logits = try a.alloc(f32, @max(prompt.len, 1) * V);
    }
    const p0 = std.Io.Clock.awake.now(io);
    var row: usize = 0;
    while (row < prompt.len) : (row += chunk) {
        const n = @min(chunk, prompt.len - row);
        if (pass_logits) |all| {
            try hy4.passRows(prompt[row..][0..n], logits, all[row * V ..][0 .. n * V]);
        } else {
            try hy4.pass(prompt[row..][0..n], logits);
        }
    }
    const prefill_ns = p0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
    try out.writeFile(io, .{ .sub_path = "prefill-logits.f32", .data = std.mem.sliceAsBytes(logits) });
    if (pass_logits) |all| try writeScores(io, a, out, prompt, all[0 .. prompt.len * V], V, chunk, prefill_ns);
    const d0 = std.Io.Clock.awake.now(io);
    var copy: std.ArrayList(u32) = .empty;
    for (0..new_tokens) |step| {
        var best: usize = 0;
        for (logits, 0..) |v, i| if (v > logits[best]) {
            best = i;
        };
        try copy.append(a, @intCast(best));
        if (step + 1 == new_tokens) break;
        const one = [1]u32{@intCast(best)};
        try hy4.pass(&one, logits);
    }
    const decode_ns = d0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
    try out.writeFile(io, .{ .sub_path = "generated.u32", .data = std.mem.sliceAsBytes(copy.items) });
    std.debug.print("prompt {d} tokens: prefill {d:.1} ms ({d:.1} tok/s); decode {d} tokens {d:.1} ms/token\ngenerated={any}\n", .{
        prompt.len,
        @as(f64, @floatFromInt(prefill_ns)) / 1e6,
        @as(f64, @floatFromInt(prompt.len)) * 1e9 / @as(f64, @floatFromInt(prefill_ns)),
        new_tokens,
        @as(f64, @floatFromInt(decode_ns)) / 1e6 / @as(f64, @floatFromInt(@max(new_tokens, 2) - 1)),
        copy.items,
    });
}

/// `exl3_reference.py --score --save-logits`'s layout: logits.f32 for rows 0..n-2, the next token's log-prob.
fn writeScores(io: std.Io, a: std.mem.Allocator, out: std.Io.Dir, prompt: []const u32, logits: []const f32, voc: usize, chunk: usize, prefill_ns: i96) !void {
    const rows = prompt.len - 1; // row i's logits predict token i + 1: no row for the last token
    try out.writeFile(io, .{ .sub_path = "logits.f32", .data = std.mem.sliceAsBytes(logits[0 .. rows * voc]) });
    const scores = try a.alloc(f32, rows);
    var nll: f64 = 0;
    for (0..rows) |i| {
        const row = logits[i * voc ..][0..voc];
        var hi: f32 = row[0];
        for (row) |v| hi = @max(hi, v);
        var sum: f64 = 0;
        for (row) |v| sum += @exp(v - hi);
        const lse = hi + @as(f32, @floatCast(@log(sum)));
        const id = prompt[i + 1];
        if (id >= voc) return error.BadHy4PassRows;
        scores[i] = row[id] - lse;
        nll -= @as(f64, scores[i]);
    }
    try out.writeFile(io, .{ .sub_path = "scores.f32", .data = std.mem.sliceAsBytes(scores) });
    const report = try std.fmt.allocPrint(a, "{{\"prompt_tokens\": {d}, \"scored\": {d}, \"nll\": {d}, \"ppl\": {d}, \"chunk\": {d}, \"prefill_s\": {d}}}\n", .{
        prompt.len, rows, nll / @as(f64, @floatFromInt(rows)), @exp(nll / @as(f64, @floatFromInt(rows))), chunk, @as(f64, @floatFromInt(prefill_ns)) / 1e9,
    });
    try out.writeFile(io, .{ .sub_path = "report.json", .data = report });
    std.debug.print("scored {d} rows: nll {d:.4} ppl {d:.4}\n", .{ rows, nll / @as(f64, @floatFromInt(rows)), @exp(nll / @as(f64, @floatFromInt(rows))) });
}
