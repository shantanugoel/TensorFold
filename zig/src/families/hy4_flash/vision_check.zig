//! One image through the CUDA vision tower, its rows written as FP32: the boundary check against the ExLlamaV3 fork's
//! tower (plan 7.9's first gate) without the text model in the way.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const vision = @import("cuda_vision.zig");

const usage = "usage: tf-hy4-vision MODEL_DIR IMAGE OUT.f32 [TOKENS] [--stages DIR] [--patches PATH]\n" ++
    "  MODEL_DIR  a pack with vit.* and its config.json\n" ++
    "  IMAGE      PNG or JPEG bytes\n" ++
    "  OUT.f32    the rows, (gh / 2) * (gw / 2 + 1) of them, 3072 FP32 each, in raster order\n" ++
    "  TOKENS     the visual-token budget an image may take (default 4224, the checkpoint's own)\n" ++
    "  --stages DIR  dump each pipeline stage as FP32 rows under DIR (a mismatch's stage)\n" ++
    "  --patches PATH  write the preprocessed patches (rows x 768 FP32) to PATH\n";

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 4) {
        std.debug.print("{s}", .{usage});
        return error.BadArgs;
    }
    const dir = args[1];
    var budget: u32 = vision.token_limit;
    var stages: ?[]const u8 = null;
    var patches_out: ?[]const u8 = null;
    var i: usize = 4;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--stages")) {
            if (i + 1 == args.len) return error.BadArgs;
            i += 1;
            stages = args[i];
        } else if (std.mem.eql(u8, args[i], "--patches")) {
            if (i + 1 == args.len) return error.BadArgs;
            i += 1;
            patches_out = args[i];
        } else budget = try std.fmt.parseInt(u32, args[i], 10);
    }
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[2], a, .limited(64 << 20));
    const cfg = (try vision.vc.Config.read(a, io, dir)) orelse return error.NoHy4VisionConfig;
    const prepared = try vision.vc.prepare(io, a, bytes, cfg, budget);
    std.debug.print("image: {d} x {d} patches, {d} tokens ({d} bytes hashed)\n", .{ prepared.gh, prepared.gw, prepared.tokens, prepared.hash });

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    if (try ctx.capability() != 121) return error.NeedGb10;
    var s = try cuda.Stream.init(&driver, true);
    defer s.deinit();
    var kernels = try @import("serve_cuda.zig").Kernels.load(a, &driver);
    defer kernels.deinit();
    var ck = try core.Checkpoint.openModel(a, io, dir);
    defer ck.close();
    var own: cuda.exl3_upload.Uploader = .{ .d = &driver, .gpa = std.heap.page_allocator, .max_buffers = 1 << 16, .slab = 0 };
    defer own.deinit();
    const tower = try vision.Tower.load(a, &driver, s, &kernels.gemm, &kernels.vision_glue, &kernels.pixels, &kernels.norms, &ck, &own, cfg, budget);
    defer tower.deinit();
    if (patches_out) |path| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = std.mem.sliceAsBytes(prepared.pixels) });
    if (stages) |path| {
        try std.Io.Dir.cwd().createDirPath(io, path);
        tower.stages = .{ .io = io, .dir = try std.Io.Dir.cwd().openDir(io, path, .{}) };
    }
    const rows = prepared.tokens;
    var out = try cuda.DeviceBuffer.alloc(&driver, @as(usize, rows) * vision.H * 2);
    defer out.free();
    const t0 = std.Io.Clock.awake.now(io);
    try tower.encode(prepared.pixels, prepared.gh, prepared.gw, out.ptr);
    try s.synchronize();
    const ms = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
    // BF16 rows out as FP32, which every reference dump is compared in.
    const words = try a.alloc(u16, @as(usize, rows) * vision.H);
    const bits = try a.alloc(u8, @as(usize, rows) * vision.H * 2);
    try out.downloadAsync(0, bits, s.handle);
    try s.synchronize();
    for (words, 0..) |*w, k| w.* = std.mem.readInt(u16, bits[k * 2 ..][0..2], .little);
    const floats = try a.alloc(f32, words.len);
    for (floats, words) |*f, w| f.* = @bitCast(@as(u32, w) << 16);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[3], .data = std.mem.sliceAsBytes(floats) });
    std.debug.print("encoded {d} rows in {d:.1} ms ({d:.1} GiB of weights and scratch)\n", .{ rows, ms, @as(f64, @floatFromInt(tower.bytes)) / (1 << 30) });
}
