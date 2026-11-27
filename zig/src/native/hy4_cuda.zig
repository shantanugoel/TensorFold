//! Hy4-flash on the native CUDA server: our v3a pack's text path (dense EXL3 groups, the indexer, the ring).
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const api = @import("engine_api");
const family = @import("hy4_family");
const settings = @import("cuda_device.zig");

const Allocator = std.mem.Allocator;

/// The registry entry: `config.json`'s model_type for this pack, and the format its weights are stored in.
pub const model_type = "hy_v4_flash";
pub const formats: []const []const u8 = &.{"exl3"};
/// The context when neither --context nor the pack names one: the indexer's recall scores cap a cache at 262,144.
pub const default_context: u32 = family.max_capacity;
/// The smallest context a server is useful at, and the largest its recall scores can rank.
pub const least_context: u32 = 4096;
/// Free bytes kept for the system and every other process after the last load-time allocation.
pub const default_floor_gib: u64 = 8;

/// The context this thread's CUDA calls go to: open's thread at startup, the lane host's thread at rounds.
threadlocal var bound: ?*const cuda.Context = null;

fn bind(o: *Owned) void {
    if (bound != &o.ctx) {
        o.ctx.makeCurrent() catch |e| std.log.err("cuCtxSetCurrent on this thread: {s}", .{@errorName(e)});
        bound = &o.ctx;
    }
}

fn bindThread(ptr: *anyopaque) anyerror!void {
    bind(@as(*Owned, @ptrCast(@alignCast(ptr))));
}

/// Everything the engine thread reads: the device, the model, the lane core and the host that serves it.
const Owned = struct {
    gpa: Allocator,
    driver: cuda.Driver,
    ctx: cuda.Context,
    s: cuda.Stream,
    k: family.Kernels,
    c: *family.Cuda,
    cfg: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    /// The host's engine with this family's request refusals in front of it.
    vtable: api.Engine.VTable,
    startup: []u8,
};

fn halt(ptr: *anyopaque, reason: []const u8) void {
    @as(*Owned, @ptrCast(@alignCast(ptr))).host.halt(reason);
}

/// The host joined first; then the model's device memory, the kernel images and the context that made them.
fn release(ptr: *anyopaque) void {
    const o: *Owned = @ptrCast(@alignCast(ptr));
    o.host.stop();
    bind(o);
    o.core.deinit();
    o.cfg.deinit(o.gpa);
    o.c.deinit();
    o.k.deinit();
    o.s.deinit();
    o.ctx.deinit();
    o.driver.close();
    o.gpa.free(o.startup);
    o.gpa.destroy(o);
}

/// False once the device context has a sticky fault.
fn healthy(ptr: *anyopaque) bool {
    return @as(*Owned, @ptrCast(@alignCast(ptr))).ctx.healthy();
}

fn memory(_: ?*anyopaque, reset_peak: bool) ?api.Memory {
    const u = cuda.usage(reset_peak);
    return .{ .active = u.device, .cache = 0, .peak = u.peak, .pinned = u.host };
}

/// Words for a request this engine cannot serve yet: tools are not served. Images are not refused here: without
/// `--vision` the server refuses an image part by name before any engine sees it, and with `--vision` the tower is
/// loaded and the plan reads them (`serve_cuda.setPlan`).
fn refusalMessage(_: *anyopaque, request: *const api.Request, out: []u8) ?[]const u8 {
    if (request.tools_json.len > 0) return put(out, "the Hy4-flash CUDA engine does not serve tools yet: send the request without tools");
    return null;
}

/// One image's LM tokens: `--vision-image-tokens` when given, else the family's own budget for `--vision` alone.
fn imageTokens(v: api.VisionOpen) u32 {
    return if (v.image_tokens_set) v.image_tokens else family.Cuda.default_image_tokens;
}

fn put(out: []u8, message: []const u8) []const u8 {
    const n = @min(message.len, out.len);
    @memcpy(out[0..n], message[0..n]);
    return out[0..n];
}

/// A refusal: the client's words in `problem`, and an error, so every errdefer of the open runs.
fn refuse(a: Allocator, problem: *[]const u8, comptime fmt: []const u8, args: anytype) error{Refused} {
    problem.* = std.fmt.allocPrint(a, fmt, args) catch fmt;
    return error.Refused;
}

/// The flags this engine's path cannot serve yet, each with the step that brings it; null when they are all absent.
fn refusal(a: Allocator, o: api.Open) !?[]const u8 {
    if (o.vision) |v| {
        // One image's rows are the tower's, four patches a token, and its scratch is sized for them at load.
        if (v.image_tokens > family.vision_token_limit) return try std.fmt.allocPrint(a, "--vision-image-tokens {d}: one image may take {d} tokens at most (four patches a token is the tower's scratch)", .{ v.image_tokens, family.vision_token_limit });
        // Video is not refused here: this tower's offer has no video half, so the server's own expansion refuses a
        // `video_url` part by name ("this server takes no video input") and /v1/models lists none.
    }
    if (o.lanes < 1 or o.lanes > family.max_slots) return try std.fmt.allocPrint(a, "--parallel {d}: the native Hy4-flash engine serves 1 to {d} sequences", .{ o.lanes, family.max_slots });
    if (o.tp != 1 or o.rank != 0) return try std.fmt.allocPrint(a, "--tp {d} --rank {d}: the native Hy4-flash engine is one device (1 and 0)", .{ o.tp, o.rank });
    if (o.drafter != null or o.mtp_confidence != null) return try std.fmt.allocPrint(a, "--drafter and --mtp-confidence: DFly2 is the pack's own component, so --mtp-drafts {d} names its drafts and no separate checkpoint or confidence head is served", .{family.drafts});
    if ((o.mtp_drafts orelse 0) > family.drafts) return try std.fmt.allocPrint(a, "--mtp-drafts {d}: DFly2 proposes {d} drafts a round at most", .{ o.mtp_drafts.?, family.drafts });
    if (o.slide) return try std.fmt.allocPrint(a, "--slide: Hy4-flash does not learn on CUDA; serve without --slide", .{});
    if (o.learn != null or o.learn_conversations) return try std.fmt.allocPrint(a, "--learn and --learn-conversations: the native Hy4-flash engine keeps no learned states; serve without them", .{});
    if (o.recall or o.record != null) return try std.fmt.allocPrint(a, "--recall and --record: the native Hy4-flash engine has neither to offer yet", .{});
    if (o.speed_up != null or o.policy.len != 0) return try std.fmt.allocPrint(a, "--speed-up and --policy: not served by the native Hy4-flash engine", .{});
    if (o.prompt_int8) return try std.fmt.allocPrint(a, "--prompt-precision int8: the native Hy4-flash engine's activations are BF16", .{});
    if (o.precision) |p| if (p != .full) return try std.fmt.allocPrint(a, "--precision: the native Hy4-flash engine runs the pack's own EXL3 groups and BF16 elsewhere", .{});
    if (o.segments) |n| if (n != 1) return try std.fmt.allocPrint(a, "--segments {d}: the native Hy4-flash engine fills one prompt chunk at a time", .{n});
    if (o.spill != .off) return try std.fmt.allocPrint(a, "--spill-gib: the native Hy4-flash engine keeps no prompt states to spill", .{});
    if (o.prompt_cache_gib) |g| if (!std.math.isFinite(g) or g < 0) return try std.fmt.allocPrint(a, "--prompt-cache-gib: a finite nonnegative budget", .{});
    if (o.keep) |n| if (n != 0) return try std.fmt.allocPrint(a, "--checkpoint-slots: the native Hy4-flash engine keeps no prefill checkpoints yet", .{});
    return null;
}

/// The drafts a load takes. Drafting pays on one busy lane and costs a page when the load is shared, so a multi-lane
/// server does not load the tower unless a shared round may draft there: `TF_HY4_DRAFT_LANES` is `plain_streams` — a
/// round that many streams wide drafts none — so above two it leaves a two-stream round drafting, and at 1 it loads the
/// tower for a measurement that never drafts. (2, the default, is the served plain path at `--parallel 2` or more.)
fn loadDrafts(sequences: usize, draft_lanes: u32, asked: u32) u32 {
    const shared_draft = draft_lanes > 2 or draft_lanes <= 1;
    return if (sequences > 1 and !shared_draft) 0 else asked;
}

/// The context to serve: --context (0: the largest this engine ranks), rounded to whole 512-slot pages.
fn contextOf(a: Allocator, o: api.Open, problem: *[]const u8) !?u32 {
    const asked: i64 = o.context orelse default_context;
    const most: i64 = family.max_capacity;
    if (asked > most) {
        problem.* = try std.fmt.allocPrint(a, "--context {d}: this engine's indexer ranks at most {d} slots, so its context stops there", .{ asked, most });
        return null;
    }
    const native: i64 = if (asked <= 0) most else asked;
    if (native < least_context) {
        problem.* = try std.fmt.allocPrint(a, "--context {d}: the native Hy4-flash engine serves {d} tokens at least", .{ asked, least_context });
        return null;
    }
    return (@as(u32, @intCast(native)) + 511) / 512 * 512;
}

/// Opens `o.dir` on CUDA; null with the reason in `problem` when it is refused.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (try refusal(a, o)) |why| {
        problem.* = why;
        return null;
    }
    if (!cuda.kernels.available) {
        problem.* = "Hy4-flash on CUDA needs the embedded native kernel images: build with -Dnvcc=<nvcc> or -Dfatbins";
        return null;
    }
    const device = try settings.setting(a, o.device, "TF_CUDA_DEVICE", 0, problem) orelse return null;
    const capacity = try contextOf(a, o, problem) orelse return null;
    const floor_gib = try @import("sys").environment.read(u64, "TF_HY4_SERVE_FLOOR_GIB", default_floor_gib);
    const h = try gpa.create(Owned);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.driver = cuda.Driver.open() catch |e| return refuse(a, problem, "no CUDA driver ({s})", .{@errorName(e)});
    errdefer h.driver.close();
    const count = h.driver.deviceCount() catch |e| return refuse(a, problem, "cannot read the CUDA device count ({s})", .{@errorName(e)});
    if (device >= count) return refuse(a, problem, "--device {d}: this machine has {d} CUDA device{s} (0 to {d})", .{ device, count, if (count == 1) "" else "s", @max(count, 1) - 1 });
    h.ctx = cuda.Context.init(&h.driver, @intCast(device)) catch |e| return refuse(a, problem, "cannot open CUDA device {d} ({s})", .{ device, @errorName(e) });
    errdefer h.ctx.deinit();
    bound = &h.ctx;
    h.s = cuda.Stream.init(&h.driver, true) catch |e| return refuse(a, problem, "cannot make a CUDA stream ({s})", .{@errorName(e)});
    errdefer h.s.deinit();
    h.k = family.Kernels.load(gpa, &h.driver) catch |e| return refuse(a, problem, "the native Hy4-flash kernel images cannot load ({s})", .{@errorName(e)});
    errdefer h.k.deinit();
    const asked = o.mtp_drafts orelse 0;
    const draft_lanes = @import("sys").environment.read(u32, "TF_HY4_DRAFT_LANES", 2) catch 2;
    const drafts = loadDrafts(o.lanes, draft_lanes, asked);
    if (asked > 0 and drafts == 0) std.log.info("hy4-cuda: --mtp-drafts {d} not loaded: a --parallel {d} server serves the plain path (a resident tower costs a shared pass about 10 %, step 6c; use --parallel 1, or TF_HY4_DRAFT_LANES=3 to draft in a two-lane round)", .{ asked, o.lanes });
    const vision_tokens: u32 = if (o.vision) |v| imageTokens(v) else 0;
    h.c = family.Cuda.init(gpa, io, &h.driver, h.s, &h.k, o.dir, capacity, drafts, o.lanes, vision_tokens, o.kv8) catch |e| return refuse(a, problem, "the native Hy4-flash engine cannot load {s} ({s})", .{ o.dir, @errorName(e) });
    errdefer h.c.deinit();
    // The last load-time allocations are in, and a GB10's memory is the system's: read the floor here, not per layer.
    const bytes = h.c.bytes();
    const avail = family.memAvailable();
    if (avail < floor_gib << 30) return refuse(a, problem, "{s}", .{try floorRefusal(a, capacity, family.bytesPerToken(h.c.m), avail, floor_gib)});
    // The depth rule prices a round from the widths' own times: without them it drafts one a round whatever
    // the block would accept (a prior). `timeWindows` leaves the caches empty, as a cold load has them.
    var facts = family.Cuda.facts(h.c);
    var windows: [family.max_window]lanes.config.Cost = undefined;
    var timed: ?family.Cuda.Timed = null;
    if (h.c.m.draft != null) timed = h.c.timeWindows() catch |e| ret: {
        std.log.warn("hy4-cuda: the window timing failed ({s}); the depth rule keeps its prior", .{@errorName(e)});
        for (0..o.lanes) |i| h.c.m.cut(i, 0);
        for (h.c.seq[0..o.lanes]) |*q| q.valid_from = 0;
        break :ret null;
    };
    if (timed) |t| {
        // A drafted round's pass carries the block's fixed charge, so the widths past one row are priced with it and
        // the slope is what each draft adds. The one-row price stays the plain step's: it drafts nothing.
        for (&windows, 0..) |*c, i| c.* = .{ .width = @intCast(i + 1), .ms = t.rows[i] + if (i > 0) t.fixed else 0 };
        facts.window_costs = &windows;
        facts.mtp_step_ms = t.per;
    }
    h.cfg = try lanes.Config.init(gpa, facts, family.max_window, drafts);
    errdefer h.cfg.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.cfg, h.c.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.startup = try std.fmt.allocPrint(gpa, "CUDA device {d}: Hy4-flash {d}-token context x {d} sequences ({d:.2} GiB KV + {d:.2} GiB indexer keys + {d:.0} MiB ring + {d:.0} MiB recall scores), {d:.2} GiB weights, {d:.1} GiB free after the load, DFly2 {s}, images {s}", .{
        device,                       family.Cuda.served(h.c),     o.lanes,
        gib(bytes.kv),                gib(bytes.index),
        @as(f64, @floatFromInt(bytes.ring)) / (1 << 20), @as(f64, @floatFromInt(bytes.scores)) / (1 << 20),
        gib(h.c.m.weight_bytes),      gib(avail),
        if (h.c.m.draft != null) "on" else "off",
        if (o.vision) |v| try std.fmt.allocPrint(a, "on ({d} tokens an image)", .{imageTokens(v)}) else "off",
    });
    errdefer gpa.free(h.startup);
    h.host = api.LaneHost.init(gpa, io, &h.core, .{
        .name = "hy4-cuda",
        .lanes = o.lanes,
        .context_window = family.Cuda.served(h.c),
        .prefill_step = family.prompt_rows,
        .structures = true,
        .logprobs = true,
        .reasoning = true,
        .policy = "Hy4-flash CUDA: dense EXL3 groups, BF16 elsewhere, the sliding layers' 1,536-slot ring, one sequence a lane",
        .startup = h.startup,
    });
    // --vision: the tower's processor behind the server's image expansion (it writes the prompt's placeholders).
    if (o.vision != null) h.host.info_.vision = @import("cuda_vision_offer.zig").offer(@as(?*family.Vision, &h.c.vision), o.vision);
    h.host.on_thread = .{ .ctx = h, .run = bindThread };
    h.host.health = .{ .ctx = h, .read = healthy };
    h.host.memory = .{ .ctx = null, .read = memory };
    try h.host.start();
    // The host's own engine, with this family's request refusals in front of it (chat.zig reads the message).
    const inner = h.host.engine();
    h.vtable = inner.vtable.*;
    h.vtable.refusal = refusalMessage;
    return .{ .engine = .{ .ctx = inner.ctx, .vtable = &h.vtable }, .close = release, .ctx = h, .halt = halt };
}

fn gib(bytes: anytype) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

/// What a load over the post-load floor says: the context that would fit, in pages and never below the smallest served.
fn floorRefusal(a: Allocator, capacity: u32, per_token: u64, avail: u64, floor_gib: u64) ![]const u8 {
    const over = (floor_gib << 30) - avail;
    const fits = if (per_token == 0) capacity else @min(over / per_token, capacity);
    const under = "a {d}-token context leaves {d:.2} GiB of MemAvailable, {d:.2} GiB under the {d} GiB floor: ";
    if (fits >= capacity) return std.fmt.allocPrint(a, under ++ "even {d} tokens would not fit now", .{ capacity, gib(avail), gib(over), floor_gib, least_context });
    const slack = @max(capacity - @as(u32, @intCast(fits)), least_context);
    return std.fmt.allocPrint(a, under ++ "serve --context {d} or less", .{ capacity, gib(avail), gib(over), floor_gib, slack / 512 * 512 });
}

test "Hy4-flash CUDA host: a multi-lane server loads the tower only when a shared round may draft" {
    // A lone stream keeps its round whole, so a one-lane server always loads it.
    try std.testing.expectEqual(@as(u32, 5), loadDrafts(1, 2, 5));
    // The served default: at --parallel 2 or more the switch is at 2, a shared round that wide drafts none, so the
    // tower stays out of the load (a resident one costs a shared pass about 10 %).
    try std.testing.expectEqual(@as(u32, 0), loadDrafts(2, 2, 5));
    try std.testing.expectEqual(@as(u32, 0), loadDrafts(4, 2, 5));
    // Above two the switch leaves a two-stream round drafting, so the tower is loaded for it.
    try std.testing.expectEqual(@as(u32, 5), loadDrafts(2, 3, 5));
    try std.testing.expectEqual(@as(u32, 5), loadDrafts(4, 4, 5));
    // At 1 it is resident and drafts never: the measurement that prices a resident tower's cost to a shared pass.
    try std.testing.expectEqual(@as(u32, 5), loadDrafts(4, 1, 5));
    // A server asked for no drafts loads none whatever the switch says.
    try std.testing.expectEqual(@as(u32, 0), loadDrafts(1, 3, 0));
}

test "Hy4-flash CUDA host: the context this engine serves: whole pages, the recall scores' width, and clear words past it" {
    const a = std.testing.allocator;
    var problem: []const u8 = "";
    defer if (problem.len > 0) a.free(problem);
    try std.testing.expectEqual(@as(?u32, 262_144), try contextOf(a, .{ .dir = "d", .model_type = model_type, .context = 0 }, &problem));
    try std.testing.expectEqual(@as(?u32, 131_072), try contextOf(a, .{ .dir = "d", .model_type = model_type, .context = 131000 }, &problem));
    try std.testing.expectEqual(@as(?u32, null), try contextOf(a, .{ .dir = "d", .model_type = model_type, .context = 524_288 }, &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "262144") != null);
}

test "Hy4-flash CUDA host: a bare serve is served, and every step this path lacks is refused with its step named" {
    const a = std.testing.allocator;
    // one sequence, and several: a lane a sequence, from one to the per-slot arrays' width
    const plain: api.Open = .{ .dir = "d", .model_type = model_type, .lanes = 1 };
    try std.testing.expect(try refusal(a, plain) == null);
    try std.testing.expect(try refusal(a, .{ .dir = "d", .model_type = model_type, .lanes = family.max_slots }) == null);
    // --vision itself is served (`--vision` alone takes the checkpoint's own budget); past the tower's scratch is not.
    try std.testing.expect(try refusal(a, .{ .dir = "d", .model_type = model_type, .lanes = 1, .vision = .{} }) == null);
    // `--kv-cache fp8` is served now: the 12 indexed layers' rows are E4M3, the ring and the indexer stay BF16.
    try std.testing.expect(try refusal(a, .{ .dir = "d", .model_type = model_type, .lanes = 1, .kv8 = true }) == null);
    // one flag a request cannot bring yet, and the words name the flag: too many image tokens, too many lanes
    for ([_]struct { api.Open, []const u8 }{
        .{ .{ .dir = "d", .model_type = model_type, .lanes = 1, .vision = .{ .image_tokens = family.vision_token_limit + 1, .image_tokens_set = true } }, "--vision-image-tokens" },
        .{ .{ .dir = "d", .model_type = model_type, .lanes = family.max_slots + 1 }, "--parallel 9" },
        .{ .{ .dir = "d", .model_type = model_type, .lanes = 1, .slide = true }, "--slide" },
    }) |case| {
        const why = (try refusal(a, case[0])).?;
        defer a.free(why);
        try std.testing.expect(std.mem.indexOf(u8, why, case[1]) != null);
    }
}

test "Hy4-flash CUDA host: --vision alone takes the family's own budget, and the flag overrides it" {
    try std.testing.expectEqual(family.Cuda.default_image_tokens, imageTokens(.{}));
    try std.testing.expectEqual(@as(u32, 1024), imageTokens(.{ .image_tokens = 1024, .image_tokens_set = true }));
}

test "Hy4-flash CUDA host: a request this engine cannot serve says which part it cannot take" {
    const a = std.testing.allocator;
    var out: [512]u8 = undefined;
    const plain: api.Request = .{ .prompt = &.{}, .max_tokens = 1 };
    try std.testing.expect(refusalMessage(undefined, &plain, &out) == null);
    // images are the server's own refusal without --vision (media.images), so a request with them is the engine's
    const one: [1]lanes.stream.Image = .{.{ .pixels = &.{}, .gh = 1, .gw = 1, .at = 0, .tokens = 1 }};
    const image: api.Request = .{ .prompt = &.{}, .max_tokens = 1, .images = &one };
    try std.testing.expect(refusalMessage(undefined, &image, &out) == null);
    const tools: api.Request = .{ .prompt = &.{}, .max_tokens = 1, .tools_json = "[]" };
    const problem = refusalMessage(undefined, &tools, &out).?;
    try std.testing.expect(std.mem.startsWith(u8, problem, "the Hy4-flash CUDA engine does not serve tools"));
    _ = a;
}

test "Hy4-flash CUDA host: the post-load floor names the context that fits, in whole pages and never below the least" {
    const a = std.testing.allocator;
    const per_token: u64 = 18_688; // v3a's growing caches at 262,144 slots, from the family's own accounting
    const floor: u64 = 8; // GiB, as TF_HY4_SERVE_FLOOR_GIB reads it
    // 0.5 GiB under the floor: about 28,727 tokens' worth, so 233,417 slots fit, rounded down to whole pages.
    const short = try floorRefusal(a, 262_144, per_token, (7 << 30) + (1 << 29), floor);
    defer a.free(short);
    try std.testing.expect(std.mem.indexOf(u8, short, "serve --context 232960 or less") != null);
    // A context far too large to rescue by shrinking it: 5 GiB short is more than the whole context's caches.
    const hopeless = try floorRefusal(a, 262_144, per_token, 3 << 30, floor);
    defer a.free(hopeless);
    try std.testing.expect(std.mem.indexOf(u8, hopeless, "even 4096 tokens would not fit now") != null);
    // The clamp: a deficit that would leave less than this engine serves suggests the least it serves, not less.
    const tight = try floorRefusal(a, 262_144, per_token, (floor << 30) - 261_000 * per_token, floor);
    defer a.free(tight);
    try std.testing.expect(std.mem.indexOf(u8, tight, "serve --context 4096 or less") != null);
}
