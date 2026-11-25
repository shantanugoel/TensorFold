//! Loading helpers for the CUDA text model: the UMA floor's MemAvailable, GiB for its prints, and a table's shard.
const std = @import("std");

pub fn gib(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

/// MemAvailable from /proc/meminfo in bytes (0 when unreadable, which trips the floor).
pub fn memAvailable() usize {
    var buf: [4096]u8 = undefined;
    const fd = std.c.open("/proc/meminfo", .{});
    if (fd < 0) return 0;
    defer _ = std.c.close(fd);
    const got = std.c.read(fd, &buf, buf.len);
    if (got <= 0) return 0;
    const n: usize = @intCast(got);
    var it = std.mem.tokenizeScalar(u8, buf[0..n], '\n');
    while (it.next()) |line| if (std.mem.startsWith(u8, line, "MemAvailable:")) {
        var f = std.mem.tokenizeScalar(u8, line["MemAvailable:".len..], ' ');
        const kb = std.fmt.parseInt(usize, f.next() orelse return 0, 10) catch return 0;
        return kb * 1024;
    };
    return 0;
}

/// The safetensors file holding `name`, through the model index (an Engram table sits alone in its shard).
pub fn shardOf(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![:0]const u8 {
    const index = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" }), a, .limited(1 << 26));
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, index, .{});
    const file = parsed.object.get("weight_map").?.object.get(name) orelse return error.MissingHy4EngramTable;
    return std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, file.string }, 0);
}
