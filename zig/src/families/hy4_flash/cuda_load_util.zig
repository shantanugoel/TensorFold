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

/// `model.safetensors.index.json`, read and parsed once a load: each Engram layer asks it for its table's shard.
pub const Shards = struct {
    parsed: std.json.Parsed(std.json.Value),
    dir: []const u8,

    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Shards {
        const path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
        defer a.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 26));
        defer a.free(bytes);
        const parsed = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
        errdefer parsed.deinit();
        if (parsed.value != .object) return error.MissingHy4EngramTable;
        if (parsed.value.object.get("weight_map")) |map| if (map != .object) return error.MissingHy4EngramTable;
        if (parsed.value.object.get("weight_map") == null) return error.MissingHy4EngramTable;
        return .{ .parsed = parsed, .dir = dir };
    }

    pub fn deinit(self: *Shards) void {
        self.parsed.deinit();
    }

    /// The safetensors file holding `name`, as `<dir>/<file>` in `buf`; null when the map does not name it.
    pub fn shardOf(self: *const Shards, buf: []u8, name: []const u8) !?[:0]const u8 {
        const map = self.parsed.value.object.get("weight_map").?.object;
        const file = map.get(name) orelse return null;
        if (file != .string) return error.MissingHy4EngramTable;
        return try std.fmt.bufPrintSentinel(buf, "{s}/{s}", .{ self.dir, file.string }, 0);
    }
};
