//! A pass's Engram rows: every table's windows planned at the pass's start, read in parallel, taken a layer at a time.
const std = @import("std");
const direct = @import("bf16_rows.zig");
const engram = @import("engram.zig");

/// One pass's Engram rows, in the layer order `start` was given the tables.
pub const Rows = struct {
    a: std.mem.Allocator,
    raw: []u16, // [tables][rows][input] BF16 rows, token/head order
    stride: usize, // u16 one table holds: rows * input
    plans: []direct.Plan,
    reads: []direct.Prefetch,
    planned: usize = 0, // plans built, in table order
    started: usize = 0, // reads started, in table order
    given: usize = 0, // tables `next` has handed back

    /// Plans every table's windows (they depend only on the token ids) and starts all their reads at once.
    pub fn start(a: std.mem.Allocator, io: std.Io, tables: []const *const direct.Table, windows: []const [engram.order]u32, input: usize, workers: usize) !Rows {
        const stride = try std.math.mul(usize, windows.len, input);
        const total = try std.math.mul(usize, stride, tables.len);
        const raw = try a.alloc(u16, total);
        const plans = a.alloc(direct.Plan, tables.len) catch |e| {
            a.free(raw);
            return e;
        };
        const reads = a.alloc(direct.Prefetch, tables.len) catch |e| {
            a.free(plans);
            a.free(raw);
            return e;
        };
        // From here a failure unwinds through `close`, which owns all three slices.
        var self: Rows = .{ .a = a, .raw = raw, .stride = stride, .plans = plans, .reads = reads };
        for (tables, 0..) |t, i| {
            self.plans[i] = direct.buildPlan(a, t.geometry, t.map, t.hash, windows) catch |e| {
                self.close(io);
                return e;
            };
            self.planned = i + 1;
            self.reads[i] = direct.Prefetch.start(a, io, t.geometry, &self.plans[i], raw[i * stride ..][0..stride], workers) catch |e| {
                self.close(io);
                return e;
            };
            self.started = i + 1;
        }
        return self;
    }

    /// The next table's rows, in the order `start` was given them: its read is waited for here.
    pub fn next(self: *Rows, io: std.Io) ![]u16 {
        if (self.given >= self.started) return error.BadEngramRows;
        const i = self.given;
        try self.reads[i].finish(io);
        self.given = i + 1;
        return self.raw[i * self.stride ..][0..self.stride];
    }

    /// Waits for any read still in flight, then frees the plans, their arenas and the rows.
    pub fn close(self: *Rows, io: std.Io) void {
        for (self.reads[0..self.started]) |*r| r.deinit(self.a, io);
        for (self.plans[0..self.planned]) |*p| p.deinit(self.a);
        self.a.free(self.reads);
        self.a.free(self.plans);
        self.a.free(self.raw);
        self.* = undefined;
    }
};
