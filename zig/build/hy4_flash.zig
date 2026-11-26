//! Host checks for native Hy4-flash tensor layouts, retained keys and draft inputs.
const std = @import("std");

pub fn tests(b: *std.Build, test_step: *std.Build.Step) void {
    const core = b.createModule(.{
        .root_source_file = b.path("zig/src/core/root.zig"),
        .target = b.graph.host,
        .link_libc = true,
    });
    const cuda = @import("cuda_runtime.zig").runtime(b, b.graph.host, .ReleaseSafe, &.{});
    const hash = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/engram_hash.zig") });
    // The server adapter's own state names a `lanes.Stream` (a lane a sequence), so its tests need the module.
    const lanes = b.createModule(.{
        .root_source_file = b.path("zig/src/core/lanes/lanes.zig"),
        .target = b.graph.host,
        .link_libc = true,
    });
    lanes.addImport("core", core);
    const module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/hy4_flash/host_tests.zig"),
        .target = b.graph.host,
        .link_libc = true,
        .imports = &.{
            .{ .name = "core", .module = core },
            .{ .name = "cuda", .module = cuda },
            .{ .name = "engram_hash_core", .module = hash },
            .{ .name = "lanes", .module = lanes },
        },
    });
    module.addImport("config_number", b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/config_number.zig") }));
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module }));
    test_step.dependOn(&run.step);
    b.step("test-hy4-flash-host", "Hy4-flash checkpoint layout, slot selection, Engram gathers and DFly2 inputs").dependOn(&run.step);
}
