//! Hy4-flash CUDA tools: the greedy text generator for EXL3 routed-expert checkpoints, and the server's family module.
const std = @import("std");

/// The family module `native/hy4_cuda.zig` opens: the step 1-3 text model and its lane backend.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/hy4_flash/serve_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("core", core);
    m.addImport("lanes", lanes);
    m.addImport("engram_hash_core", b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/engram_hash.zig"), .target = target, .optimize = optimize }));
    m.addImport("config_number", b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/config_number.zig") }));
    return m;
}

pub fn install(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module) void {
    const root = b.createModule(.{
        .root_source_file = b.path("zig/src/families/hy4_flash/cuda_generate.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } },
    });
    root.addImport("engram_hash_core", b.createModule(.{
        .root_source_file = b.path("zig/src/families/deepseek_v41/engram_hash.zig"),
        .target = target,
        .optimize = optimize,
    }));
    root.addImport("config_number", b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/config_number.zig") }));
    const exe = b.addExecutable(.{ .name = "tf-hy4-cuda-generate", .root_module = root });
    b.step("hy4-cuda-generate", "Build the Hy4-flash CUDA greedy generator for EXL3 checkpoints").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// `tf-hy4-cuda-draft`: the DFly2 verify window on a device (one row against the same row inside the block).
pub fn installDraft(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) void {
    const root = b.createModule(.{
        .root_source_file = b.path("zig/src/families/hy4_flash/draft_check.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    root.addImport("lanes", lanes);
    root.addImport("cuda", cuda);
    root.addImport("core", core);
    root.addImport("engram_hash_core", b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/engram_hash.zig"), .target = target, .optimize = optimize }));
    root.addImport("config_number", b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/config_number.zig") }));
    const exe = b.addExecutable(.{ .name = "tf-hy4-cuda-draft", .root_module = root });
    b.step("hy4-cuda-draft", "Build the Hy4-flash DFly2 verify-window check that needs a device").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// `tf-hy4-cuda-reuse`: the server's prefix reuse against a fresh fill on a device (the caches' two fills must match).
pub fn installReuse(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) void {
    const root = b.createModule(.{
        .root_source_file = b.path("zig/src/families/hy4_flash/reuse_check.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    root.addImport("lanes", lanes);
    root.addImport("cuda", cuda);
    root.addImport("core", core);
    root.addImport("engram_hash_core", b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/engram_hash.zig"), .target = target, .optimize = optimize }));
    root.addImport("config_number", b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/config_number.zig") }));
    const exe = b.addExecutable(.{ .name = "tf-hy4-cuda-reuse", .root_module = root });
    b.step("hy4-cuda-reuse", "Build the Hy4-flash prefix-reuse check that needs a device").dependOn(&b.addInstallArtifact(exe, .{}).step);
}
