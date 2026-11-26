//! A Hy4-flash pass's device scratch on CUDA, sized once at `max_rows`, and the family's fixed geometry.
const std = @import("std");
const cuda = @import("cuda");
const ihc = @import("ihc.zig");
const m = @import("mla_weights.zig");
const attn = @import("mla_attention.zig");
const engram_cuda = @import("engram_cuda.zig");

pub const H = 3072;
pub const S = ihc.streams;
pub const L = 1536;
pub const D = 13824;
pub const E = 512;
pub const K = 16;
pub const Q = 1024;
/// Keys a row of an indexed layer attends at most, and the embedding width of one KV slot.
pub const sparse_stride = 2048;
pub const sliding_stride = 512;
/// Streams the layer halves rotate through: the pass never reads and writes one buffer at once.
pub const rotation = 3;
/// Rows a pass takes at most: the widest row the per-pass scratch and the recall lists are sized for.
pub const row_cap = 1024;
/// Rows the hyper head, the final norm and the LM head run at once: one block of the EXL3 linear's own loop.
pub const head_block = 128;

/// A sliding layer's KV ring: `rows` slots for the positions written last, a position's slot its `pos % rows`.
pub const ring = struct {
    /// Keys a sliding row attends at most: the checkpoint's sliding window.
    pub const window = sliding_stride;
    /// Rows one pass writes at most.
    pub const pass_rows = row_cap;
    /// Slots a ring holds: a pass's own rows plus the window its first row still reads, at a 512-key chunk.
    pub const rows = 3 * 512;

    /// The slot position `pos` is written to and read from; a reader addresses keys by slot, so no row ever moves.
    pub fn slot(pos: usize) usize {
        return pos % rows;
    }

    /// The oldest position a row at `pos` attends, and the width of its key list.
    pub fn first(pos: usize) usize {
        return if (pos + 1 > window) pos + 1 - window else 0;
    }

    /// Refuse a pass whose own rows would overwrite a slot its earlier rows' windows still read.
    pub fn check(pass: usize) !void {
        if (pass == 0 or pass + window - 1 > rows) return error.BadHy4RingPass;
    }
};

test "the ring holds a pass and the window its first row reads" {
    try ring.check(ring.pass_rows);
    try std.testing.expectError(error.BadHy4RingPass, ring.check(ring.rows - ring.window + 2));
    try std.testing.expectError(error.BadHy4RingPass, ring.check(0));
    try std.testing.expectEqual(@as(usize, 0), ring.rows % sliding_stride); // a ring starts and ends on a chunk
    // A pass that starts just past a wrap: every slot it writes and its first row's window reads is its own.
    var taken: [ring.rows]bool = @splat(false);
    const start = 4 * ring.rows - 3;
    for (ring.first(start)..start + ring.pass_rows) |pos| {
        const slot = ring.slot(pos);
        try std.testing.expect(!taken[slot]);
        taken[slot] = true;
    }
    const used = std.mem.count(bool, &taken, &.{true});
    try std.testing.expectEqual(ring.pass_rows + ring.window - 1, @as(usize, used));
    // A row's key list is its window, oldest first, and a key's slot is the key's own position's.
    for ([_]usize{ 0, 1, ring.window - 1, ring.window, 7 * ring.rows + 100 }) |pos| {
        const first = ring.first(pos);
        try std.testing.expectEqual(@min(pos + 1, ring.window), pos + 1 - first);
        for (first..pos + 1) |key| try std.testing.expectEqual(key % ring.rows, ring.slot(key));
    }
}

pub const Scratch = struct {
    streams: [rotation]cuda.DeviceBuffer, // [rows, S, H]
    collapsed: u64,
    post: u64,
    hidden: u64,
    norm_tmp: u64,
    q_raw: u64,
    q_tmp: u64,
    q_resid: u64,
    q: u64,
    kv: u64,
    gate: u64,
    absorb: u64,
    latent_out: u64,
    values: u64,
    branch: u64,
    branch_norm: u64,
    attn_scratch: u64,
    attn_scratch_bytes: usize,
    positions: cuda.DeviceBuffer,
    table: u64,
    slots: cuda.DeviceBuffer,
    counts: cuda.DeviceBuffer,
    write_slots: cuda.DeviceBuffer,
    write_ring: cuda.DeviceBuffer,
    media: u64,
    latent: u64,
    logits: u64,
    picks: u64,
    probabilities: u64,
    mix: u64,
    route_out: u64,
    ff_gate: u64,
    ff_up: u64,
    ff_inner: u64,
    shared: u64,
    e_raw: cuda.DeviceBuffer,
    e_normalized: u64,
    e_keys: u64,
    e_value: u64,
    head_hidden: u64,
    head_norm: u64,
    lm_logits: u64,
};

/// Zeroed device bytes the upload owns; the caller keeps the address.
fn zeros(own: *cuda.exl3_upload.Uploader, bytes: usize) !u64 {
    const z = try std.heap.page_allocator.alloc(u8, bytes);
    defer std.heap.page_allocator.free(z);
    @memset(z, 0);
    return own.dev(z);
}

pub fn alloc(d: *const cuda.Driver, own: *cuda.exl3_upload.Uploader, max_rows: usize, vocab: usize) !Scratch {
    const r = max_rows;
    const shape = try attn.Shape.init(@intCast(r), 64, sparse_stride);
    var sc: Scratch = undefined;
    // The head reads a whole block of the pass's last rows, so the streams it reads run a block past the pass.
    for (&sc.streams) |*x| x.* = try cuda.DeviceBuffer.alloc(d, (r + head_block - 1) * S * H * 2);
    sc.collapsed = try zeros(own, r * H * 2);
    sc.post = try zeros(own, r * S * 4);
    sc.hidden = try zeros(own, r * H * 2);
    sc.norm_tmp = try zeros(own, @max(r, head_block) * D * 2);
    sc.q_raw = try zeros(own, r * Q * 2);
    sc.q_tmp = try zeros(own, r * Q * 2);
    sc.q_resid = try zeros(own, r * Q * 2);
    sc.q = try zeros(own, r * 64 * m.query_stride * 2);
    sc.kv = try zeros(own, r * m.absorbed_stride * 2);
    sc.gate = try zeros(own, r * 64 * m.value_width * 2);
    sc.absorb = try zeros(own, r * 64 * m.absorbed_stride * 2);
    sc.latent_out = try zeros(own, r * 64 * m.latent * 2);
    sc.values = try zeros(own, r * 64 * m.value_width * 2);
    sc.branch = try zeros(own, r * H * 2);
    sc.branch_norm = try zeros(own, r * H * 2);
    sc.attn_scratch_bytes = shape.scratchBytes();
    sc.attn_scratch = try zeros(own, sc.attn_scratch_bytes);
    sc.positions = try cuda.DeviceBuffer.alloc(d, r * 12);
    sc.table = try zeros(own, r * 64 * 2);
    sc.slots = try cuda.DeviceBuffer.alloc(d, r * sparse_stride * 4);
    sc.counts = try cuda.DeviceBuffer.alloc(d, r * 4);
    sc.write_slots = try cuda.DeviceBuffer.alloc(d, r * 4);
    sc.write_ring = try cuda.DeviceBuffer.alloc(d, r * 4);
    sc.media = try zeros(own, r * 4);
    sc.latent = try zeros(own, r * L * 2);
    sc.logits = try zeros(own, r * E * 4);
    sc.picks = try zeros(own, r * K * 4);
    sc.probabilities = try zeros(own, r * K * 4);
    sc.mix = try zeros(own, r * L * 2);
    sc.route_out = try zeros(own, r * H * 2);
    sc.ff_gate = try zeros(own, r * D * 2);
    sc.ff_up = try zeros(own, r * D * 2);
    sc.ff_inner = try zeros(own, r * D * 2);
    sc.shared = try zeros(own, r * H * 2);
    sc.e_raw = try cuda.DeviceBuffer.alloc(d, r * engram_cuda.input * 2);
    sc.e_normalized = try zeros(own, r * engram_cuda.input * 2);
    sc.e_keys = try zeros(own, S * r * H * 2);
    sc.e_value = try zeros(own, r * H * 2);
    sc.head_hidden = try zeros(own, @max(r, head_block) * H * 2);
    sc.head_norm = try zeros(own, @max(r, head_block) * H * 2);
    sc.lm_logits = try zeros(own, head_block * vocab * 4);
    return sc;
}
