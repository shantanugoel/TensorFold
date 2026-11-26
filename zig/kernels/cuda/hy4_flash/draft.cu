// DFly2's two row-sized ops: the MTP fusion on the block's anchor row, and a draft layer's residual add.
#include <cuda_bf16.h>
#include <stdint.h>

// Slot 0 of the query block: the token's embedding plus the fusion gate's share of the projection.
extern "C" __global__ void __launch_bounds__(256) tf_hy4_draft_fuse(
    const __nv_bfloat16 *token, const __nv_bfloat16 *fused, const __nv_bfloat16 *gate,
    __nv_bfloat16 *out, int width) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= width) return;
    const float share = __fmul_rn(__bfloat162float(gate[column]), __bfloat162float(fused[column]));
    out[column] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(token[column]), share));
}

// A draft layer's residual: out = bf16(branch + residual), rows at `count` values each.
extern "C" __global__ void __launch_bounds__(256) tf_hy4_draft_add(
    const __nv_bfloat16 *branch, const __nv_bfloat16 *residual, __nv_bfloat16 *out, int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += int64_t(gridDim.x) * blockDim.x)
        out[i] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(branch[i]), __bfloat162float(residual[i])));
}
