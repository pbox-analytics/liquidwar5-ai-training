"""
Custom Triton kernels for GPU-accelerated Liquid War operations.

The gradient spread is the bottleneck (~48% of tick time). This kernel
fuses multiple iterations of the min-neighbor spread into a single
GPU launch, using shared memory tiling to minimize global memory access.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gradient_spread_kernel(
    grad_ptr,       # (B*T, H, W) gradient tensor
    wall_ptr,       # (B, H, W) wall mask (1=wall)
    B_T: tl.constexpr,  # B * T
    H: tl.constexpr,
    W: tl.constexpr,
    INF: tl.constexpr,
    ITERATIONS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Spread gradient: each cell adopts min(neighbors) + 1.

    Each program handles one (batch*team, block_y, block_x) tile.
    Runs ITERATIONS of spread within the kernel to avoid launch overhead.
    """
    # Program IDs
    pid_bt = tl.program_id(0)  # batch * team index
    pid_by = tl.program_id(1)  # block row
    pid_bx = tl.program_id(2)  # block col

    # This program's tile boundaries
    y_start = pid_by * BLOCK_H
    x_start = pid_bx * BLOCK_W

    # Batch index for wall lookup (walls shared across teams)
    b_idx = pid_bt // (B_T // tl.cdiv(B_T, 1))  # approximate

    # Load tile into registers
    y_offsets = y_start + tl.arange(0, BLOCK_H)
    x_offsets = x_start + tl.arange(0, BLOCK_W)

    # Masks for valid positions
    y_mask = y_offsets < H
    x_mask = x_offsets < W
    tile_mask = y_mask[:, None] & x_mask[None, :]

    # Base offset for this batch*team slice
    base = pid_bt * H * W

    for _iter in range(ITERATIONS):
        # Load current gradient values for this tile
        offsets = base + y_offsets[:, None] * W + x_offsets[None, :]
        grad = tl.load(grad_ptr + offsets, mask=tile_mask, other=INF)

        # For each of 8 neighbors, load and compute min(neighbor + 1)
        # We need to handle boundaries by loading INF for out-of-bounds

        # Top neighbor (y-1)
        y_top = y_offsets - 1
        top_mask = (y_top >= 0)[:, None] & x_mask[None, :]
        top_off = base + y_top[:, None] * W + x_offsets[None, :]
        top = tl.load(grad_ptr + top_off, mask=top_mask & tile_mask, other=INF)
        grad = tl.minimum(grad, top + 1)

        # Bottom neighbor (y+1)
        y_bot = y_offsets + 1
        bot_mask = (y_bot < H)[:, None] & x_mask[None, :]
        bot_off = base + y_bot[:, None] * W + x_offsets[None, :]
        bot = tl.load(grad_ptr + bot_off, mask=bot_mask & tile_mask, other=INF)
        grad = tl.minimum(grad, bot + 1)

        # Left neighbor (x-1)
        x_left = x_offsets - 1
        left_mask = y_mask[:, None] & (x_left >= 0)[None, :]
        left_off = base + y_offsets[:, None] * W + x_left[None, :]
        left = tl.load(grad_ptr + left_off, mask=left_mask & tile_mask, other=INF)
        grad = tl.minimum(grad, left + 1)

        # Right neighbor (x+1)
        x_right = x_offsets + 1
        right_mask = y_mask[:, None] & (x_right < W)[None, :]
        right_off = base + y_offsets[:, None] * W + x_right[None, :]
        right = tl.load(grad_ptr + right_off, mask=right_mask & tile_mask, other=INF)
        grad = tl.minimum(grad, right + 1)

        # Top-left (y-1, x-1)
        tl_mask = (y_top >= 0)[:, None] & (x_left >= 0)[None, :]
        tl_off = base + y_top[:, None] * W + x_left[None, :]
        tl_val = tl.load(grad_ptr + tl_off, mask=tl_mask & tile_mask, other=INF)
        grad = tl.minimum(grad, tl_val + 1)

        # Top-right (y-1, x+1)
        tr_mask = (y_top >= 0)[:, None] & (x_right < W)[None, :]
        tr_off = base + y_top[:, None] * W + x_right[None, :]
        tr_val = tl.load(grad_ptr + tr_off, mask=tr_mask & tile_mask, other=INF)
        grad = tl.minimum(grad, tr_val + 1)

        # Bottom-left (y+1, x-1)
        bl_mask = (y_bot < H)[:, None] & (x_left >= 0)[None, :]
        bl_off = base + y_bot[:, None] * W + x_left[None, :]
        bl_val = tl.load(grad_ptr + bl_off, mask=bl_mask & tile_mask, other=INF)
        grad = tl.minimum(grad, bl_val + 1)

        # Bottom-right (y+1, x+1)
        br_mask = (y_bot < H)[:, None] & (x_right < W)[None, :]
        br_off = base + y_bot[:, None] * W + x_right[None, :]
        br_val = tl.load(grad_ptr + br_off, mask=br_mask & tile_mask, other=INF)
        grad = tl.minimum(grad, br_val + 1)

        # Store result
        tl.store(grad_ptr + offsets, grad, mask=tile_mask)


def triton_gradient_spread(gradient, walls, iterations=4):
    """Spread gradient using Triton kernel.

    Args:
        gradient: (B, T, H, W) float32 tensor
        walls: (B, H, W) bool tensor
        iterations: Number of spread iterations

    Modifies gradient in-place.
    """
    B, T, H, W = gradient.shape

    # Reshape to (B*T, H, W) for the kernel
    grad_flat = gradient.view(B * T, H, W)

    # Wall mask expanded to (B*T, H, W) — walls are INF and shouldn't change
    wall_expanded = walls.unsqueeze(1).expand(B, T, H, W).reshape(B * T, H, W)

    BLOCK_H = 32
    BLOCK_W = 32

    grid = (B * T,
            triton.cdiv(H, BLOCK_H),
            triton.cdiv(W, BLOCK_W))

    _gradient_spread_kernel[grid](
        grad_flat,
        wall_expanded,
        B * T, H, W,
        999999,  # INF
        iterations,
        BLOCK_H, BLOCK_W,
    )

    # Restore walls to INF (kernel might have modified them)
    gradient[walls.unsqueeze(1).expand_as(gradient)] = 999999
