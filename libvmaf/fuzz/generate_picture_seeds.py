#!/usr/bin/env python3
"""generate_picture_seeds.py -- create seed inputs for fuzz_read_pictures."""
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "corpus", "pictures")
os.makedirs(out_dir, exist_ok=True)

widths  = [64, 128, 192, 576]
heights = [64, 128, 108, 324]

for dim_idx in range(4):
    w, h = widths[dim_idx], heights[dim_idx]
    for bpc_flag in [0, 1]:
        bpc = 10 if (bpc_flag & 1) else 8
        bps = 2 if bpc > 8 else 1
        luma = w * h * bps
        chroma = (w // 2) * (h // 2) * bps
        frame = luma + 2 * chroma
        header = bytes([dim_idx, bpc_flag])
        # Fill with mid-gray
        mid = 128 if bpc == 8 else 0  # 0x0200 little-endian = 512
        pixel_data = bytes([mid]) * frame
        seed = header + pixel_data
        name = os.path.join(out_dir, f"seed_{w}x{h}_{bpc}bit.bin")
        with open(name, "wb") as f:
            f.write(seed)
