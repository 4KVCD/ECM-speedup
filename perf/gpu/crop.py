# crop.py <in.yuv> <w> <h> <x> <y> <cw> <ch> <frames> <out.yuv>: crops an 8-bit 4:2:0 sequence (x, y, cw, ch even);
# small pictures put many blocks at the picture's edges, where the reference availability changes
import sys

src, w, h, x, y, cw, ch, frames, dst = sys.argv[1], *map(int, sys.argv[2:9]), sys.argv[9]
frame = w * h * 3 // 2
with open(src, "rb") as f, open(dst, "wb") as o:
    for _ in range(frames):
        data = f.read(frame)
        if len(data) < frame:
            break
        # luma, then the two chroma planes at half the size
        for pw, ph, px, py, pcw, pch, off in [
            (w, h, x, y, cw, ch, 0),
            (w // 2, h // 2, x // 2, y // 2, cw // 2, ch // 2, w * h),
            (w // 2, h // 2, x // 2, y // 2, cw // 2, ch // 2, w * h + (w // 2) * (h // 2)),
        ]:
            for r in range(pch):
                start = off + (py + r) * pw + px
                o.write(data[start:start + pcw])
