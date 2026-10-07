#!/usr/bin/env python3
"""Extract frames of an 8-bit 4:2:0 .y4m into a raw .yuv file, optionally cropped.

usage: y4m2yuv.py in.y4m out.yuv num_frames [crop_w crop_h crop_x crop_y]
"""
import sys


def main():
    if len(sys.argv) not in (4, 8):
        print(__doc__)
        return 1
    src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with open(src, "rb") as f:
        header = f.readline().split()
        w = int(next(t for t in header if t.startswith(b"W"))[1:])
        h = int(next(t for t in header if t.startswith(b"H"))[1:])
        cw, ch, cx, cy = (int(v) for v in sys.argv[4:8]) if len(sys.argv) == 8 else (w, h, 0, 0)
        assert cw % 2 == 0 and ch % 2 == 0 and cx % 2 == 0 and cy % 2 == 0
        frame_size = w * h * 3 // 2
        written = 0
        with open(dst, "wb") as o:
            for _ in range(n):
                tag = f.readline()
                if not tag.startswith(b"FRAME"):
                    break
                frame = f.read(frame_size)
                planes = [(frame[: w * h], w, cw, ch, cx, cy)]
                cs = w * h // 4
                for p in range(2):
                    data = frame[w * h + p * cs: w * h + (p + 1) * cs]
                    planes.append((data, w // 2, cw // 2, ch // 2, cx // 2, cy // 2))
                for data, stride, pw, ph, px, py in planes:
                    for y in range(py, py + ph):
                        o.write(data[y * stride + px: y * stride + px + pw])
                written += 1
    print(f"{dst}: {cw}x{ch}, {written} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
