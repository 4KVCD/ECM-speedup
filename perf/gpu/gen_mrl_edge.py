# gen_mrl_edge.py <out.yuv> <w> <h> <frames> <seed> [p16]: synthetic 4:2:0 content (deterministic) on which the
# encoder chooses far reference lines (MRL) near the picture's top: rows 0-3 noise, below them vertical stripes of two
# grey levels (with a little noise), 8 wide alternating. With p16 the stripes fill rows 4-11 only (noise below) and
# have a period of 16 columns (4 of one level, 12 of the other): for a block at x = 16k the left column of reference
# line 12 (x - 13) is the only one of its level
import random, sys

out, w, h, frames, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
period16 = len(sys.argv) > 6
rnd = random.Random(seed)
with open(out, "wb") as f:
    for fr in range(frames):
        a, b = rnd.randrange(40, 100), rnd.randrange(150, 220)
        y = bytearray(w * h)
        for r in range(h):
            for c in range(w):
                if r < 4 or (period16 and r >= 12):
                    v = rnd.randrange(0, 256)
                elif period16:
                    v = (a if c % 16 < 4 else b) + rnd.randrange(-2, 3)
                else:
                    v = (a if (c // 8) % 2 == 0 else b) + rnd.randrange(-2, 3)
                y[r * w + c] = max(0, min(255, v))
        f.write(y)
        f.write(bytes([128]) * (w * h // 2))
