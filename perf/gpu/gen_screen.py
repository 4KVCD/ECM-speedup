# gen_screen.py <out.yuv> [width height frames]: synthetic screen-like 4:2:0 8-bit content (flat areas, sharp-edged
# rectangles, text-like glyphs, a few gradients), for tools encoders mostly choose on screen content (transform skip,
# BDPCM). Deterministic (fixed seed); each frame scrolls the content a little.
import random
import sys

out = sys.argv[1]
w, h, frames = (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (352, 288, 4)
rnd = random.Random(20261011)

# the canvas, wider than the picture so that frames can scroll: Y, U, V at full resolution (subsampled when written)
cw, ch = w + 64, h
Y = [[235] * cw for _ in range(ch)]
U = [[128] * cw for _ in range(ch)]
V = [[128] * cw for _ in range(ch)]

def fill(x0, y0, x1, y1, y, u, v):
    for yy in range(max(0, y0), min(ch, y1)):
        for xx in range(max(0, x0), min(cw, x1)):
            Y[yy][xx], U[yy][xx], V[yy][xx] = y, u, v

# windows and buttons
for _ in range(40):
    x0, y0 = rnd.randrange(cw), rnd.randrange(ch)
    fill(x0, y0, x0 + rnd.randrange(8, 120), y0 + rnd.randrange(8, 80), rnd.choice([16, 40, 90, 160, 200, 235]),
         rnd.choice([64, 128, 160, 200]), rnd.choice([64, 100, 128, 180]))
# horizontal and vertical gradients
for _ in range(4):
    x0, y0, gw, gh = rnd.randrange(cw), rnd.randrange(ch), rnd.randrange(16, 96), rnd.randrange(16, 64)
    vertical = rnd.random() < 0.5
    for yy in range(y0, min(ch, y0 + gh)):
        for xx in range(x0, min(cw, x0 + gw)):
            Y[yy][xx] = 16 + 219 * ((yy - y0) if vertical else (xx - x0)) // (gh if vertical else gw)
# text-like glyphs: random 5x7 bitmaps, one or two pixels per bit, in lines
for _ in range(60):
    x, y, scale = rnd.randrange(cw - 80), rnd.randrange(ch - 16), rnd.choice([1, 1, 2])
    fg = rnd.choice([16, 30, 235, 220])
    for _ in range(rnd.randrange(3, 10)):
        glyph = [rnd.getrandbits(5) for _ in range(7)]
        for gy in range(7):
            for gx in range(5):
                if glyph[gy] >> gx & 1:
                    fill(x + gx * scale, y + gy * scale, x + (gx + 1) * scale, y + (gy + 1) * scale, fg, 128, 128)
        x += 6 * scale

with open(out, "wb") as f:
    for n in range(frames):
        dx = 4 * n
        f.write(bytes(Y[yy][xx + dx] for yy in range(h) for xx in range(w)))
        for P in (U, V):
            f.write(bytes(P[yy][xx + dx] for yy in range(0, h, 2) for xx in range(0, w, 2)))
print(f"{out}: {w}x{h}, {frames} frames")
