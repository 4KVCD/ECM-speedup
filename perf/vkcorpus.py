#!/usr/bin/env python3
"""Test bitstream corpus for decoder work: builds the streams, records the stock decoder's output and checks other
decoder builds against it.

usage:
  vkcorpus.py prepare                       derived source clips (crops, synthetic screen content) in C:/Code/seq/corpus
  vkcorpus.py encode [--only a,b] [--jobs N] [--exe ENC]   encodes missing streams into C:/Code/corpus
  vkcorpus.py ref [--only a,b]              decodes with the stock DecoderApp, stores YUV md5 and frame count
  vkcorpus.py check EXE [--only a,b] [--env K=V] [--args "..."] [--jobs N]
                                            decodes with EXE and compares every YUV md5 with the reference
  vkcorpus.py list                          prints the streams and their state

Encodes run on the efficiency cores (affinity mask EBITS) and decodes too, unless --cpus is given, so that the
performance cores stay free for measurements. The manifest is C:/Code/corpus/manifest.json.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import struct
import subprocess
import sys
import time

ROOT = "C:/Code"
CFG = f"{ROOT}/vk/cfg"
SEQ = f"{ROOT}/seq"
SRC = f"{SEQ}/corpus"
OUT = f"{ROOT}/corpus"
MANIFEST = f"{OUT}/manifest.json"
STOCK_DEC = f"{ROOT}/bin/stock/DecoderApp.exe"
DEFAULT_ENC = f"{ROOT}/bin/pgo5/EncoderApp.exe"
EBITS = 0x3FC3C  # logical CPUs 2-5 and 10-17: the efficiency cores of the Core Ultra 7 265K

AI = "encoder_intra_ecm.cfg"
RA = "encoder_randomaccess_ecm.cfg"
RA16 = "encoder_randomaccess_gop16_ecm.cfg"
LD = "encoder_lowdelay_ecm.cfg"
LDP = "encoder_lowdelay_P_ecm.cfg"

# Source clips: name -> (file, width, height, fps, frames available, chroma format, bit depth)
CLIPS = {
    "foreman176": ("foreman_176x144.yuv", 176, 144, 30, 10, 420, 8),
    "foreman352": ("foreman_352x288.yuv", 352, 288, 30, 10, 420, 8),
    "foreman128": ("foreman_128x128_257.yuv", 128, 128, 30, 257, 420, 8),
    "parkjoy256": ("ParkJoy_256x256.yuv", 256, 256, 50, 10, 420, 8),
    "parkjoy416": ("ParkJoy_416x240.yuv", 416, 240, 50, 4, 420, 8),
    "fourpeople256": ("FourPeople_256x256.yuv", 256, 256, 60, 10, 420, 8),
    "crowdrun256": ("CrowdRun_256x256.yuv", 256, 256, 50, 10, 420, 8),
    # made by `prepare`
    "foreman176_33": ("corpus/foreman_176x144_33.yuv", 176, 144, 30, 33, 420, 8),
    "parkjoy832": ("corpus/ParkJoy_832x480_12.yuv", 832, 480, 50, 12, 420, 8),
    "crowdrun832": ("corpus/CrowdRun_832x480_12.yuv", 832, 480, 50, 12, 420, 8),
    "foreman176_10b": ("corpus/foreman_176x144_10bit.yuv", 176, 144, 30, 10, 420, 10),
    "foreman176_444": ("corpus/foreman_176x144_444.yuv", 176, 144, 30, 10, 444, 8),
    "foreman176_422": ("corpus/foreman_176x144_422.yuv", 176, 144, 30, 10, 422, 8),
    "screen256": ("corpus/screen_256x256.yuv", 256, 256, 30, 10, 420, 8),
    "screen256_444": ("corpus/screen_256x256_444.yuv", 256, 256, 30, 10, 444, 8),
    "odd200": ("corpus/foreman_200x136.yuv", 200, 136, 30, 10, 420, 8),
}

WPP = ["-c", f"{CFG}/partitioning/entropy_coding_synchronization.cfg"]
SUBPIC3 = ["-c", f"{CFG}/partitioning/subpicture_3_slice_2_tile.cfg"]
TILES_RASTER = ["--EnablePicPartitioning=1", "--TileColumnWidthArray=4", "--TileRowHeightArray=2",
                "--RasterScanSlices=1", "--RasterSliceSizes=1"]
TILES_RECT = ["--EnablePicPartitioning=1", "--TileColumnWidthArray=2", "--TileRowHeightArray=2",
              "--RasterScanSlices=0", "--RectSliceFixedWidth=1", "--RectSliceFixedHeight=1"]
NO_XTILE_LF = ["--DisableLoopFilterAcrossTiles=1", "--DisableLoopFilterAcrossSlices=1"]
SCC = ["-c", f"{CFG}/per-class/classF.cfg"]
SCC2 = ["-c", f"{CFG}/per-class/classSCC.cfg"]

# name -> (main cfg, clip, frames, qp, extra encoder args)
STREAMS = {
    # common test conditions, several contents and QPs
    "ai_foreman352_q22": (AI, "foreman352", 2, 22, ["--TemporalSubsampleRatio=1"]),
    "ai_foreman352_q37": (AI, "foreman352", 1, 37, []),
    "ai_parkjoy_q27": (AI, "parkjoy256", 2, 27, ["--TemporalSubsampleRatio=1"]),
    "ai_fourpeople_q32": (AI, "fourpeople256", 1, 32, []),
    "ai_crowdrun_q42": (AI, "crowdrun256", 1, 42, []),
    "ra_foreman_q32": (RA, "foreman176_33", 33, 32, []),
    "ra_foreman_q22": (RA, "foreman176", 9, 22, []),
    "ra_parkjoy_q37": (RA, "parkjoy256", 9, 37, []),
    "ra_fourpeople_q27": (RA, "fourpeople256", 9, 27, []),
    "ra_crowdrun_q32": (RA, "crowdrun256", 9, 32, []),
    "ra16_foreman_q32": (RA16, "foreman176", 10, 32, []),
    "ra_long_q32": (RA, "foreman128", 65, 32, []),
    "ld_foreman_q32": (LD, "foreman176", 10, 32, []),
    "ld_parkjoy_q27": (LD, "parkjoy256", 5, 27, []),
    "ld_fourpeople_q37": (LD, "fourpeople256", 9, 37, []),
    "ldp_foreman_q32": (LDP, "foreman176", 10, 32, []),
    "ldp_crowdrun_q27": (LDP, "crowdrun256", 5, 27, []),
    # larger pictures: several CTU rows and columns
    "ra_parkjoy832_q32": (RA, "parkjoy832", 5, 32, []),
    "ai_crowdrun832_q27": (AI, "crowdrun832", 1, 27, []),
    "odd_ra_q32": (RA, "odd200", 5, 32, []),
    "odd_ai_q27": (AI, "odd200", 1, 27, []),
    # bit depths
    "ra_10bit_q32": (RA, "foreman176_10b", 5, 32, ["--InputBitDepth=10"]),
    "ra_internal8_q32": (RA, "foreman176", 5, 32, ["--InternalBitDepth=8"]),
    "ai_internal8_q27": (AI, "foreman176", 1, 27, ["--InternalBitDepth=8"]),
    "ra_internal12_q32": (RA, "foreman176", 5, 32, ["--InternalBitDepth=12"]),
    # chroma formats
    "ai_444_q27": (AI, "foreman176_444", 1, 27, ["--InputChromaFormat=444", "-c", f"{CFG}/444/yuv444.cfg"]),
    "ra_444_q32": (RA, "foreman176_444", 5, 32, ["--InputChromaFormat=444", "-c", f"{CFG}/444/yuv444.cfg"]),
    "ai_422_q27": (AI, "foreman176_422", 1, 27, ["--InputChromaFormat=422"]),
    "ra_422_q32": (RA, "foreman176_422", 5, 32, ["--InputChromaFormat=422"]),
    "ai_400_q27": (AI, "foreman176", 1, 27, ["--ChromaFormatIDC=400"]),
    "ra_400_q32": (RA, "foreman176", 5, 32, ["--ChromaFormatIDC=400"]),
    "ai_rgb_q27": (AI, "foreman176_444", 1, 27, ["--InputChromaFormat=444", "-c", f"{CFG}/per-class/formatRGB.cfg"]),
    "ra_rgb_q32": (RA, "foreman176_444", 5, 32, ["--InputChromaFormat=444", "-c", f"{CFG}/per-class/formatRGB.cfg"]),
    # screen content tools
    "ai_scc_q27": (AI, "screen256", 1, 27, SCC),
    "ra_scc_q32": (RA, "screen256", 9, 32, SCC),
    "ld_scc_q32": (LD, "screen256", 5, 32, SCC),
    "ai_scc2_q32": (AI, "screen256", 1, 32, SCC2),
    "ai_scc444_q27": (AI, "screen256_444", 1, 27, SCC + ["--InputChromaFormat=444"]),
    # lossless
    "ai_lossless": (AI, "foreman176", 1, 0, ["-c", f"{CFG}/lossless/lossless.cfg"]),
    "ra_lossless": (RA, "foreman176", 3, 0, ["-c", f"{CFG}/lossless/lossless.cfg"]),
    "ra_lossless_mixed": (RA, "foreman176", 3, 22, ["-c", f"{CFG}/lossless/lossless_mixed.cfg"]),
    # partitioning
    "ra_wpp_q32": (RA, "parkjoy416", 4, 32, WPP),
    "ai_wpp_q27": (AI, "crowdrun832", 1, 27, WPP),
    "ra_tiles_raster_q32": (RA, "parkjoy832", 3, 32, TILES_RASTER),
    "ra_tiles_rect_q32": (RA, "parkjoy832", 3, 32, TILES_RECT + NO_XTILE_LF),
    "ai_tiles_rect_q27": (AI, "crowdrun832", 1, 27, TILES_RECT),
    "ra_subpic_q32": (RA, "parkjoy832", 3, 32, SUBPIC3),
    # reference picture resampling
    "ra_rpr2x_q32": (RA, "foreman176_33", 17, 32, ["-c", f"{CFG}/rpr/scale2.0x.cfg"]),
    "ra_rpr15x_q32": (RA, "foreman176_33", 17, 32, ["-c", f"{CFG}/rpr/scale1.5x.cfg"]),
    # quantization variants
    "ra_scalinglist_q32": (RA, "foreman176", 5, 32, ["-c", f"{CFG}/scaling_list/sample_scaling_list.cfg",
                                                    f"--ScalingListFile={CFG}/scaling_list/sample_scaling_list_file.txt"]),
    "ai_scalinglist_q27": (AI, "foreman176", 1, 27, ["-c", f"{CFG}/scaling_list/sample_scaling_list.cfg",
                                                    f"--ScalingListFile={CFG}/scaling_list/sample_scaling_list_file.txt"]),
    "ra_dqp_q32": (RA, "foreman176", 5, 32, ["--MaxDeltaQP=4", "--MaxCuDQPSubdiv=2", "--AdaptiveQP=1"]),
    "ra_signhide_q32": (RA, "foreman176", 5, 32, ["--DepQuant=0", "--SignHideFlag=1"]),
    "ai_signhide_q27": (AI, "foreman176", 1, 27, ["--DepQuant=0", "--SignHideFlag=1"]),
    "ra_chromaqp_q32": (RA, "foreman176", 5, 32, ["--CbQpOffset=3", "--CrQpOffset=-2", "--MaxCuChromaQpOffsetSubdiv=2",
                                                 "--SliceChromaQPOffsetPeriodicity=2", "--SliceCbQpOffsetIntraOrPeriodic=2"]),
    # in-loop filter variants
    "ra_nolmcs_q32": (RA, "foreman176", 5, 32, ["--LMCSEnable=0"]),
    "ra_nodbk_q32": (RA, "foreman176", 5, 32, ["--LoopFilterDisable=1"]),
    "ra_dbkparams_q32": (RA, "foreman176", 5, 32, ["--LoopFilterBetaOffset_div2=3", "--LoopFilterTcOffset_div2=-2",
                                                  "--LoopFilterCbBetaOffset_div2=-1", "--LoopFilterCrTcOffset_div2=2"]),
    "ra_noalf_q32": (RA, "foreman176", 5, 32, ["--ALF=0", "--CCALF=0"]),
    "ra_nosao_q32": (RA, "foreman176", 5, 32, ["--SAO=0", "--CCSAO=0", "--BIF=0"]),
    # weighted prediction
    "ra_wp_q32": (RA, "foreman176", 5, 32, ["--WeightedPredP=1", "--WeightedPredB=1"]),
    "ldp_wp_q32": (LDP, "foreman176", 5, 32, ["--WeightedPredP=1"]),
    # tool groups switched off, which takes other code paths
    "ra_off1_q32": (RA, "foreman176", 5, 32, ["-c", f"{CFG}/groupoff/offgroup1.cfg"]),
    "ra_off2_q32": (RA, "foreman176", 5, 32, ["-c", f"{CFG}/groupoff/offgroup2.cfg"]),
    "ra_off3_q32": (RA, "foreman176", 5, 32, ["-c", f"{CFG}/groupoff/offgroup3.cfg"]),
    "ra_off4_q32": (RA, "foreman176", 5, 32, ["-c", f"{CFG}/groupoff/offgroup4.cfg"]),
    "ra_off5_q32": (RA, "foreman176", 5, 32, ["-c", f"{CFG}/groupoff/offgroup5.cfg"]),
    "ra_off15_q32": (RA, "foreman176", 5, 32, ["-c", f"{CFG}/groupoff/offgroup1-5.cfg"]),
    "ai_off15_q27": (AI, "foreman176", 1, 27, ["-c", f"{CFG}/groupoff/offgroup1-5.cfg"]),
    # VTM-like tool sets
    "ai_vtm_q32": (f"tool-on/encoder_intra_vtm.cfg", "foreman176", 1, 32, []),
    "ra_vtm_q32": (f"tool-on/encoder_randomaccess_vtm.cfg", "foreman176", 9, 32, []),
    "ld_vtm_q32": (f"tool-on/encoder_lowdelay_vtm.cfg", "foreman176", 5, 32, []),
    "ldp_vtm_q32": (f"tool-on/encoder_lowdelay_P_vtm.cfg", "foreman176", 5, 32, []),
    # partitioning limits and CTU sizes
    "ra_ctu64_q32": (RA, "foreman176", 5, 32, ["--CTUSize=64"]),
    "ai_ctu64_q27": (AI, "foreman176", 1, 27, ["--CTUSize=64"]),
    "ra_ctu256_q32": (RA, "parkjoy416", 4, 32, ["--CTUSize=256"]),
    "ra_nodualtree_q32": (RA, "foreman176", 5, 32, ["--DualITree=0"]),
    "ai_nodualtree_q27": (AI, "foreman176", 1, 27, ["--DualITree=0"]),
    # decoding refresh and reference structures
    "ra_idr_q32": (RA, "foreman176_33", 33, 32, ["--DecodingRefreshType=2"]),
    "ld_gdr_q32": (LD, "foreman176", 10, 32, ["--GdrEnabled=1", "--GdrPeriod=4", "--GdrPocStart=2", "--GdrInterval=4"]),
    "ai_fast_q32": (AI, "foreman176", 1, 32, ["-c", f"{CFG}/alternative-addon/reduced_runtime1.cfg"]),
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_affinity(proc, mask):
    import ctypes
    ctypes.windll.kernel32.SetProcessAffinityMask(int(proc._handle), ctypes.c_size_t(mask))


def load_manifest():
    if os.path.exists(MANIFEST):
        return json.load(open(MANIFEST))
    return {}


def save_manifest(m):
    os.makedirs(OUT, exist_ok=True)
    tmp = MANIFEST + ".tmp"
    json.dump(m, open(tmp, "w"), indent=1, sort_keys=True)
    os.replace(tmp, MANIFEST)


def selected(only):
    names = list(STREAMS)
    if only:
        want = only.split(",")
        names = [n for n in names if any(n == w or (w.endswith("*") and n.startswith(w[:-1])) for w in want)]
    return names


# ---- prepare: derived clips --------------------------------------------------------------------------------------

def read_y4m_frames(path, nframes, w, h):
    """Yields 4:2:0 8-bit frames (y, u, v as bytes) from a y4m file of size w x h."""
    with open(path, "rb") as f:
        f.readline()
        for _ in range(nframes):
            line = f.readline()
            if not line.startswith(b"FRAME"):
                return
            y = f.read(w * h)
            u = f.read(w * h // 4)
            v = f.read(w * h // 4)
            yield y, u, v


def crop(plane, pw, x, y, w, h):
    return b"".join(plane[(y + r) * pw + x:(y + r) * pw + x + w] for r in range(h))


def upsample_chroma(plane, cw, ch, fx, fy):
    rows = []
    for r in range(ch):
        row = plane[r * cw:(r + 1) * cw]
        if fx == 2:
            row = bytes(b for b in row for _ in range(2))
        rows.extend([row] * fy)
    return b"".join(rows)


def read_yuv420(path, w, h, n):
    fs = w * h * 3 // 2
    with open(path, "rb") as f:
        for _ in range(n):
            d = f.read(fs)
            if len(d) < fs:
                return
            yield d[:w * h], d[w * h:w * h * 5 // 4], d[w * h * 5 // 4:]


def make_screen(path, w, h, n, chroma444):
    """Synthetic screen content: flat windows, repeated glyphs and a gradient, with a scrolling text block."""
    import random
    rnd = random.Random(12345)
    glyphs = [[rnd.random() < 0.4 for _ in range(64)] for _ in range(24)]
    text = [rnd.randrange(len(glyphs)) for _ in range(4000)]
    with open(path, "wb") as f:
        for t in range(n):
            y = bytearray(w * h)
            for r in range(h):
                for c in range(w):
                    y[r * w + c] = 235 if r < 32 else (16 + (c * 219) // w if r > h - 24 else 200)
            # window with text that scrolls one glyph row up every other frame
            scroll = (t // 2) * 8
            for gy in range(8, 26):
                for gx in range(2, 30):
                    g = glyphs[text[(gy + scroll // 8) * 31 + gx]]
                    for py in range(8):
                        for px in range(8):
                            if g[py * 8 + px]:
                                yy, xx = 40 + (gy - 8) * 10 + py, gx * 8 + px
                                if yy < h - 24 and xx < w:
                                    y[yy * w + xx] = 20
            # a moving solid box
            bx = 16 + 6 * t
            for r in range(36, 60):
                for c in range(bx, min(w, bx + 40)):
                    y[r * w + c] = 90
            cw, ch = (w, h) if chroma444 else (w // 2, h // 2)
            u = bytearray([128] * (cw * ch))
            v = bytearray([128] * (cw * ch))
            s = 1 if chroma444 else 2
            for r in range(ch):
                for c in range(cw):
                    if r * s < 32:
                        u[r * cw + c], v[r * cw + c] = 160, 100
                    elif 36 <= r * s < 60 and bx <= c * s < bx + 40:
                        u[r * cw + c], v[r * cw + c] = 90, 200
            f.write(bytes(y) + bytes(u) + bytes(v))


def cmd_prepare(args):
    os.makedirs(SRC, exist_ok=True)
    fsrc = f"{SEQ}/src/foreman_cif.y4m"

    def out(name):
        return f"{SEQ}/{name}"

    if not os.path.exists(out(CLIPS["foreman176_33"][0])):
        with open(out(CLIPS["foreman176_33"][0]), "wb") as f:
            for y, u, v in read_y4m_frames(fsrc, 33, 352, 288):
                f.write(crop(y, 352, 88, 72, 176, 144) + crop(u, 176, 44, 36, 88, 72) + crop(v, 176, 44, 36, 88, 72))
    if not os.path.exists(out(CLIPS["odd200"][0])):
        with open(out(CLIPS["odd200"][0]), "wb") as f:
            for y, u, v in read_y4m_frames(fsrc, 10, 352, 288):
                f.write(crop(y, 352, 72, 72, 200, 136) + crop(u, 176, 36, 36, 100, 68) + crop(v, 176, 36, 36, 100, 68))
    for key, src in (("parkjoy832", "src/park_joy_1080p50_12f.y4m"), ("crowdrun832", "src2/crowd_run_1080p50_12f.y4m")):
        if not os.path.exists(out(CLIPS[key][0])):
            with open(out(CLIPS[key][0]), "wb") as f:
                for y, u, v in read_y4m_frames(f"{SEQ}/{src}", 12, 1920, 1080):
                    f.write(crop(y, 1920, 544, 300, 832, 480) + crop(u, 960, 272, 150, 416, 240) +
                            crop(v, 960, 272, 150, 416, 240))
    base = f"{SEQ}/foreman_176x144.yuv"
    if not os.path.exists(out(CLIPS["foreman176_10b"][0])):
        with open(out(CLIPS["foreman176_10b"][0]), "wb") as f:
            for y, u, v in read_yuv420(base, 176, 144, 10):
                # 10-bit samples with varied low bits, so the extra precision is used
                data = y + u + v
                f.write(b"".join(struct.pack("<H", (s << 2) | ((i * 7 + s) & 3)) for i, s in enumerate(data)))
    for key, fx, fy in (("foreman176_444", 2, 2), ("foreman176_422", 2, 1)):
        if not os.path.exists(out(CLIPS[key][0])):
            with open(out(CLIPS[key][0]), "wb") as f:
                for y, u, v in read_yuv420(base, 176, 144, 10):
                    f.write(y + upsample_chroma(u, 88, 72, fx, fy) + upsample_chroma(v, 88, 72, fx, fy))
    if not os.path.exists(out(CLIPS["screen256"][0])):
        make_screen(out(CLIPS["screen256"][0]), 256, 256, 10, False)
    if not os.path.exists(out(CLIPS["screen256_444"][0])):
        make_screen(out(CLIPS["screen256_444"][0]), 256, 256, 10, True)
    for k, c in CLIPS.items():
        p = f"{SEQ}/{c[0]}"
        print(f"{k:16s} {os.path.getsize(p) if os.path.exists(p) else 'MISSING':>10}  {p}")


# ---- encode -------------------------------------------------------------------------------------------------------

def encode_one(name, exe):
    cfg, clip, frames, qp, extra = STREAMS[name]
    f, w, h, fps, avail, fmt, bd = CLIPS[clip]
    assert frames <= avail, name
    bs = f"{OUT}/{name}.bin"
    log = f"{OUT}/{name}.enc.log"
    cmd = [exe, "-c", f"{CFG}/{cfg}", *extra, "-i", f"{SEQ}/{f}", "-wdt", str(w), "-hgt", str(h), "-fr", str(fps),
           "-f", str(frames), "-q", str(qp), "-b", bs + ".part", "-o", "NUL"]
    if not any(a.startswith("--InputBitDepth") for a in extra):
        cmd.append(f"--InputBitDepth={bd}")
    if not any(a.startswith("--InputChromaFormat") for a in extra):
        cmd.append(f"--InputChromaFormat={fmt}")
    t0 = time.time()
    with open(log, "w") as lf:
        lf.write(" ".join(cmd) + "\n")
        lf.flush()
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        set_affinity(p, EBITS)
        rc = p.wait()
    if rc == 0 and os.path.exists(bs + ".part"):
        os.replace(bs + ".part", bs)
    return name, rc, time.time() - t0


def cmd_encode(args):
    os.makedirs(OUT, exist_ok=True)
    names = [n for n in selected(args.only) if args.force or not os.path.exists(f"{OUT}/{n}.bin")]
    print(f"encoding {len(names)} streams with {args.exe}", flush=True)
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        for name, rc, dt in ex.map(lambda n: encode_one(n, args.exe), names):
            print(f"{name:24s} rc={rc} {dt:7.1f}s", flush=True)


# ---- decode, reference and check ------------------------------------------------------------------------------------

def decode(exe, name, tag, env=None, extra=(), cpu=None):
    bs = f"{OUT}/{name}.bin"
    yuv = f"{OUT}/tmp/{name}.{tag}.yuv"
    os.makedirs(f"{OUT}/tmp", exist_ok=True)
    e = dict(os.environ)
    e.update(env or {})
    t0 = time.time()
    with open(yuv + ".log", "w") as lf:
        p = subprocess.Popen([exe, "-b", bs, "-o", yuv, *extra], stdout=lf, stderr=subprocess.STDOUT, env=e)
        set_affinity(p, (1 << cpu) if cpu is not None else EBITS)
        rc = p.wait()
    dt = time.time() - t0
    res = {"rc": rc, "time": round(dt, 3)}
    if rc == 0 and os.path.exists(yuv):
        res["md5"] = md5(yuv)
        res["bytes"] = os.path.getsize(yuv)
    else:
        res["log"] = open(yuv + ".log", errors="replace").read()[-2000:]
    for p in (yuv, yuv + ".log"):
        if rc == 0 and os.path.exists(p):
            os.remove(p)
    return name, res


def cmd_ref(args):
    m = load_manifest()
    names = [n for n in selected(args.only) if os.path.exists(f"{OUT}/{n}.bin")]
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        for name, res in ex.map(lambda n: decode(STOCK_DEC, n, "ref"), names):
            if res["rc"] != 0:
                print(f"{name:24s} STOCK DECODE FAILED rc={res['rc']}")
                continue
            m[name] = {"md5": res["md5"], "bytes": res["bytes"], "stock_time": res["time"],
                       "bin_md5": md5(f"{OUT}/{name}.bin"), "def": list(map(str, STREAMS[name][:4])) + [STREAMS[name][4]]}
            print(f"{name:24s} {res['md5']}  {res['time']:6.2f}s", flush=True)
    save_manifest(m)


def cmd_check(args):
    m = load_manifest()
    names = [n for n in selected(args.only) if n in m]
    env = dict(kv.split("=", 1) for kv in args.env) if args.env else {}
    extra = args.args.split() if args.args else []
    cpus = [int(c) for c in args.cpus.split(",")] if args.cpus else None
    bad = 0
    t0 = time.time()

    def run(i_n):
        i, n = i_n
        return decode(args.exe, n, "chk", env, extra, cpus[i % len(cpus)] if cpus else None)

    jobs = len(cpus) if cpus else args.jobs
    with cf.ThreadPoolExecutor(jobs) as ex:
        for name, res in ex.map(run, enumerate(names)):
            ref = m[name]
            ok = res["rc"] == 0 and res.get("md5") == ref["md5"]
            bad += not ok
            if not ok or args.verbose:
                state = "OK" if ok else ("CRASH rc=%d" % res["rc"] if res["rc"] else "MISMATCH")
                print(f"{name:24s} {state:12s} {res['time']:7.2f}s (stock {ref['stock_time']:.2f}s)", flush=True)
                if res["rc"]:
                    print("   " + res.get("log", "")[-600:].replace("\n", "\n   "))
    print(f"{len(names) - bad}/{len(names)} identical, {time.time() - t0:.1f}s")
    return 1 if bad else 0


def cmd_list(args):
    m = load_manifest()
    for n in selected(args.only):
        state = "ref" if n in m else ("bin" if os.path.exists(f"{OUT}/{n}.bin") else "-")
        print(f"{n:24s} {state}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p = sub.add_parser("encode")
    p.add_argument("--only")
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--exe", default=DEFAULT_ENC)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("ref")
    p.add_argument("--only")
    p.add_argument("--jobs", type=int, default=12)
    p = sub.add_parser("check")
    p.add_argument("exe")
    p.add_argument("--only")
    p.add_argument("--env", action="append")
    p.add_argument("--args")
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--cpus")
    p.add_argument("-v", "--verbose", action="store_true")
    p = sub.add_parser("list")
    p.add_argument("--only")
    args = ap.parse_args()
    return {"prepare": cmd_prepare, "encode": cmd_encode, "ref": cmd_ref, "check": cmd_check, "list": cmd_list}[
        args.cmd](args) or 0


if __name__ == "__main__":
    sys.exit(main())
