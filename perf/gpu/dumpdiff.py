# dumpdiff.py <prefix> <POC>...: compares the records a verification decode (--GpuRecon=2) and an ON decode
# (--GpuRecon=3) of the same stream uploaded, both run with ECM_GPU_DUMP=<prefix> (GpuIntraRec::endPicture). With the
# reconstruction on, the CPU derives modes from samples it has not reconstructed, so every field the host takes from
# the CPU's mode of a block whose mode the GPU derives (Tu::derive & DERIVE_MODE) must be one intra_rec derives again:
# any other difference is a bug that verification cannot see. Reports per picture the fields that differ beyond those
# (and beyond the availability, read ranges and ISP side, which intra_deps computes), with the first blocks.
import struct, sys

TU = 101   # int32 per Tu (intra_common.glsl)
FIELDS = ["comp", "x", "y", "w", "h", "dirMode", "isModeVer", "intraPredAngle", "absInvAngle", "useCubic", "unitW",
          "unitH", "bitDepth", "cbf", "coeffOffset", "dqScale", "dqRightShift", "dqInMin", "dqInMax", "trMin", "trMax",
          "shift1", "shift2", "recMin", "recMax", "dbgOffset", "plIdx", "refFilter", "applyPDPC", "gradPDPC",
          "angularScale", "fusion", "disableRefFilter"] + [f"avail{i}" for i in range(14)] + \
         [f"readMax{i}" for i in range(4)] + ["nzRows", "nzCols", "bf", "qp", "lmcs", "crsX", "crsY", "crsW", "crsH",
          "crsAvail", "quant", "jccr", "jccrOffset", "mtxV", "mtxH", "transformSkip", "mipMatrix", "mipTransposed",
          "mipSizeId", "secKind", "secMatrix", "secRowLength", "secNumIn", "secScan", "secLayout", "secSize",
          "secTranspose", "mrl", "bfAvail0", "bfAvail1", "bfAvail2", "isp", "cuX", "cuY", "cuW", "cuH", "regX", "regW",
          "ispSide", "ispPrev", "spCoded", "spOffset", "resiFrom", "cu", "derive", "mtsSet", "secIdx", "secBucket",
          "lm", "cflm"]
assert len(FIELDS) == TU
# intra_rec derives these for DERIVE_MODE blocks (applyModeParams, DIMD), the transforms for DERIVE_TRANSFORM ones
DERIVED   = {"dirMode", "isModeVer", "intraPredAngle", "absInvAngle", "useCubic", "refFilter", "applyPDPC", "gradPDPC",
             "angularScale", "fusion", "bf", "plIdx"}
TRANSFORM = {"mtxV", "mtxH", "secMatrix", "secTranspose", "secLayout"}
# a TMRL block's line comes from its node too (tmrlBlockParams; Tu::derive & DERIVE_TMRL)
DERIVE_TMRL = 131072
# computed by intra_deps on the GPU
SCHEDULING = {f for f in FIELDS if f.startswith(("avail", "readMax", "bfAvail"))} | {"ispSide"}
# the angular fields planar / DC / MIP blocks keep from the CPU's previous block (never read for them), and PDPC's
# scale and gradient flag, which initPredIntraParams sets for positive angles only (and only those read)
ANGULAR  = {"intraPredAngle", "absInvAngle", "angularScale", "gradPDPC"}
POSITIVE = {"angularScale", "gradPDPC"}

def load(path):
    data, parts, o = open(path, "rb").read(), [], 0
    while o < len(data):
        n = struct.unpack_from("<Q", data, o)[0]
        parts.append(data[o + 8:o + 8 + n])
        o += 8 + n
    return parts   # Tu records, levels, sign prediction data, CU records with their modes, DIMD histograms and OBIC neighbours

def main():
    prefix, pocs = sys.argv[1], sys.argv[2:]
    bad = False
    for poc in pocs:
        ver, on = load(f"{prefix}_{poc}_ver.bin"), load(f"{prefix}_{poc}_on.bin")
        for name, a, b in zip(("levels", "sign prediction", "CUs"), ver[1:], on[1:]):
            if a != b:
                bad = True
                print(f"POC {poc}: the {name} differ")
        ta = struct.unpack(f"<{len(ver[0]) // 4}i", ver[0])
        tb = struct.unpack(f"<{len(on[0]) // 4}i", on[0])
        if len(ta) != len(tb):
            bad = True
            print(f"POC {poc}: {len(ta) // TU} blocks against {len(tb) // TU}")
            continue
        counts, shown = {}, 0
        for r in range(len(ta) // TU):
            ra, rb = ta[r * TU:(r + 1) * TU], tb[r * TU:(r + 1) * TU]
            derive = ra[FIELDS.index("derive")]
            ignored = SCHEDULING | (DERIVED if derive & 1 else set()) | (TRANSFORM if derive & 64 else set()) \
                      | ({"mrl"} if derive & DERIVE_TMRL else set())
            if ra[FIELDS.index("dirMode")] <= 1 or ra[FIELDS.index("mipMatrix")] >= 0:
                ignored |= ANGULAR
            elif ra[FIELDS.index("intraPredAngle")] <= 0:
                ignored |= POSITIVE
            for k, f in enumerate(FIELDS):
                if ra[k] != rb[k] and f not in ignored:
                    counts[f] = counts.get(f, 0) + 1
                    if shown < 8:
                        shown += 1
                        print(f"  POC {poc} block {r} (comp {ra[0]} at {ra[1]},{ra[2]} {ra[3]}x{ra[4]}, derive {derive}): {f} "
                              f"verification {ra[k]} ON {rb[k]}")
        bad |= bool(counts)
        print(f"POC {poc}: " + (f"fields differing {counts}" if counts else "the records agree"))
    sys.exit(1 if bad else 0)

main()
