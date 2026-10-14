"""Print logical processors grouped by efficiency class (higher class = faster P-cores)."""
import ctypes
from ctypes import wintypes

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
n = wintypes.DWORD(0)
k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(n))
buf = ctypes.create_string_buffer(n.value)
assert k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(n))
off, classes = 0, {}
while off < n.value:
    rel, size = ctypes.c_uint32.from_buffer(buf, off).value, ctypes.c_uint32.from_buffer(buf, off + 4).value
    if rel == 0:  # RelationProcessorCore: Flags(1) EfficiencyClass(1) Reserved(20) GroupCount(2) GROUP_AFFINITY[]
        eff = buf.raw[off + 9]
        mask = ctypes.c_uint64.from_buffer(buf, off + 8 + 24).value
        classes.setdefault(eff, []).extend(i for i in range(64) if mask >> i & 1)
    off += size
for eff in sorted(classes):
    print(f"efficiency class {eff}: {len(classes[eff])} logical CPUs {classes[eff]}")
