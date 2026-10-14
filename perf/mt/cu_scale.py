#!/usr/bin/env python3
"""CuThreads scaling: each test at each thread count, one run at a time on 4 P-cores (8 above 4 threads), with the
peak commit of the process (job object).

usage: cu_scale.py EXE test[,test...] count[,count...]      e.g. cu_scale.py bin/x/EncoderApp.exe ai_foreman 1,2,4,8
"""
import ctypes
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import bench  # noqa: E402

exe, tests, counts = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(",")
peak = {}


class ExtLimit(ctypes.Structure):   # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32), ("SchedulingClass", ctypes.c_uint32),
                ("ReadOperationCount", ctypes.c_uint64), ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64), ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64), ("OtherTransferCount", ctypes.c_uint64),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


orig_close = bench.CpuJob.close


def close(self):
    """records the job's peak commit before the job object goes"""
    info = ExtLimit()
    if self.k32.QueryInformationJobObject(ctypes.c_void_p(self.job), 9, ctypes.byref(info), ctypes.sizeof(info), None):
        peak["last"] = info.PeakJobMemoryUsed
    orig_close(self)


bench.CpuJob.close = close
mask_now = [0]


def pin(proc, cpu):
    if not ctypes.windll.kernel32.SetProcessAffinityMask(int(proc._handle), ctypes.c_size_t(mask_now[0])):
        raise OSError("SetProcessAffinityMask failed")


bench.pin = pin
tag = os.path.basename(os.path.dirname(os.path.abspath(exe)))
for t in tests:
    base = None
    for n in counts:
        mask_now[0] = 0xC03C3 if int(n) > 4 else 0xC3   # the 265K's P-cores: 8, or logical CPUs 0, 1, 6, 7
        out = os.path.join(bench.ROOT, "runs", "cuscale", f"{tag}_{t}_{n}")
        os.makedirs(out, exist_ok=True)
        _, r = bench.run_test(exe, out, t, [f"--CuThreads={n}"] if n != "1" else [], fixed_core=0)
        base = base or r["time"]
        print(f"{t:12s} CuThreads={n:2s} {r['time']:8.2f} s  {base / r['time']:5.2f}x  cpu {r['cpu']:8.2f} s  "
              f"peak {peak.get('last', 0) / 2**20:7.0f} MB  {r['md5']}", flush=True)
