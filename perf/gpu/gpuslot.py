# gpuslot.py: a machine-wide limit on concurrent GPU decoders (several sessions or agents verifying at once). The Intel
# driver compiles an intra_rec pipeline in 20-30 GB of commit (a decoder that compiles several grew to 33 GB), so 3
# cold decoders beside builds exhausted the 64 GB PC (2026-10-22: 177 GB of 178 GB committed). `with slot():` takes one
# of ECM_GPU_SLOTS (default 2) slots - directories C:/Code/ECM/runs/gpuslots/<n> made atomically, holding the owner's
# PID; a slot whose owner process is gone is taken over -, and only while the machine has ECM_GPU_MIN_FREE_GB (default
# 24) of physical memory and twice that of commit free; it frees the slot at the end.
import ctypes, os, threading, time
from contextlib import contextmanager

ROOT = 'C:/Code/ECM/runs/gpuslots'
SLOTS = int(os.environ.get('ECM_GPU_SLOTS', '2'))
MIN_FREE = float(os.environ.get('ECM_GPU_MIN_FREE_GB', '24')) * (1 << 30)


class _MemStatus(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def _memory_ok():
    m = _MemStatus()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys >= MIN_FREE and m.ullAvailPageFile >= 2 * MIN_FREE
_local = threading.Lock()


def _alive(pid):
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return bool(ok) and code.value == 259     # STILL_ACTIVE


def _take():
    os.makedirs(ROOT, exist_ok=True)
    while True:
        with _local:
            for n in range(SLOTS if _memory_ok() else 0):
                d = f'{ROOT}/{n}'
                try:
                    os.mkdir(d)
                except FileExistsError:
                    try:
                        pid = int(open(f'{d}/pid').read())
                    except (OSError, ValueError):
                        pid = 0    # (being taken right now, or its owner died between mkdir and the pid file)
                        try:
                            if time.time() - os.path.getmtime(d) < 60:
                                continue
                        except OSError:
                            continue
                    if pid and _alive(pid):
                        continue
                    try:
                        os.remove(f'{d}/pid')
                        os.rmdir(d)
                    except OSError:
                        pass
                    continue
                open(f'{d}/pid', 'w').write(str(os.getpid()))
                return d
        time.sleep(2)


@contextmanager
def slot():
    d = _take()
    try:
        yield
    finally:
        try:
            os.remove(f'{d}/pid')
            os.rmdir(d)
        except OSError:
            pass
