# gpuslot.py: a machine-wide limit on concurrent GPU decoders (several sessions or agents verifying at once). The Intel
# driver compiles an intra_rec pipeline in up to ~20 GB of RAM, so more than 3 cold decoders at a time exhaust a 64 GB
# PC. `with slot():` takes one of ECM_GPU_SLOTS (default 3) slots - directories C:/Code/ECM/runs/gpuslots/<n> made
# atomically, holding the owner's PID; a slot whose owner process is gone is taken over - and frees it at the end.
import ctypes, os, threading, time
from contextlib import contextmanager

ROOT = 'C:/Code/ECM/runs/gpuslots'
SLOTS = int(os.environ.get('ECM_GPU_SLOTS', '3'))
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
            for n in range(SLOTS):
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
