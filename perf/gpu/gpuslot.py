# gpuslot.py: a machine-wide limit on memory-heavy jobs (several sessions or agents verifying and building at once).
# The user's budget: the system's committed memory stays below 60 GB (ECM_MEM_LIMIT_GB). The Intel driver compiles an
# intra_rec pipeline in 20-35 GB of commit (a decoder that compiled several grew to 33 GB), so one cold decoder at a
# time (2026-10-22: 3 decoders and the agents' glslang -O builds reached 177 GB). `with slot():` takes one of
# ECM_GPU_SLOTS (default 2) slots - directories C:/Code/ECM/runs/gpuslots/<n> made atomically, holding the owner's PID;
# a slot whose owner process is gone is taken over - once the committed memory plus `need` GB (ECM_GPU_NEED_GB,
# default 10 for a decoder) fits the budget, and frees it at the end. As a command it runs one under a slot of its own
# kind (builds: `python perf/gpu/gpuslot.py --kind build --need 12 -- powershell -ExecutionPolicy Bypass -File
# perf\build.ps1 ...`), so a build waits for memory too, but not for a decoder's slot. run() runs a decoder under a slot
# and stops it (its own child) where the machine's commit passes the budget less 1 GB, then runs it again once memory
# allows: one cold intra_rec pipeline peaked at 16.9 GB of commit (MPM + CCLM, 760 s), larger levels take more.
import ctypes, os, subprocess, sys, tempfile, threading, time
from contextlib import contextmanager

ROOT = 'C:/Code/ECM/runs/gpuslots'
SLOTS = int(os.environ.get('ECM_GPU_SLOTS', '2'))
LIMIT = float(os.environ.get('ECM_MEM_LIMIT_GB', '60')) * (1 << 30)
NEED = float(os.environ.get('ECM_GPU_NEED_GB', '10'))
_local = threading.Lock()


class _MemStatus(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def committed():
    m = _MemStatus()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullTotalPageFile - m.ullAvailPageFile


def _alive(pid):
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return bool(ok) and code.value == 259     # STILL_ACTIVE


def _first_in_line(kind, ticket):
    """whether ticket is the oldest live waiter's for slots of this kind (FIFO: a freed slot went to whichever waiter
    polled first, so a decoder thread could wait while another thread of the same partial.py took every slot again)"""
    oldest = None
    for name in os.listdir(f'{ROOT}/queue'):
        k, t, pid, tid = name.rsplit('-', 3)
        if k != kind:
            continue
        if not _alive(int(pid)):
            try:
                os.remove(f'{ROOT}/queue/{name}')
            except OSError:
                pass
            continue
        if oldest is None or (int(t), name) < oldest:
            oldest = (int(t), name)
    return oldest is None or oldest[1] == ticket


def _take(kind, slots, need):
    os.makedirs(f'{ROOT}/queue', exist_ok=True)
    ticket = f'{kind or "dec"}-{time.time_ns()}-{os.getpid()}-{threading.get_ident()}'
    open(f'{ROOT}/queue/{ticket}', 'w').close()
    try:
        return _take_in_turn(kind, slots, need, ticket)
    finally:
        try:
            os.remove(f'{ROOT}/queue/{ticket}')
        except OSError:
            pass


def _take_in_turn(kind, slots, need, ticket):
    while True:
        with _local:
            first = _first_in_line(kind or 'dec', ticket)
            for n in range(slots if first and committed() + need * (1 << 30) <= LIMIT else 0):
                d = f'{ROOT}/{kind}{n}'
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
        time.sleep(5)


@contextmanager
def slot(kind='', slots=None, need=None):
    d = _take(kind, SLOTS if slots is None else slots, NEED if need is None else need)
    try:
        yield
    finally:
        try:
            os.remove(f'{d}/pid')
            os.rmdir(d)
        except OSError:
            pass


def run(cmd, env=None, need=None):
    """cmd under a slot with the memory watchdog: (returncode, stdout, stderr)"""
    while True:
        with slot(need=need), tempfile.TemporaryFile('w+') as out, tempfile.TemporaryFile('w+') as err:
            p = subprocess.Popen(cmd, stdout=out, stderr=err, text=True, env=env)
            killed = False
            while p.poll() is None:
                if committed() > LIMIT - (1 << 30):
                    p.kill()
                    p.wait()
                    killed = True
                    break
                time.sleep(2)
            if not killed:
                out.seek(0)
                err.seek(0)
                return p.returncode, out.read(), err.read()
        print(f'[gpuslot] {os.path.basename(cmd[0])} {" ".join(cmd[1:3])}: stopped at the memory budget, again later',
              flush=True)
        time.sleep(60)


if __name__ == '__main__':   # gpuslot.py [--kind K] [--need GB] -- command...
    args = sys.argv[1:]
    kind, need = 'job', NEED
    while args and args[0] != '--':
        if args[0] == '--kind':
            kind = args[1]
        elif args[0] == '--need':
            need = float(args[1])
        args = args[2:]
    with slot(kind, 1, need):
        sys.exit(subprocess.run(args[1:]).returncode)
