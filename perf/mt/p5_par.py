#!/usr/bin/env python3
"""Runs several p5_test.py jobs at the same time, each on its own CPU mask: the frame-thread output checks of a step in
a few minutes instead of one after another (the times are then not comparable; time on an idle machine).
usage: p5_par.py OUT EXE TAG JOB [JOB ...]
       JOB = "hexmask:tests:threads[:extra encoder args]", e.g. C3:ra65:8  3C00:ra33:4:--FramePeriods=2
The references are the "ref" lines of p5_refs.txt (single-thread encodes identical to stock)."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = os.path.join(HERE, "p5_refs.txt")


def main():
    out, exe, tag = sys.argv[1:4]
    procs = []
    for i, job in enumerate(sys.argv[4:]):
        mask, tests, threads, *extra = job.split(":")
        extra = " ".join(extra).split()
        job_tag = tag + "".join("_" + a.lstrip("-").replace("=", "") for a in extra)
        log = open(f"{out}.{i}", "w")
        env = dict(os.environ, P5_MASK=mask, PYTHONUTF8="1")
        p = subprocess.Popen([sys.executable, os.path.join(HERE, "p5_test.py"), REFS, exe, job_tag, tests, threads]
                             + extra, stdout=log, stderr=subprocess.STDOUT, env=env)
        procs.append((p, log))
    with open(out, "w") as o:
        for i, (p, log) in enumerate(procs):
            p.wait()
            log.close()
            with open(f"{out}.{i}") as part:
                o.write(part.read())
            os.remove(f"{out}.{i}")
        o.write("DONE\n")
    with open(out) as o:
        print(o.read(), end="")


if __name__ == "__main__":
    main()
