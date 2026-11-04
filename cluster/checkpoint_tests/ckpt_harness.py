"""Shared driver for the checkpoint / resume tests.

Each package test defines

    production(n_iters)  -> array (n_iters, 2): (loc, scale) after every iteration,
                            produced by the *current* simpleVI code path
    make_impl()          -> a Resumable object (fresh, in the current process)

A Resumable has
    init()               start a brand-new fit (same initialisation as production)
    run_chunk(m)         advance m iterations, return array (m, 2) of (loc, scale)
    get_state()          everything needed to continue the fit (picklable)
    set_state(state)     restore that state into a freshly made impl

The test then checks two things:

  [resume]      a fit that is SIGKILLed at arbitrary points and resumed in a new
                process from its last checkpoint gives exactly (bit for bit) the
                same trajectory as the same resumable fit run without interruption.
  [production]  the uninterrupted resumable fit equals the current production
                code path bit for bit (if not, results from the resumable code
                are not interchangeable with results already produced).
"""
import argparse
import json
import os
import pickle
import random
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------- checkpoint io
def save_checkpoint(path, payload):
    """Write atomically: a kill mid-write leaves the previous checkpoint intact."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- one (resumable) worker process
def run_worker(make_impl, n_iters, ckpt_path, every, die_after=None, sleep=0.0):
    """Run (or resume) the resumable fit up to n_iters, checkpointing every `every` iterations.

    die_after: SIGKILL this process right after that iteration (no checkpoint written for it
               unless it is also a checkpoint boundary), to simulate preemption.
    """
    impl = make_impl()
    ckpt_path = Path(ckpt_path)
    if ckpt_path.exists():
        ck = load_checkpoint(ckpt_path)
        impl.set_state(ck["state"])
        done, traj = ck["done"], list(ck["traj"])
        print(f"RESUMED at {done}", flush=True)
    else:
        impl.init()
        done, traj = 0, []
        print("FRESH start", flush=True)

    while done < n_iters:
        next_ckpt = (done // every + 1) * every
        stop = min(next_ckpt, n_iters)
        if die_after is not None and done < die_after < stop:
            stop = die_after
        rows = np.asarray(impl.run_chunk(stop - done))
        traj.extend(rows)
        done = stop
        if done % every == 0 or done == n_iters:
            save_checkpoint(ckpt_path, {"done": done, "traj": np.asarray(traj), "state": impl.get_state()})
        print(f"PROGRESS {done}", flush=True)
        if die_after is not None and done == die_after:
            os.kill(os.getpid(), signal.SIGKILL)
        if sleep:
            time.sleep(sleep)
    np.save(ckpt_path.with_suffix(".final.npy"), np.asarray(traj))
    print("FINISHED", flush=True)


# --------------------------------------------------------------------------- orchestration
def _child_cmd(script, n_iters, ckpt, every, die_after=None, sleep=0.0):
    cmd = [sys.executable, "-u", str(script), "--worker", "--n-iters", str(n_iters),
           "--ckpt", str(ckpt), "--every", str(every), "--sleep", str(sleep)]
    if die_after is not None:
        cmd += ["--die-after", str(die_after)]
    return cmd


def run_scenario(script, n_iters, workdir, name, every, die_afters=(), random_kills=0, sleep=0.0, rng=None):
    """Run a worker repeatedly (killing it as requested) until it finishes; return its trajectory + log."""
    ckpt = Path(workdir) / f"{name}.pkl"
    final = ckpt.with_suffix(".final.npy")
    log = []
    plan = [("die_after", d) for d in die_afters] + [("random", None)] * random_kills
    for kind, value in plan + [("finish", None)]:
        if kind == "die_after":
            res = subprocess.run(_child_cmd(script, n_iters, ckpt, every, die_after=value),
                                 capture_output=True, text=True)
            log.append(f"die_after={value}: exit={res.returncode} "
                       f"{[l for l in res.stdout.splitlines() if not l.startswith('PROGRESS')]}")
            if res.returncode != -signal.SIGKILL:
                raise RuntimeError(f"{name}: worker did not die as planned\n{res.stdout}\n{res.stderr}")
        elif kind == "random":
            # kill from the outside at an arbitrary moment (possibly mid-step or mid-checkpoint-write)
            errlog = open(Path(workdir) / f"{name}.stderr.txt", "a")
            proc = subprocess.Popen(_child_cmd(script, n_iters, ckpt, every, sleep=sleep),
                                    stdout=subprocess.PIPE, stderr=errlog, text=True)
            start_line = proc.stdout.readline().strip()
            target = rng.randint(1, max(2, n_iters // 2))
            last = None
            for line in proc.stdout:
                if line.startswith("PROGRESS"):
                    last = int(line.split()[1])
                    if last >= target:
                        time.sleep(rng.uniform(0, max(sleep, 1e-3)))
                        proc.kill()
                        break
            proc.wait()
            errlog.close()
            log.append(f"random kill: '{start_line}', killed after PROGRESS {last} (target {target}), "
                       f"exit={proc.returncode}")
        else:
            res = subprocess.run(_child_cmd(script, n_iters, ckpt, every), capture_output=True, text=True)
            log.append(f"finish: exit={res.returncode} "
                       f"{[l for l in res.stdout.splitlines() if not l.startswith('PROGRESS')]}")
            if res.returncode != 0:
                raise RuntimeError(f"{name}: final worker failed\n{res.stdout}\n{res.stderr}")
    return np.load(final), log


def compare(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return {"identical": False, "note": f"shape {a.shape} vs {b.shape}"}
    identical = bool(np.array_equal(a, b, equal_nan=True))
    diff = np.abs(a - b)
    first = int(np.argmax(np.any(diff > 0, axis=1))) + 1 if not identical else None
    return {"identical": identical, "max_abs_diff": float(np.nanmax(diff)) if diff.size else 0.0,
            "first_differing_iter": first}


def main(script, package, variants, n_iters_default=1000):
    """variants: {name: (production_fn, make_impl)}"""
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--variant", default=None)
    p.add_argument("--n-iters", type=int, default=n_iters_default)
    p.add_argument("--ckpt")
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--die-after", type=int, default=None)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--out", default=None, help="write the JSON report here")
    args = p.parse_args()

    if args.worker:
        variant = Path(args.ckpt).name.split("__")[0]
        run_worker(variants[variant][1], args.n_iters, args.ckpt, args.every, args.die_after, args.sleep)
        return

    rng = random.Random(12345)
    report = {"package": package, "n_iters": args.n_iters, "python": sys.executable, "variants": {}}
    workdir = Path(tempfile.mkdtemp(prefix=f"ckpt_{package}_"))
    n = args.n_iters
    for vname, (production_fn, _) in variants.items():
        print(f"\n===== {package} / {vname} =====", flush=True)
        t0 = time.time()
        prod = np.asarray(production_fn(n))
        t_prod = time.time() - t0
        print(f"production: {n} iters in {t_prod:.1f}s, final (loc, scale) = {prod[-1]}", flush=True)

        t0 = time.time()
        straight, _ = run_scenario(script, n, workdir, f"{vname}__straight", every=n)
        t_straight = time.time() - t0
        t0 = time.time()
        straight_ck1, _ = run_scenario(script, n, workdir, f"{vname}__straight_every1", every=1)
        t_every1 = time.time() - t0

        scenarios = {
            "kill_once_every1":        dict(every=1,   die_afters=[137]),
            "kill_twice_every1":       dict(every=1,   die_afters=[1, n - 1]),
            "kill_between_ckpts_e100": dict(every=100, die_afters=[350, 777]),
            "random_kills_every1":     dict(every=1,   random_kills=3, sleep=0.002),
        }
        results = {
            "production_vs_resumable_uninterrupted": compare(prod, straight),
            "every1_checkpointing_vs_no_checkpointing": compare(straight_ck1, straight),
        }
        logs = {}
        for sname, kw in scenarios.items():
            traj, log = run_scenario(script, n, workdir, f"{vname}__{sname}", rng=rng, **kw)
            results[f"resume:{sname}"] = compare(traj, straight)
            logs[sname] = log
        report["variants"][vname] = {
            "results": results, "logs": logs,
            "seconds": {"production": t_prod, "resumable_no_ckpt_incl_startup": t_straight,
                        "resumable_ckpt_every_iter_incl_startup": t_every1},
        }
        for k, v in results.items():
            flag = "PASS" if v["identical"] else "DIFF"
            extra = "" if v["identical"] else f"  max|diff|={v.get('max_abs_diff')}, first differs at iter {v.get('first_differing_iter')}"
            print(f"  [{flag}] {k}{extra}", flush=True)
        for sname, log in logs.items():
            for line in log:
                print(f"      {sname}: {line}", flush=True)

    out = Path(args.out) if args.out else Path(script).parent / "reports" / f"report_{package}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport written to {out}\n(work dir {workdir})")
