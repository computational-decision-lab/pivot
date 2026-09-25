"""Launch a v4 cohort (pilot / calibration / confirmation) with N parallel single-thread workers.

Stops the whole cohort on the first worker failure (no silent partial cohorts).
Re-running with --resume continues unfinished seeds only (episode cache is content-addressed).
"""
import argparse, concurrent.futures, datetime, hashlib, json, os, signal, subprocess, sys, threading, time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(p, s):
    q = p.with_suffix(".tmp"); q.write_text(json.dumps(s, indent=2) + "\n"); q.replace(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True); p.add_argument("--crossbench-root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=20); p.add_argument("--max-seconds", type=int, default=21600)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed-subset", type=str, default="", help="comma-separated subset of protocol seeds to run on THIS machine (sharding); protocol file itself is unchanged")
    a = p.parse_args()
    protocol = json.loads(a.protocol.read_text())
    assert protocol["status"] in ("FROZEN", "PILOT_FROZEN"), "protocol must be frozen before launch"
    assert 1 <= a.workers <= 22 and 0 < a.max_seconds <= 43200
    for path, h in protocol.get("source_file_hashes", {}).items():
        assert sha(path) == h, f"Frozen source changed: {path}"
    a.output.mkdir(parents=True, exist_ok=a.resume)
    write(a.output / "protocol.json", protocol)
    seeds = protocol["seeds"]
    if a.seed_subset:
        seeds = [int(x) for x in a.seed_subset.split(",")]
        assert set(seeds) <= set(protocol["seeds"]), "subset must be within frozen protocol seeds"
        write(a.output / "shard.json", {"seeds_on_this_machine": seeds})
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", TF_ENABLE_ONEDNN_OPTS="0", TF_CPP_MIN_LOG_LEVEL="3", PYTHONHASHSEED="0")
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"]:
        env[key] = "1"
    stopped = threading.Event(); lock = threading.Lock(); active = {}; results = []; start = time.monotonic()

    def done(seed):
        s = a.output / f"seed_{seed}" / "status.json"
        return s.exists() and json.loads(s.read_text()).get("status") == "complete"

    def run(seed):
        if a.resume and done(seed):
            return {"seed": seed, "status": "complete", "returncode": 0, "skipped_already_complete": True}
        if stopped.is_set():
            return {"seed": seed, "status": "not_started_after_failure"}
        remaining = max(1, int(a.max_seconds - (time.monotonic() - start)))
        cmd = [sys.executable, str(root / "melting_native_v4_hetero.py"), "--protocol", str(a.protocol.resolve()), "--seed", str(seed),
               "--output", str((a.output / f"seed_{seed}").resolve()), "--model-path", str(a.model_path.resolve()),
               "--crossbench-root", str(a.crossbench_root.resolve()), "--max-seconds", str(remaining)] + (["--resume"] if a.resume else [])
        with (a.output / f"seed_{seed}.log").open("a") as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            with lock: active[seed] = process
            try:
                code = process.wait(timeout=remaining + 30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try: code = process.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); code = process.wait()
            finally:
                with lock: active.pop(seed, None)
        r = {"seed": seed, "status": "complete" if code == 0 else "failed", "returncode": code}
        write(a.output / f"seed_{seed}.job.json", r)
        return r

    write(a.output / "status.json", {"status": "running", "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "jobs": []})
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run, s) for s in seeds]):
            r = future.result(); results.append(r)
            if r["status"] == "failed":
                stopped.set()
                with lock:
                    for proc in active.values():
                        if proc.poll() is None:
                            try: os.killpg(proc.pid, signal.SIGTERM)
                            except ProcessLookupError: pass
            write(a.output / "status.json", {"status": "running" if not stopped.is_set() else "stopping_after_failure", "jobs": results})
            print(json.dumps(r), flush=True)
    ok = all(r["status"] == "complete" for r in results)
    write(a.output / "status.json", {"status": "complete" if ok else "incomplete", "jobs": results, "seconds": time.monotonic() - start})
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
