"""Multi-seed driver for highway_panel.py (one subprocess per seed; --resume skips complete seeds;
--seed-subset for multi-machine sharding). If the protocol is FROZEN and carries source_file_hashes,
every listed file must match before anything is launched."""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(obj, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--max-seconds", type=int, default=7200)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--seed-subset", type=str, default="")
    a = ap.parse_args()
    protocol = json.loads(a.protocol.read_text())
    assert protocol["status"] in ("FROZEN", "PILOT_FROZEN", "DISCOVERY"), protocol["status"]
    for path, h in protocol.get("source_file_hashes", {}).items():
        assert sha(path) == h, f"Frozen source changed: {path}"
    a.output.mkdir(parents=True, exist_ok=a.resume)
    write(a.output / "protocol.json", protocol)
    seeds = protocol["seeds"]
    if a.seed_subset:
        seeds = [int(x) for x in a.seed_subset.strip(",").split(",")]
        assert set(seeds) <= set(protocol["seeds"]), "subset must be within protocol seeds"
        write(a.output / "shard.json", {"seeds_on_this_machine": seeds})
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0", "SDL_VIDEODRIVER": "dummy"}
    panel = Path(__file__).resolve().parent / "highway_panel.py"

    def done(seed):
        s = a.output / f"seed_{seed}" / "status.json"
        return s.exists() and json.loads(s.read_text()).get("status") == "complete"

    def run(seed):
        if a.resume and done(seed):
            return {"seed": seed, "status": "complete", "skipped_already_complete": True}
        cmd = [sys.executable, str(panel), "--protocol", str(a.protocol.resolve()), "--seed", str(seed),
               "--output", str((a.output / f"seed_{seed}").resolve()), "--max-seconds", str(a.max_seconds)]
        with (a.output / f"seed_{seed}.log").open("a") as f:
            code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        st = a.output / f"seed_{seed}" / "status.json"
        status = json.loads(st.read_text()).get("status") if st.exists() else "failed"
        r = {"seed": seed, "status": status if code == 0 else ("timeout" if code == 3 else "failed"), "returncode": code}
        write(a.output / f"seed_{seed}.job.json", r)
        return r

    write(a.output / "status.json", {"status": "running", "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(run, seeds))
    n_ok = sum(r["status"] == "complete" for r in results)
    final = "complete" if n_ok == len(seeds) else ("timeout" if any(r["status"] == "timeout" for r in results) else "failed")
    write(a.output / "status.json", {"status": final, "complete": n_ok, "total": len(seeds), "jobs": results,
                                     "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({"status": final, "complete": n_ok, "total": len(seeds)}))
    sys.exit(0 if final == "complete" else 1)


if __name__ == "__main__":
    main()
