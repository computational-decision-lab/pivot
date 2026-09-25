"""Freeze the confirmation protocol deterministically (no hand edits).

Preconditions checked here:
  * pilot_report.json says pilot_go == true
  * LORO pre-flight summary passes the calibration protocol's gate
  * calibration cohort has all 12 roots complete
Writes melting_v4_confirm.json with status FROZEN and sha256 of every code file and of the
calibration posterior spec. Refuses to overwrite an existing frozen file.
"""
import argparse, datetime, glob, hashlib, json, sys
from pathlib import Path


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, required=True)
    ap.add_argument("--pilot-report", type=Path, required=True)
    ap.add_argument("--loro-summary", type=Path, required=True)
    ap.add_argument("--calibration-protocol", type=Path, required=True)
    ap.add_argument("--calibration-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--also-hash", type=str, nargs="*", default=[],
                    help="extra glob(s) of source files to lock, e.g. code/highway_v1/*.py (Melting Pot needs none)")
    a = ap.parse_args()
    if a.out.exists():
        sys.exit(f"refusing to overwrite existing frozen protocol {a.out}")
    pilot = json.loads(a.pilot_report.read_text())
    assert pilot.get("pilot_go") is True, f"pilot did not pass: {pilot.get('pilot_checks')}"
    cal_protocol = json.loads(a.calibration_protocol.read_text())
    complete = [d for d in a.calibration_dir.glob("seed_*") if (d / "status.json").exists()
                and json.loads((d / "status.json").read_text()).get("status") == "complete"]
    assert sorted(int(d.name.split("_")[1]) for d in complete) == sorted(cal_protocol["seeds"]), "calibration cohort incomplete"
    loro = json.loads(a.loro_summary.read_text())
    hl = str(max(cal_protocol["adaptation_episodes"]) if "adaptation_episodes" in cal_protocol
             else max(cal_protocol["adaptation_steps"].values()))
    tbl = {(r["adaptation"], r["method"]): r for r in loro["method_table_cap192"]}
    kg = tbl[(int(hl), cal_protocol["primary_method"])]["mean_gain"]; un = tbl[(int(hl), cal_protocol["primary_comparator"])]["mean_gain"]
    nohf = tbl[(int(hl), "no_hf_v2")]["mean_gain"]
    gate = cal_protocol["preflight_gate_before_confirmation_freeze"]
    checks = {"loro_long_kg_minus_uniform_mean_positive": kg - un > 0,
              "loro_long_kg_not_much_worse_than_no_hf": (nohf - kg) <= float(gate["loro_long_kg_not_worse_than_no_hf_v2_by_more_than"]),
              "coverage_in_range": all(0.80 <= v <= 1.00 for v in loro["root_held_out_predictive_coverage_95"].values())}
    if not all(checks.values()):
        sys.exit(f"pre-flight gate FAILED: {checks}. STOP and report; do not tune.")
    draft = json.loads(a.draft.read_text())
    code_dir = Path(__file__).resolve().parent
    extra = [Path(p).resolve() for g in a.also_hash for p in glob.glob(g)]
    assert all(g == [] or glob.glob(g) for g in a.also_hash), f"--also-hash matched nothing: {a.also_hash}"
    draft.update(status="FROZEN", frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 draft_sha256=sha(a.draft), pilot_report_sha256=sha(a.pilot_report), loro_summary_sha256=sha(a.loro_summary),
                 preflight_checks=checks, calibration_protocol_sha256=sha(a.calibration_protocol),
                 calibration_root_summary_sha256={d.name: sha(d / "summary.json") for d in sorted(complete)},
                 source_file_hashes={str(p): sha(p) for p in sorted(list(code_dir.glob("*.py")) + extra)})
    a.out.write_text(json.dumps(draft, indent=2) + "\n")
    print("FROZEN_CONFIRM_PROTOCOL", sha(a.out))


if __name__ == "__main__":
    main()
