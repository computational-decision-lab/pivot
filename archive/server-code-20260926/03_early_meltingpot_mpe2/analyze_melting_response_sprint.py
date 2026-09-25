#!/usr/bin/env python3
"""Read-only scientific analysis of the bounded response-learning sprint.

Fresh response training and an optional reanalysis of earlier evidence have
separate provenance, CSVs, plots and conclusions. This script never runs an
environment, trains a model, or selects a PIVOT winner.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import tarfile
import tempfile

import numpy as np

VERSION = "melting_response_sprint_v1"
STATISTICS_SEED = 20260916
BOOTSTRAP_DRAWS = 10000


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message):
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and
            math.isclose(float(a), float(b), abs_tol=1e-8, rel_tol=1e-8), message)


def sigmoid(x):
    return 1/(1+math.exp(-x)) if x >= 0 else math.exp(x)/(1+math.exp(x))


def weighted(means, p, q, role):
    require(0 <= p <= 1 and 0 <= q <= 1, "invalid mixture probability")
    return float(sum((p if a else 1-p)*(q if b else 1-q)*means[f"{a}{b}"][role]
                     for a in (0, 1) for b in (0, 1)))


def mean_ci(values, name):
    x = np.asarray(values, dtype=float)
    require(x.ndim == 1 and len(x) and np.isfinite(x).all(), "invalid bootstrap input")
    seed = int(hashlib.sha256(f"{STATISTICS_SEED}:{name}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    samples = x[rng.integers(len(x), size=(BOOTSTRAP_DRAWS, len(x)))].mean(1)
    return {"mean": float(x.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist(), "n_seeds": len(x)}


def csv_rows(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["status"]
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes = {}

    def load(self, path, expected_hash=None):
        path = Path(path).resolve()
        require(path.is_relative_to(self.root), "evidence path escaped batch root")
        h = sha(path)
        self.hashes[str(path.relative_to(self.root))] = h
        if expected_hash is not None:
            require(h == expected_hash, f"changed evidence: {path.name}")
        return read(path)


def verify_block(evidence, path, native, replicas, config):
    block = evidence.load(path)
    require(block["complete"] and not block["training_reads_this_block"], "invalid evaluation block")
    require(block["replicates_per_stratum"] == replicas and len(block["rows"]) == 4*replicas,
            "wrong conditional evaluation count")
    groups = {}
    for row in block["rows"]:
        require(row == native[row["episode_id"]], "evaluation block differs from persisted native episode")
        sig = row["signature"]
        require(sig["sampling_mode"] == "forced_evaluation_stratum" and sig["focal_logit"] is None and sig["response_logit"] is None,
                "evaluation stratum was not exactly forced")
        a, b = sig["forced_specialists"]
        r = sig["key"][-1]
        require(a in (0, 1) and b in (0, 1) and 0 <= r < replicas and (a,b,r) not in groups,
                "duplicate/invalid evaluation stratum")
        require(row["sampled_specialists"] == [a, b], "forced Specialist identity mismatch")
        groups[a,b,r] = row
    for a in (0, 1):
        for b in (0, 1):
            for role in ("focal_return", "response_return"):
                close(block["conditional_return_means"][f"{a}{b}"][role],
                      np.mean([groups[a,b,r][role] for r in range(replicas)]), "conditional mean is not the raw-return mean")
    close(block["actual_env_frames"], sum(row["env_steps"] for row in block["rows"]), "evaluation frame count mismatch")
    return block


def inspect_panel(evidence, folder, seed, batch_protocol, source_dir):
    info = evidence.load(folder/"protocol.json")
    config = info["config"]
    require(info["version"] == VERSION and config["seed"] == seed, "wrong sprint version/seed")
    require(config["target_probabilities"] == [.25, .5, .75] and config["checkpoints"] == [0,4,16,32], "wrong target/checkpoint grid")
    require(config["train_episodes"] == 32 and config["horizon"] == 2500 and config["stratum_replicates"] == 4,
            "engineering smoke or changed response budget")
    require(config["baseline"] == "running_mean" and config["initial_response_logit"] == 0, "changed learner initialization")
    for name, expected in info["sources_sha256"].items():
        path = Path(source_dir)/Path(name).name
        require(path.is_file() and sha(path) == expected, f"source mismatch: {name}")
    state, summary = evidence.load(folder/"status.json"), evidence.load(folder/"summary.json")
    require(state["status"] == summary["status"] == "complete", "unfinished panel")
    require(summary["complete"] and summary["actual_complete_episodes"] == 128
            and summary["actual_training_episodes"] == 96
            and summary["actual_evaluation_episodes"] == 32
            and summary["actual_partial_episodes"] == 0, "summary episode accounting mismatch")
    require(summary["protocol_sha256"] == sha(folder/"protocol.json"), "protocol seal mismatch")
    costs, registry = evidence.load(folder/"episode_costs.json"), evidence.load(folder/"seeds.json")
    require(len(costs) == 128 and all(c["complete"] for c in costs.values()), "expected 96 training + 32 evaluation episodes")
    require(len(set(registry.values())) == len(registry), "seed registry collision")
    native, contexts, training_contexts, evaluation_contexts = {}, set(), set(), set()
    counts, frames = Counter(), Counter()
    for rel, cost in costs.items():
        row = evidence.load(folder/rel)
        sig = row["signature"]
        require(row["complete"] and not row.get("event_parse_errors"), "incomplete/invalid native episode")
        require(row["episode_id"] == digest(sig["key"])[:24] and row["train_seed"] == seed,
                "native episode identity mismatch")
        close(row["env_steps"], cost["env_steps"], "native frame ledger mismatch")
        require(0 < row["env_steps"] <= 2500 and cost["key"] == sig["key"], "invalid episode length/key")
        require(all(math.isfinite(row[k]) for k in ("focal_return", "response_return")), "nonfinite native return")
        template = sig["template"]
        require(registry[json.dumps([*template,"environment"])] == sig["environment_seed"], "unregistered environment seed")
        for role in (0,1):
            require(registry[json.dumps([*template,"policy",role])] == sig["policy_seeds"][role], "unregistered low-level seed")
        context = (sig["environment_seed"], *sig["policy_seeds"])
        contexts.add(context)
        if sig["sampling_mode"] == "sampled_training":
            category = "training"
            training_contexts.add(context)
            for role, name in enumerate(("focal", "response")):
                uniform_seed = registry[json.dumps([*template,"mixture",role])]
                require(uniform_seed == sig["mixture_uniform_seeds"][role], "training mixture seed mismatch")
                u = float(np.random.default_rng(uniform_seed).random())
                close(sig["mixture_uniforms"][role], u, "training mixture draw mismatch")
                require(row["sampled_specialists"][role] == int(u < sigmoid(sig[f"{name}_logit"])), "incorrect sampled skill")
        else:
            require(sig["sampling_mode"] == "forced_evaluation_stratum", "unknown rollout role")
            category = "evaluation"
            evaluation_contexts.add(context)
        native[row["episode_id"]] = row
        counts[category] += 1
        frames[category] += row["env_steps"]
    require(counts == {"training":96,"evaluation":32} and not training_contexts & evaluation_contexts,
            "training/evaluation overlap or wrong count")
    require(len(training_contexts) == 32 and len(evaluation_contexts) == 8,
            "paired target training or evaluation random templates changed")
    close(summary["actual_env_frames"], sum(frames.values()), "summary frame ledger mismatch")
    frozen = evidence.load(folder/"freeze_training.json")
    require(frozen["protocol_sha256"] == sha(folder/"protocol.json")
            and frozen["config_sha256"] == digest(config)
            and summary["training_seal_sha256"] == sha(folder/"freeze_training.json")
            and frozen["all_training_frozen_before_evaluation"]
            and not frozen["training_reads_evaluation"], "training freeze contract mismatch")
    # The exact serialization is checked below when a checkpoint is used; all
    # training histories are independently reconstructed from saved native rows.
    histories, checkpoints = {}, {}
    for target, p in enumerate(config["target_probabilities"]):
        path = folder/"training"/(digest(("training",target))[:24]+".json")
        history = evidence.load(path)
        require(history["complete"] and len(history["history"]) == 32 and history["role"] == "response", "incomplete response learning")
        close(history["initial_logit"], 0, "wrong initial response policy")
        close(sigmoid(history["other_logit"]), p, "wrong frozen focal target")
        theta, previous_sum = 0.0, 0.0
        checkpoints[target,0] = theta
        for i, step in enumerate(history["history"]):
            row = native[step["episode_id"]]
            require(row["signature"]["sampling_mode"] == "sampled_training" and row["signature"]["key"] == ["training",target,i],
                    "learning history references wrong trajectory")
            close(row["signature"]["response_logit"], theta, "response update not inherited")
            close(sigmoid(row["signature"]["focal_logit"]), p, "focal changed during response learning")
            scaled = float(np.clip(row["response_return"]/config["reward_scale"],-config["reward_clip"],config["reward_clip"]))
            baseline = previous_sum/i if i else 0.0
            gradient = (scaled-baseline)*(row["sampled_specialists"][1]-sigmoid(theta))
            after = float(np.clip(theta+config["response_learning_rate"]*gradient,-config["logit_clip"],config["logit_clip"]))
            for key, value in (("native_return",row["response_return"]),("baseline_before",baseline),("gradient",gradient),("logit_after",after)):
                close(step[key], value, "response REINFORCE update mismatch")
            theta, previous_sum = after, previous_sum+scaled
            if i+1 in config["checkpoints"]:
                checkpoints[target,i+1] = theta
        close(history["final_logit"],theta,"wrong final learned state")
        histories[target] = {"path":str(path.relative_to(folder)),"sha256":sha(path)}
    freeze_text = json.dumps(frozen, sort_keys=True)
    require(all(value["sha256"] in freeze_text for value in histories.values()), "training freeze omitted/changed a history")
    blocks = {name:verify_block(evidence,folder/"evaluation"/f"block_{name}.json",native,4,config) for name in ("A","B")}
    require(all(block["signature"]["frozen_context"]["training_seal_sha256"] == sha(folder/"freeze_training.json")
                and block["signature"]["frozen_context"]["all_probabilities_frozen_before_block"]
                for block in blocks.values()), "evaluation does not reference frozen training")
    require(set(blocks["A"]["evaluation_seeds"]).isdisjoint(blocks["B"]["evaluation_seeds"]), "evaluation blocks share seeds")
    rows = evidence.load(folder/"results.json")
    require(summary["results_sha256"] == sha(folder/"results.json"), "result file seal mismatch")
    require(len(rows) == 24, "incomplete checkpoint/target/block evaluation grid")
    keys = {(r["target_id"],r["training_episodes"],r["eval_block"]) for r in rows}
    expected_keys = {(j,t,b) for j in range(3) for t in config["checkpoints"] for b in ("A","B")}
    require(keys == expected_keys, "duplicate/missing checkpoint result")
    verified = []
    for row in rows:
        target,t,b = row["target_id"],row["training_episodes"],row["eval_block"]
        p,q = config["target_probabilities"][target],sigmoid(checkpoints[target,t])
        block = blocks[b]
        means = block["conditional_return_means"]
        require(row["seed"] == seed and row["evaluation_block_sha256"] == sha(folder/"evaluation"/f"block_{b}.json"), "checkpoint evaluation provenance mismatch")
        expected = {"target_probability":p,"response_probability":q,"q_initial":.5,
                    "focal_return":weighted(means,p,q,"focal_return"),"response_return":weighted(means,p,q,"response_return"),
                    "frozen_focal_return":weighted(means,p,.5,"focal_return"),"frozen_response_return":weighted(means,p,.5,"response_return")}
        expected["response_own_gain"] = expected["response_return"]-expected["frozen_response_return"]
        expected["focal_response_effect"] = expected["focal_return"]-expected["frozen_focal_return"]
        expected["stag_minus_hare_response_advantage"] = weighted(means,p,1,"response_return")-weighted(means,p,0,"response_return")
        for field,value in expected.items():
            close(row[field],value,f"checkpoint estimate mismatch: {field}")
        verified.append(dict(row))
    return {"seed":seed,"config":config,"rows":verified,"blocks":blocks,"contexts":contexts,
            "counts":dict(counts),"frames":dict(frames),"sources_sha256":info["sources_sha256"],"freeze_sha256":sha(folder/"freeze_training.json")}


def summarize_new(panels, protocol):
    raw = [row for panel in panels for row in panel["rows"]]
    seeds = sorted(p["seed"] for p in panels)
    fields = ("response_probability","response_own_gain","response_return","frozen_response_return","focal_response_effect","stag_minus_hare_response_advantage")
    seed_rows, probe_rows, consistency_rows = [],[],[]
    for panel in panels:
        lookup = {(r["target_id"],r["training_episodes"],r["eval_block"]):r for r in panel["rows"]}
        for j,p in enumerate(panel["config"]["target_probabilities"]):
            for t in panel["config"]["checkpoints"]:
                a,b = lookup[j,t,"A"],lookup[j,t,"B"]
                out = {"seed":panel["seed"],"target_probability":p,"training_episodes":t,
                       **{field:float(np.mean([a[field],b[field]])) for field in fields}}
                out["probability_change"] = out["response_probability"]-.5
                seed_rows.append(out)
                if t == 32:
                    consistency_rows.append({"seed":panel["seed"],"target_probability":p,
                                             "own_gain_block_A":a["response_own_gain"],"own_gain_block_B":b["response_own_gain"],
                                             "probability_change":out["probability_change"]})
                if p == .5:
                    continue
                for block_name in ("A","B"):
                    r,reference = lookup[j,t,block_name],lookup[1,t,block_name]
                    proxy = r["frozen_focal_return"]-reference["frozen_focal_return"]
                    adapted = r["focal_return"]-reference["focal_return"]
                    probe_rows.append({"seed":panel["seed"],"target_probability":p,"training_episodes":t,"eval_block":block_name,
                                       "frozen_probe_gain":proxy,"responsive_probe_gain":adapted,"response_effect":adapted-proxy})
    curves = []
    for p in (.25,.5,.75):
        for t in (0,4,16,32):
            rows = [r for r in seed_rows if r["target_probability"]==p and r["training_episodes"]==t]
            curves.append({"target_probability":p,"training_episodes":t,
                           **{field:mean_ci([r[field] for r in rows],f"fresh:{p}:{t}:{field}") for field in (*fields,"probability_change")}})
    pooled = []
    for t in (0,4,16,32):
        record = {"training_episodes":t}
        for field in fields:
            values = [np.mean([r[field] for r in seed_rows if r["seed"]==seed and r["training_episodes"]==t]) for seed in seeds]
            record[field] = mean_ci(values,f"fresh:pooled:{t}:{field}")
        pooled.append(record)
    paired_probes = []
    for seed in seeds:
        for p in (.25,.75):
            for t in (0,4,16,32):
                rows = [r for r in probe_rows if r["seed"]==seed and r["target_probability"]==p and r["training_episodes"]==t]
                paired_probes.append({"seed":seed,"target_probability":p,"training_episodes":t,
                                     **{f:float(np.mean([r[f] for r in rows])) for f in ("frozen_probe_gain","responsive_probe_gain","response_effect")}})
    last = [r for r in paired_probes if r["training_episodes"]==32]
    positive = [r for r in last if r["frozen_probe_gain"]>0]
    reversals = sum(r["responsive_probe_gain"]<0 for r in positive)
    outer_gains = [float(np.mean([r["response_own_gain"] for r in seed_rows
                    if r["seed"] == seed and r["training_episodes"] == 32
                    and r["target_probability"] in (.25, .75)])) for seed in seeds]
    direction = []
    for seed in seeds:
        endpoints = {r["target_probability"]: r["response_probability"] for r in seed_rows
                     if r["seed"] == seed and r["training_episodes"] == 32}
        direction.append(endpoints[.75] - endpoints[.25])
    probe_contrasts = []
    for p in (.25, .75):
        for t in (0, 4, 16, 32):
            selected = [r for r in paired_probes if r["target_probability"] == p and r["training_episodes"] == t]
            probe_contrasts.append({"target_probability": p, "training_episodes": t,
                **{field: mean_ci([r[field] for r in selected], f"probe:{p}:{t}:{field}")
                   for field in ("frozen_probe_gain", "responsive_probe_gain", "response_effect")}})
    empirical_surface = {str(a)+str(b): {
        role: float(np.mean([panel["blocks"][block]["conditional_return_means"][str(a)+str(b)][role]
                             for panel in panels for block in ("A", "B")]))
        for role in ("focal_return", "response_return")}
        for a in (0, 1) for b in (0, 1)}
    return {"status":"complete","fresh_results_available":True,"n_seeds":len(seeds),"seeds":seeds,
            "curves":curves,"pooled_curves":pooled,
            "primary_diagnostic":mean_ci(outer_gains, "fresh:outer:32:response_own_gain"),
            "all_target_own_gain":pooled[-1]["response_own_gain"],
            "response_direction":mean_ci(direction, "fresh:paired_target_direction:32"),
            "probe_contrasts":probe_contrasts,
            "empirical_conditional_means":empirical_surface,
            "seed_results":seed_rows,"probe_seed_results":paired_probes,"block_consistency":consistency_rows,
            "noisy_probe_reversal":{"numerator":reversals,"denominator":len(positive),"note":"Prespecified focal-probability probes, not trained candidate upgrades; signs of noisy estimates only."},
            "accounting":{"episodes":sum(sum(p["counts"].values()) for p in panels),
                          "training_episodes":sum(p["counts"]["training"] for p in panels),
                          "evaluation_episodes":sum(p["counts"]["evaluation"] for p in panels),
                          "environment_frames":sum(sum(p["frames"].values()) for p in panels)},
            "statistics":{"unit":"independent response-training/environment seed; average both evaluation blocks within seed first",
                          "bootstrap_draws":BOOTSTRAP_DRAWS,"fixed_seed":STATISTICS_SEED,"inference":"descriptive pointwise 95% bootstrap intervals; no significance search or PIVOT test"},
            "limitations":["One frozen official low-level network; only an episodic Bernoulli skill mixture is learned.",
                           "Focal probabilities are prespecified probes, not newly trained candidate upgrades.",
                           "Checkpoints share one learning trajectory and the same two independent evaluation blocks; they are correlated.",
                           "Different focal targets use paired training random templates and distinct learner states; targets and checkpoints are correlated within each seed.",
                           "No PIVOT method comparison, full neural PPO training, or claim of optimal response is made."]}


def analyze_new(batch_root, source_dir):
    evidence = Evidence(batch_root)
    report = {"status":"partial","fresh_results_available":False,"validation_errors":[],"panel_progress":[]}
    try:
        protocol,state = evidence.load(evidence.root/"protocol.json"),evidence.load(evidence.root/"status.json")
        report["registered_protocol"] = protocol
        seeds = protocol.get("seeds",protocol.get("test_seeds"))
        require(isinstance(seeds,list) and len(seeds)>=4 and len(set(seeds))==len(seeds),"missing/invalid planned sprint seeds")
        ready = state["status"]=="complete"
        folders = {}
        for seed in seeds:
            choices = [evidence.root/f"seed_{seed}",evidence.root/"panels"/f"seed_{seed}"]
            folder = next((p for p in choices if (p/"status.json").exists()),choices[0])
            folders[seed] = folder
            status = evidence.load(folder/"status.json")["status"] if (folder/"status.json").exists() else "missing"
            report["panel_progress"].append({"seed":seed,"status":status})
            ready &= status=="complete"
        require(ready,"planned batch is incomplete; no completed-subset response effects reported")
        panels = [inspect_panel(evidence,folders[seed],seed,protocol,source_dir) for seed in seeds]
        seen = set()
        for panel in panels:
            require(not seen & panel["contexts"],"independent seeds reused a full rollout random context")
            seen.update(panel["contexts"])
        canonical = {k:v for k,v in panels[0]["config"].items() if k!="seed"}
        require(all({k:v for k,v in p["config"].items() if k!="seed"}==canonical for p in panels),"cross-seed configuration drift")
        require(all(p["sources_sha256"]==panels[0]["sources_sha256"] for p in panels),"cross-seed source drift")
        report.update(summarize_new(panels,protocol))
        replay_checks = []
        for name in ("replay_check.json", "replay_followup.json"):
            path = evidence.root/name
            if path.exists():
                replay_checks.append(evidence.load(path))
        report["native_replay_checks"] = replay_checks
        if replay_checks:
            total = sum(x["additional_native_replay_episodes"] for x in replay_checks)
            matched = sum(bool(row["match"]) for x in replay_checks for row in x["checks"])
            report["replay_accounting"] = {"additional_episodes": total, "matched": matched,
                "additional_frames": sum(x["additional_native_frames"] for x in replay_checks),
                "used_in_effect_estimation": False}
            report["exact_replay_status"] = "PASS" if matched == total else "REPLAY_MISMATCH"
            if matched != total:
                report["limitations"].append("At least one independent native replay differed despite identical declared seeds; exact trajectory reproducibility is not established. Original observations are retained; no failed replay is substituted into the study.")
    except (ValueError,KeyError,TypeError,OSError,IndexError) as error:
        report["validation_errors"].append(f"{type(error).__name__}: {error}")
    return report,evidence.hashes


def configure_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,
                         "savefig.dpi":170,"figure.facecolor":"white","axes.facecolor":"white"})
    return plt


def draw_new(report, output):
    require(report["status"]=="complete" and report["fresh_results_available"],"partial batches have no effect plots")
    plt = configure_plotting()
    colors = { .25:"#0E7490", .5:"#7C3AED", .75:"#B45309" }
    fig,axes = plt.subplots(1,3,figsize=(15,4.7),layout="constrained")
    for p in (.25,.5,.75):
        rows = [r for r in report["curves"] if r["target_probability"]==p]
        for ax,field in zip(axes,("response_probability","response_own_gain","focal_response_effect")):
            x=[r["training_episodes"] for r in rows]; y=[r[field]["mean"] for r in rows]
            ax.plot(x,y,"o-",color=colors[p],label=f"Focal stag probability {p:.2f}")
            ax.fill_between(x,[r[field]["ci95"][0] for r in rows],[r[field]["ci95"][1] for r in rows],color=colors[p],alpha=.12)
            ax.set_xticks([0,4,16,32]);ax.set_xlabel("Response training episodes")
    axes[0].axhline(.5,color="gray",lw=.8,ls="--");axes[0].set_ylabel("Responder stag probability")
    axes[0].set_title("A  What the responder learns")
    axes[1].axhline(0,color="gray",lw=.8,ls="--");axes[1].set_ylabel("Own native-return change vs initial responder")
    axes[1].set_title("B  Does independent payoff improve?")
    axes[2].axhline(0,color="gray",lw=.8,ls="--");axes[2].set_ylabel("Focal native-return change from response")
    axes[2].set_title("C  Effect on the fixed focal policy")
    axes[0].legend(fontsize=8,loc="best")
    fig.suptitle(f"NEW training | Frozen official skills, learned episodic mixture | {report['n_seeds']} seed clusters\nIndependent evaluation blocks A/B averaged within seed; correlated checkpoints; pointwise 95% CIs",fontsize=12)
    for suffix in ("png","pdf"):fig.savefig(output/f"new_response_learning.{suffix}")
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout="constrained")
    for p in (.25,.5,.75):
        rows=[r for r in report["block_consistency"] if r["target_probability"]==p]
        axes[0].scatter([r["own_gain_block_A"] for r in rows],[r["own_gain_block_B"] for r in rows],color=colors[p],label=f"p={p:.2f}",alpha=.8)
    values=[r[k] for r in report["block_consistency"] for k in ("own_gain_block_A","own_gain_block_B")]
    lo,hi=min(values),max(values);pad=max(1,(hi-lo)*.07)
    axes[0].plot([lo-pad,hi+pad],[lo-pad,hi+pad],"--",color="gray",lw=.8)
    axes[0].axhline(0,color="gray",lw=.5);axes[0].axvline(0,color="gray",lw=.5)
    axes[0].set(xlabel="Responder own-gain estimate: block A",ylabel="Responder own-gain estimate: block B",title="A  Independent measurement check, t=32")
    axes[0].legend(fontsize=9)
    for p in (.25,.75):
        rows=[r for r in report["probe_seed_results"] if r["target_probability"]==p and r["training_episodes"]==32]
        axes[1].scatter([r["frozen_probe_gain"] for r in rows],[r["responsive_probe_gain"] for r in rows],color=colors[p],label=f"p={p:.2f} vs p=0.50",alpha=.8)
    axes[1].axhline(0,color="gray",lw=.6);axes[1].axvline(0,color="gray",lw=.6)
    points = [r for r in report["probe_seed_results"] if r["training_episodes"] == 32]
    limits = [r[f] for r in points for f in ("frozen_probe_gain", "responsive_probe_gain")] + [0]
    low, high = min(limits), max(limits)
    margin = max(1., .07*(high-low))
    low, high = low-margin, high+margin
    axes[1].plot([low, high], [low, high], "--", color="gray", lw=.8, zorder=0)
    axes[1].fill_between([0, high], low, 0, color="#DC2626", alpha=.055, zorder=-1)
    axes[1].set_xlim(low, high); axes[1].set_ylim(low, high)
    axes[1].set(xlabel="Probe gain against initial responder",ylabel="Probe gain after respective response learning",title="B  Fixed policy probes, t=32")
    axes[1].legend(fontsize=9)
    fig.suptitle("NEW evidence | Probe policies are prespecified, not learned candidate upgrades\nMultiple points share a seed; noisy sign changes are not confirmed harmful upgrades",fontsize=12)
    for suffix in ("png","pdf"):fig.savefig(output/f"new_measurement_and_probes.{suffix}")
    plt.close(fig)
    grid = np.linspace(0, 1, 101)
    pp, qq = np.meshgrid(grid, grid)
    means = report["empirical_conditional_means"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for ax, role, title in zip(axes, ("focal_return", "response_return"),
                               ("A  Focal policy payoff", "B  Responder payoff")):
        surface = sum((pp if a else 1-pp)*(qq if b else 1-qq)*means[f"{a}{b}"][role]
                      for a in (0, 1) for b in (0, 1))
        m = ax.pcolormesh(pp, qq, surface, shading="auto", cmap="viridis", rasterized=True)
        fig.colorbar(m, ax=ax, label="Mean native episode return")
        ax.axhline(.5, color="white", ls="--", lw=1, label="Frozen responder")
        for p in (.25, .5, .75):
            rows = [r for r in report["curves"] if r["target_probability"] == p]
            q = [r["response_probability"]["mean"] for r in rows]
            ax.plot([p]*len(q), q, "o-", color=colors[p], mec="white", ms=5, lw=2)
            ax.annotate("32", (p, q[-1]), xytext=(5, 3), textcoords="offset points", color="white", fontsize=9)
        ax.set(xlim=(0,1), ylim=(0,1), xlabel="Focal stag probability p",
               ylabel="Responder stag probability q", title=title)
    fig.suptitle("NEW native evaluation | Payoff landscape and learned response paths\nSurface: exact mixture weights of measured conditional means; paths: mean q at 0/4/16/32 episodes", fontsize=12)
    for suffix in ("png", "pdf"): fig.savefig(output/f"new_native_payoff_landscape.{suffix}")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained", sharey=True)
    for ax, p in zip(axes, (.25, .75)):
        rows = [r for r in report["probe_contrasts"] if r["target_probability"] == p]
        x = [r["training_episodes"] for r in rows]
        baseline = rows[0]["frozen_probe_gain"]
        ax.axhline(baseline["mean"], color="#64748B", ls="--", lw=1.8, label="Frozen-opponent backtest")
        ax.axhspan(*baseline["ci95"], color="#64748B", alpha=.10)
        y = [r["responsive_probe_gain"]["mean"] for r in rows]
        lo = [r["responsive_probe_gain"]["ci95"][0] for r in rows]
        hi = [r["responsive_probe_gain"]["ci95"][1] for r in rows]
        ax.plot(x, y, "o-", color=colors[p], lw=2, label="Deployment after response learning")
        ax.fill_between(x, lo, hi, color=colors[p], alpha=.16)
        ax.axhline(0, color="#111827", lw=.8)
        ax.set_xticks([0, 4, 16, 32])
        ax.set(xlabel="Responder learning episodes", title=f"Policy replacement: p=0.50 to p={p:.2f}")
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("Paired focal improvement (native episode return)")
    fig.suptitle(f"Does the same policy change remain an improvement?\nBoth incumbent and replacement induce their own learned response; {report['n_seeds']} seed clusters, pointwise 95% CIs", fontsize=12)
    for suffix in ("png", "pdf"): fig.savefig(output/f"new_update_gain_horizons.{suffix}")
    plt.close(fig)


def report_markdown(report):
    if report["status"]!="complete":
        return "# 响应学习小试：尚未完整验收\n\n计划种子未全部完成或证据检查失败，因此不报告已完成子集的方法/机制效果。\n\n"+"\n".join(f"- {e}" for e in report["validation_errors"])+"\n"
    d=report["primary_diagnostic"];a=report["accounting"]
    lines=["# 新训练：对手响应学习小试", "",f"完整验收 {report['n_seeds']} 个计划种子，共 {a['episodes']} 局（训练 {a['training_episodes']}、评估 {a['evaluation_episodes']}），{a['environment_frames']:,} 环境帧。", "",
           "固定我方选择stag的概率为0.25、0.5、0.75；只学习对手每局选stag/hare的概率，官方低层网络不更新。每个条件沿同一训练轨迹保存0/4/16/32局检查点，全部冻结后用两个独立真实评估块评分。", "",
           f"32局时，两个外侧目标下，对手自身原生收益相对初始策略的变化：**{d['mean']:+.4f}，95%区间 [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]**。先平均每种子的p=0.25/0.75及A/B块，再跨种子计算；不是把所有点当独立实验。中间p=0.5作为弱响应对照单独保留。", "",
           f"响应方向差（面对p=0.75时的stag概率减去面对p=0.25）：**{report['response_direction']['mean']:+.4f}，95%区间 {report['response_direction']['ci95']}**。", "",
           "这是能力与测量诊断，不是新的PIVOT方法对比。概率改变说明更新发生，独立收益区间才说明本次是否测到了改善；正点估计不自动表示可靠改善。", "",
           "| 我方stag概率 | 对手训练局数 | 对手stag概率 | 对手自身收益变化 [95%区间] |", "|---:|---:|---:|---:|"]
    for r in report["curves"]:
        g=r["response_own_gain"]
        lines.append(f"| {r['target_probability']:.2f} | {r['training_episodes']} | {r['response_probability']['mean']:.3f} | {g['mean']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] |")
    lines += ["", "图一展示学习概率、独立自收益和对我方的影响；图二检查A/B两块是否给出相近结果，并比较预设我方策略探针相对p=0.5的收益。后者不是训练生成的候选，也不是新算法胜利。", "",
              "第三张图把新评估对局得到的条件收益画成我方/对手的收益地图，并叠加真实学习轨迹的平均概率。地图各点是同一批原生对局的混合期望，不是新增训练、额外测试或精确已知的游戏收益。", "",
              "四格条件均值来自真实Melting Pot对局，按已冻结概率积分；仍有评估噪声。各检查点共享同一评估块，不同目标条件使用配对的训练随机模板、各自独立初始化的学习状态，区间是以seed为单位的逐点描述性区间。", "",
              "[逐种子数据](new_seed_results.csv) · [策略探针](new_probe_seed_results.csv) · [A/B测量复核](new_block_consistency.csv) · [完整汇总](summary.json) · [证据哈希](manifest.json)", ""]
    if report.get("replay_accounting"):
        replay = report["replay_accounting"]
        lines += [f"额外原生重放检查：{replay['matched']}/{replay['additional_episodes']}局逐帧与回报完全一致；这些额外对局不进入上面的效果统计。", ""]
        if report.get("exact_replay_status") != "PASS":
            lines += ["复现限制：相同声明种子未保证所有对局逐帧完全一致。因此本次保留原始证据与配对随机模板，但不宣称环境轨迹已经达到逐帧确定性复现。", ""]
    return "\n".join(lines)


def save_new(output, report, hashes, plots=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    write(output/"summary.json",report)
    for name,key in (("new_seed_results.csv","seed_results"),("new_probe_seed_results.csv","probe_seed_results"),("new_block_consistency.csv","block_consistency"),("panel_progress.csv","panel_progress")):
        csv_rows(output/name,report.get(key,[]))
    (output/"新训练结果.md").write_text(report_markdown(report))
    if report["status"]=="complete" and plots:
        draw_new(report,output)
    elif report["status"]!="complete":
        for name in ("new_response_learning","new_measurement_and_probes","new_native_payoff_landscape","new_update_gain_horizons"):
            for suffix in ("png","pdf"):
                (output/f"{name}.{suffix}").unlink(missing_ok=True)
    names=[p for p in output.iterdir() if p.is_file() and p.name!="manifest.json"]
    write(output/"manifest.json",{"status":report["status"],"analysis_source_sha256":sha(__file__),
                                "input_hashes":hashes,"output_sha256":{p.name:sha(p) for p in names},
                                "note":"Fresh training only. Any earlier-data mechanism reanalysis is saved separately."})


def self_test():
    require(mean_ci([2]*8,"constant")["ci95"]==[2,2],"bootstrap fixture")
    means={f"{a}{b}":{"focal_return":3+5*a+7*b+11*a*b} for a in (0,1) for b in (0,1)}
    close(weighted(means,.23,.67,"focal_return"),3+5*.23+7*.67+11*.23*.67,"weighted expected return")
    with tempfile.TemporaryDirectory(prefix="response-analysis-check-") as tmp:
        root=Path(tmp);report,hashes=analyze_new(root,root)
        require(report["status"]=="partial" and not report["fresh_results_available"],"missing evidence gate")
        save_new(root/"output",report,hashes)
        require(not list((root/"output").glob("*.png")),"no subset effect plots")
    print("PASS: seed bootstrap, weighted expectations and incomplete-batch plot gate; no user fixture outputs retained")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-root",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--source-dir",type=Path,default=Path(__file__).resolve().parent)
    p.add_argument("--no-plots",action="store_true")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:
        self_test();return 0
    if args.batch_root is None or args.output is None:p.error("--batch-root and --output are required")
    report,hashes=analyze_new(args.batch_root,args.source_dir)
    save_new(args.output,report,hashes,not args.no_plots)
    print(json.dumps({"status":report["status"],"errors":report["validation_errors"],"output":str(args.output)},ensure_ascii=False))
    return 0 if report["status"]=="complete" else 2


if __name__=="__main__":
    raise SystemExit(main())
