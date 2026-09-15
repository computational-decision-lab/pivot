# Modern Agent Evaluation Design

## Purpose

This design adds an agent-agnostic evaluation layer for the Improvement
Fidelity study.  The existing controlled and finance evidence remains frozen;
the new layer can only add independently traceable evidence and cannot alter a
previous result because it is inconvenient or unfavorable.

## Scientific object

The unit is a directed replacement transition
`tau = (pi_t, pi_t+1)`.  A self-improvement operator proposes candidates using
only the public observer plane.  The deployment plane evaluates the same
incumbent/candidate pair in fresh, matched sandboxes.  A separate assessment
plane is never exposed to proposal, calibration, or promotion code.

## Components

- `experiments/v15/protocol.py` defines immutable scaffold policies, candidate
  provenance, transition records, hashes, and terminal states.
- `experiments/v15/planes.py` enforces role-based access to proxy, gate, and
  assessment task pools.
- `experiments/v15/sandbox.py` provides fresh paired sandbox execution for the
  local development reference and records initial-state hashes.
- `experiments/v15/operators.py` contains two independent, deterministic
  proposal mechanisms.  Both consume only proxy feedback and registered
  resource limits.
- `experiments/v15/control_plane.py` detects optional Inspect AI, mini-SWE, and
  Pi adapters without silently substituting a different scientific scaffold.
- `experiments/v15/dev.py` runs a small construct-validity smoke test.  Its
  output is explicitly DEV-only and is never promoted to confirmatory evidence.
- `experiments/v15/reports.py` and `experiments/v15/finalize.py` generate the
  protocol, audit, and release reports.

## Confirmatory boundary

The confirmatory lock freezes task hashes, operator prompts, budgets, seeds,
metrics, footprint features, baselines, and stopping rules before any external
agent run.  The pinned Inspect AI, mini-SWE-agent, and Pi runtimes are now
available and have completed bounded DEV smoke checks.  The full confirmatory
study is still intentionally unopened because it consumes the registered
external model/container budget and requires an explicit execution
authorization.  The implementation reports that state as `NOT_RUN` rather
than promoting DEV observations or replacing them with synthetic evidence.

## Figure and manuscript policy

Existing figures are copied to a neutral release path and audited with source
bundles.  No structural redesign is introduced by this layer.  The manuscript
imports neutral result-macro filenames so internal build labels do not appear
in the paper source or rendered text.
