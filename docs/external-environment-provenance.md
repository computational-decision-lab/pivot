# V7 External Environment Provenance

The first external-environment candidate is Farama's MPE2
`simple_adversary_v3`, accessed through PettingZoo 1.27.0 and MPE2 1.1.0.
The public PettingZoo repository is MIT licensed and was checked at commit
`7221e4b15873a8b16555fda038a77a6ae595e88d` on 2026-08-25. The adapter records
these versions in every rollout result.

This is an environment-source record, not a scientific pass. E3b and E7b may
promote claims only after the adapter passes the preregistered construct gates,
uses held-out seeds/opponent settings, and receives a frozen confirmatory
classification. If the source fails a gate, the run is retained as
`DESIGN_INVALID` and the claim is narrowed.

The first E3b observer score (a discrete static-target hit rate) failed the
non-perfect-proxy gate at scale and is retained as a design-invalid artifact.
The development redesign uses a short-horizon direct replay of the public
environment reward with fixed replay opponents. It changes only the cheap
proxy horizon and opponent response, not the public environment dynamics or
paired actor evaluation.

The optional dependencies are reproducible with:

```bash
python -m pip install -e '.[external]'
```

No live trading, network data, or private labels enter this adapter.

For the strategic experiment, opponent family A is a reactive observation
policy and held-out family B uses an explicit finite-action best-response
optimization over the focal action signal. Neither opponent calls PIVOT,
reads future test labels, or receives the focal evaluator's deployment return.
