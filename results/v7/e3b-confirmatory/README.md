# E3b Confirmatory Result

This frozen result uses the version-pinned Farama MPE2 adapter with a
short-horizon direct-replay reward proxy (observer horizon 12), a full actor
rollout, paired seeds, 30 self-improvement rounds, 8 candidates per round,
and a matched two-query high-fidelity budget. The confirmatory pool contains
772 independent paired trajectories and requires 536 under the registered
trajectory-level power rule.

All five construct-validity gates pass. The powered state is
`HYPOTHESIS_NOT_SUPPORTED`: PIVOT-VOI minus Proxy Only CTI is `-3.8672173`,
so this external environment does not support the registered CTI improvement
claim. This is a valid null result, not an implementation failure. The earlier
discrete observer attempt remains separately frozen under
`e3b-confirmatory-invalid-proxy` as `DESIGN_INVALID`.
