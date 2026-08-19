from pivot.core.policy import Policy
from pivot.footprint.generic import compute_update_footprint


def test_footprint_preserves_component_features() -> None:
    incumbent = Policy.from_mapping({"intensity": 0.2, "bias": 0.1})
    candidate = Policy.from_mapping({"intensity": 0.6, "bias": 0.1})
    result = compute_update_footprint(incumbent, candidate, [0.0, 0.5, 1.0])
    required = {
        "mean_kl",
        "max_kl",
        "action_shift",
        "entropy_change",
        "occupancy_divergence",
        "support_expansion",
        "trajectory_divergence",
        "episode_length_change",
    }
    assert required <= result.components.keys()
    assert result.distance > 0


def test_identical_policies_have_zero_footprint() -> None:
    policy = Policy.from_mapping({"intensity": 0.2})
    result = compute_update_footprint(policy, policy, [0.0, 1.0])
    assert result.distance == 0.0
