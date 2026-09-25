"""Contract and numerical tests for deterministic Kuhn rollouts."""

from __future__ import annotations

import unittest

import numpy as np
import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score

try:
    from kuhn_rollout_sampler import StreamPurpose, sample_kuhn_returns
except ModuleNotFoundError:  # Allows discovery from the repository root too.
    from colin_pivot_cloud.openspiel_fallback.kuhn_rollout_sampler import (
        StreamPurpose,
        sample_kuhn_returns,
    )


class KuhnRolloutSamplerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.game = pyspiel.load_game("kuhn_poker")
        self.player_zero = policy.TabularPolicy(self.game)
        self.player_one = policy.TabularPolicy(self.game)

        # Use different, non-uniform behavior for the two players so the exact
        # value check detects player-policy swaps and hard-coded uniform play.
        for index, state in enumerate(self.player_zero.states):
            if state.current_player() == 0:
                self.player_zero.action_probability_array[index] = [0.80, 0.20]
            if state.current_player() == 1:
                self.player_one.action_probability_array[index] = [0.25, 0.75]
        self.policies = (self.player_zero, self.player_one)

    def test_exact_replay_and_policies_remain_fixed(self) -> None:
        before = tuple(item.action_probability_array.copy() for item in self.policies)
        arguments = dict(
            game=self.game,
            policies=self.policies,
            num_hands=512,
            seed=20260917,
            stream_key=(StreamPurpose.QUERY, 4, 0),
        )
        first = sample_kuhn_returns(**arguments)
        replay = sample_kuhn_returns(**arguments)

        np.testing.assert_array_equal(first, replay)
        for original, item in zip(before, self.policies):
            np.testing.assert_array_equal(original, item.action_probability_array)

    def test_independent_streams_and_block_splitting(self) -> None:
        seed = 314159
        stream = (StreamPurpose.QUERY, 7, 2)
        complete = sample_kuhn_returns(
            self.game, self.policies, 257, seed=seed, stream_key=stream
        )
        split = np.concatenate(
            [
                sample_kuhn_returns(
                    self.game,
                    self.policies,
                    73,
                    seed=seed,
                    stream_key=stream,
                    start_hand=0,
                ),
                sample_kuhn_returns(
                    self.game,
                    self.policies,
                    184,
                    seed=seed,
                    stream_key=stream,
                    start_hand=73,
                ),
            ]
        )
        audit = sample_kuhn_returns(
            self.game,
            self.policies,
            257,
            seed=seed,
            stream_key=(StreamPurpose.AUDIT, 7, 2),
        )

        np.testing.assert_array_equal(complete, split)
        self.assertFalse(np.array_equal(complete, audit))

    def test_sample_means_match_expected_game_score(self) -> None:
        exact = np.asarray(
            expected_game_score.policy_value(
                self.game.new_initial_state(), list(self.policies)
            ),
            dtype=float,
        )
        sampled = sample_kuhn_returns(
            self.game,
            self.policies,
            80_000,
            seed=20260917,
            stream_key=(StreamPurpose.CALIBRATION, 11, 0),
        )

        np.testing.assert_allclose(sampled.mean(axis=0), exact, atol=0.025, rtol=0.0)
        np.testing.assert_allclose(sampled.sum(axis=1), 0.0, atol=0.0, rtol=0.0)

    def test_rejects_ambiguous_or_invalid_seed_namespaces(self) -> None:
        with self.assertRaisesRegex(ValueError, "stream_key"):
            sample_kuhn_returns(self.game, self.policies, 1, seed=1, stream_key=())
        with self.assertRaisesRegex(ValueError, "num_hands"):
            sample_kuhn_returns(self.game, self.policies, 0, seed=1, stream_key=(0,))
        with self.assertRaisesRegex(ValueError, "seed"):
            sample_kuhn_returns(self.game, self.policies, 1, seed=-1, stream_key=(0,))


if __name__ == "__main__":
    unittest.main()
