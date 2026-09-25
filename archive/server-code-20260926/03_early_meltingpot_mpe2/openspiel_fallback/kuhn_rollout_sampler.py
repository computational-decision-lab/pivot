"""Deterministic Monte Carlo rollouts for a fixed pair of Kuhn policies.

The seed namespace for every random draw is

``(seed, *stream_key, absolute_hand_index, random_source)``.

``random_source`` separates chance, player 0, and player 1.  Consequently,
policy-dependent action counts cannot advance the chance stream or the other
player's stream.  Absolute hand indices also make a rollout invariant to block
size, call order, and parallel scheduling.

Use a distinct ``stream_key`` for every independent use.  For example, the
first coordinate can be ``StreamPurpose.QUERY`` or ``StreamPurpose.AUDIT`` and
the remaining coordinates can identify the candidate and block.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Sequence

import numpy as np
import pyspiel
from open_spiel.python import policy


_UINT32_MAX = np.iinfo(np.uint32).max
_PROBABILITY_TOLERANCE = 1e-10


class StreamPurpose(IntEnum):
    """Stable top-level coordinates for scientifically separate streams."""

    CALIBRATION = 0
    QUERY = 1
    AUDIT = 2


class _RandomSource(IntEnum):
    CHANCE = 0
    PLAYER_ZERO = 1
    PLAYER_ONE = 2


def _nonnegative_integer(value: int, *, name: str, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be nonnegative")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{name} must be no greater than {maximum}")
    return normalized


def _normalize_stream_key(stream_key: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(stream_key, (int, np.integer)) and not isinstance(
        stream_key, (bool, np.bool_)
    ):
        raw_key = (stream_key,)
    else:
        if isinstance(stream_key, (str, bytes)):
            raise TypeError("stream_key must contain integers")
        try:
            raw_key = tuple(stream_key)
        except TypeError as exc:
            raise TypeError(
                "stream_key must be an integer or a sequence of integers"
            ) from exc
    if not raw_key:
        raise ValueError("stream_key must contain at least one coordinate")
    return tuple(
        _nonnegative_integer(value, name="stream_key coordinate", maximum=_UINT32_MAX)
        for value in raw_key
    )


def _validate_inputs(
    game: pyspiel.Game,
    policies: Sequence[policy.TabularPolicy],
) -> tuple[policy.TabularPolicy, policy.TabularPolicy]:
    if game.get_type().short_name != "kuhn_poker" or game.num_players() != 2:
        raise ValueError("sample_kuhn_returns requires a two-player kuhn_poker game")
    fixed_policies = tuple(policies)
    if len(fixed_policies) != 2:
        raise ValueError("exactly two policies are required")
    if not all(isinstance(item, policy.TabularPolicy) for item in fixed_policies):
        raise TypeError("both policies must be OpenSpiel TabularPolicy instances")
    if any(item.game_type.short_name != "kuhn_poker" for item in fixed_policies):
        raise ValueError("both policies must be defined for kuhn_poker")
    return fixed_policies[0], fixed_policies[1]


def _rng(
    seed: int,
    stream_key: tuple[int, ...],
    hand_index: int,
    source: _RandomSource,
) -> np.random.Generator:
    sequence = np.random.SeedSequence(
        seed,
        spawn_key=stream_key + (hand_index, int(source)),
    )
    # Name the bit generator explicitly so a future change to NumPy's
    # ``default_rng`` choice cannot silently alter the replay contract.
    return np.random.Generator(np.random.PCG64(sequence))


def _sample_action(
    actions: Sequence[int],
    probabilities: Sequence[float],
    rng: np.random.Generator,
    *,
    distribution_name: str,
) -> int:
    action_array = np.asarray(actions, dtype=int)
    probability_array = np.asarray(probabilities, dtype=float)
    if action_array.ndim != 1 or len(action_array) == 0:
        raise ValueError(f"{distribution_name} has no actions")
    if probability_array.shape != action_array.shape:
        raise ValueError(
            f"{distribution_name} has mismatched actions and probabilities"
        )
    if not np.all(np.isfinite(probability_array)):
        raise ValueError(f"{distribution_name} contains a non-finite probability")
    if np.any(probability_array < -_PROBABILITY_TOLERANCE):
        raise ValueError(f"{distribution_name} contains a negative probability")
    probability_array = np.maximum(probability_array, 0.0)
    total = float(np.sum(probability_array))
    if not np.isclose(total, 1.0, rtol=0.0, atol=_PROBABILITY_TOLERANCE):
        raise ValueError(f"{distribution_name} probabilities sum to {total}, not 1")

    # Normalizing removes harmless floating-point drift from behavioral-policy
    # mixtures.  Inverse-CDF sampling also makes the draw rule explicit.
    cumulative = np.cumsum(probability_array / total)
    selected = int(np.searchsorted(cumulative, rng.random(), side="right"))
    return int(action_array[min(selected, len(action_array) - 1)])


def sample_kuhn_returns(
    game: pyspiel.Game,
    policies: Sequence[policy.TabularPolicy],
    num_hands: int,
    *,
    seed: int,
    stream_key: int | Sequence[int],
    start_hand: int = 0,
) -> np.ndarray:
    """Samples terminal returns from two fixed tabular Kuhn policies.

    Args:
        game: A two-player OpenSpiel ``kuhn_poker`` game.
        policies: ``(player_zero_policy, player_one_policy)``.  The policies are
            read only and remain fixed for the entire batch.
        num_hands: Number of independently seeded hands to sample.
        seed: Nonnegative root seed.
        stream_key: Nonempty integer namespace for this independent stream.
            A recommended query key is
            ``(StreamPurpose.QUERY, candidate_index, block_index)``; audit uses
            ``StreamPurpose.AUDIT`` in the first coordinate.
        start_hand: Absolute index of the first hand.  Use this when splitting a
            stream into blocks; concatenated blocks exactly reproduce one call.

    Returns:
        A ``(num_hands, 2)`` float array of terminal player returns.
    """
    fixed_policies = _validate_inputs(game, policies)
    count = _nonnegative_integer(num_hands, name="num_hands")
    if count == 0:
        raise ValueError("num_hands must be positive")
    root_seed = _nonnegative_integer(seed, name="seed")
    first_hand = _nonnegative_integer(
        start_hand, name="start_hand", maximum=_UINT32_MAX
    )
    if first_hand + count - 1 > _UINT32_MAX:
        raise ValueError(
            "the final absolute hand index exceeds the uint32 seed namespace"
        )
    normalized_key = _normalize_stream_key(stream_key)

    returns = np.empty((count, 2), dtype=float)
    for local_index in range(count):
        hand_index = first_hand + local_index
        generators = (
            _rng(root_seed, normalized_key, hand_index, _RandomSource.CHANCE),
            _rng(root_seed, normalized_key, hand_index, _RandomSource.PLAYER_ZERO),
            _rng(root_seed, normalized_key, hand_index, _RandomSource.PLAYER_ONE),
        )
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                action = _sample_action(
                    [item[0] for item in outcomes],
                    [item[1] for item in outcomes],
                    generators[_RandomSource.CHANCE],
                    distribution_name="chance distribution",
                )
            else:
                player_id = int(state.current_player())
                if player_id not in (0, 1):
                    raise ValueError(
                        f"unexpected nonterminal Kuhn player id {player_id}"
                    )
                legal_actions = state.legal_actions(player_id)
                action_probabilities = fixed_policies[player_id].action_probabilities(
                    state, player_id
                )
                action = _sample_action(
                    legal_actions,
                    [action_probabilities.get(item, 0.0) for item in legal_actions],
                    generators[player_id + 1],
                    distribution_name=f"player {player_id} policy",
                )
            state.apply_action(action)
        terminal_returns = np.asarray(state.returns(), dtype=float)
        if terminal_returns.shape != (2,):
            raise ValueError("Kuhn terminal state did not return two player values")
        returns[local_index] = terminal_returns
    return returns
