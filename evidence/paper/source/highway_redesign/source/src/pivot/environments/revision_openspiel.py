from __future__ import annotations

import hashlib
import importlib
import math
from collections.abc import Mapping
from typing import Any, Literal

from pivot.core.policy import Policy
from pivot.core.result import RolloutResult

OpenSpielMode = Literal["observer", "actor", "strategic"]


class OpenSpielKuhnWorld:
    """Exact finite-game strategic-response smoke using OpenSpiel Kuhn poker.

    Player 0 is the focal PIVOT policy. Observer mode holds player 1 fixed at
    uniform random. Actor and strategic modes solve player 1's tabular best
    response with OpenSpiel, then integrate player 0's utility over the full
    game tree. No trajectory is sampled, so ``seed`` is accepted only for the
    V9 world interface and cannot change the result.

    ``compute_cost`` contains only adapter-visible exact utility-tree visits.
    OpenSpiel's best-response implementation is opaque to this adapter, so its
    call count and a bookkeeping proxy are reported separately and never added
    to measured compute cost.
    """

    environment_id = "openspiel_kuhn_strategic_response"
    environment_family = "openspiel_finite_game"
    game_name = "kuhn_poker"
    focal_player = 0
    response_player = 1
    paired_query_cost_unit = "exact_utility_tree_node_visits_upper_bound"
    paired_query_cost_is_upper_bound = True
    paired_query_opaque_solver_calls_upper_bound = 2

    def __init__(self) -> None:
        try:
            self._pyspiel = importlib.import_module("pyspiel")
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "OpenSpielKuhnWorld requires the pinned optional dependency; "
                "install requirements/revision-openspiel.txt"
            ) from error

        self._game = self._pyspiel.load_game(self.game_name)
        self._native_delta_bounds = self._derive_native_delta_bounds()
        self._full_tree_node_count = self._structural_tree_node_count()
        self._decision_states = self._collect_decision_states()
        self._focal_infosets = self._infosets_for_player(self.focal_player)
        self._response_infosets = self._infosets_for_player(self.response_player)

    @property
    def single_evaluation_cost_upper_bound(self) -> float:
        return float(self._full_tree_node_count)

    @property
    def paired_query_cost(self) -> float:
        """Full-tree visit upper bound for an incumbent/candidate query."""
        return 2.0 * self.single_evaluation_cost_upper_bound

    @property
    def native_delta_bounds(self) -> dict[str, object]:
        """Exact paired-difference bounds from the native finite game's payoffs.

        These are deterministic game-theoretic bounds, not clipping bounds and
        not statistical confidence evidence.  The terminal-tree traversal in
        ``_derive_native_delta_bounds`` verifies OpenSpiel's native utility
        extrema for this exact game before exposing the paired range.
        """
        return dict(self._native_delta_bounds)

    def evaluate(
        self,
        policy: Policy,
        *,
        seed: int,
        mode: OpenSpielMode = "observer",
    ) -> RolloutResult:
        """Return player 0's exact expected utility under the requested response."""
        if mode not in ("observer", "actor", "strategic"):
            raise ValueError(f"unsupported OpenSpiel evaluation mode: {mode}")

        focal_table = self.focal_policy_table(policy)
        response_proxy_work = 0
        opaque_response_solver_calls = 0
        response_actions: dict[str, int] = {}
        response_solver_value: float | None = None
        if mode == "observer":
            opponent_table = self._uniform_table(self._response_infosets)
            opponent_name = "uniform_random"
            response_solver: str | None = None
            simulator_calls = 1
        else:
            response_actions, response_solver_value = self._solve_response(focal_table)
            opponent_table = {
                infoset: {
                    action: 1.0 if action == response_actions[infoset] else 0.0
                    for action in legal_actions
                }
                for infoset, legal_actions in self._response_infosets.items()
            }
            opponent_name = "tabular_best_response"
            response_solver = "pyspiel.TabularBestResponse"
            response_proxy_work = len(self._decision_states) + len(response_actions) + 1
            opaque_response_solver_calls = 1
            simulator_calls = 2

        value, utility_work = self._exact_focal_utility(focal_table, opponent_table)
        metadata: dict[str, object] = {
            "environment_id": self.environment_id,
            "environment_family": self.environment_family,
            "game": self.game_name,
            "mode": mode,
            "seed": int(seed),
            "focal_player": self.focal_player,
            "response_player": self.response_player,
            "opponent_policy": opponent_name,
            "response_solver": response_solver,
            "response_solver_infosets": len(response_actions),
            "response_solver_actions": dict(sorted(response_actions.items())),
            "response_solver_value": response_solver_value,
            "opaque_response_solver_calls": opaque_response_solver_calls,
            "response_solver_cost_measured": False,
            "response_solver_proxy_work_units": response_proxy_work,
            "response_solver_proxy_definition": (
                "decision_histories_plus_solved_information_states_plus_root_value"
            ),
            "response_solver_proxy_scope": (
                "bookkeeping_proxy_not_measured_solver_work_excluded_from_compute_cost"
            ),
            "evaluation_method": "exact_game_tree",
            "sampled_return": False,
            "seed_affects_value": False,
            "seed_role": "interface_only_not_an_independent_replicate",
            "independent_replicate": False,
            "utility_tree_work_units": utility_work,
            "utility_tree_work_unit_definition": "recursive_game_tree_node_visits",
            "compute_cost_scope": "adapter_visible_exact_utility_tree_node_visits_only",
            "compute_cost_complete": opaque_response_solver_calls == 0,
            "single_evaluation_cost_upper_bound": self.single_evaluation_cost_upper_bound,
            "paired_query_cost": self.paired_query_cost,
            "paired_query_cost_unit": self.paired_query_cost_unit,
            "paired_query_cost_is_upper_bound": self.paired_query_cost_is_upper_bound,
            "paired_query_opaque_solver_calls_upper_bound": (
                self.paired_query_opaque_solver_calls_upper_bound
            ),
            "native_delta_bounds": self.native_delta_bounds,
            "native_delta_bounds_are_clipping": False,
            "native_delta_bounds_are_confidence_evidence": False,
            "claim_scope": "technical_smoke_only_no_training_claim",
        }
        return RolloutResult(
            value=value,
            environment_steps=utility_work,
            simulator_calls=simulator_calls,
            compute_cost=float(utility_work),
            metadata=metadata,
        )

    def focal_policy_table(self, policy: Policy) -> dict[str, dict[int, float]]:
        """Map all focal parameters to deterministic legal-action probabilities."""
        table: dict[str, dict[int, float]] = {}
        for infoset, legal_actions in self._focal_infosets.items():
            if len(legal_actions) != 2:
                raise ValueError("Kuhn focal policy expects exactly two legal actions")
            logit = self._focal_logit(policy.parameters, infoset)
            probability_action_one = self._sigmoid(logit)
            table[infoset] = {
                legal_actions[0]: 1.0 - probability_action_one,
                legal_actions[1]: probability_action_one,
            }
        return table

    def _collect_decision_states(self) -> dict[str, Any]:
        states: dict[str, Any] = {}

        def visit(state: Any) -> None:
            if state.is_terminal():
                return
            if state.is_chance_node():
                transitions = state.chance_outcomes()
            else:
                states[state.history_str()] = state.clone()
                transitions = ((action, 1.0) for action in state.legal_actions())
            for action, _ in transitions:
                visit(state.child(action))

        visit(self._game.new_initial_state())
        return states

    def _structural_tree_node_count(self) -> int:
        node_count = 0

        def visit(state: Any) -> None:
            nonlocal node_count
            node_count += 1
            if state.is_terminal():
                return
            transitions = (
                state.chance_outcomes()
                if state.is_chance_node()
                else ((action, 1.0) for action in state.legal_actions())
            )
            for action, _ in transitions:
                visit(state.child(action))

        visit(self._game.new_initial_state())
        return node_count

    def _infosets_for_player(self, player: int) -> dict[str, tuple[int, ...]]:
        infosets: dict[str, tuple[int, ...]] = {}
        for state in self._decision_states.values():
            if state.current_player() != player:
                continue
            key = state.information_state_string(player)
            legal_actions = tuple(state.legal_actions(player))
            previous = infosets.setdefault(key, legal_actions)
            if previous != legal_actions:
                raise ValueError(f"inconsistent legal actions for information state {key!r}")
        return dict(sorted(infosets.items()))

    def _solve_response(
        self, focal_table: Mapping[str, Mapping[int, float]]
    ) -> tuple[dict[str, int], float]:
        joint_policy: dict[str, list[tuple[int, float]]] = {}
        uniform_response = self._uniform_table(self._response_infosets)
        for infoset, probabilities in {**focal_table, **uniform_response}.items():
            joint_policy[infoset] = sorted(probabilities.items())
        solver = self._pyspiel.TabularBestResponse(
            self._game,
            self.response_player,
            joint_policy,
        )
        actions = {
            str(infoset): int(action)
            for infoset, action in solver.get_best_response_actions().items()
        }
        if set(actions) != set(self._response_infosets):
            raise RuntimeError("OpenSpiel best response did not cover every opponent information state")
        response_value = float(solver.value_from_state(self._game.new_initial_state()))
        return actions, response_value

    def _exact_focal_utility(
        self,
        focal_table: Mapping[str, Mapping[int, float]],
        opponent_table: Mapping[str, Mapping[int, float]],
    ) -> tuple[float, int]:
        node_visits = 0

        def value(state: Any) -> float:
            nonlocal node_visits
            node_visits += 1
            if state.is_terminal():
                return float(state.returns()[self.focal_player])
            if state.is_chance_node():
                transitions = state.chance_outcomes()
            else:
                player = state.current_player()
                infoset = state.information_state_string(player)
                table = focal_table if player == self.focal_player else opponent_table
                transitions = table[infoset].items()
            return sum(
                float(probability) * value(state.child(action))
                for action, probability in transitions
                if probability > 0.0
            )

        return value(self._game.new_initial_state()), node_visits

    def _derive_native_delta_bounds(self) -> dict[str, object]:
        native_min = float(self._game.min_utility())
        native_max = float(self._game.max_utility())
        terminal_values: list[float] = []

        def visit(state: Any) -> None:
            if state.is_terminal():
                terminal_values.append(float(state.returns()[self.focal_player]))
                return
            transitions = (
                state.chance_outcomes()
                if state.is_chance_node()
                else ((action, 1.0) for action in state.legal_actions())
            )
            for action, _ in transitions:
                visit(state.child(action))

        visit(self._game.new_initial_state())
        if not terminal_values:
            raise ValueError("OpenSpiel game has no terminal utility outcomes")
        terminal_min = min(terminal_values)
        terminal_max = max(terminal_values)
        values = (native_min, native_max, terminal_min, terminal_max)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("OpenSpiel native utility bounds must be finite")
        if native_min != terminal_min or native_max != terminal_max:
            raise ValueError(
                "OpenSpiel native utility bounds do not match exhaustive terminal traversal"
            )
        lower = native_min - native_max
        upper = native_max - native_min
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("OpenSpiel paired delta bounds must be finite and non-degenerate")
        return {
            "lower": lower,
            "upper": upper,
            "source": "openspiel_exact_game_utility_bounds",
            "game": self.game_name,
            "focal_player": self.focal_player,
            "min_utility": native_min,
            "max_utility": native_max,
            "terminal_min_utility": terminal_min,
            "terminal_max_utility": terminal_max,
        }

    @staticmethod
    def _uniform_table(
        infosets: Mapping[str, tuple[int, ...]],
    ) -> dict[str, dict[int, float]]:
        return {
            infoset: {action: 1.0 / len(actions) for action in actions}
            for infoset, actions in infosets.items()
        }

    @staticmethod
    def _focal_logit(parameters: Mapping[str, float], infoset: str) -> float:
        card_strength = float(int(infoset[0]) - 1)
        facing_bet = infoset.endswith("pb")
        logit = 0.8 * card_strength - (0.25 if facing_bet else 0.0)
        for name, value in sorted(parameters.items()):
            if name == "intensity":
                coefficient = 2.0
            elif name == "bias":
                coefficient = 1.0 + 0.35 * card_strength - (0.2 if facing_bet else 0.0)
            else:
                digest = hashlib.sha256(f"{name}|{infoset}".encode()).digest()
                unit = int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)
                coefficient = 2.0 * unit - 1.0
            logit += coefficient * float(value)
        return logit

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0.0:
            inverse = math.exp(-value)
            return 1.0 / (1.0 + inverse)
        exponential = math.exp(value)
        return exponential / (1.0 + exponential)


__all__ = ["OpenSpielKuhnWorld", "OpenSpielMode"]
