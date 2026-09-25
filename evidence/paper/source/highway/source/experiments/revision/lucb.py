"""Paired LUCB1 state machine with an exact known-zero incumbent.

The caller owns observations and their reproducible seed streams. This module
only reserves paid pulls, validates returned paired deltas, updates LUCB1
confidence bounds, and distinguishes certificates from budget censoring.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

LUCBStatus = Literal[
    "running",
    "certified",
    "budget_censored",
    "initialization_infeasible",
]
PullPhase = Literal["initialization", "critical"]


@dataclass(frozen=True)
class LUCBConfig:
    candidate_ids: tuple[str, ...]
    epsilon: float
    delta: float
    lower_bound: float
    upper_bound: float
    budget: float
    pull_cost: float
    incumbent_id: str = "incumbent"

    def __post_init__(self) -> None:
        if not self.candidate_ids:
            raise ValueError("Paired LUCB1 requires at least one candidate")
        if any(not candidate_id for candidate_id in self.candidate_ids):
            raise ValueError("candidate IDs must be nonempty")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate IDs must be unique")
        if not self.incumbent_id or self.incumbent_id in self.candidate_ids:
            raise ValueError("incumbent ID must be nonempty and distinct from candidate IDs")
        for name in ("epsilon", "delta", "lower_bound", "upper_bound", "budget", "pull_cost"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must be between zero and one")
        if self.lower_bound >= self.upper_bound:
            raise ValueError("lower_bound must be below upper_bound")
        if not self.lower_bound <= 0.0 <= self.upper_bound:
            raise ValueError("declared reward bounds must contain the known-zero incumbent")
        if self.budget < 0.0:
            raise ValueError("budget must be nonnegative")
        if self.pull_cost <= 0.0:
            raise ValueError("pull_cost must be positive")


@dataclass(frozen=True)
class PullRequest:
    reservation_id: int
    candidate_id: str
    replicate_index: int
    phase: PullPhase
    iteration: int
    reserved_cost: float


@dataclass(frozen=True)
class LUCBReservation:
    reservation_id: int
    phase: PullPhase
    iteration: int
    requests: tuple[PullRequest, ...]
    reserved_cost: float
    critical_high: str | None = None
    critical_low: str | None = None


@dataclass(frozen=True)
class PullObservation:
    delta: float
    is_exact: bool
    cost: float


@dataclass(frozen=True)
class ArmSnapshot:
    candidate_id: str
    pull_count: int
    mean: float | None
    radius: float | None
    lower_confidence: float | None
    upper_confidence: float | None
    is_exact: bool


@dataclass(frozen=True)
class LUCBResult:
    status: LUCBStatus
    certified: bool
    initialized: bool
    recommendation: str | None
    arms: tuple[ArmSnapshot, ...]
    total_pulls: int
    spent_cost: float
    reserved_cost: float
    completed_iterations: int
    round_counter: int
    critical_high: str | None
    critical_low: str | None
    stopping_gap: float | None
    reason: str | None

    def arm(self, candidate_id: str) -> ArmSnapshot:
        for arm in self.arms:
            if arm.candidate_id == candidate_id:
                return arm
        raise KeyError(candidate_id)


@dataclass
class _Arm:
    candidate_id: str
    pull_count: int = 0
    total: float = 0.0
    exact_value: float | None = None

    @property
    def is_exact(self) -> bool:
        return self.exact_value is not None

    @property
    def mean(self) -> float | None:
        if self.exact_value is not None:
            return self.exact_value
        if self.pull_count == 0:
            return None
        return self.total / self.pull_count


class PairedLUCB1:
    """Stateful LUCB1 scheduler; observation generation stays outside the algorithm."""

    def __init__(self, config: LUCBConfig) -> None:
        self.config = config
        self._order = (config.incumbent_id, *config.candidate_ids)
        self._arms = {candidate_id: _Arm(candidate_id) for candidate_id in self._order}
        self._arms[config.incumbent_id].exact_value = 0.0
        self._status: LUCBStatus = "running"
        self._reason: str | None = None
        self._initialized = False
        self._spent_cost = 0.0
        self._completed_iterations = 0
        self._round_counter = 0
        self._next_reservation_id = 0
        self._reservation: LUCBReservation | None = None
        self._outstanding: dict[PullRequest, None] = {}
        self._last_critical: tuple[str, str, float] | None = None

    def ask(self) -> LUCBReservation | None:
        """Reserve the entire initialization or next critical LUCB iteration."""

        if self._reservation is not None:
            raise RuntimeError("cannot ask while a reservation has outstanding observations")
        if self._status != "running":
            return None
        if not self._initialized:
            required = len(self.config.candidate_ids) * self.config.pull_cost
            if not self._affordable(required):
                self._status = "initialization_infeasible"
                self._reason = "budget below the registered K-pull initialization floor"
                return None
            return self._reserve(
                self.config.candidate_ids,
                phase="initialization",
                iteration=0,
            )

        high, low, gap = self._critical_pair()
        self._last_critical = (high, low, gap)
        if gap < self.config.epsilon:
            self._status = "certified"
            self._reason = "strict LUCB separation condition satisfied"
            return None

        pull_ids = tuple(
            candidate_id for candidate_id in (high, low) if not self._arms[candidate_id].is_exact
        )
        if not pull_ids:
            raise RuntimeError("nonterminal LUCB state has no uncertain critical arm")
        required = len(pull_ids) * self.config.pull_cost
        if not self._affordable(required):
            self._status = "budget_censored"
            self._reason = "remaining budget cannot fund the complete critical iteration"
            return None
        return self._reserve(
            pull_ids,
            phase="critical",
            iteration=self._round_counter,
            critical_high=high,
            critical_low=low,
        )

    def observe(
        self,
        request: PullRequest,
        *,
        delta: float,
        is_exact: bool,
        cost: float,
    ) -> None:
        """Commit one paid observation belonging to the active reservation."""

        if request not in self._outstanding:
            raise ValueError("pull request is not outstanding")
        if not isinstance(is_exact, bool):
            raise TypeError("is_exact must be boolean")
        if not math.isfinite(delta):
            raise ValueError("observed delta must be finite")
        if not self.config.lower_bound <= delta <= self.config.upper_bound:
            raise ValueError("observed delta is outside declared reward bounds")
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError("observed cost must be finite and nonnegative")
        if cost > request.reserved_cost + 1e-12:
            raise ValueError("observed cost exceeds reserved pull cost")
        arm = self._arms[request.candidate_id]
        if arm.is_exact:
            raise ValueError("exact arm cannot receive another observation")
        if request.replicate_index != arm.pull_count:
            raise ValueError("pull replicate index does not match the arm observation count")
        if is_exact and arm.pull_count:
            raise ValueError("an arm cannot change from stochastic to exact")

        arm.pull_count += 1
        arm.total += float(delta)
        if is_exact:
            arm.exact_value = float(delta)
        self._spent_cost += float(cost)
        del self._outstanding[request]
        if self._outstanding:
            return

        reservation = self._reservation
        if reservation is None:  # pragma: no cover - protected by outstanding membership
            raise RuntimeError("completed pulls have no reservation")
        self._reservation = None
        if reservation.phase == "initialization":
            self._initialized = True
            self._round_counter = 2
        else:
            self._completed_iterations += 1
            self._round_counter += 1

    def result(self) -> LUCBResult:
        snapshots = tuple(self._snapshot(candidate_id) for candidate_id in self._order)
        recommendation = self._empirical_best() if self._initialized else None
        if self._last_critical is None:
            critical_high = critical_low = None
            stopping_gap = None
        else:
            critical_high, critical_low, stopping_gap = self._last_critical
        return LUCBResult(
            status=self._status,
            certified=self._status == "certified",
            initialized=self._initialized,
            recommendation=recommendation,
            arms=snapshots,
            total_pulls=sum(
                self._arms[candidate_id].pull_count for candidate_id in self.config.candidate_ids
            ),
            spent_cost=self._spent_cost,
            reserved_cost=sum(request.reserved_cost for request in self._outstanding),
            completed_iterations=self._completed_iterations,
            round_counter=self._round_counter,
            critical_high=critical_high,
            critical_low=critical_low,
            stopping_gap=stopping_gap,
            reason=self._reason,
        )

    def _reserve(
        self,
        candidate_ids: tuple[str, ...],
        *,
        phase: PullPhase,
        iteration: int,
        critical_high: str | None = None,
        critical_low: str | None = None,
    ) -> LUCBReservation:
        reservation_id = self._next_reservation_id
        self._next_reservation_id += 1
        requests = tuple(
            PullRequest(
                reservation_id=reservation_id,
                candidate_id=candidate_id,
                replicate_index=self._arms[candidate_id].pull_count,
                phase=phase,
                iteration=iteration,
                reserved_cost=self.config.pull_cost,
            )
            for candidate_id in candidate_ids
        )
        reservation = LUCBReservation(
            reservation_id=reservation_id,
            phase=phase,
            iteration=iteration,
            requests=requests,
            reserved_cost=len(requests) * self.config.pull_cost,
            critical_high=critical_high,
            critical_low=critical_low,
        )
        self._reservation = reservation
        self._outstanding = dict.fromkeys(requests)
        return reservation

    def _critical_pair(self) -> tuple[str, str, float]:
        high = self._empirical_best()
        competitors = (candidate_id for candidate_id in self._order if candidate_id != high)
        low = max(competitors, key=lambda candidate_id: self._bounds(candidate_id)[1])
        high_lower = self._bounds(high)[0]
        low_upper = self._bounds(low)[1]
        return high, low, low_upper - high_lower

    def _empirical_best(self) -> str:
        return max(self._order, key=lambda candidate_id: self._required_mean(candidate_id))

    def _required_mean(self, candidate_id: str) -> float:
        mean = self._arms[candidate_id].mean
        if mean is None:
            raise RuntimeError("LUCB arm has not completed initialization")
        return mean

    def _bounds(self, candidate_id: str) -> tuple[float, float, float]:
        arm = self._arms[candidate_id]
        mean = self._required_mean(candidate_id)
        if arm.is_exact:
            return mean, mean, 0.0
        if self._round_counter < 1 or arm.pull_count < 1:
            raise RuntimeError("confidence bound requested before initialization")
        radius = (self.config.upper_bound - self.config.lower_bound) * math.sqrt(
            math.log((5.0 / 4.0) * len(self._order) * self._round_counter**4 / self.config.delta)
            / (2.0 * arm.pull_count)
        )
        return mean - radius, mean + radius, radius

    def _snapshot(self, candidate_id: str) -> ArmSnapshot:
        arm = self._arms[candidate_id]
        if arm.mean is None:
            lower = upper = radius = None
        else:
            lower, upper, radius = self._bounds(candidate_id)
        return ArmSnapshot(
            candidate_id=candidate_id,
            pull_count=arm.pull_count,
            mean=arm.mean,
            radius=radius,
            lower_confidence=lower,
            upper_confidence=upper,
            is_exact=arm.is_exact,
        )

    def _affordable(self, reserved_cost: float) -> bool:
        return self._spent_cost + reserved_cost <= self.config.budget + 1e-12


def run_paired_lucb(
    state: PairedLUCB1,
    query: Callable[[PullRequest], PullObservation],
) -> LUCBResult:
    """Drive a state with a caller-owned observation callback."""

    while (reservation := state.ask()) is not None:
        for request in reservation.requests:
            observation = query(request)
            if not isinstance(observation, PullObservation):
                raise TypeError("LUCB query callback must return PullObservation")
            state.observe(
                request,
                delta=observation.delta,
                is_exact=observation.is_exact,
                cost=observation.cost,
            )
    return state.result()
