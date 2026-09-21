"""Locked, decimal CNY reservations; charges include compute, disk and transfer.

Reservations must include the full bounded runtime and cleanup allowance BEFORE
RunInstances. The ledger is an admission gate, not a substitute for a provider
billing query or a controller that enforces the reserved runtime.
"""

from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .archive import atomic_json


class BudgetExceeded(RuntimeError):
    pass


def _amount(value) -> Decimal:
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("cost must be a finite nonnegative number") from exc
    if not number.is_finite() or number < 0:
        raise ValueError("cost must be a finite nonnegative number")
    return number


class BudgetLedger:
    LIMIT = Decimal(500)

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _locked(self):
        with self.path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = (
                json.loads(self.path.read_text())
                if self.path.exists()
                else {"currency": "CNY", "limit": "500", "actual_cost_cny": "0", "reservations": {}}
            )
            if data["currency"] != "CNY" or _amount(data["limit"]) != self.LIMIT:
                raise ValueError("budget ledger does not match the fixed CNY 500 limit")
            yield data

    def read(self):
        with self._locked() as data:
            return data

    def _remaining(self, data) -> Decimal:
        charges = _amount(data["actual_cost_cny"])
        reserved = sum(
            (
                _amount(row["total"])
                for row in data["reservations"].values()
                if row["status"] == "reserved"
            ),
            Decimal(0),
        )
        return max(Decimal(0), self.LIMIT - charges - reserved)

    def remaining(self) -> Decimal:
        with self._locked() as data:
            return self._remaining(data)

    def reserve(self, run_id: str, *, compute, disks, network, contingency):
        parts = {
            name: str(_amount(value))
            for name, value in {
                "compute": compute,
                "disks": disks,
                "network": network,
                "contingency": contingency,
            }.items()
        }
        total = sum((_amount(value) for value in parts.values()), Decimal(0))
        if not run_id or total <= 0:
            raise ValueError("reservation must name a run with positive bounded cost")
        record = {"status": "reserved", "parts": parts, "total": str(total)}
        with self._locked() as data:
            previous = data["reservations"].get(run_id)
            if previous:
                if previous != record:
                    raise ValueError("existing reservation cannot be changed or reused")
                return previous
            if total > self._remaining(data):
                raise BudgetExceeded(
                    "BUDGET_LIMIT_REACHED: request exceeds remaining CNY 500 allocation"
                )
            data["reservations"][run_id] = record
            atomic_json(self.path, data)
        return record

    def settle(self, run_id: str, *, compute, disks, network, other):
        parts = {
            name: str(_amount(value))
            for name, value in {
                "compute": compute,
                "disks": disks,
                "network": network,
                "other": other,
            }.items()
        }
        total = sum((_amount(value) for value in parts.values()), Decimal(0))
        with self._locked() as data:
            previous = data["reservations"].get(run_id)
            if not previous:
                raise ValueError("cannot settle an unregistered run")
            if previous["status"] == "settled":
                if previous["actual_parts"] != parts:
                    raise ValueError("settled cost is immutable")
                return previous
            # Never hide a real bill above the reservation; prevent further admissions.
            data["actual_cost_cny"] = str(_amount(data["actual_cost_cny"]) + total)
            previous.update(
                {"status": "settled", "actual_parts": parts, "actual_total": str(total)}
            )
            atomic_json(self.path, data)
            return previous
