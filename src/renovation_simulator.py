"""Property Value Simulator: hypothetical, model-based "what-if" scenarios.

IMPORTANT: The model learned associations from observational historical sales.
The difference between two predictions shows how *the model's estimate*
responds to changed inputs. It is not evidence that a renovation would cause the
same change in market value, and no renovation costs or returns are implied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .prediction import PredictionEngine

SIMULATOR_DISCLAIMER = (
    "Hypothetical model-based scenario, not a renovation investment calculator. "
    "Differences reflect how the model's estimate responds to changed inputs, learned from "
    "historical Ames, Iowa sales (2006–2010). They do not establish that a renovation would "
    "cause this change in real market value, and they exclude renovation costs."
)


@dataclass
class ScenarioResult:
    name: str
    changes: dict
    original_price: float
    modified_price: float
    original_record: pd.DataFrame
    modified_record: pd.DataFrame
    changed_fields: pd.DataFrame
    lower: float = float("nan")
    upper: float = float("nan")
    warnings: list[str] = field(default_factory=list)

    @property
    def difference(self) -> float:
        return self.modified_price - self.original_price

    @property
    def percent(self) -> float:
        return self.difference / self.original_price * 100


def changed_fields(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in before.columns:
        a, b = before.iloc[0][c], after.iloc[0][c]
        same = (pd.isna(a) and pd.isna(b)) or (a == b)
        if not same:
            rows.append({"field": c, "before": a, "after": b})
    return pd.DataFrame(rows, columns=["field", "before", "after"])


def simulate(engine: PredictionEngine, base_inputs: dict, changes: dict, name: str = "Scenario",
             base_record: pd.Series | None = None) -> ScenarioResult:
    """Predict the original profile and the profile with ``changes`` applied."""
    if not changes:
        raise ValueError("A scenario needs at least one changed input.")
    original = engine.value(base_inputs, base=base_record)
    modified_inputs = {**base_inputs, **changes}
    modified = engine.value(modified_inputs, base=base_record)
    return ScenarioResult(
        name=name, changes=changes, original_price=original.estimate, modified_price=modified.estimate,
        original_record=original.record, modified_record=modified.record,
        changed_fields=changed_fields(original.record, modified.record),
        lower=modified.interval.lower, upper=modified.interval.upper,
        warnings=sorted(set(original.warnings + modified.warnings)),
    )


def simulate_many(engine: PredictionEngine, base_inputs: dict, scenarios: dict[str, dict],
                  base_record: pd.Series | None = None) -> list[ScenarioResult]:
    return [simulate(engine, base_inputs, ch, name, base_record) for name, ch in scenarios.items() if ch]


def sensitivity_curve(engine: PredictionEngine, base_inputs: dict, feature: str, values,
                      base_record: pd.Series | None = None) -> pd.DataFrame:
    """Model estimate as one input sweeps through ``values`` (others fixed)."""
    recs = []
    for v in values:
        rec, _ = engine.build_record({**base_inputs, feature: v}, base=base_record)
        recs.append(rec)
    preds = engine.predict_frame(pd.concat(recs, ignore_index=True))
    return pd.DataFrame({feature: list(values), "estimate": preds})


def scenarios_frame(results: list[ScenarioResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Scenario": r.name,
        "Changes": ", ".join(f"{k} → {v}" for k, v in r.changes.items()),
        "Original estimate": r.original_price,
        "Modified estimate": r.modified_price,
        "Difference": r.difference,
        "Difference %": r.percent,
    } for r in results])


def log_contribution_shift(expl_before: pd.Series, expl_after: pd.Series, top: int = 8) -> pd.DataFrame:
    idx = expl_before.index.union(expl_after.index)
    d = expl_after.reindex(idx, fill_value=0) - expl_before.reindex(idx, fill_value=0)
    d = d[np.abs(d) > 1e-6]
    d = d.reindex(d.abs().sort_values(ascending=False).index).head(top)
    return pd.DataFrame({"feature": d.index, "log_shift": d.to_numpy(), "pct": (np.exp(d.to_numpy()) - 1) * 100})
