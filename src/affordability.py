"""Housing affordability explorer: standard amortising-loan arithmetic.

Mathematically separate from the machine-learning valuation model. Results are
illustrative calculations under user-supplied assumptions, not a mortgage
approval decision, personalised financial advice, or a lending offer. Taxes,
insurance, fees and mortgage insurance are not included.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

AFFORDABILITY_DISCLAIMER = (
    "Illustrative calculation only. Not a mortgage approval, lending offer or financial advice. "
    "Excludes property tax, insurance, fees and mortgage insurance."
)


class AffordabilityInputError(ValueError):
    pass


@dataclass
class AffordabilityResult:
    property_price: float
    down_payment: float
    loan_amount: float
    annual_rate_pct: float
    years: int
    monthly_payment: float
    down_payment_pct: float
    total_repayment: float
    total_interest: float
    monthly_income: float | None
    payment_to_income_pct: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def monthly_payment(principal: float, annual_rate_pct: float, years: int) -> float:
    """Level payment of a fully amortising loan: P·r / (1 − (1 + r)^−n)."""
    if principal < 0:
        raise AffordabilityInputError("Loan amount cannot be negative.")
    n = int(round(years * 12))
    if n <= 0:
        raise AffordabilityInputError("Loan duration must be at least one month.")
    if principal == 0:
        return 0.0
    r = annual_rate_pct / 100 / 12
    if r == 0:
        return principal / n
    return principal * r / (1 - (1 + r) ** -n)


def calculate(property_price: float, down_payment: float, annual_rate_pct: float, years: int,
              monthly_income: float | None = None, loan_amount: float | None = None) -> AffordabilityResult:
    errors = []
    if not np.isfinite(property_price) or property_price <= 0:
        errors.append("Property price must be a positive number.")
    if down_payment < 0:
        errors.append("Down payment cannot be negative.")
    if property_price > 0 and down_payment > property_price:
        errors.append("Down payment cannot exceed the property price.")
    if not 0 <= annual_rate_pct <= 30:
        errors.append("Annual interest rate must be between 0% and 30%.")
    if not 1 <= years <= 50:
        errors.append("Loan duration must be between 1 and 50 years.")
    if monthly_income is not None and monthly_income < 0:
        errors.append("Monthly income cannot be negative.")
    if loan_amount is not None and loan_amount < 0:
        errors.append("Loan amount cannot be negative.")
    if errors:
        raise AffordabilityInputError(" ".join(errors))

    loan = property_price - down_payment if loan_amount is None else loan_amount
    pay = monthly_payment(loan, annual_rate_pct, years)
    total = pay * years * 12
    pti = (pay / monthly_income * 100) if monthly_income else None
    return AffordabilityResult(
        property_price=float(property_price), down_payment=float(down_payment), loan_amount=float(loan),
        annual_rate_pct=float(annual_rate_pct), years=int(years), monthly_payment=float(pay),
        down_payment_pct=float(down_payment / property_price * 100), total_repayment=float(total),
        total_interest=float(total - loan), monthly_income=monthly_income, payment_to_income_pct=pti,
    )


def amortization_schedule(loan: float, annual_rate_pct: float, years: int) -> pd.DataFrame:
    """Yearly summary of interest paid, principal paid and remaining balance."""
    pay = monthly_payment(loan, annual_rate_pct, years)
    r = annual_rate_pct / 100 / 12
    bal = float(loan)
    rows = []
    for yr in range(1, years + 1):
        yi = yp = 0.0
        for _ in range(12):
            interest = bal * r
            principal = min(pay - interest, bal)
            bal -= principal
            yi += interest
            yp += principal
        rows.append({"year": yr, "interest": yi, "principal": yp, "balance": max(bal, 0.0)})
    return pd.DataFrame(rows)


def rate_sensitivity(loan: float, years: int, rates) -> pd.DataFrame:
    return pd.DataFrame({"annual_rate_pct": list(rates),
                         "monthly_payment": [monthly_payment(loan, r, years) for r in rates]})
