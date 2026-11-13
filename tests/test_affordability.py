"""Affordability calculations (pure arithmetic, no data needed)."""

import pytest

from src.affordability import (
    AffordabilityInputError, amortization_schedule, calculate, monthly_payment, rate_sensitivity,
)


def test_known_payment():
    # $200,000 at 6% for 30 years → $1,199.10 (standard amortisation table value)
    assert monthly_payment(200_000, 6.0, 30) == pytest.approx(1199.10, abs=0.01)


def test_zero_rate_and_zero_loan():
    assert monthly_payment(120_000, 0.0, 10) == pytest.approx(1000.0)
    assert monthly_payment(0, 5.0, 30) == 0.0


def test_full_calculation():
    r = calculate(250_000, 50_000, 6.0, 30, monthly_income=8000)
    assert r.loan_amount == 200_000
    assert r.down_payment_pct == pytest.approx(20.0)
    assert r.monthly_payment == pytest.approx(1199.10, abs=0.01)
    assert r.total_repayment == pytest.approx(r.monthly_payment * 360)
    assert r.total_interest == pytest.approx(r.total_repayment - 200_000)
    assert r.payment_to_income_pct == pytest.approx(1199.10 / 8000 * 100, abs=0.01)


def test_manual_loan_amount():
    r = calculate(300_000, 60_000, 5.0, 20, loan_amount=150_000)
    assert r.loan_amount == 150_000


@pytest.mark.parametrize("args", [(-1, 0, 5, 30), (100, 200, 5, 30), (100_000, 0, 40, 30), (100_000, 0, 5, 0),
                                  (100_000, -5, 5, 30)])
def test_invalid_inputs(args):
    with pytest.raises(AffordabilityInputError):
        calculate(*args)


def test_schedule_pays_off_loan():
    s = amortization_schedule(200_000, 6.0, 30)
    assert len(s) == 30
    assert s["principal"].sum() == pytest.approx(200_000, abs=1)
    assert s["balance"].iloc[-1] == pytest.approx(0, abs=1)
    assert s["interest"].iloc[0] > s["interest"].iloc[-1]


def test_rate_sensitivity_monotonic():
    s = rate_sensitivity(200_000, 30, [3, 4, 5, 6])
    assert s["monthly_payment"].is_monotonic_increasing
