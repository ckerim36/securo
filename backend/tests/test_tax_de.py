"""German tax engine: statutory arithmetic, checked against the law's own
structure (continuous tariff, thresholds) and hand-computed cases."""
from __future__ import annotations

from decimal import Decimal

import pytest

from app.tax_de.estimate import (
    TaxProfile,
    WorkExpenses,
    company_car_benefit_monthly,
    deduction_value,
    estimate,
)
from app.tax_de.income_tax import flat_tax, income_tax, solidarity_surcharge, tariff
from app.tax_de.params import available_years, load_year
from app.tax_de.social import employee_contributions

D = Decimal


@pytest.fixture(params=available_years())
def p(request):
    return load_year(request.param)


def test_no_tax_up_to_basic_allowance(p):
    assert tariff(p.basic_allowance, p) == 0
    assert tariff(p.basic_allowance + 100, p) > 0


def test_tariff_is_continuous_at_zone_boundaries(p):
    # A statute typo or a mistyped coefficient shows up as a jump here.
    for zone in p.zones:
        if zone.upper is None:
            continue
        below = tariff(zone.upper, p)
        above = tariff(zone.upper + 1, p)
        assert 0 <= above - below <= 1, (zone.upper, below, above)


def test_tariff_is_monotonic_and_top_rate_is_45(p):
    last = D(0)
    for zve in range(0, 400_001, 2_500):
        t = tariff(D(zve), p)
        assert t >= last
        last = t
    assert tariff(D(500_000), p) - tariff(D(499_000), p) == 450


def test_splitting_doubles_half_income(p):
    assert income_tax(D(100_000), p, joint=True) == 2 * tariff(D(50_000), p)


def test_soli_threshold_and_taper(p):
    t = p.soli_threshold_single
    assert solidarity_surcharge(t, p) == 0
    # Just above the threshold the taper applies, not the full 5.5 %.
    assert solidarity_surcharge(t + 1000, p) == (D(1000) * p.soli_taper_rate).quantize(D("0.01"))
    # Far above it, the full rate.
    assert solidarity_surcharge(D(100_000), p) == D("5500.00")


def test_contributions_capped_at_ceilings(p):
    high = employee_contributions(D(200_000), p, childless=True, age=27)
    at_cap = employee_contributions(p.bbg_pension_annual, p, childless=True, age=27)
    assert high.pension == at_cap.pension
    assert high.health == employee_contributions(p.bbg_health_annual, p, childless=True, age=27).health
    assert high.pension == (p.bbg_pension_annual * p.rate_pension / 2 / 100).quantize(D("0.01"))


def test_childless_care_surcharge_from_23(p):
    young = employee_contributions(D(50_000), p, childless=True, age=22)
    older = employee_contributions(D(50_000), p, childless=True, age=23)
    assert older.care > young.care


def test_company_car_plug_in_hybrid_with_user_payment():
    # 72,000 list price, 0.5 % rule, no commute (full home office), 200 paid by the employee.
    assert company_car_benefit_monthly(D(72_000), share_percent=D("0.5"), user_payment_monthly=D(200)) == D(160)
    # List price is rounded down to full hundreds.
    assert company_car_benefit_monthly(D(72_099), share_percent=D(1)) == D(720)
    # Commute surcharge: 0.03 % per km of the reduced base.
    assert company_car_benefit_monthly(D(72_000), share_percent=D("0.5"), commute_km=10) == D(360) + D("108.0")
    # Einzelbewertung below 15 trips: 0.002 % per km and trip.
    assert company_car_benefit_monthly(
        D(72_000), share_percent=D("0.5"), commute_km=10, commute_trips_per_month=4,
    ) == D(360) + D("28.8")


def test_flat_tax_with_and_without_church():
    p = load_year(2026)
    plain = flat_tax(D(1000), p)
    assert plain["tax"] == D("250.00") and plain["soli"] == D("13.75")
    church = flat_tax(D(1000), p, church_rate=D("0.09"))
    assert church["tax"] == D("244.49")  # 1000 / (4 + 0.09)


def _profile(**over) -> TaxProfile:
    base = dict(
        year=2026, age=27, salary=D(60_000),
        expenses=WorkExpenses(home_office_days=0, account_fee_flat=False),
    )
    base.update(over)
    return TaxProfile(**base)


def test_work_expenses_below_allowance_use_allowance():
    r = estimate(_profile())
    assert r["work_expenses"]["applied"] == r["work_expenses"]["employee_allowance"]
    assert r["work_expenses"]["above_allowance"] == 0


def test_home_office_flat_is_capped():
    p = load_year(2026)
    r = estimate(_profile(expenses=WorkExpenses(home_office_days=300, account_fee_flat=False)))
    line = next(x for x in r["work_expenses"]["lines"] if x["key"] == "home_office")
    assert line["amount"] == float(p.home_office_max)


def test_deduction_value_respects_allowance_and_marginal_rate():
    small = deduction_value(_profile(), extra_work_expenses=D(500))
    assert small["tax_saved"] == 0  # still below the Arbeitnehmer-Pauschbetrag
    prof = _profile(expenses=WorkExpenses(home_office_days=210, account_fee_flat=False))
    big = deduction_value(prof, extra_work_expenses=D(1000))
    marginal = estimate(prof)["marginal_rate_percent"]
    assert abs(big["effective_rate_percent"] - marginal) < 1.5


def test_refund_against_given_wage_tax():
    r = estimate(_profile(wage_tax_withheld=D(20_000)))
    assert r["wage_tax_withheld_is_estimate"] is False
    assert r["refund"] == pytest.approx(20_000 - r["total_tax"], abs=0.01)


def test_medical_costs_only_above_reasonable_burden():
    r = estimate(_profile(medical_costs=D(1_800)))
    assert r["extraordinary"]["deductible"] == max(0.0, 1_800 - r["extraordinary"]["reasonable_burden"])


def test_tariff_reference_values_2026():
    p = load_year(2026)
    # Hand-computed from § 32a Abs. 1 idF SteFeG.
    assert tariff(D(12_348), p) == 0
    assert tariff(D(50_000), p) == 10_548
    assert tariff(D(69_879), p) == 18_213  # 0.42 * 69,879 - 11,135.63


def test_union_fees_count_on_top_of_allowance_2026():
    with_union = estimate(_profile(expenses=WorkExpenses(union_fees=D(300), account_fee_flat=False)))
    assert with_union["work_expenses"]["applied"] == 1_230 + 300


@pytest.mark.asyncio
async def test_mcp_tools_return_json_ready_results():
    from mcp_server.tools.tax_de import tax_de_company_car, tax_de_deduction_value, tax_de_estimate, tax_de_parameters

    params = await tax_de_parameters(session=None, ctx=None, year=2026)
    assert params["basic_allowance"] == 12_348.0 and 2026 in params["available_years"]

    r = await tax_de_estimate(
        session=None, ctx=None, salary=60_000, age=27,
        expenses={"home_office_days": 210, "software_subscriptions": 600},
    )
    assert r["work_expenses"]["actual_total"] == 1_260 + 600 + 16
    assert isinstance(r["refund"], float)

    v = await tax_de_deduction_value(session=None, ctx=None, salary=60_000, age=27, extra_work_expenses=2_000)
    assert v["tax_saved"] > 0

    car = await tax_de_company_car(session=None, ctx=None, list_price=72_000, share_percent=0.5, user_payment_monthly=200)
    assert car == {"benefit_monthly": 160.0, "benefit_annual": 1_920.0}
