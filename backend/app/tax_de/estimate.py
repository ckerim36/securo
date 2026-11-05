"""Annual income tax estimate for an employee, line by line.

Every amount comes back with the rule it follows, so the agent can explain
the result instead of restating a number it cannot justify.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import ROUND_DOWN, Decimal

from app.tax_de.income_tax import church_tax, flat_tax, floor_euro, income_tax, solidarity_surcharge
from app.tax_de.params import TaxYearParams, load_year
from app.tax_de.social import Contributions, employee_contributions

ZERO = Decimal(0)
HUNDRED = Decimal(100)


def company_car_benefit_monthly(
    list_price: Decimal,
    *,
    share_percent: Decimal = Decimal(1),
    commute_km: int = 0,
    commute_trips_per_month: int | None = None,
    user_payment_monthly: Decimal = ZERO,
) -> Decimal:
    """Geldwerter Vorteil per month under the list price method (§ 8 Abs. 2,
    § 6 Abs. 1 Nr. 4 EStG). `share_percent` is 1, 0.5 (plug-in hybrid or
    EV above the cap) or 0.25 (EV). The commute surcharge is 0.03 % per km
    of the same reduced base, or 0.002 % per km and trip when fewer than
    15 trips a month are recorded (Einzelbewertung). A user payment
    (Nutzungsentgelt) reduces the benefit, not below zero.
    """
    base_price = (list_price / 100).to_integral_value(rounding=ROUND_DOWN) * 100
    reduced = base_price * share_percent  # the reduction applies to the surcharge too
    private = base_price * share_percent / HUNDRED
    surcharge = ZERO
    if commute_km > 0:
        if commute_trips_per_month is not None and commute_trips_per_month < 15:
            surcharge = reduced * Decimal("0.00002") * commute_km * commute_trips_per_month
        else:
            surcharge = reduced * Decimal("0.0003") * commute_km
    return max(ZERO, private + surcharge - user_payment_monthly)


@dataclass
class WorkExpenses:
    home_office_days: int = 0
    # Arbeitszimmer when it is the centre of all professional activity:
    # either the flat annual amount or documented actual costs.
    home_office_room: bool = False
    home_office_room_actual: Decimal = ZERO
    commute_km: int = 0
    commute_days: int = 0
    work_equipment: Decimal = ZERO
    software_subscriptions: Decimal = ZERO
    training: Decimal = ZERO
    phone_internet: Decimal = ZERO
    travel_unreimbursed: Decimal = ZERO
    professional_fees: Decimal = ZERO  # Berufsverbände
    union_fees: Decimal = ZERO  # Gewerkschaft
    legal_insurance_work_share: Decimal = ZERO
    account_fee_flat: bool = True
    other: Decimal = ZERO


@dataclass
class TaxProfile:
    year: int = 2026
    age: int = 30
    joint: bool = False
    childless: bool = True
    church_rate: Decimal = ZERO  # 0.08 or 0.09 when liable
    saxony: bool = False
    # Pay, annual, gross: everything the employer pays in cash.
    salary: Decimal = ZERO
    bonus: Decimal = ZERO
    overtime_pay: Decimal = ZERO
    tax_free_pay: Decimal = ZERO  # e.g. § 3b night/Sunday surcharges; informational
    company_car_benefit_annual: Decimal = ZERO
    # Contributions
    statutory_health: bool = True
    health_extra_rate: Decimal | None = None
    sick_pay_entitlement: bool = True
    bav_conversion: Decimal = ZERO  # Entgeltumwandlung into § 3 Nr. 63
    ruerup: Decimal = ZERO
    other_insurance: Decimal = ZERO  # liability, accident, supplementary health ...
    donations: Decimal = ZERO
    medical_costs: Decimal = ZERO  # unreimbursed, § 33
    household_services_labour: Decimal = ZERO  # § 35a Abs. 2
    craftsmen_labour: Decimal = ZERO  # § 35a Abs. 3
    expenses: WorkExpenses = field(default_factory=WorkExpenses)
    # Capital income (gross, after Teilfreistellung) and what was withheld
    capital_income: Decimal = ZERO
    wage_tax_withheld: Decimal | None = None
    soli_withheld: Decimal | None = None


@dataclass
class Line:
    key: str
    amount: Decimal
    note: str


def _commute(e: WorkExpenses, p: TaxYearParams) -> Decimal:
    if e.commute_km <= 0 or e.commute_days <= 0:
        return ZERO
    first = min(e.commute_km, 20) * p.commute_rate_first_km
    rest = max(0, e.commute_km - 20) * p.commute_rate_from_21_km
    return (first + rest) * e.commute_days


def _home_office(e: WorkExpenses, p: TaxYearParams) -> tuple[Decimal, str]:
    day_flat = min(e.home_office_days * p.home_office_per_day, p.home_office_max)
    room = max(p.home_office_room_flat, e.home_office_room_actual) if e.home_office_room else ZERO
    if room > day_flat:
        return room, "Arbeitszimmer (§ 4 Abs. 5 Nr. 6b EStG), Jahrespauschale oder tatsächliche Kosten"
    return day_flat, f"Homeoffice-Pauschale {e.home_office_days} Tage, gedeckelt (§ 4 Abs. 5 Nr. 6c EStG)"


def work_expense_lines(e: WorkExpenses, p: TaxYearParams) -> list[Line]:
    ho, ho_note = _home_office(e, p)
    lines = [
        Line("home_office", ho, ho_note),
        Line("commute", _commute(e, p), "Entfernungspauschale § 9 Abs. 1 Nr. 4 EStG"),
        Line("work_equipment", e.work_equipment, "Arbeitsmittel § 9 Abs. 1 Nr. 6 EStG"),
        Line("software_subscriptions", e.software_subscriptions, "beruflich genutzte Software/Abos (beruflicher Anteil)"),
        Line("training", e.training, "Fortbildungskosten"),
        Line("phone_internet", e.phone_internet, "Telekommunikation, beruflicher Anteil"),
        Line("travel_unreimbursed", e.travel_unreimbursed, "nicht erstattete Reisekosten inkl. Verpflegungsmehraufwand"),
        Line("professional_fees", e.professional_fees, "Beiträge zu Berufsverbänden"),
        Line("legal_insurance_work_share", e.legal_insurance_work_share, "Rechtsschutz, Anteil Arbeitsrecht"),
        Line("account_fee", p.account_fee_flat if e.account_fee_flat else ZERO, "Kontoführung, Pauschale der Verwaltungspraxis"),
        Line("other", e.other, "sonstige Werbungskosten"),
    ]
    return [ln for ln in lines if ln.amount > 0]


def _reasonable_burden(total_income: Decimal, p: TaxYearParams) -> Decimal:
    """§ 33 Abs. 3, applied in steps (BFH VI R 75/14)."""
    burden, lower = ZERO, ZERO
    for upper, rate in p.reasonable_burden_single:
        top = total_income if upper is None else min(total_income, upper)
        if top > lower:
            burden += (top - lower) * rate / HUNDRED
        if upper is None or total_income <= upper:
            break
        lower = upper
    return floor_euro(burden)


def _vorsorge(c: Contributions, prof: TaxProfile, p: TaxYearParams) -> list[Line]:
    pension_total = c.pension + c.employer_pension + prof.ruerup
    capped = min(pension_total, p.pension_contribution_cap_single * (2 if prof.joint else 1))
    pension = max(ZERO, capped * p.pension_deductible_share / HUNDRED - c.employer_pension)
    health = c.health * (1 - p.sick_pay_reduction / HUNDRED) if prof.sick_pay_entitlement else c.health
    basic = health + c.care
    other = c.unemployment + prof.other_insurance
    cap = p.other_insurance_cap_employee * (2 if prof.joint else 1)
    rest = max(basic, min(cap, basic + other))
    return [
        Line("pension_contributions", floor_euro(pension), "Altersvorsorge § 10 Abs. 1 Nr. 2, Abs. 3 EStG (AN-Anteil RV + Rürup)"),
        Line(
            "health_care_and_other",
            floor_euro(rest),
            "Basis-KV/PV voll, übrige Vorsorge nur bis zum Höchstbetrag § 10 Abs. 4 EStG"
            + (" (bereits durch Basis-KV/PV ausgeschöpft)" if basic >= cap else ""),
        ),
    ]


def _tax_on(zve: Decimal, prof: TaxProfile, p: TaxYearParams) -> tuple[Decimal, Decimal, Decimal]:
    est = income_tax(zve, p, joint=prof.joint)
    soli = solidarity_surcharge(est, p, joint=prof.joint)
    church = church_tax(est, prof.church_rate) if prof.church_rate else ZERO
    return est, soli, church


def approximate_wage_tax(prof: TaxProfile, p: TaxYearParams | None = None) -> Decimal:
    """Lohnsteuer + Soli the employer withholds over the year, roughly:
    tariff on pay minus the employee allowance, the special expenses
    allowance and the Vorsorgepauschale. The real Programmablaufplan has
    more detail; prefer the figure from the Lohnsteuerbescheinigung."""
    p = p or load_year(prof.year)
    pay = prof.salary + prof.bonus + prof.overtime_pay + prof.company_car_benefit_annual - prof.bav_conversion
    c = employee_contributions(
        pay, p, health_extra_rate=prof.health_extra_rate, childless=prof.childless,
        age=prof.age, saxony=prof.saxony, statutory_health=prof.statutory_health,
    )
    kv_base = min(pay, p.bbg_health_annual)
    extra = p.rate_health_avg_extra if prof.health_extra_rate is None else prof.health_extra_rate
    health = kv_base * (p.rate_health_reduced + extra) / 2 / HUNDRED if prof.statutory_health else ZERO
    allowance = c.pension + health + c.care
    if p.wage_tax_allowance_includes_unemployment:
        allowance += c.unemployment
    zve = pay - p.employee_allowance - p.special_expenses_allowance - allowance
    est, soli, church = _tax_on(zve, prof, p)
    return est + soli + church


def estimate(prof: TaxProfile) -> dict:
    p = load_year(prof.year)
    gross = prof.salary + prof.bonus + prof.overtime_pay + prof.company_car_benefit_annual
    taxable_pay = gross - prof.bav_conversion
    contrib = employee_contributions(
        taxable_pay, p, health_extra_rate=prof.health_extra_rate, childless=prof.childless,
        age=prof.age, saxony=prof.saxony, statutory_health=prof.statutory_health,
    )

    wk_lines = work_expense_lines(prof.expenses, p)
    wk_actual = sum((ln.amount for ln in wk_lines), ZERO)
    wk = max(wk_actual, p.employee_allowance)
    union = prof.expenses.union_fees
    if union > 0:
        # From 2026 union fees come on top of the allowance (§ 9a S. 3).
        wk = wk + union if p.union_fees_on_top_of_allowance else max(wk_actual + union, p.employee_allowance)
        wk_lines.append(Line("union_fees", union, "Gewerkschaftsbeiträge (§ 9a S. 3 EStG: neben dem Pauschbetrag)"))
        wk_actual += union
    employment_income = max(ZERO, taxable_pay - wk)

    vorsorge = _vorsorge(contrib, prof, p)
    special = prof.donations if prof.donations > p.special_expenses_allowance else p.special_expenses_allowance
    total_income = employment_income
    burden = _reasonable_burden(total_income, p)
    extraordinary = max(ZERO, prof.medical_costs - burden)

    zve = floor_euro(total_income - sum((ln.amount for ln in vorsorge), ZERO) - special - extraordinary)
    est_tariff, _, _ = _tax_on(zve, prof, p)
    reduction = min(prof.household_services_labour * p.household_services_rate / HUNDRED, p.household_services_max)
    reduction += min(prof.craftsmen_labour * p.craftsmen_rate / HUNDRED, p.craftsmen_max)
    est = max(ZERO, est_tariff - floor_euro(reduction))
    soli = solidarity_surcharge(est, p, joint=prof.joint)
    church = church_tax(est, prof.church_rate) if prof.church_rate else ZERO

    # Marginal rate on the next 1,000 euros of taxable income, Soli and
    # church tax included: what one more euro of deductions is worth.
    base_total = sum(_tax_on(zve, prof, p), ZERO)
    marginal = (sum(_tax_on(zve + 1000, prof, p), ZERO) - base_total) / 10

    saver = p.saver_allowance_single * (2 if prof.joint else 1)
    capital = flat_tax(prof.capital_income - saver, p, church_rate=prof.church_rate)

    withheld = prof.wage_tax_withheld
    withheld_is_estimate = withheld is None
    if withheld is None:
        withheld = approximate_wage_tax(prof, p)
    elif prof.soli_withheld is not None:
        withheld += prof.soli_withheld
    total = est + soli + church

    def f(v: Decimal) -> float:
        return float(round(v, 2))

    return {
        "year": p.year,
        "gross_pay": f(gross),
        "taxable_pay": f(taxable_pay),
        "social_insurance": {
            "pension": f(contrib.pension), "unemployment": f(contrib.unemployment),
            "health": f(contrib.health), "care": f(contrib.care), "total": f(contrib.total),
        },
        "work_expenses": {
            "lines": [{"key": ln.key, "amount": f(ln.amount), "note": ln.note} for ln in wk_lines],
            "actual_total": f(wk_actual),
            "employee_allowance": f(p.employee_allowance),
            "applied": f(wk),
            "above_allowance": f(max(ZERO, wk - p.employee_allowance)),
        },
        "employment_income": f(employment_income),
        "vorsorge": [{"key": ln.key, "amount": f(ln.amount), "note": ln.note} for ln in vorsorge],
        "special_expenses_other": f(special),
        "extraordinary": {"costs": f(prof.medical_costs), "reasonable_burden": f(burden), "deductible": f(extraordinary)},
        "taxable_income": f(zve),
        "income_tax": f(est),
        "tax_reduction_35a": f(floor_euro(reduction)),
        "solidarity_surcharge": f(soli),
        "church_tax": f(church),
        "total_tax": f(total),
        "average_rate_percent": f(total / gross * 100) if gross else 0.0,
        "marginal_rate_percent": f(marginal),
        "wage_tax_withheld": f(withheld),
        "wage_tax_withheld_is_estimate": withheld_is_estimate,
        "refund": f(withheld - total),
        "capital_income": {
            "income": f(prof.capital_income), "saver_allowance": f(saver),
            "flat_tax_total": f(capital["total"]),
        },
        "assumptions": [
            "Sozialversicherung als Jahreswert auf min(Entgelt, Jahres-BBG); gilt für ganzjährige Beschäftigung.",
            "Keine Einkünfte aus Gewerbe, Vermietung oder Ausland; kein Progressionsvorbehalt.",
            "Ohne Lohnsteuerbescheinigung ist die einbehaltene Lohnsteuer eine Näherung." if withheld_is_estimate else
            "Einbehaltene Lohnsteuer laut Angabe.",
        ],
        "sources": p.sources,
    }


def deduction_value(prof: TaxProfile, extra_work_expenses: Decimal = ZERO, extra_special_expenses: Decimal = ZERO) -> dict:
    """What an additional deduction is worth in tax, all else equal."""
    base = estimate(prof)
    changed = replace(
        prof,
        expenses=replace(prof.expenses, other=prof.expenses.other + extra_work_expenses),
        donations=prof.donations + extra_special_expenses,
    )
    after = estimate(changed)
    saved = base["total_tax"] - after["total_tax"]
    amount = float(extra_work_expenses + extra_special_expenses)
    return {
        "amount": amount,
        "tax_saved": round(saved, 2),
        "effective_rate_percent": round(saved / amount * 100, 2) if amount else 0.0,
        "note": "Werbungskosten wirken erst oberhalb des Arbeitnehmer-Pauschbetrags; "
                "Spenden erst oberhalb des Sonderausgaben-Pauschbetrags.",
    }
