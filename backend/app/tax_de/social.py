"""Employee social insurance contributions (Arbeitnehmeranteil).

Annual approximation: contributions on min(annual pay, annual BBG). That
is exact for a full year of employment whose monthly pay stays above or
below the ceiling, and matches the anteilige Jahres-BBG for one-off pay
such as a bonus. It ignores the Märzklausel and part-year employment.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.tax_de.income_tax import floor_cent
from app.tax_de.params import TaxYearParams

HUNDRED = Decimal(100)


@dataclass(frozen=True)
class Contributions:
    pension: Decimal
    unemployment: Decimal
    health: Decimal
    care: Decimal
    # Employer share of the pension contribution; tax-free and subtracted
    # again in the Vorsorge calculation.
    employer_pension: Decimal

    @property
    def total(self) -> Decimal:
        return self.pension + self.unemployment + self.health + self.care


def employee_contributions(
    pay: Decimal,
    p: TaxYearParams,
    *,
    health_extra_rate: Decimal | None = None,
    childless: bool = True,
    age: int = 30,
    saxony: bool = False,
    statutory_health: bool = True,
) -> Contributions:
    """`pay` is the annual Arbeitsentgelt subject to contributions,
    including the company car benefit and one-off payments. Rates are
    percentages as in the parameter file."""
    extra = p.rate_health_avg_extra if health_extra_rate is None else health_extra_rate
    rv_base = min(pay, p.bbg_pension_annual)
    kv_base = min(pay, p.bbg_health_annual)
    care_rate = p.rate_care / 2
    if saxony:
        # Sachsen kept the Buß- und Bettag; employees carry 0.5 points more.
        care_rate += Decimal("0.5")
    if childless and age >= p.care_childless_min_age:
        care_rate += p.rate_care_childless_extra
    pension = floor_cent(rv_base * p.rate_pension / 2 / HUNDRED)
    return Contributions(
        pension=pension,
        unemployment=floor_cent(rv_base * p.rate_unemployment / 2 / HUNDRED),
        health=floor_cent(kv_base * (p.rate_health_general + extra) / 2 / HUNDRED) if statutory_health else Decimal(0),
        care=floor_cent(kv_base * care_rate / HUNDRED) if statutory_health else Decimal(0),
        employer_pension=pension,
    )
