"""Per-year legal parameters, loaded from `years/<year>.toml`."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

YEARS_DIR = Path(__file__).parent / "years"


def _d(v: Any) -> Decimal:
    # Through str so 0.42 stays 0.42 and not its binary approximation.
    return Decimal(str(v))


@dataclass(frozen=True)
class TariffZone:
    """One zone of § 32a Abs. 1 EStG, valid for zvE up to `upper` (inclusive).

    Progressive zones compute (a * t + b) * t + c with
    t = (zvE - offset) / 10000; linear zones compute rate * zvE - minus.
    """

    upper: Decimal | None
    a: Decimal = Decimal(0)
    b: Decimal = Decimal(0)
    c: Decimal = Decimal(0)
    offset: Decimal = Decimal(0)
    rate: Decimal | None = None
    minus: Decimal = Decimal(0)


@dataclass(frozen=True)
class TaxYearParams:
    year: int
    sources: dict[str, str]
    # § 32a
    basic_allowance: Decimal
    zones: tuple[TariffZone, ...]
    # Solidaritätszuschlag
    soli_rate: Decimal
    soli_threshold_single: Decimal
    soli_taper_rate: Decimal
    # Werbungskosten / Sonderausgaben
    employee_allowance: Decimal
    special_expenses_allowance: Decimal
    home_office_per_day: Decimal
    home_office_max: Decimal
    home_office_room_flat: Decimal
    commute_rate_first_km: Decimal
    commute_rate_from_21_km: Decimal
    account_fee_flat: Decimal
    union_fees_on_top_of_allowance: bool
    # Vorsorgeaufwendungen
    pension_contribution_cap_single: Decimal
    pension_deductible_share: Decimal
    other_insurance_cap_employee: Decimal
    sick_pay_reduction: Decimal
    # Sozialversicherung
    bbg_pension_annual: Decimal
    bbg_health_annual: Decimal
    rate_pension: Decimal
    rate_unemployment: Decimal
    rate_health_general: Decimal
    rate_health_reduced: Decimal
    rate_health_avg_extra: Decimal
    rate_care: Decimal
    rate_care_childless_extra: Decimal
    care_childless_min_age: int
    compulsory_insurance_limit: Decimal
    # Lohnsteuer approximation
    wage_tax_allowance_includes_unemployment: bool
    # Betriebliche Altersversorgung
    bav_tax_free: Decimal
    bav_social_free: Decimal
    # Kapitalerträge
    saver_allowance_single: Decimal
    flat_tax_rate: Decimal
    fund_base_rate: Decimal
    equity_fund_exemption: Decimal
    # § 33 zumutbare Belastung, ledig ohne Kinder: (upper bound, rate)
    reasonable_burden_single: tuple[tuple[Decimal | None, Decimal], ...]
    # § 35a
    household_services_rate: Decimal
    household_services_max: Decimal
    craftsmen_rate: Decimal
    craftsmen_max: Decimal


def _zone(raw: dict[str, Any]) -> TariffZone:
    return TariffZone(
        upper=_d(raw["upper"]) if raw.get("upper") is not None else None,
        a=_d(raw.get("a", 0)),
        b=_d(raw.get("b", 0)),
        c=_d(raw.get("c", 0)),
        offset=_d(raw.get("offset", 0)),
        rate=_d(raw["rate"]) if raw.get("rate") is not None else None,
        minus=_d(raw.get("minus", 0)),
    )


def available_years() -> list[int]:
    return sorted(int(p.stem) for p in YEARS_DIR.glob("*.toml"))


@lru_cache
def load_year(year: int) -> TaxYearParams:
    path = YEARS_DIR / f"{year}.toml"
    if not path.exists():
        raise ValueError(f"No German tax parameters for {year}. Available: {available_years()}")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    t, s, w, v, sv, k, b, h = (
        raw["tariff"], raw["soli"], raw["werbungskosten"], raw["vorsorge"],
        raw["sozialversicherung"], raw["kapital"], raw["bav"], raw["haushalt"],
    )
    return TaxYearParams(
        year=int(raw["year"]),
        sources=dict(raw.get("sources", {})),
        basic_allowance=_d(t["basic_allowance"]),
        zones=tuple(_zone(z) for z in t["zones"]),
        soli_rate=_d(s["rate"]),
        soli_threshold_single=_d(s["threshold_single"]),
        soli_taper_rate=_d(s["taper_rate"]),
        employee_allowance=_d(w["employee_allowance"]),
        special_expenses_allowance=_d(w["special_expenses_allowance"]),
        home_office_per_day=_d(w["home_office_per_day"]),
        home_office_max=_d(w["home_office_max"]),
        home_office_room_flat=_d(w["home_office_room_flat"]),
        commute_rate_first_km=_d(w["commute_rate_first_km"]),
        commute_rate_from_21_km=_d(w["commute_rate_from_21_km"]),
        account_fee_flat=_d(w["account_fee_flat"]),
        union_fees_on_top_of_allowance=bool(w.get("union_fees_on_top_of_allowance", False)),
        pension_contribution_cap_single=_d(v["pension_contribution_cap_single"]),
        pension_deductible_share=_d(v["pension_deductible_share"]),
        other_insurance_cap_employee=_d(v["other_insurance_cap_employee"]),
        sick_pay_reduction=_d(v["sick_pay_reduction"]),
        bbg_pension_annual=_d(sv["bbg_pension_annual"]),
        bbg_health_annual=_d(sv["bbg_health_annual"]),
        rate_pension=_d(sv["rate_pension"]),
        rate_unemployment=_d(sv["rate_unemployment"]),
        rate_health_general=_d(sv["rate_health_general"]),
        rate_health_reduced=_d(sv["rate_health_reduced"]),
        rate_health_avg_extra=_d(sv["rate_health_avg_extra"]),
        rate_care=_d(sv["rate_care"]),
        rate_care_childless_extra=_d(sv["rate_care_childless_extra"]),
        care_childless_min_age=int(sv["care_childless_min_age"]),
        compulsory_insurance_limit=_d(sv["compulsory_insurance_limit"]),
        wage_tax_allowance_includes_unemployment=bool(v["wage_tax_allowance_includes_unemployment"]),
        bav_tax_free=_d(b["tax_free"]),
        bav_social_free=_d(b["social_free"]),
        saver_allowance_single=_d(k["saver_allowance_single"]),
        flat_tax_rate=_d(k["flat_tax_rate"]),
        fund_base_rate=_d(k["fund_base_rate"]),
        equity_fund_exemption=_d(k["equity_fund_exemption"]),
        reasonable_burden_single=tuple(
            (_d(u) if u is not None else None, _d(r))
            for u, r in ((step.get("upper"), step["rate"]) for step in raw["zumutbare_belastung"]["single"])
        ),
        household_services_rate=_d(h["services_rate"]),
        household_services_max=_d(h["services_max"]),
        craftsmen_rate=_d(h["craftsmen_rate"]),
        craftsmen_max=_d(h["craftsmen_max"]),
    )
