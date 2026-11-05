"""§ 32a EStG tariff, Solidaritätszuschlag, Kirchensteuer, flat tax."""
from __future__ import annotations

from decimal import ROUND_DOWN, Decimal

from app.tax_de.params import TaxYearParams

CENT = Decimal("0.01")
EURO = Decimal("1")
TEN_THOUSAND = Decimal(10000)


def floor_euro(v: Decimal) -> Decimal:
    return v.quantize(EURO, rounding=ROUND_DOWN) if v > 0 else Decimal(0)


def floor_cent(v: Decimal) -> Decimal:
    return v.quantize(CENT, rounding=ROUND_DOWN) if v > 0 else Decimal(0)


def tariff(zve: Decimal, p: TaxYearParams) -> Decimal:
    """Einkommensteuer for one person by § 32a Abs. 1 (Grundtarif).

    zvE is rounded down to whole euros first, the tax down to whole euros
    last, as the statute prescribes.
    """
    x = floor_euro(zve)
    if x <= p.basic_allowance:
        return Decimal(0)
    for zone in p.zones:
        if zone.upper is None or x <= zone.upper:
            if zone.rate is not None:
                return floor_euro(zone.rate * x - zone.minus)
            t = (x - zone.offset) / TEN_THOUSAND
            return floor_euro((zone.a * t + zone.b) * t + zone.c)
    raise ValueError("tariff zones do not cover the income")  # pragma: no cover - data error


def income_tax(zve: Decimal, p: TaxYearParams, *, joint: bool = False) -> Decimal:
    """Grundtarif, or the splitting procedure (§ 32a Abs. 5) when `joint`."""
    if joint:
        return tariff(zve / 2, p) * 2
    return tariff(zve, p)


def solidarity_surcharge(est: Decimal, p: TaxYearParams, *, joint: bool = False) -> Decimal:
    """SolZG § 3, § 4: nothing up to the threshold, then a taper of
    `taper_rate` on the excess until it reaches the full 5.5 %."""
    threshold = p.soli_threshold_single * (2 if joint else 1)
    if est <= threshold:
        return Decimal(0)
    return floor_cent(min(est * p.soli_rate, (est - threshold) * p.soli_taper_rate))


def church_tax(est: Decimal, rate: Decimal) -> Decimal:
    """Kirchensteuer as a share of the income tax (8 % BY/BW, 9 % elsewhere).
    Without children the base is the income tax itself."""
    return floor_cent(est * rate)


def flat_tax(taxable_capital_income: Decimal, p: TaxYearParams, *, church_rate: Decimal = Decimal(0)) -> dict:
    """Abgeltungsteuer § 32d with Soli; Kirchensteuer lowers the base rate
    (§ 32d Abs. 1 S. 3: 25 % / (1 + church rate))."""
    base = max(Decimal(0), taxable_capital_income)
    rate = p.flat_tax_rate / (1 + p.flat_tax_rate * church_rate) if church_rate else p.flat_tax_rate
    tax = floor_cent(base * rate)
    soli = floor_cent(tax * p.soli_rate)
    church = floor_cent(tax * church_rate)
    return {"tax": tax, "soli": soli, "church": church, "total": tax + soli + church}
