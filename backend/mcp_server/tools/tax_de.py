"""German income tax tools for the advisor agent.

Pure calculations over app.tax_de: no ledger access, no writes. The agent
fills the inputs from the user's profile and transactions, and quotes the
results instead of doing tax arithmetic itself.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.tax_de.estimate import TaxProfile, WorkExpenses, company_car_benefit_monthly, deduction_value, estimate
from app.tax_de.params import available_years, load_year
from mcp_server.auth import CallContext
from mcp_server.registry import tool

_MONEY = {"type": "number", "minimum": 0}

_EXPENSE_PROPS: dict[str, Any] = {
    "home_office_days": {"type": "integer", "minimum": 0, "maximum": 366},
    "home_office_room": {"type": "boolean", "description": "Separates Arbeitszimmer als Mittelpunkt der Tätigkeit"},
    "home_office_room_actual": {**_MONEY, "description": "Tatsächliche Arbeitszimmerkosten im Jahr"},
    "commute_km": {"type": "integer", "minimum": 0, "description": "Einfache Entfernung zur ersten Tätigkeitsstätte"},
    "commute_days": {"type": "integer", "minimum": 0, "maximum": 366},
    "work_equipment": _MONEY,
    "software_subscriptions": {**_MONEY, "description": "Beruflicher Anteil von Software/KI-Abos im Jahr"},
    "training": _MONEY,
    "phone_internet": {**_MONEY, "description": "Beruflicher Anteil Telefon/Internet im Jahr"},
    "travel_unreimbursed": {**_MONEY, "description": "Nicht erstattete Reisekosten inkl. Verpflegungsmehraufwand"},
    "professional_fees": {**_MONEY, "description": "Beiträge zu Berufsverbänden"},
    "union_fees": {**_MONEY, "description": "Gewerkschaftsbeiträge (ab 2026 neben dem Pauschbetrag)"},
    "legal_insurance_work_share": _MONEY,
    "account_fee_flat": {"type": "boolean", "default": True},
    "other": _MONEY,
}

_PROFILE_PROPS: dict[str, Any] = {
    "year": {"type": "integer", "default": 2026},
    "age": {"type": "integer", "minimum": 16, "maximum": 100},
    "joint": {"type": "boolean", "default": False, "description": "Zusammenveranlagung (Splitting)"},
    "childless": {"type": "boolean", "default": True},
    "church_rate": {"type": "number", "enum": [0, 0.08, 0.09], "default": 0},
    "saxony": {"type": "boolean", "default": False},
    "salary": {**_MONEY, "description": "Laufendes Bruttogehalt im Jahr"},
    "bonus": {**_MONEY, "description": "Einmalzahlungen/Bonus brutto im Jahr"},
    "overtime_pay": {**_MONEY, "description": "Steuerpflichtige Überstundenvergütung brutto im Jahr"},
    "company_car_benefit_annual": {**_MONEY, "description": "Geldwerter Vorteil Firmenwagen nach Nutzungsentgelt, Jahr"},
    "statutory_health": {"type": "boolean", "default": True},
    "health_extra_rate": {"type": "number", "minimum": 0, "maximum": 10, "description": "Zusatzbeitrag der Krankenkasse in %"},
    "sick_pay_entitlement": {"type": "boolean", "default": True},
    "bav_conversion": {**_MONEY, "description": "Entgeltumwandlung in die bAV, Jahr"},
    "ruerup": {**_MONEY, "description": "Beiträge Basisrente (Rürup), Jahr"},
    "other_insurance": {**_MONEY, "description": "Haftpflicht, Unfall, Zusatzversicherungen u. a., Jahr"},
    "donations": _MONEY,
    "medical_costs": {**_MONEY, "description": "Selbst getragene Krankheitskosten, Jahr"},
    "household_services_labour": _MONEY,
    "craftsmen_labour": _MONEY,
    "capital_income": {**_MONEY, "description": "Kapitalerträge nach Teilfreistellung, Jahr"},
    "wage_tax_withheld": {**_MONEY, "description": "Einbehaltene Lohnsteuer laut Lohnsteuerbescheinigung"},
    "soli_withheld": _MONEY,
    "expenses": {"type": "object", "properties": _EXPENSE_PROPS, "additionalProperties": False},
}


def _dec(v: Any) -> Decimal:
    return Decimal(str(v))


_INT_FIELDS = {"year", "age", "home_office_days", "commute_km", "commute_days"}


def _expenses(raw: dict[str, Any]) -> WorkExpenses:
    kwargs: dict[str, Any] = {}
    for k, v in raw.items():
        if v is None:
            continue
        if k in _INT_FIELDS:
            kwargs[k] = int(v)
        elif isinstance(v, bool):
            kwargs[k] = v
        else:
            kwargs[k] = _dec(v)
    return WorkExpenses(**kwargs)


def _profile(args: dict[str, Any]) -> TaxProfile:
    expenses = _expenses(args.pop("expenses", None) or {})
    kwargs: dict[str, Any] = {}
    for k, v in args.items():
        if v is None:
            continue
        if k in _INT_FIELDS:
            kwargs[k] = int(v)
        elif isinstance(v, bool):
            kwargs[k] = v
        elif isinstance(v, int | float):
            kwargs[k] = _dec(v)
        else:
            kwargs[k] = v
    return TaxProfile(expenses=expenses, **kwargs)


@tool(
    name="tax_de_parameters",
    description=(
        "German tax and social insurance figures for a year (Grundfreibetrag, Pauschbeträge, "
        "Beitragsbemessungsgrenzen, Beitragssätze, Sparerpauschbetrag) with their legal sources. "
        "Use before quoting any statutory amount."
    ),
    parameters={
        "type": "object",
        "properties": {"year": {"type": "integer", "default": 2026}},
        "additionalProperties": False,
    },
    tags=["read", "tax_de"],
)
async def tax_de_parameters(*, session: AsyncSession, ctx: CallContext, year: int = 2026) -> dict[str, Any]:
    p = load_year(int(year))
    out: dict[str, Any] = {"available_years": available_years()}
    for k, v in p.__dict__.items():
        if k == "zones":
            continue
        if isinstance(v, Decimal):
            out[k] = float(v)
        elif isinstance(v, tuple):
            out[k] = [[float(a) if a is not None else None, float(b)] for a, b in v]
        else:
            out[k] = v
    return out


@tool(
    name="tax_de_estimate",
    description=(
        "Annual German income tax estimate for an employee: Werbungskosten vs. Pauschbetrag, "
        "Vorsorgeaufwendungen, zu versteuerndes Einkommen, Einkommensteuer, Soli, Kirchensteuer, "
        "marginal and average rate, and the expected refund against withheld wage tax. "
        "Amounts are annual euros. Use for every question about refunds or tax burden."
    ),
    parameters={"type": "object", "properties": _PROFILE_PROPS, "additionalProperties": False},
    tags=["read", "tax_de"],
)
async def tax_de_estimate(*, session: AsyncSession, ctx: CallContext, **kwargs: Any) -> dict[str, Any]:
    return estimate(_profile(dict(kwargs)))


@tool(
    name="tax_de_deduction_value",
    description=(
        "How much tax an additional deduction saves for this profile, e.g. 'what is a 1,500 EUR "
        "software subscription worth if deducted?'. Accounts for the Arbeitnehmer-Pauschbetrag."
    ),
    parameters={
        "type": "object",
        "properties": {
            **_PROFILE_PROPS,
            "extra_work_expenses": _MONEY,
            "extra_special_expenses": _MONEY,
        },
        "additionalProperties": False,
    },
    tags=["read", "tax_de"],
)
async def tax_de_deduction_value(
    *,
    session: AsyncSession,
    ctx: CallContext,
    extra_work_expenses: float = 0,
    extra_special_expenses: float = 0,
    **kwargs: Any,
) -> dict[str, Any]:
    return deduction_value(
        _profile(dict(kwargs)),
        extra_work_expenses=_dec(extra_work_expenses),
        extra_special_expenses=_dec(extra_special_expenses),
    )


@tool(
    name="tax_de_company_car",
    description=(
        "Monthly taxable benefit of a company car under the list price method (1 %, 0.5 % for "
        "plug-in hybrids, 0.25 % for EVs) including the commute surcharge and the user's own payment."
    ),
    parameters={
        "type": "object",
        "properties": {
            "list_price": {**_MONEY, "description": "Bruttolistenpreis inkl. Sonderausstattung"},
            "share_percent": {"type": "number", "enum": [1, 0.5, 0.25], "default": 1},
            "commute_km": {"type": "integer", "minimum": 0, "default": 0},
            "commute_trips_per_month": {"type": "integer", "minimum": 0},
            "user_payment_monthly": {**_MONEY, "default": 0},
        },
        "required": ["list_price"],
        "additionalProperties": False,
    },
    tags=["read", "tax_de"],
)
async def tax_de_company_car(
    *,
    session: AsyncSession,
    ctx: CallContext,
    list_price: float,
    share_percent: float = 1,
    commute_km: int = 0,
    commute_trips_per_month: int | None = None,
    user_payment_monthly: float = 0,
) -> dict[str, Any]:
    monthly = company_car_benefit_monthly(
        _dec(list_price),
        share_percent=_dec(share_percent),
        commute_km=int(commute_km),
        commute_trips_per_month=commute_trips_per_month,
        user_payment_monthly=_dec(user_payment_monthly),
    )
    return {"benefit_monthly": float(round(monthly, 2)), "benefit_annual": float(round(monthly * 12, 2))}
