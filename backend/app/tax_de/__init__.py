"""German personal income tax: the numbers an advisor agent may quote.

Everything here is deterministic arithmetic on published law. The agent
explains and compares; it never computes a tax figure itself. Each tax year
is a TOML file under `years/`, so a new year is a data change reviewed
against its source, not a code change.

Scope is employees (Anlage N), Vorsorgeaufwendungen, außergewöhnliche
Belastungen, § 35a, and capital income under the flat tax. Business,
rental and foreign income are out of scope and say so in the result.
"""
