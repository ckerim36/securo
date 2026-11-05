# Steuer- & Finanzberater (DE) – lokale Einrichtung

Dieser Fork ergänzt Securo um einen deutschen Steuer-Rechenkern
(`backend/app/tax_de/`), vier Werkzeuge für den KI-Agenten
(`tax_de_parameters`, `tax_de_estimate`, `tax_de_deduction_value`,
`tax_de_company_car`) und Unterstützung für Claude Opus 5.5.

## 1. Voraussetzungen

- Docker Desktop (Windows/macOS) oder Docker Engine (Linux)
- Ein Anthropic-API-Schlüssel (console.anthropic.com → API Keys)

## 2. Starten

```bash
git clone https://github.com/ckerim36/securo.git
cd securo
git checkout claude/personal-tax-finance-advisor-m25mfg
cp .env.example .env
```

In `.env` setzen:

```
SECRET_KEY=<lange Zufallszeichenkette>
AGENTS_ENABLED=true
COMPOSE_PROFILES=agents
AGENTS_MCP_JWT_SECRET=<zweite lange Zufallszeichenkette>
# optional, Standard ist high
AGENTS_ANTHROPIC_EFFORT=high
TZ=Europe/Berlin
```

Zufallswerte erzeugen: `openssl rand -hex 32`

```bash
docker compose up --build -d
```

Dann http://localhost:3000 öffnen und ein Konto anlegen.

## 3. Berater anlegen

1. **Agents → Verbindungen**: Anbieter *Anthropic*, API-Schlüssel eintragen.
2. **Agents → Neuer Agent**: Modell `claude-opus-5-5`, System-Prompt aus
   [`agent-prompt.md`](agent-prompt.md) einfügen und den Abschnitt „Profil“
   mit den eigenen Angaben füllen.
3. Ein neuer Agent darf alle Werkzeuge nutzen; die `tax_de_*`-Werkzeuge sind
   damit sofort aktiv.

## 4. Bankdaten

Entweder Bank-Sync über Enable Banking (siehe Securo-Doku) oder Kontoauszüge
als CAMT/CSV importieren (**Transaktionen → Import**).

## 5. Datenschutz

Securo und die Datenbank laufen nur auf dem eigenen Rechner. An Anthropic gehen
ausschließlich die Chat-Nachrichten samt der Werkzeugergebnisse, die der Agent
für eine Antwort abruft. Persönliche Daten nie in dieses (öffentliche)
Repository committen.

## 6. Steuerjahr aktualisieren

Neue Jahreswerte = neue Datei `backend/app/tax_de/years/<jahr>.toml`, jede
Zahl mit Quelle. Die Tests prüfen automatisch Stetigkeit des Tarifs,
Soli-Freigrenze und Beitragsbemessungsgrenzen für jedes vorhandene Jahr.
