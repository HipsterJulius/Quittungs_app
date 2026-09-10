"""
LLM-Extraktion: wandelt den rohen OCR-Text eines Kassenbons in strukturierte
Daten um (Händler, Datum, Produkte mit Preis+Kategorie, Gesamtbetrag).

Nutzt ein lokales LLM über Ollama (Standard: qwen2.5:7b-instruct). Ollama
muss installiert sein und als Hintergrunddienst laufen.

Voraussetzungen:
    pip install ollama
    ollama pull qwen2.5:7b-instruct
"""

import json
import re

import ollama

MODEL = "qwen2.5:7b-instruct"

# Feste Kategorie-Liste statt freiem Text: sorgt dafür, dass später in
# Excel/Auswertungen immer dieselben Kategorienamen auftauchen, statt dass
# das LLM bei jedem Bon leicht andere Formulierungen erfindet.
KATEGORIEN = [
    "Obst & Gemüse",
    "Fleisch & Wurst",
    "Milchprodukte",
    "Backwaren",
    "Süßigkeiten & Snacks",
    "Getränke",
    "Drogerie & Hygiene",
    "Sonstiges",
]

SYSTEM_PROMPT = f"""Du extrahierst strukturierte Daten aus dem OCR-Text eines deutschen Kassenbons.

Gib AUSSCHLIESSLICH gültiges JSON zurück - keine Erklärungen, kein Markdown, keine Code-Blöcke.

Schema:
{{
  "haendler": string,
  "datum": string im Format YYYY-MM-DD (oder null, falls nicht erkennbar),
  "gesamtbetrag": number,
  "produkte": [
    {{"name": string, "preis": number, "kategorie": string}}
  ]
}}

Regeln:
- "kategorie" MUSS exakt einer dieser Werte sein: {", ".join(KATEGORIEN)}
- Wähle bei Unsicherheit die naheliegendste Kategorie. "Sonstiges" nur, wenn wirklich nichts passt.
- OCR-Artefakte ignorieren, durch Zeilenumbrüche zerrissene Produktnamen sinnvoll zusammenführen.
- Mengenangaben wie "2 x 0,85 €" oder "3 × 0,79 €" gehören zum vorherigen Produkt, NICHT als eigene Zeile.
- Pfand als eigenes Produkt mit Kategorie "Getränke" behandeln, falls vorhanden.
- "gesamtbetrag" ist der Betrag hinter "ZU ZAHLEN" (nicht die Summe der Einzelposten - dient später zur Kontrolle).
- Datum auf dem Bon steht meist als TT.MM.JJ oder TT.MM.JJJJ - immer nach YYYY-MM-DD umwandeln.
"""

# Ein Few-Shot-Beispiel verbessert die Zuverlässigkeit bei 7B-Modellen deutlich.
EXAMPLE_INPUT = """ALDI
Lütemannskamp 4, 49838 Lengerich
FISHERMANS FRIEND MINT 0,95 € 1
H-MILCH 1,5%, 1L 1,70 € 1
BIO ZWIEBELN 1KG 1,99 € 1
3 × 0,79 €
ZU ZAHLEN 4,64 €
Datum 17.08.26"""

EXAMPLE_OUTPUT = json.dumps(
    {
        "haendler": "ALDI",
        "datum": "2026-08-17",
        "gesamtbetrag": 4.64,
        "produkte": [
            {"name": "Fishermans Friend Mint", "preis": 0.95, "kategorie": "Süßigkeiten & Snacks"},
            {"name": "H-Milch 1,5%, 1L", "preis": 1.70, "kategorie": "Milchprodukte"},
            {"name": "Bio Zwiebeln 1KG", "preis": 1.99, "kategorie": "Obst & Gemüse"},
        ],
    },
    ensure_ascii=False,
)


def extract_structured_data(ocr_text: str, model: str = MODEL) -> dict:
    """
    Schickt den OCR-Text an das lokale LLM und gibt das geparste JSON als
    dict zurück. Wirft ValueError, wenn das Modell kein valides JSON liefert.
    """
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": EXAMPLE_INPUT},
            {"role": "assistant", "content": EXAMPLE_OUTPUT},
            {"role": "user", "content": ocr_text},
        ],
        options={"temperature": 0},
    )

    raw = response["message"]["content"].strip()
    raw = _strip_markdown_fences(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM hat kein valides JSON geliefert:\n{raw}") from e

    _validate_and_clean(data)
    return data


def _strip_markdown_fences(text: str) -> str:
    """Manche Modelle liefern trotz Anweisung ```json ... ``` - das entfernen."""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1) if match else text


def _validate_and_clean(data: dict) -> None:
    required = {"haendler", "datum", "gesamtbetrag", "produkte"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Fehlende Felder im LLM-Output: {missing}")

    for produkt in data["produkte"]:
        # Lieber "Sonstiges" erzwingen als eine erfundene Kategorie durchlassen -
        # das hält die spätere Excel-Auswertung konsistent.
        if produkt.get("kategorie") not in KATEGORIEN:
            produkt["kategorie"] = "Sonstiges"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Nutzung: python llm_extractor.py <textdatei_oder_text>")
        sys.exit(1)

    arg = sys.argv[1]
    try:
        with open(arg, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        text = arg  # Direkt als Text übergeben

    result = extract_structured_data(text)
    print(json.dumps(result, indent=2, ensure_ascii=False))
