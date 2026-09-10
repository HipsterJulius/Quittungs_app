"""
Hauptskript: steuert die komplette Pipeline von Bon-Foto bis Excel-Zeile.

Ablauf pro Bild:
    1. Vorverarbeitung (EXIF-Korrektur, Crop)      -> receipt_preprocessing
    2. OCR-Texterkennung (PaddleOCR, gefiltert)    -> ocr_extractor
    3. Strukturierte Extraktion (lokales LLM)      -> llm_extractor
    4. Anhängen an die Haushaltsbuch-Excel-Datei   -> excel_writer

Voraussetzungen: alle Module (receipt_preprocessing.py, ocr_extractor.py,
llm_extractor.py, excel_writer.py) im selben Ordner. Ollama muss laufen.

Erwartete Projektstruktur (src/, input/, output/ als Geschwister-Ordner):
    quittungen_project/
        src/            <- dieses Skript liegt hier
            main.py
            ...
        input/          <- Bon-Fotos hierhin legen
        output/         <- haushaltsbuch.xlsx wird hier angelegt

Ohne weitere Argumente verarbeitet das Skript automatisch alles in input/
und schreibt nach output/haushaltsbuch.xlsx - unabhängig davon, aus
welchem Verzeichnis es gestartet wird.

Aufruf:
    python main.py                                   # nutzt input/ und output/
    python main.py bon.jpg                            # einzelnes Bild
    python main.py --input-dir ./anderer_ordner --output ./anderswo/buch.xlsx
"""

import argparse
import sys
from pathlib import Path

from ocr_extractor import extract_text, MIN_CONFIDENCE as DEFAULT_MIN_CONFIDENCE
from llm_extractor import extract_structured_data
from excel_writer import append_receipt

# Projekt-Root = Elternordner von src/ (wo dieses Skript liegt) - so
# funktionieren die Standardpfade unabhängig vom aktuellen Arbeitsverzeichnis.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "input"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "output" / "haushaltsbuch.xlsx"

BILDENDUNGEN = ("*.jpg", "*.jpeg", "*.png")


def process_receipt(image_path: str, output_path: str, min_confidence: float) -> None:
    print(f"\n--- {image_path} ---")

    print("  [1/3] OCR...")
    text, dropped = extract_text(image_path, min_confidence)
    if dropped:
        print(f"        {len(dropped)} Zeile(n) unter Confidence {min_confidence} verworfen")

    print("  [2/3] LLM-Extraktion...")
    try:
        data = extract_structured_data(text)
    except ValueError as e:
        print(f"        FEHLER: {e}")
        print(f"        Bon wird übersprungen: {image_path}")
        return

    print(
        f"        Händler: {data['haendler']}, Datum: {data['datum']}, "
        f"{len(data['produkte'])} Produkte, Gesamt: {data['gesamtbetrag']} €"
    )

    print("  [3/3] In Excel schreiben...")
    purchase_id = append_receipt(data, output_path)
    print(f"        -> {output_path} (Einkaufs-ID: {purchase_id})")


def main():
    parser = argparse.ArgumentParser(description="Kassenbon-Scanner Pipeline")
    parser.add_argument(
        "images", nargs="*",
        help="Pfad(e) zu einzelnen Bon-Fotos (optional - ohne Angabe wird input/ gescannt)",
    )
    parser.add_argument(
        "--input-dir", default=str(DEFAULT_INPUT_DIR),
        help=f"Ordner mit Bon-Fotos (Standard: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT_FILE),
        help=f"Ziel-Excel-Datei (Standard: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
        help=f"OCR-Confidence-Schwellwert (Standard: {DEFAULT_MIN_CONFIDENCE})",
    )
    args = parser.parse_args()

    image_paths = list(args.images)

    # Nur den Input-Ordner scannen, wenn keine einzelnen Bilder als
    # Argumente übergeben wurden - so kann man gezielt EIN Testbild
    # verarbeiten, ohne dass automatisch der ganze Ordner mitläuft.
    if not image_paths:
        input_dir = Path(args.input_dir)
        if input_dir.exists():
            for pattern in BILDENDUNGEN:
                image_paths += [str(p) for p in input_dir.glob(pattern)]

    if not image_paths:
        print(f"Keine Bilder gefunden (weder als Argument noch in {args.input_dir}).")
        print("Nutzung: python main.py bon.jpg [bon2.jpg ...]")
        print(f"     oder: Bilder in {DEFAULT_INPUT_DIR} legen und ohne Argumente starten")
        sys.exit(1)

    for image_path in image_paths:
        process_receipt(image_path, args.output, args.min_confidence)

    print(f"\nFertig. Ergebnisse in: {args.output}")


if __name__ == "__main__":
    main()
