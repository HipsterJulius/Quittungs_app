"""
PaddleOCR-basierte Texterkennung für Kassenbons.

Nutzt die Vorverarbeitung aus receipt_preprocessing.py (EXIF-Korrektur +
robustes Cropping) und übergibt den Crop OHNE Binarisierung an PaddleOCR -
PaddleOCR ist auf reale Fotos trainiert und liefert auf dem farbigen Crop
bessere Ergebnisse als auf hart binarisiertem Schwarz/Weiß (im Unterschied
zu Tesseract).

Confidence-Filter: PaddleOCR gibt pro Zeile einen Score zwischen 0 und 1
zurück. Zeilen unter MIN_CONFIDENCE werden separat ausgegeben statt den
nächsten Schritt (LLM-Extraktion) mit Rauschen zu füttern.

Modell-Instanz ist ein Lazy Singleton: Das Laden dauert beim ersten Aufruf
spürbar (Modell-Download/-Initialisierung), danach ist jede weitere
Erkennung schnell. Beim späteren Batch-Betrieb über den Input-Ordner immer
dieselbe Instanz wiederverwenden, nicht pro Bild neu instanziieren.

Voraussetzungen:
    pip install paddlepaddle==3.2.2 paddleocr opencv-python pillow
    (paddlepaddle 3.3.x hat einen bekannten CPU-Inferenz-Bug, siehe Chat)

Aufruf (Einzelbild, zum Testen):
    python ocr_extractor.py bon.jpg
    python ocr_extractor.py bon.jpg 0.9   # abweichender Confidence-Schwellwert
"""

import sys
from dataclasses import dataclass

from receipt_preprocessing import fix_orientation, crop_receipt

MIN_CONFIDENCE = 0.85

_ocr_engine = None  # Lazy Singleton


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR

        _ocr_engine = PaddleOCR(
            lang="german",
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    return _ocr_engine


@dataclass
class OcrLine:
    text: str
    confidence: float


def extract_lines(image_path: str) -> list[OcrLine]:
    """Führt Vorverarbeitung + OCR aus, gibt ALLE erkannten Zeilen zurück
    (ungefiltert, inkl. niedriger Confidence)."""
    oriented = fix_orientation(image_path)
    cropped = crop_receipt(oriented)

    ocr = get_ocr_engine()
    result = ocr.predict(cropped)

    lines = []
    for res in result:
        for text, score in zip(res["rec_texts"], res["rec_scores"]):
            lines.append(OcrLine(text=text, confidence=float(score)))
    return lines


def extract_text(
    image_path: str, min_confidence: float = MIN_CONFIDENCE
) -> tuple[str, list[OcrLine]]:
    """
    Gibt (sauberer_text, verworfene_zeilen) zurück.

    sauberer_text: nur Zeilen >= min_confidence, zeilenweise getrennt -
    das ist die Eingabe für den nächsten Schritt (LLM-Extraktion).

    verworfene_zeilen: zur Kontrolle/Logging, z.B. um zu prüfen, ob
    wichtige Infos (Preise, Produktnamen) verloren gehen und der
    Schwellwert ggf. angepasst werden muss.
    """
    lines = extract_lines(image_path)
    kept = [l for l in lines if l.confidence >= min_confidence]
    dropped = [l for l in lines if l.confidence < min_confidence]

    clean_text = "\n".join(l.text for l in kept)
    return clean_text, dropped


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Nutzung: python ocr_extractor.py <bild_pfad> [min_confidence]")
        sys.exit(1)

    path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else MIN_CONFIDENCE

    text, dropped = extract_text(path, threshold)

    print("=" * 60)
    print(f"ERKANNTER TEXT (confidence >= {threshold})")
    print("=" * 60)
    print(text)

    if dropped:
        print("\n" + "=" * 60)
        print(f"VERWORFEN (confidence < {threshold}) - zur Kontrolle")
        print("=" * 60)
        for l in dropped:
            print(f"{l.text!r}  (conf={l.confidence:.2f})")
