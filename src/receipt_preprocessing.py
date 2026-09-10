"""
Quittungs-Vorverarbeitung: EXIF-Korrektur -> Crop -> Binarisierung -> OCR

Getestet mit einem echten ALDI-Kassenbon (Foto auf unruhigem, ungleichmäßig
beleuchtetem Hintergrund). Wichtigste Erkenntnisse aus dem Test:

1. EXIF-Rotation IMMER zuerst korrigieren. Viele Handyfotos speichern die
   Rohpixel quer und drehen nur über ein EXIF-Flag "richtig". OpenCV und
   Tesseract ignorieren dieses Flag -> Bild wird sonst seitwärts gelesen.

2. Für das Cropping KEINEN globalen Helligkeits-Threshold (Otsu) auf dem
   ganzen Foto verwenden. Bei ungleichmäßigem Umgebungslicht (z.B. eine
   helle und eine dunkle Bildecke) wird der Hintergrund fälschlich als
   "hell wie der Bon" erkannt. Kantenbasiert (Canny) auf einer
   herunterskalierten Version ist deutlich robuster.

3. Für die Binarisierung NACH dem Crop (jetzt gleichmäßigerer Hintergrund)
   funktioniert adaptiveThreshold zuverlässiger als Otsu, weil Schattenwurf
   und ins Papier gedruckte Wasserzeichen/Logos lokal unterschiedlich hell
   sind - ein globaler Schwellwert erwischt dann entweder Schatten mit oder
   frisst echten Text.

4. WICHTIGSTER PUNKT für deutsche Bons: das deutsche Tesseract-Sprachmodell
   (`deu`) installieren und verwenden. Ohne das Sprachpaket "korrigiert"
   Tesseract deutsche Wörter Richtung englischer Muster und die
   Trefferquote sinkt spürbar - unabhängig von der Bildqualität.

    Linux:   sudo apt install tesseract-ocr-deu
    macOS:   brew install tesseract-lang

Benötigte Pakete: opencv-python, pillow, pytesseract
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps


def fix_orientation(path: str) -> np.ndarray:
    """Lädt ein Bild und wendet die EXIF-Rotation korrekt an."""
    img = ImageOps.exif_transpose(Image.open(path))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def crop_receipt(img: np.ndarray, downscale: float = 0.25) -> np.ndarray:
    """
    Findet den Bon per Kantenerkennung (robuster als Helligkeits-Threshold
    bei ungleichmäßigem Hintergrundlicht) und schneidet ihn mit etwas
    Puffer aus.
    """
    small = cv2.resize(img, None, fx=downscale, fy=downscale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(blurred, 30, 100)
    edged = cv2.dilate(edged, np.ones((7, 7), np.uint8), iterations=2)
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: kein Bon gefunden, Originalbild unverändert zurückgeben
        return img

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    x, y, w, h = (int(v / downscale) for v in (x, y, w, h))

    pad = 15
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    return img[y0:y1, x0:x1]


def binarize(img: np.ndarray) -> np.ndarray:
    """
    Binarisiert den (bereits gecropten) Bon. adaptiveThreshold statt Otsu,
    da Schattenwurf/Wasserzeichen auf dem Papier selbst ungleichmäßig hell
    sein können.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 15
    )


def ocr_receipt(path: str, lang: str = "deu", debug_dir: str | None = None) -> str:
    """
    Führt die komplette Pipeline aus und gibt den rohen OCR-Text zurück.

    lang: Tesseract-Sprachcode. 'deu' für deutsche Bons (Sprachpaket muss
          installiert sein, siehe Modul-Docstring). 'deu+eng' bei
          gemischtem Text (z.B. "Mastercard", "EUR").
    debug_dir: optionaler Pfad, um Zwischenschritte als Bilder zu speichern
               (hilfreich, um die Pipeline an neuen Bon-Typen zu justieren).
    """
    oriented = fix_orientation(path)
    cropped = crop_receipt(oriented)
    binarized = binarize(cropped)

    if debug_dir:
        cv2.imwrite(f"{debug_dir}/1_oriented.jpg", oriented)
        cv2.imwrite(f"{debug_dir}/2_cropped.jpg", cropped)
        cv2.imwrite(f"{debug_dir}/3_binarized.jpg", binarized)

    # --psm 6: einzelner gleichmäßiger Textblock - passt zur Bon-Struktur
    # besser als die automatische Seitensegmentierung (psm 3), die bei
    # der zweispaltigen Produkt/Preis-Struktur durcheinanderkommt.
    return pytesseract.image_to_string(binarized, lang=lang, config="--psm 6")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Nutzung: python receipt_preprocessing.py <bild_pfad> [lang]")
        sys.exit(1)

    image_path = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else "deu"
    print(ocr_receipt(image_path, lang=language, debug_dir="."))
