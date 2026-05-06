# KKTC Plate Recognition System

A computer vision project that automatically reads vehicle license plates from images and matches them against a registered vehicle database. Built with Python, OpenCV, and EasyOCR.

---

## Overview

This system is designed for **Northern Cyprus (KKTC)** license plates. It monitors a folder for incoming plate images, runs OCR to extract the plate number, corrects common misreadings caused by KKTC's unique plate font (especially the slashed zero), and looks up the result in a local vehicle database.

Key capabilities:

- Real-time folder monitoring — drop an image, get a result instantly
- Multi-version image preprocessing to maximize OCR accuracy
- Automatic correction of OCR errors (O/0, I/1, U/O, 0/6/8 confusions)
- Supports KKTC standard, KKTC rental (red), and Turkish Republic plate formats
- Registers and manages vehicle records through an interactive CLI menu

---

## Team

| Member | Role |
|---|---|
| **Oğuz Gürkan Altunok** | AI & Image Processing |
| **Ali Can Altun** | Data Management & Algorithm |
| **Özgür Evecen** | System Integration & Testing |

### Contributions

**Oğuz Gürkan Altunok — AI & Image Processing**
- Image preprocessing pipeline (noise reduction, contrast enhancement, sharpening)
- Blue EU indicator strip detection and cropping to prevent OCR confusion
- Red plate detection via HSV color masking for rental vehicle plates
- EasyOCR integration with multi-version image strategy (4 preprocessed variants per image)
- HEIC/HEIF format support for iPhone photos

**Ali Can Altun — Data Management & Algorithm**
- Regex-based format validation for all KKTC and TR plate standards
- OCR error correction (letter-to-digit and digit-to-letter mapping per position)
- Unified variant generation system: handles KKTC's slashed-zero ambiguity (0/6/8) and OCR letter confusion (O/U) simultaneously using combinatorial substitution
- Over-long OCR output recovery via 2-round character deletion algorithm
- Vehicle database CRUD operations (register, edit, lookup)

**Özgür Evecen — System Integration & Testing**
- Watchdog-based real-time folder monitoring
- Main orchestrator (`PlateRecognizer`) connecting all components
- Interactive CLI menu with scanning, registration, and editing modes
- File management: processed images auto-moved to `processed/` folder with collision handling
- End-to-end testing across multiple plate types, lighting conditions, and angles

---

## Supported Plate Formats

| Type | Pattern | Example |
|---|---|---|
| KKTC Old | 2 letters + 3 digits | UD 724, AA 101 |
| KKTC New | 2 letters + 3 digits + 1 letter | AA 006 V, AB 232 C |
| KKTC Rental Old | 3 letters + 3 digits | ZAA 123 |
| KKTC Rental New | 3 letters + 3 digits + 1 letter (Z/T prefix, red plate) | ZAA 123 A |
| Turkish Republic | City code + letters + digits | 34 ABC 123, 06 A 1234 |

---

## How It Works

```
Image Input
    │
    ▼
Image Preprocessor
  ├─ Crop blue EU strip (prevents OCR false reads)
  ├─ Detect red background (rental plates)
  └─ Generate 4 preprocessed versions:
       denoised / CLAHE enhanced / adaptive threshold / sharpened
    │
    ▼
OCR Engine (EasyOCR)
  ├─ Run on original + all preprocessed versions
  └─ Collect all text regions with confidence scores
    │
    ▼
Plate Validator
  ├─ Expand candidates (remove 1-2 artifact chars from over-long reads)
  ├─ Validate format (Regex per plate type)
  ├─ Fix per-position errors (digit↔letter corrections)
  └─ Generate all confusion variants (0/6/8 + O/U substitutions)
    │
    ▼
Database Lookup
  ├─ Check plate + all variants against vehicle database
  └─ Prefer DB-matched result; fall back to highest-confidence valid plate
    │
    ▼
Output: plate number, type, confidence, owner (if registered)
```

---

## Installation

**Requirements:** Python 3.9+

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
opencv-python-headless>=4.8.0
easyocr>=1.7.0
numpy>=1.24.0
Pillow>=10.0.0
pillow-heif>=0.13.0
watchdog>=4.0.0
```

> **Note:** The first run downloads the EasyOCR language models (~300 MB). An internet connection is required.

---

## Usage

```bash
python main.py
```

```
========================================
  CAR PLATE RECOGNITION SYSTEM
========================================
  1. Start Plate Scanning
  2. Register Vehicle
  3. Edit Record
  4. Exit
========================================
```

### Option 1 — Start Plate Scanning

Processes any images already in `plates/`, then watches the folder for new ones. Press `Ctrl+C` to stop and return to the menu.

Drop any supported image file into `plates/` and the result appears within seconds:

```
[2026-05-07 01:49:22] Image processed: IMG_0319.jpg
  Plate           : UD724
  Plate Type      : KKTC_OLD
  Confidence Score: %55.1
  Status          : REGISTERED
  Owner           : Gürkan Altunok
----------------------------------------------------------------
```

```
[2026-05-07 01:49:53] Image processed: IMG_0324.jpg
  Plate           : UD111
  Plate Type      : KKTC_OLD
  Confidence Score: %44.1
  Status          : UNREGISTERED
  Possible Plates : OD111
----------------------------------------------------------------
```

Processed images are automatically moved to `processed/`.

### Option 2 — Register Vehicle

```
Enter Plate: UD724
Enter Name Surname: Gurkan Altunok
[SUCCESS] Plate 'UD724' registered to 'Gurkan Altunok'.
```

### Option 3 — Edit Record

Updates the owner name for an existing plate.

---

## Supported Image Formats

`.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.webp` `.heic` `.heif`

---

## Project Structure

```
carplate/
├── main.py                  # All application logic
├── plates/                  # Drop plate images here for scanning
├── processed/               # Processed images are moved here
├── data/
│   ├── vehicles.txt         # Plate-owner records (plate,name per line)
│   └── results.txt          # Recognition log (auto-generated)
├── requirements.txt
├── gorev_dagilimi.md        # Task distribution (Turkish)
└── README.md
```

---

## Data Files

**`data/vehicles.txt`** — one record per line, comma-separated:
```
AB123C,Ahmet Yilmaz
UD724,Gurkan Altunok
UN681,Ozgur Evecen
```

**`data/results.txt`** — automatic log of every recognition:
```
[2026-05-07 01:49:22] File: IMG_0319.jpg | Plate: UD724 | Type: KKTC_OLD | Conf: %55.1 | Status: REGISTERED | Owner: Gurkan Altunok
```

---

## Technical Notes

### KKTC Plate Font

KKTC license plates use a modified digit font where the number **0** has a diagonal slash on its upper right. This causes OCR engines to frequently misread it as **6**, **8**, or the letter **O**. The validator generates all valid substitution combinations and cross-checks each against the database.

### OCR Artifact Recovery

When the camera angle or lighting introduces extra characters (e.g., reading `ODI1011` instead of `UD111`), the system applies up to two rounds of single-character deletion across all OCR candidates, then re-validates. This recovers correct 5-7 character plates from noisy reads.

### Image Preprocessing Strategy

Each image is passed to OCR four times with different preprocessing applied:

1. Bilateral denoised (edge-preserving smoothing)
2. CLAHE contrast enhanced
3. Adaptive threshold (binary)
4. Sharpened

Results from all versions are pooled and the best-scoring valid plate is selected.

---

## Built With

- [OpenCV](https://opencv.org/) — image processing
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — text recognition
- [Pillow](https://python-pillow.org/) + [pillow-heif](https://github.com/bigcat88/pillow_heif) — image loading and HEIC support
- [watchdog](https://github.com/gorakhargosh/watchdog) — real-time folder monitoring

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Türkçe

### Proje Hakkında

Bu proje, Kuzey Kıbrıs Türk Cumhuriyeti (KKTC) araç plakalarını görüntülerden otomatik olarak tanıyan ve kayıtlı araç veritabanıyla eşleştiren bir bilgisayarlı görü sistemidir. Python, OpenCV ve EasyOCR kullanılarak geliştirilmiştir.

### Özellikler

- Klasör izleme: `plates/` klasörüne bırakılan görsel otomatik olarak işlenir
- KKTC'ye özgü plaka fontundaki kesik sıfır (0) karakterinin 6, 8 ve O ile karışmasını otomatik düzeltir
- Çoklu görüntü ön işleme ile OCR doğruluğu artırılır
- KKTC eski, yeni, kiralık ve Türkiye Cumhuriyeti plaka formatlarını destekler
- Araç kayıt, düzenleme ve sorgulama için interaktif menü

### Ekip

| Üye | Görev |
|---|---|
| **Oğuz Gürkan Altunok** | Yapay Zeka ve Görüntü İşleme |
| **Ali Can Altun** | Veri Yönetimi ve Algoritma |
| **Özgür Evecen** | Sistem Entegrasyonu ve Test |

### Kurulum

```bash
pip install -r requirements.txt
python main.py
```

### Desteklenen Plaka Formatları

| Tip | Format | Örnek |
|---|---|---|
| KKTC Eski | 2 harf + 3 rakam | UD 724, AA 101 |
| KKTC Yeni | 2 harf + 3 rakam + 1 harf | AA 006 V |
| KKTC Kiralık Eski | 3 harf + 3 rakam | ZAA 123 |
| KKTC Kiralık Yeni | 3 harf + 3 rakam + 1 harf (Z/T ile başlar, kırmızı plaka) | ZAA 123 A |
| Türkiye Cumhuriyeti | İl kodu + harf + rakam | 34 ABC 123 |
