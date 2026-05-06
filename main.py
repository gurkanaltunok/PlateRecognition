import os
import sys
import re
import time
import shutil
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import easyocr
from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_AVAILABLE = True
except ImportError:
    HEIC_AVAILABLE = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PLATES_DIR = BASE_DIR / "plates"
PROCESSED_DIR = BASE_DIR / "processed"
DATA_DIR = BASE_DIR / "data"
VEHICLES_TXT = DATA_DIR / "vehicles.txt"
RESULTS_TXT = DATA_DIR / "results.txt"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".heic", ".heif"}

LOG_FORMAT = "[%(asctime)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
logger = logging.getLogger("CarPlateSystem")


# ---------------------------------------------------------------------------
#  1. Image Preprocessing
# ---------------------------------------------------------------------------

class ImagePreprocessor:
    def load_image(self, image_path: str) -> np.ndarray:
        ext = Path(image_path).suffix.lower()

        if ext in (".heic", ".heif"):
            if not HEIC_AVAILABLE:
                raise ValueError("pillow-heif is required for HEIC support: pip install pillow-heif")
            pil_img = Image.open(image_path).convert("RGB")
            image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return image

    def _crop_left_strip(self, image: np.ndarray) -> np.ndarray:
        """Crops the blue EU/country indicator strip on the left to prevent OCR confusion."""
        h, w = image.shape[:2]
        left_w = int(w * 0.15)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        left_hsv = hsv[:, :left_w]
        blue_mask = cv2.inRange(left_hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        if np.count_nonzero(blue_mask) / blue_mask.size > 0.15:
            return image[:, int(w * 0.12):]
        return image

    def _is_red_plate(self, image: np.ndarray) -> bool:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([170, 50, 50]), np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 | mask2
        red_ratio = np.count_nonzero(red_mask) / red_mask.size
        return red_ratio > 0.15

    def _process_red_plate(self, image: np.ndarray) -> list:
        versions = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        red_mask = mask1 | mask2
        text_mask = cv2.bitwise_not(red_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        text_clean = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel)
        versions.append(text_clean)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        versions.append(cv2.bitwise_not(gray))
        return versions

    def _process_normal_plate(self, image: np.ndarray) -> list:
        versions = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if w < 400:
            scale = 400 / w
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        denoised = cv2.bilateralFilter(gray, 11, 17, 17)
        versions.append(denoised)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        versions.append(enhanced)

        thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        versions.append(cleaned)

        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)
        versions.append(sharpened)

        return versions

    def process(self, image: np.ndarray) -> list:
        image = self._crop_left_strip(image)
        if self._is_red_plate(image):
            return self._process_red_plate(image)
        else:
            return self._process_normal_plate(image)


# ---------------------------------------------------------------------------
#  2. OCR Engine
# ---------------------------------------------------------------------------

class PlateOCR:
    def __init__(self):
        logger.info("Loading OCR engine...")
        self.reader = easyocr.Reader(["tr", "en"], gpu=False, verbose=False)
        logger.info("OCR engine is ready.")

    def read(self, original_image: np.ndarray, processed_versions: list) -> list:
        all_results = []
        images_to_try = [original_image] + processed_versions

        for image in images_to_try:
            raw_results = self.reader.readtext(image, detail=1, paragraph=False, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
            raw_free = self.reader.readtext(image, detail=1, paragraph=False)
            combined_raw = raw_results + raw_free

            if not combined_raw:
                continue

            img_h = image.shape[0]
            plate_zone_max_y = img_h * 0.70

            sorted_results = sorted(combined_raw, key=lambda r: min(pt[1] for pt in r[0]))
            plate_parts = []
            
            for (bbox, text, conf) in sorted_results:
                y_top = min(pt[1] for pt in bbox)
                cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
                if not cleaned:
                    continue

                all_results.append((cleaned, conf))
                if y_top <= plate_zone_max_y:
                    plate_parts.append((cleaned, conf))

            if len(plate_parts) >= 2:
                combined = "".join(p[0] for p in plate_parts)
                avg_conf = sum(p[1] for p in plate_parts) / len(plate_parts)
                all_results.append((combined, avg_conf))

        return all_results


# ---------------------------------------------------------------------------
#  3. Plate Format Validation
# ---------------------------------------------------------------------------

class PlateValidator:
    KKTC_OLD_PATTERN = re.compile(r"^[A-Z]{2}\d{3}$")
    KKTC_NEW_PATTERN = re.compile(r"^[A-Z]{2}\d{3}[A-Z]$")
    KKTC_RENTAL_OLD_PATTERN = re.compile(r"^[A-Z]{3}\d{3}$")
    KKTC_RENTAL_NEW_PATTERN = re.compile(r"^[ZT][A-Z]{2}\d{3}[A-Z]$")

    TR_PATTERNS = [
        re.compile(r"^\d{2}[A-Z]\d{4}$"),
        re.compile(r"^\d{2}[A-Z]\d{5}$"),
        re.compile(r"^\d{2}[A-Z]{2}\d{3}$"),
        re.compile(r"^\d{2}[A-Z]{2}\d{4}$"),
        re.compile(r"^\d{2}[A-Z]{3}\d{2}$"),
        re.compile(r"^\d{2}[A-Z]{3}\d{3}$"),
    ]

    DIGIT_TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "6": "G"}
    LETTER_TO_DIGIT = {
        "O": "0", "o": "0", "Q": "0", "D": "0",
        "G": "6", "g": "0", "U": "0",
        "I": "1", "l": "1", "L": "1",
        "S": "5", "s": "5",
        "B": "8", "b": "6",
        "Z": "2", "z": "2",
        "T": "7", "A": "4",
    }
    # KKTC plates use a slashed "0" that OCR often reads as 6, 8, or O
    KKTC_CONFUSIONS = {
        "letters": {"O": "U", "U": "O"},
        "digits":  {"0": ["6", "8"], "6": ["0", "8"], "8": ["0", "6"]},
    }

    def _clean(self, text: str) -> str:
        return re.sub(r"[^A-Za-z0-9]", "", text)

    def _fix_kktc_old(self, text: str) -> str:
        if len(text) != 5: return text
        fixed = list(text)
        for i in range(2):
            c = fixed[i].upper()
            fixed[i] = self.DIGIT_TO_LETTER.get(c, c) if c.isdigit() else c
        for i in range(2, 5):
            c = fixed[i]
            if c.isalpha():
                mapped = self.LETTER_TO_DIGIT.get(c) or self.LETTER_TO_DIGIT.get(c.upper(), c.upper())
                fixed[i] = mapped
        return "".join(fixed)

    def _fix_kktc_new(self, text: str) -> str:
        if len(text) != 6: return text
        fixed = list(text)
        for i in range(2):
            c = fixed[i].upper()
            fixed[i] = self.DIGIT_TO_LETTER.get(c, c) if c.isdigit() else c
        for i in range(2, 5):
            c = fixed[i]
            if c.isalpha():
                mapped = self.LETTER_TO_DIGIT.get(c) or self.LETTER_TO_DIGIT.get(c.upper(), c.upper())
                fixed[i] = mapped
        c = fixed[5].upper()
        fixed[5] = self.DIGIT_TO_LETTER.get(c, c) if c.isdigit() else c
        return "".join(fixed)

    def _all_variants(self, plate: str, plate_type: str) -> list:
        """Generate all alternative readings by substituting confused chars in each position."""
        from itertools import product as iproduct

        # letter_pos: positions that must be letters (O/U confusion applied)
        # digit_pos:  positions that must be digits (0/6/8 confusion applied)
        layout = {
            "KKTC_OLD":        {"letter_pos": {0, 1},          "digit_pos": {2, 3, 4}},
            "KKTC_NEW":        {"letter_pos": {0, 1, 5},       "digit_pos": {2, 3, 4}},
            "KKTC_RENTAL_OLD": {"letter_pos": {0, 1, 2},       "digit_pos": {3, 4, 5}},
            "KKTC_RENTAL_NEW": {"letter_pos": {0, 1, 2, 6},    "digit_pos": {3, 4, 5}},
        }
        if plate_type not in layout:
            return []

        letter_pos = layout[plate_type]["letter_pos"]
        digit_pos  = layout[plate_type]["digit_pos"]
        letter_swap = self.KKTC_CONFUSIONS["letters"]
        digit_swap  = self.KKTC_CONFUSIONS["digits"]

        options = []
        for i, c in enumerate(plate):
            if i in letter_pos and c in letter_swap:
                options.append([c, letter_swap[c]])
            elif i in digit_pos and c in digit_swap:
                options.append([c] + digit_swap[c])
            else:
                options.append([c])

        variants = set()
        for combo in iproduct(*options):
            v = "".join(combo)
            if v != plate:
                variants.add(v)
        return list(variants)

    def _fix_kktc_rental_old(self, text: str) -> str:
        if len(text) != 6: return text
        fixed = list(text)
        for i in range(3):
            c = fixed[i].upper()
            fixed[i] = self.DIGIT_TO_LETTER.get(c, c) if c.isdigit() else c
        for i in range(3, 6):
            c = fixed[i]
            if c.isalpha():
                mapped = self.LETTER_TO_DIGIT.get(c) or self.LETTER_TO_DIGIT.get(c.upper(), c.upper())
                fixed[i] = mapped
        return "".join(fixed)

    def _fix_kktc_rental_new(self, text: str) -> str:
        if len(text) != 7: return text
        fixed = list(text)
        for i in range(3):  # pos 0-2: letters
            c = fixed[i].upper()
            fixed[i] = self.DIGIT_TO_LETTER.get(c, c) if c.isdigit() else c
        for i in range(3, 6):  # pos 3-5: digits
            c = fixed[i]
            if c.isalpha():
                mapped = self.LETTER_TO_DIGIT.get(c) or self.LETTER_TO_DIGIT.get(c.upper(), c.upper())
                fixed[i] = mapped
        c = fixed[6].upper()  # pos 6: letter
        fixed[6] = self.DIGIT_TO_LETTER.get(c, c) if c.isdigit() else c
        return "".join(fixed)

    def validate(self, raw_text: str):
        cleaned = self._clean(raw_text)

        if len(cleaned) == 5:
            fixed = self._fix_kktc_old(cleaned)
            if self.KKTC_OLD_PATTERN.match(fixed):
                return True, fixed, "KKTC_OLD"

        if len(cleaned) == 6:
            fixed = self._fix_kktc_new(cleaned)
            if self.KKTC_NEW_PATTERN.match(fixed):
                return True, fixed, "KKTC_NEW"

        if len(cleaned) == 6:
            fixed = self._fix_kktc_rental_old(cleaned)
            if self.KKTC_RENTAL_OLD_PATTERN.match(fixed):
                return True, fixed, "KKTC_RENTAL_OLD"

        if len(cleaned) == 7:
            fixed = self._fix_kktc_rental_new(cleaned)
            if self.KKTC_RENTAL_NEW_PATTERN.match(fixed):
                return True, fixed, "KKTC_RENTAL_NEW"

        cleaned_upper = cleaned.upper()
        for pattern in self.TR_PATTERNS:
            if pattern.match(cleaned_upper):
                return True, cleaned_upper, "TR"

        return False, cleaned_upper, "UNKNOWN"

    def _expand_candidates(self, ocr_results: list) -> list:
        """Try 1- and 2-char deletions to recover correct plates from over-long OCR reads."""
        expanded = []
        seen = set()
        for text, conf in ocr_results:
            if text not in seen:
                seen.add(text)
                expanded.append((text, conf))

        round1 = []
        for text, conf in ocr_results:
            if len(text) > 5:
                for i in range(len(text)):
                    deleted = text[:i] + text[i + 1:]
                    if deleted not in seen:
                        seen.add(deleted)
                        round1.append((deleted, conf * 0.85))
        expanded.extend(round1)

        for text, conf in round1:
            if len(text) > 5:
                for i in range(len(text)):
                    deleted = text[:i] + text[i + 1:]
                    if deleted not in seen:
                        seen.add(deleted)
                        expanded.append((deleted, conf * 0.85))

        return expanded

    def find_best_plate(self, ocr_results: list, db=None):
        candidates = []

        for raw_text, confidence in self._expand_candidates(ocr_results):
            is_valid, normalized, plate_type = self.validate(raw_text)
            if not is_valid or plate_type == "UNKNOWN":
                continue
            candidates.append((normalized, plate_type, confidence))
            for v in self._all_variants(normalized, plate_type):
                is_v, norm_v, pt_v = self.validate(v)
                if is_v and pt_v != "UNKNOWN":
                    candidates.append((norm_v, pt_v, confidence * 0.9))

        if not candidates:
            if ocr_results:
                best = max(ocr_results, key=lambda x: x[1])
                return self._clean(best[0]).upper(), "UNKNOWN", best[1]
            return "", "UNKNOWN", 0.0

        if db:
            db_hits = [(p, pt, c) for p, pt, c in candidates if db.find(p)]
            if db_hits:
                candidates = db_hits

        freq = {}
        best_conf = {}
        best_type = {}
        for plate, ptype, conf in candidates:
            freq[plate] = freq.get(plate, 0) + 1
            if plate not in best_conf or conf > best_conf[plate]:
                best_conf[plate] = conf
                best_type[plate] = ptype

        # Shorter formats are preferred; rental plates are 7 chars and rarely misread as shorter
        priority = {"KKTC_OLD": 0, "KKTC_NEW": 1, "KKTC_RENTAL_OLD": 2, "KKTC_RENTAL_NEW": 3, "TR": 4}
        ranked = sorted(freq, key=lambda p: (priority.get(best_type[p], 5), -freq[p], -best_conf[p]))

        best = ranked[0]
        return best, best_type[best], best_conf[best]


# ---------------------------------------------------------------------------
#  4. Vehicle Database
# ---------------------------------------------------------------------------

class VehicleDatabase:
    def __init__(self, txt_path: str):
        self.txt_path = txt_path
        self.vehicles = {}
        self.reload()

    def reload(self):
        self.vehicles = {}
        if not os.path.exists(self.txt_path):
            logger.warning(f"Vehicle database not found: {self.txt_path}")
            return

        try:
            with open(self.txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(',', 1)
                    if len(parts) == 2:
                        plate = re.sub(r"[^A-Z0-9]", "", parts[0].upper())
                        owner_name = parts[1].strip()
                        if plate:
                            self.vehicles[plate] = {"plate": plate, "owner_name": owner_name}
            logger.info(f"Vehicle database loaded: {len(self.vehicles)} records.")
        except Exception as e:
            logger.error(f"TXT read error: {e}")

    def register_vehicle(self, plate: str, owner_name: str):
        plate = re.sub(r"[^A-Z0-9]", "", plate.upper())
        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write(f"{plate},{owner_name}\n")
        print(f"\n[SUCCESS] Plate '{plate}' registered to '{owner_name}'.")
        self.reload()

    def edit_vehicle(self, plate: str, new_owner_name: str):
        plate = re.sub(r"[^A-Z0-9]", "", plate.upper())
        if plate not in self.vehicles:
            return False
            
        lines = []
        try:
            with open(self.txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            pass
            
        with open(self.txt_path, "w", encoding="utf-8") as f:
            for line in lines:
                parts = line.strip().split(',', 1)
                if len(parts) == 2 and re.sub(r"[^A-Z0-9]", "", parts[0].upper()) == plate:
                    f.write(f"{plate},{new_owner_name}\n")
                else:
                    f.write(line)
                    
        print(f"\n[SUCCESS] Plate '{plate}' updated to '{new_owner_name}'.")
        self.reload()
        return True

    def find(self, plate: str):
        normalized = re.sub(r"[^A-Z0-9]", "", plate.upper())
        return self.vehicles.get(normalized)


# ---------------------------------------------------------------------------
#  5. Result Logger
# ---------------------------------------------------------------------------

class ResultLogger:
    def __init__(self, txt_path: str):
        self.txt_path = txt_path

    def log(self, result: dict):
        status_map = {
            "REGISTERED": "REGISTERED", 
            "UNREGISTERED": "UNREGISTERED", 
            "OCR_FAILED": "OCR_FAILED", 
            "ERROR": "ERROR",
            "FAILED": "FAILED"
        }
        status_en = status_map.get(result['match_status'], result['match_status'])
        
        line = f"[{result['timestamp']}] File: {result['file_name']} | Plate: {result['read_plate']} | Type: {result['plate_type']} | Conf: {result['confidence_score']} | Status: {status_en} | Owner: {result['owner']}\n"
        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write(line)


# ---------------------------------------------------------------------------
#  6. Main Orchestrator
# ---------------------------------------------------------------------------

class PlateRecognizer:
    def __init__(self, db: "VehicleDatabase" = None):
        self.preprocessor = ImagePreprocessor()
        self.ocr = PlateOCR()
        self.validator = PlateValidator()
        self.db = db if db is not None else VehicleDatabase(str(VEHICLES_TXT))
        self.result_logger = ResultLogger(str(RESULTS_TXT))

    def recognize(self, image_path: str) -> dict:
        filename = os.path.basename(image_path)
        result = {
            "timestamp": datetime.now().strftime(DATE_FORMAT),
            "file_name": filename,
            "read_plate": "",
            "plate_type": "",
            "confidence_score": "",
            "match_status": "FAILED",
            "owner": "",
        }

        try:
            original = self.preprocessor.load_image(image_path)
            processed_versions = self.preprocessor.process(original)
            ocr_results = self.ocr.read(original, processed_versions)

            if not ocr_results:
                result["match_status"] = "OCR_FAILED"
                self._print_result(result)
                self.result_logger.log(result)
                return result

            plate, plate_type, confidence = self.validator.find_best_plate(ocr_results, db=self.db)

            result["read_plate"] = plate
            result["plate_type"] = plate_type
            result["confidence_score"] = f"%{confidence * 100:.1f}"

            vehicle = self.db.find(plate)
            if vehicle:
                result["match_status"] = "REGISTERED"
                result["owner"] = vehicle["owner_name"]
            else:
                result["match_status"] = "UNREGISTERED"
                variants = self.validator._all_variants(plate, plate_type)
                if variants:
                    result["variants"] = variants

        except Exception as e:
            result["match_status"] = "ERROR"
            logger.error(f"Processing error ({filename}): {e}")

        self._print_result(result)
        self.result_logger.log(result)
        return result

    def _print_result(self, result: dict):
        print()
        print(f"[{result['timestamp']}] Image processed: {result['file_name']}")
        
        plate = result['read_plate'] or '(unreadable)'
        print(f"  Plate           : {plate}")
        print(f"  Plate Type      : {result['plate_type']}")
        print(f"  Confidence Score: {result['confidence_score'] or '-'}")
        
        status = result['match_status']
        if status == "REGISTERED":
            print(f"  Status          : REGISTERED")
            print(f"  Owner           : {result['owner']}")
        elif status == "UNREGISTERED":
            print(f"  Status          : UNREGISTERED")
        else:
            print(f"  Status          : {status}")

        variants = result.get("variants", [])
        if variants and status != "REGISTERED":
            shown = variants[:5]
            print(f"  Possible Plates : {', '.join(shown)}{' ...' if len(variants) > 5 else ''}")

        print(f"  Saved to log file.")
        print("-" * 64)


# ---------------------------------------------------------------------------
#  7. Folder Monitor (Watchdog)
# ---------------------------------------------------------------------------

def _is_image_file(filepath: str) -> bool:
    return Path(filepath).suffix.lower() in SUPPORTED_EXTENSIONS

def _wait_for_file_ready(filepath: str, timeout: float = 5.0):
    prev_size = -1
    waited = 0.0
    interval = 0.3
    while waited < timeout:
        try:
            curr_size = os.path.getsize(filepath)
            if curr_size == prev_size and curr_size > 0:
                return True
            prev_size = curr_size
        except OSError:
            pass
        time.sleep(interval)
        waited += interval
    return os.path.exists(filepath)


class PlateFileHandler(FileSystemEventHandler):
    def __init__(self, recognizer: PlateRecognizer):
        super().__init__()
        self.recognizer = recognizer

    def on_created(self, event):
        if event.is_directory:
            return
        filepath = event.src_path
        if not _is_image_file(filepath):
            return

        logger.info(f"New image detected: {os.path.basename(filepath)}")
        _wait_for_file_ready(filepath)
        self.recognizer.recognize(filepath)
        _move_to_processed(filepath)


def _move_to_processed(filepath: str):
    try:
        dest = PROCESSED_DIR / os.path.basename(filepath)
        counter = 1
        original_dest = dest
        while dest.exists():
            stem = original_dest.stem
            suffix = original_dest.suffix
            dest = PROCESSED_DIR / f"{stem}_{counter}{suffix}"
            counter += 1
        shutil.move(filepath, str(dest))
    except Exception as e:
        logger.error(f"Could not move file: {e}")


def process_existing_files(recognizer: PlateRecognizer):
    image_files = sorted(
        f for f in PLATES_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_files:
        logger.info("No images found in plates/ directory.")
        return 0

    print(f"\n  {len(image_files)} images to process.")
    print("=" * 64)

    for i, img_path in enumerate(image_files, 1):
        print(f"\n  [{i}/{len(image_files)}]", end="")
        recognizer.recognize(str(img_path))
        _move_to_processed(str(img_path))

    return len(image_files)


def _ensure_directories():
    PLATES_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)


def start_scanning(db):
    if not WATCHDOG_AVAILABLE:
        print("\n[ERROR] 'watchdog' library not installed. Install with: pip install watchdog")
        return

    print("\n" + "=" * 64)
    print("  SCANNING MODE ACTIVE")
    print(f"  Monitoring Directory : {PLATES_DIR}")
    print(f"  Processed Directory  : {PROCESSED_DIR}")
    print(f"  Press Ctrl+C to stop scanning and return to Main Menu")
    print("=" * 64)

    recognizer = PlateRecognizer(db=db)
    process_existing_files(recognizer)

    handler = PlateFileHandler(recognizer)
    observer = Observer()
    observer.schedule(handler, str(PLATES_DIR), recursive=False)
    observer.start()

    print("  Waiting for new images...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Stopping scanning mode...")
        observer.stop()

    observer.join()
    print("  Scanning stopped. Returned to menu.")
    print("=" * 64)


def main():
    _ensure_directories()
    db = VehicleDatabase(str(VEHICLES_TXT))

    while True:
        print("\n" + "=" * 40)
        print("  CAR PLATE RECOGNITION SYSTEM")
        print("=" * 40)
        print("  1. Start Plate Scanning")
        print("  2. Register Vehicle")
        print("  3. Edit Record")
        print("  4. Exit")
        print("=" * 40)

        choice = input("Select an option (1-4): ").strip()

        if choice == '1':
            start_scanning(db)
        elif choice == '2':
            plate = input("Enter Plate: ").strip()
            owner_name = input("Enter Name Surname: ").strip()
            if plate and owner_name:
                db.register_vehicle(plate, owner_name)
            else:
                print("[ERROR] Plate and Name cannot be empty.")
        elif choice == '3':
            plate = input("Enter Plate to edit: ").strip()
            record = db.find(plate)
            if record:
                print(f"Current Owner: {record['owner_name']}")
                new_owner_name = input("Enter New Name Surname: ").strip()
                if new_owner_name:
                    db.edit_vehicle(plate, new_owner_name)
                else:
                    print("[ERROR] Name cannot be empty.")
            else:
                print(f"\n[ERROR] Plate '{plate}' is not registered.")
        elif choice == '4':
            print("Exiting...")
            sys.exit(0)
        else:
            print("[ERROR] Invalid option. Please select 1-4.")


if __name__ == "__main__":
    main()
