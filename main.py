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
ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

LOG_FORMAT = "[%(asctime)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
logger = logging.getLogger("CarPlateSystem")


# ---------------------------------------------------------------------------
#  Plate format definitions
# ---------------------------------------------------------------------------
#
#  L = letter slot, D = digit slot. Each layout is checked end-to-end against
#  candidate texts; mismatched characters are corrected via the confusion maps
#  below before the final regex pattern check.
# ---------------------------------------------------------------------------

PLATE_LAYOUTS = [
    # (type_name, layout, final_pattern, extra_constraint)
    ("KKTC_OLD",        list("LLDDD"),    re.compile(r"^[A-Z]{2}\d{3}$"),         None),
    ("KKTC_NEW",        list("LLDDDL"),   re.compile(r"^[A-Z]{2}\d{3}[A-Z]$"),    None),
    ("KKTC_RENTAL_OLD", list("LLLDDD"),   re.compile(r"^[A-Z]{3}\d{3}$"),         None),
    ("KKTC_RENTAL_NEW", list("LLLDDDL"),  re.compile(r"^[ZT][A-Z]{2}\d{3}[A-Z]$"), lambda s: s[0] in "ZT"),
    ("TR_OLD_1",        list("DDLDDDD"),  re.compile(r"^\d{2}[A-Z]\d{4}$"),       None),
    ("TR_OLD_2",        list("DDLDDDDD"), re.compile(r"^\d{2}[A-Z]\d{5}$"),       None),
    ("TR_NEW_1",        list("DDLLDDD"),  re.compile(r"^\d{2}[A-Z]{2}\d{3}$"),    None),
    ("TR_NEW_2",        list("DDLLDDDD"), re.compile(r"^\d{2}[A-Z]{2}\d{4}$"),    None),
    ("TR_NEW_3",        list("DDLLLDD"),  re.compile(r"^\d{2}[A-Z]{3}\d{2}$"),    None),
    ("TR_NEW_4",        list("DDLLLDDD"), re.compile(r"^\d{2}[A-Z]{3}\d{3}$"),    None),
]

# Two-tier region preference: any KKTC layout outranks any TR layout. The user
# of this system is in TRNC, so a TR-pattern candidate should never beat a
# valid KKTC reading even if the TR text happens to be longer.
KKTC_TYPES = {"KKTC_OLD", "KKTC_NEW", "KKTC_RENTAL_OLD", "KKTC_RENTAL_NEW"}

# When two narrow letters touch in stylised plate fonts, EasyOCR sometimes
# reads the joined glyph as a single wide character. Each expansion is a list
# of (substitute, prior_bonus) — the bonus reflects how common that two-letter
# combination is in KKTC plates relative to the others. These are applied as
# additional candidates, never as overrides.
WIDE_LETTER_EXPANSIONS = {
    "W": [("UD", 0.10), ("VV", 0.04), ("UU", 0.02),
          ("II", 0.0),  ("VY", 0.0),  ("YV", 0.0)],
    "M": [("IN", 0.04), ("NI", 0.03), ("NN", 0.0),
          ("MN", 0.0),  ("NM", 0.0)],
    "N": [("II", 0.0),  ("IN", 0.0),  ("NI", 0.0)],
}

# Common OCR confusions when a letter is read in a digit slot (and vice versa).
LETTER_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0", "U": "0",
    "I": "1", "L": "1", "J": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
    "A": "4",
}
DIGIT_TO_LETTER = {
    "0": "O", "1": "I", "2": "Z", "4": "A",
    "5": "S", "6": "G", "7": "T", "8": "B",
}

# Format priority (lower index = higher priority). Used as a tiebreaker so that
# longer matches beat shorter ones — preventing "AA172M" from being trimmed to
# "AA172" because the latter happens to be a valid KKTC_OLD.
FORMAT_PRIORITY = {name: i for i, (name, *_rest) in enumerate(reversed(PLATE_LAYOUTS))}


# ---------------------------------------------------------------------------
#  1. Image preprocessing — generate multiple variants
# ---------------------------------------------------------------------------

class ImagePreprocessor:
    TARGET_WIDTH = 1600

    def load_image(self, image_path: str) -> np.ndarray:
        ext = Path(image_path).suffix.lower()
        if ext in (".heic", ".heif"):
            if not HEIC_AVAILABLE:
                raise ValueError("pillow-heif is required for HEIC support")
            pil_img = Image.open(image_path).convert("RGB")
            image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return image

    def variants(self, image: np.ndarray) -> list:
        """Returns a list of (name, image) preprocessing variants."""
        h, w = image.shape[:2]
        if w < self.TARGET_WIDTH:
            scale = self.TARGET_WIDTH / w
            big = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            big = image

        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_inv = cv2.bitwise_not(otsu)

        return [
            ("color",    big),
            ("gray",     gray),
            ("clahe",    clahe),
            ("otsu",     otsu),
            ("otsu_inv", otsu_inv),
        ]


# ---------------------------------------------------------------------------
#  2. OCR engine — collect detections from every variant
# ---------------------------------------------------------------------------

class PlateOCR:
    def __init__(self):
        logger.info("Loading OCR engine (this can take a moment on first run)...")
        self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        logger.info("OCR engine ready.")

    def detect_all(self, variants: list) -> list:
        """Run OCR on every variant. Returns list of detection dicts containing
        text, confidence, bounding box and source variant name. Allowlist and
        free-text passes are deduplicated when they produce identical readings.
        """
        seen = {}                              # key -> detection
        for name, img in variants:
            for use_allow in (True, False):
                kwargs = {"detail": 1, "paragraph": False}
                if use_allow:
                    kwargs["allowlist"] = ALLOWLIST
                try:
                    results = self.reader.readtext(img, **kwargs)
                except Exception as exc:
                    logger.warning(f"OCR failed on {name}: {exc}")
                    continue
                for bbox, text, conf in results:
                    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
                    if not cleaned:
                        continue
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    x0, x1 = int(min(xs)), int(max(xs))
                    y0, y1 = int(min(ys)), int(max(ys))
                    key = (name, cleaned, x0, y0, x1, y1)
                    if key in seen and seen[key]["conf"] >= conf:
                        continue
                    seen[key] = {
                        "text":   cleaned,
                        "conf":   float(conf),
                        "bbox":   (x0, y0, x1, y1),
                        "source": name,
                    }
        return list(seen.values())


# ---------------------------------------------------------------------------
#  2b. Visual Y/V/U disambiguator
# ---------------------------------------------------------------------------
#
#  KKTC plate fonts often render Y with such a short bottom stem that EasyOCR
#  reads it as V (and vice versa for some U/V cases). We disambiguate by
#  cropping each suspect character cell from a clean grayscale image and
#  measuring the longest contiguous vertical run in the central columns.
#
#    - Y has a strong vertical stem reaching at least ~half of the char height
#    - V tapers to a point: no long central run
#    - U has empty middle except for the rounded base: short central run
# ---------------------------------------------------------------------------

def _isolate_character(char_bin: np.ndarray) -> np.ndarray:
    """Pick the most central, character-sized connected component out of the
    binary cell. This strips plate-frame noise bands and side-letter bleed.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(char_bin, connectivity=8)
    if n < 2:
        return char_bin
    h, w = char_bin.shape
    best_idx, best_score = -1, -1.0
    for i in range(1, n):
        x, y, w_c, h_c, area = stats[i]
        if h_c < h * 0.40 or area < 30:
            continue                       # too short or too small
        if w_c >= w * 0.95 and h_c <= h * 0.25:
            continue                       # horizontal noise band
        cx = x + w_c / 2.0
        center_dist = abs(cx - w / 2.0) / max(w, 1)
        score = area * (1.0 - min(center_dist, 1.0))
        if score > best_score:
            best_score = score
            best_idx = i
    if best_idx < 0:
        return char_bin
    mask = (labels == best_idx).astype(np.uint8) * 255
    x, y, w_c, h_c, _ = stats[best_idx]
    return mask[y:y + h_c, x:x + w_c]


def _count_horizontal_runs(row: np.ndarray) -> int:
    """Number of contiguous foreground runs in a single row."""
    runs = 0
    in_run = False
    for px in row:
        if px > 127:
            if not in_run:
                runs += 1
                in_run = True
        else:
            in_run = False
    return runs


def _stroke_merge_y_ratio(char_bin: np.ndarray) -> float:
    """Find where the two diagonal strokes of V/Y merge into a single stroke.
    Returns the relative y position (0=top, 1=bottom).

      - Y merges in the upper-middle (~0.45-0.55): the fork joins above the stem
      - V merges only near the bottom (~0.75-0.90): the diagonals converge to a tip
    """
    iso = _isolate_character(char_bin)
    h, _ = iso.shape
    if h < 8:
        return 1.0
    saw_two = False
    for y in range(h):
        runs = _count_horizontal_runs(iso[y])
        if runs >= 2:
            saw_two = True
        elif saw_two and runs == 1:
            return y / h
        elif saw_two and runs == 0:
            return 1.0
    return 1.0


def disambiguate_letters(detection: dict, gray_image: np.ndarray) -> str:
    """Return the detection text with V/Y characters corrected based on the
    vertical-stem test. The image is the same-resolution grayscale variant
    used during OCR, so the bbox is directly usable.
    """
    text = detection["text"]
    if not any(c in text for c in "VY"):
        return text

    x0, y0, x1, y1 = detection["bbox"]
    x0 = max(x0, 0); y0 = max(y0, 0)
    x1 = min(x1, gray_image.shape[1]); y1 = min(y1, gray_image.shape[0])
    if x1 - x0 < 10 or y1 - y0 < 10:
        return text

    word = gray_image[y0:y1, x0:x1]
    n = len(text)
    cell_w = (x1 - x0) / n

    out = []
    for i, ch in enumerate(text):
        if ch not in "VY":
            out.append(ch)
            continue
        cx0 = int(i * cell_w)
        cx1 = int((i + 1) * cell_w)
        cell = word[:, cx0:cx1]
        if cell.size < 50:
            out.append(ch)
            continue
        _, cell_bin = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        merge_y = _stroke_merge_y_ratio(cell_bin)
        # Y's fork merges into the stem in the upper-middle of the character.
        if merge_y < 0.65:
            out.append("Y")
        # V's diagonals only meet near the bottom of the character.
        elif merge_y > 0.72:
            out.append("V")
        else:
            out.append(ch)  # ambiguous → trust OCR
    return "".join(out)


# ---------------------------------------------------------------------------
#  3. Candidate construction (handles single + multi-line plates)
# ---------------------------------------------------------------------------

def _expand_wide_letters(text: str) -> list:
    """Generate (variant, prior_bonus) tuples by replacing wide-letter glyphs
    with their plausible two-letter substitutes."""
    variants = []
    for i, ch in enumerate(text):
        if ch in WIDE_LETTER_EXPANSIONS:
            for sub, sub_bonus in WIDE_LETTER_EXPANSIONS[ch]:
                variants.append((text[:i] + sub + text[i + 1:], sub_bonus))
    return variants


def build_candidates(detections: list, gray_image: np.ndarray = None) -> list:
    """Build candidate (text, conf) pairs.

    Each individual detection is a candidate. Additionally, vertically-stacked
    detections from the same variant are concatenated to recover two-line
    plates (e.g. "AA" / "583Z"). When a clean grayscale image is provided,
    each detection also yields a Y/V-disambiguated variant.
    """
    cands = []  # list of (text, conf, bonus)
    for det in detections:
        cands.append((det["text"], det["conf"], 0.0))
        if gray_image is not None:
            corrected = disambiguate_letters(det, gray_image)
            if corrected != det["text"]:
                # Disambiguator overrode V/Y — give it a small advantage so it
                # wins ties against the raw OCR reading.
                cands.append((corrected, det["conf"], 0.15))

    by_source = {}
    for det in detections:
        by_source.setdefault(det["source"], []).append(det)

    for source_dets in by_source.values():
        # ----- vertical stacking (two-line plates) -----
        ordered_y = sorted(source_dets, key=lambda d: d["bbox"][1])
        n = len(ordered_y)
        for span in (2, 3):
            for i in range(n - span + 1):
                group = ordered_y[i:i + span]
                ok = True
                for a, b in zip(group, group[1:]):
                    a_h = a["bbox"][3] - a["bbox"][1]
                    gap = b["bbox"][1] - a["bbox"][3]
                    if gap < -a_h * 0.5:
                        ok = False
                        break
                if not ok:
                    continue
                text = "".join(d["text"] for d in group)
                conf = sum(d["conf"] for d in group) / len(group)
                cands.append((text, conf, 0.0))
                if gray_image is not None:
                    corrected = "".join(disambiguate_letters(d, gray_image) for d in group)
                    if corrected != text:
                        cands.append((corrected, conf, 0.15))

        # ----- horizontal concatenation of same-row fragments -----
        # When OCR splits a plate ("U" + "736"), join detections that share a
        # vertical band, ordered left-to-right. Only join fragments with
        # similar heights and minimal X overlap so that nested mis-detections
        # ("VI" inside "V360") are not stitched together.
        def _compatible(a, b):
            ax0, ay0, ax1, ay1 = a["bbox"]
            bx0, by0, bx1, by1 = b["bbox"]
            ah = ay1 - ay0
            bh = by1 - by0
            aw = ax1 - ax0
            bw = bx1 - bx0
            if ah <= 0 or bh <= 0 or aw <= 0 or bw <= 0:
                return False
            if min(ah, bh) / max(ah, bh) < 0.6:           # heights too different
                return False
            y_ov = min(ay1, by1) - max(ay0, by0)
            if y_ov < min(ah, bh) * 0.5:                  # not on same row
                return False
            x_ov = min(ax1, bx1) - max(ax0, bx0)
            if x_ov > min(aw, bw) * 0.30:                 # too much X overlap
                return False
            return True

        rows = []
        for det in sorted(source_dets, key=lambda d: d["bbox"][0]):
            placed = False
            for row in rows:
                if _compatible(row[-1], det):
                    row.append(det)
                    placed = True
                    break
            if not placed:
                rows.append([det])
        for row in rows:
            if len(row) < 2 or len(row) > 4:
                continue
            text = "".join(d["text"] for d in row)
            conf = sum(d["conf"] for d in row) / len(row)
            cands.append((text, conf, 0.0))
            if gray_image is not None:
                corrected = "".join(disambiguate_letters(d, gray_image) for d in row)
                if corrected != text:
                    cands.append((corrected, conf, 0.15))

    # Wide-letter expansion: add W/M/N → 2-letter alternates as extra candidates.
    extra = []
    for raw, conf, bonus in cands:
        if any(c in raw for c in WIDE_LETTER_EXPANSIONS):
            for variant, sub_bonus in _expand_wide_letters(raw):
                # Penalise expansions so they only win when the original OCR
                # candidate fails to validate; the prior bonus then breaks ties
                # between mutually-valid expansions (UD beats VV beats UU…).
                extra.append((variant, conf * 0.85, bonus + sub_bonus))
    cands.extend(extra)
    return cands


# ---------------------------------------------------------------------------
#  4. Plate validation & format fitting
# ---------------------------------------------------------------------------

class PlateValidator:
    def _fit_layout(self, text: str, layout: list):
        """Coerce text into a layout. Returns (corrected, n_corrections) or None."""
        if len(text) != len(layout):
            return None
        out = []
        n_corr = 0
        for c, role in zip(text, layout):
            if role == "L":
                if c.isalpha():
                    out.append(c)
                elif c in DIGIT_TO_LETTER:
                    out.append(DIGIT_TO_LETTER[c])
                    n_corr += 1
                else:
                    return None
            else:  # D
                if c.isdigit():
                    out.append(c)
                elif c.upper() in LETTER_TO_DIGIT:
                    out.append(LETTER_TO_DIGIT[c.upper()])
                    n_corr += 1
                else:
                    return None
        return "".join(out), n_corr

    def validate(self, raw_text: str) -> list:
        """Return list of (corrected_plate, type, n_corrections) for all matching layouts."""
        cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
        matches = []
        for type_name, layout, pattern, extra in PLATE_LAYOUTS:
            fit = self._fit_layout(cleaned, layout)
            if fit is None:
                continue
            corrected, n_corr = fit
            if not pattern.match(corrected):
                continue
            if extra and not extra(corrected):
                continue
            matches.append((corrected, type_name, n_corr))
        return matches


# ---------------------------------------------------------------------------
#  5. Candidate scoring & selection
# ---------------------------------------------------------------------------

class PlateSelector:
    def __init__(self, validator: PlateValidator):
        self.validator = validator

    def _score(self, conf: float, plate_type: str, length: int, bonus: float) -> float:
        score = conf
        score += 0.04 * length                           # longer matches are more informative
        score += 0.03 * FORMAT_PRIORITY.get(plate_type, 0)
        score += bonus                                   # disambiguator override etc.
        return score

    def select(self, candidates: list, db=None):
        """Pick the highest ranked valid candidate.

        Ranking is lexicographic on (db_match, region_class, -n_corr, score):
            1. plates registered in the DB always win
            2. KKTC layouts outrank TR layouts (this system runs in TRNC)
            3. fewer character corrections wins
            4. ties broken by score (confidence, length, format priority, bonus)
        """
        scored = []
        for cand in candidates:
            if len(cand) == 3:
                raw_text, conf, bonus = cand
            else:
                raw_text, conf = cand
                bonus = 0.0
            for corrected, ptype, n_corr in self.validator.validate(raw_text):
                in_db = bool(db and db.find(corrected))
                region_class = 1 if ptype in KKTC_TYPES else 0
                score = self._score(conf, ptype, len(corrected), bonus)
                rank = (1 if in_db else 0, region_class, -n_corr, score)
                scored.append((rank, corrected, ptype, conf))

        if not scored:
            if candidates:
                best_raw = max(candidates, key=lambda c: c[1])
                return best_raw[0], "UNKNOWN", best_raw[1]
            return "", "UNKNOWN", 0.0

        best_per_plate = {}
        for rank, plate, ptype, conf in scored:
            key = (plate, ptype)
            if key not in best_per_plate or rank > best_per_plate[key][0]:
                best_per_plate[key] = (rank, conf)

        winner = max(best_per_plate.items(), key=lambda kv: kv[1][0])
        (plate, ptype), (_rank, conf) = winner
        return plate, ptype, conf


# ---------------------------------------------------------------------------
#  6. Vehicle database (txt-backed)
# ---------------------------------------------------------------------------

class VehicleDatabase:
    def __init__(self, txt_path: str):
        self.txt_path = txt_path
        self.vehicles = {}
        self.reload()

    @staticmethod
    def _normalize(plate: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", plate.upper())

    def reload(self):
        self.vehicles = {}
        if not os.path.exists(self.txt_path):
            logger.warning(f"Vehicle database not found: {self.txt_path}")
            return
        try:
            with open(self.txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(",", 1)
                    if len(parts) != 2:
                        continue
                    plate = self._normalize(parts[0])
                    owner = parts[1].strip()
                    if plate:
                        self.vehicles[plate] = {"plate": plate, "owner_name": owner}
            logger.info(f"Vehicle database loaded: {len(self.vehicles)} records.")
        except Exception as e:
            logger.error(f"Vehicle DB read error: {e}")

    def register_vehicle(self, plate: str, owner_name: str):
        plate = self._normalize(plate)
        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write(f"{plate},{owner_name}\n")
        print(f"\n[SUCCESS] Plate '{plate}' registered to '{owner_name}'.")
        self.reload()

    def edit_vehicle(self, plate: str, new_owner_name: str):
        plate = self._normalize(plate)
        if plate not in self.vehicles:
            return False
        with open(self.txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open(self.txt_path, "w", encoding="utf-8") as f:
            for line in lines:
                parts = line.strip().split(",", 1)
                if len(parts) == 2 and self._normalize(parts[0]) == plate:
                    f.write(f"{plate},{new_owner_name}\n")
                else:
                    f.write(line)
        print(f"\n[SUCCESS] Plate '{plate}' updated to '{new_owner_name}'.")
        self.reload()
        return True

    def find(self, plate: str):
        return self.vehicles.get(self._normalize(plate))


# ---------------------------------------------------------------------------
#  7. Result logger
# ---------------------------------------------------------------------------

class ResultLogger:
    def __init__(self, txt_path: str):
        self.txt_path = txt_path

    def log(self, result: dict):
        line = (
            f"[{result['timestamp']}] "
            f"File: {result['file_name']} | "
            f"Plate: {result['read_plate']} | "
            f"Type: {result['plate_type']} | "
            f"Conf: {result['confidence_score']} | "
            f"Status: {result['match_status']} | "
            f"Owner: {result['owner']}\n"
        )
        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write(line)


# ---------------------------------------------------------------------------
#  8. Main orchestrator
# ---------------------------------------------------------------------------

class PlateRecognizer:
    def __init__(self, db: "VehicleDatabase" = None):
        self.preprocessor = ImagePreprocessor()
        self.ocr = PlateOCR()
        self.validator = PlateValidator()
        self.selector = PlateSelector(self.validator)
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
            image = self.preprocessor.load_image(image_path)
            variants = self.preprocessor.variants(image)
            detections = self.ocr.detect_all(variants)

            if not detections:
                result["match_status"] = "OCR_FAILED"
                self._print_result(result)
                self.result_logger.log(result)
                return result

            # Use the upscaled grayscale variant for the visual disambiguator
            # (its resolution matches the OCR bbox coordinates).
            gray_for_disambig = next((v for n, v in variants if n == "gray"), None)
            candidates = build_candidates(detections, gray_image=gray_for_disambig)
            plate, ptype, confidence = self.selector.select(candidates, db=self.db)

            result["read_plate"] = plate
            result["plate_type"] = ptype
            result["confidence_score"] = f"%{confidence * 100:.1f}"

            vehicle = self.db.find(plate) if plate else None
            if vehicle:
                result["match_status"] = "REGISTERED"
                result["owner"] = vehicle["owner_name"]
            elif ptype == "UNKNOWN":
                result["match_status"] = "OCR_FAILED"
            else:
                result["match_status"] = "UNREGISTERED"

        except Exception as e:
            result["match_status"] = "ERROR"
            logger.error(f"Processing error ({filename}): {e}")

        self._print_result(result)
        self.result_logger.log(result)
        return result

    def _print_result(self, result: dict):
        print()
        print(f"[{result['timestamp']}] Image processed: {result['file_name']}")
        print(f"  Plate           : {result['read_plate'] or '(unreadable)'}")
        print(f"  Plate Type      : {result['plate_type']}")
        print(f"  Confidence      : {result['confidence_score'] or '-'}")
        status = result["match_status"]
        if status == "REGISTERED":
            print(f"  Status          : REGISTERED")
            print(f"  Owner           : {result['owner']}")
        else:
            print(f"  Status          : {status}")
        print(f"  Saved to log file.")
        print("-" * 64)


# ---------------------------------------------------------------------------
#  9. Folder watcher & menu
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


def _move_to_processed(filepath: str):
    try:
        dest = PROCESSED_DIR / os.path.basename(filepath)
        original = dest
        counter = 1
        while dest.exists():
            dest = PROCESSED_DIR / f"{original.stem}_{counter}{original.suffix}"
            counter += 1
        shutil.move(filepath, str(dest))
    except Exception as e:
        logger.error(f"Could not move file: {e}")


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

        if choice == "1":
            start_scanning(db)
        elif choice == "2":
            plate = input("Enter Plate: ").strip()
            owner_name = input("Enter Name Surname: ").strip()
            if plate and owner_name:
                db.register_vehicle(plate, owner_name)
            else:
                print("[ERROR] Plate and Name cannot be empty.")
        elif choice == "3":
            plate = input("Enter Plate to edit: ").strip()
            record = db.find(plate)
            if record:
                print(f"Current Owner: {record['owner_name']}")
                new_owner = input("Enter New Name Surname: ").strip()
                if new_owner:
                    db.edit_vehicle(plate, new_owner)
                else:
                    print("[ERROR] Name cannot be empty.")
            else:
                print(f"\n[ERROR] Plate '{plate}' is not registered.")
        elif choice == "4":
            print("Exiting...")
            sys.exit(0)
        else:
            print("[ERROR] Invalid option. Please select 1-4.")


if __name__ == "__main__":
    main()
