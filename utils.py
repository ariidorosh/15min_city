# utils.py
from __future__ import annotations

import hashlib
import json
import os
import re
import math
from typing import Dict, List, Tuple

_RE_TOKENS = re.compile(r"[\w\u0400-\u04FF]+", flags=re.UNICODE)
_RE_NORM_FILENAME = re.compile(r"[^0-9a-zA-Z\u0400-\u04FF]+", flags=re.UNICODE)
LatLon = Tuple[float, float]  # (lat, lon)


def ensure_dir(path: str) -> None:
    """Створює директорію, якщо її ще нема."""
    if path:
        os.makedirs(path, exist_ok=True)


def tokens(text: str) -> List[str]:
    """Токени для пошуку/матчінгу (працює і з кирилицею)."""
    return _RE_TOKENS.findall((text or "").lower())


def norm_name(filename: str) -> str:
    """
    Нормалізація назви файла для матчінгу:
    все, що не літера/цифра -> пробіл, потім стиснути пробіли.
    """
    s = (filename or "").lower()
    s = _RE_NORM_FILENAME.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_name(s: str) -> str:
    """
    Безпечне ім'я для файлів (legacy-friendly):
    - lower
    - коми -> пробіли
    - пробіли/дефіси -> _
    - прибирає все, крім літер/цифр/_
    """
    s = (s or "").strip().lower()
    s = s.replace(",", " ")
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^0-9a-zA-Z_\u0400-\u04FF]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"


def tags_signature(tags: Dict[str, List[str]]) -> str:
    """Коротка сигнатура від tags (на майбутнє для “кешу по тегах”)."""
    norm = {
        k: sorted({str(v).strip() for v in vals if v is not None and str(v).strip()})
        for k, vals in tags.items()
    }
    payload = json.dumps(dict(sorted(norm.items())), ensure_ascii=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]


def haversine_m(a: LatLon, b: LatLon) -> float:
    """Приблизна відстань по сфері між двома (lat, lon) у метрах."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    s = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(min(1.0, math.sqrt(s)))
    return 6371000.0 * c


def _norm_cache_token(s: str) -> str:
    """
    Нормалізація під кеш/матчінг файлів:
    усе зайве -> '_', потім стиснути '_'.
    """
    s = (s or "").lower()
    s = re.sub(r"[^a-zа-яіїєґ0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def extract_required_tokens(full_place: str) -> List[str]:
    """
    Витягує “обовʼязкові” токени з повного рядка місця (як його повертає Nominatim),
    щоб точніше матчити кеш (наприклад, відрізняти області).

    Приклад: "Lviv, Lviv Oblast, Ukraine" -> ["lviv", "lviv_oblast"]
    """
    parts = [p.strip() for p in (full_place or "").split(",") if p.strip()]
    city_tok = _norm_cache_token(parts[0]) if parts else ""
    region_tok = ""

    for p in parts[1:]:
        if re.search(r"область|oblast|voivodeship|місто київ|city of kyiv", p, flags=re.I):
            region_tok = _norm_cache_token(p)
            break

    return [t for t in (city_tok, region_tok) if t]


def parse_latlon_text(text: str, name: str) -> LatLon:
    """
    Приймає "lat, lon" або "lat lon".
    Повертає (lat, lon) як float.
    """
    s = (text or "").strip()
    if not s:
        raise ValueError(f"{name}: порожнє значення")

    parts = re.split(r"[,\s]+", s)
    parts = [p for p in parts if p]
    if len(parts) != 2:
        raise ValueError(f"{name}: формат має бути 'lat, lon' (наприклад 49.8397, 24.0297)")

    lat = float(parts[0])
    lon = float(parts[1])

    if math.isnan(lat) or math.isnan(lon):
        raise ValueError(f"{name}: координати не можуть бути NaN")

    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        raise ValueError(f"{name}: координати поза діапазоном (-90..90, -180..180)")

    return (lat, lon)
