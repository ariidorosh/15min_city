from __future__ import annotations
from typing import Dict, List, Mapping, Union

TagList = List[str]
FallbackBlock = Dict[str, TagList]                     # напр. {"building": ["school","university"]}
GroupSpec = Dict[str, Union[TagList, FallbackBlock]]   # напр. {"amenity":[...], "__fallback__": {...}}
LevelSpec = Dict[str, GroupSpec]                       # напр. {"education": {...}, "health": {...}}

# Повний список POI
POI_CATEGORIES: Dict[str, List[str]] = {
    "amenity": [
        # Healthcare
        "hospital", "clinic", "doctors", "dentist", "pharmacy", "nursing_home",
        # Education
        "school", "kindergarten", "college", "university", "library",
        # Food & drink
        "cafe", "restaurant", "fast_food", "bar", "pub", "ice_cream",
        # Transport
        "bus_station", "taxi", "bicycle_parking", "ferry_terminal", "parking", "car_sharing",
        # Government & public service
        "townhall", "courthouse", "police", "fire_station", "embassy", "post_office",
        "community_centre", "social_facility", "shelter", "marketplace",
        # Other
        "toilets", "bank", "atm", "place_of_worship", "recycling",
        # Large venues
        "theatre", "cinema"
    ],
    "shop": [
        # Food
        "supermarket", "convenience", "bakery", "butcher", "greengrocer", "confectionery",
        # Clothes & accessories
        "clothes", "shoes", "jewelry", "sports",
        # Household
        "furniture", "electronics", "hardware", "doityourself", "florist",
        # Transport
        "car", "bicycle", "motorcycle", "car_repair", "tyres",
        # Other
        "books", "stationery", "beauty", "hairdresser", "toys", "pet"
    ],
    "building": [
        "apartments", "house", "detached", "commercial", "industrial", "retail", "warehouse",
        "school", "hospital", "university", "hotel", "civic", "train_station",
        "church", "mosque", "temple", "sports_hall", "hut", "shed", "garage",
        "theatre", "museum", "arts_centre", "stadium"
    ],
    "tourism": [
        "hotel", "motel", "guest_house", "hostel", "museum", "gallery", "attraction",
        "theme_park", "viewpoint", "zoo", "aquarium", "camp_site", "picnic_site", "chalet"
    ],
    "leisure": [
        "park", "garden", "playground", "sports_centre", "stadium", "swimming_pool",
        "fitness_centre", "pitch", "golf_course", "track", "ice_rink", "dog_park", "beach_resort"
    ],
    "healthcare": [
        "hospital", "clinic", "doctors", "dentist", "pharmacy", "rehabilitation", "blood_donation"
    ],
    "office": [
        "company", "government", "ngo", "insurance", "lawyer", "real_estate", "telecommunication"
    ],
    "landuse": [
        "residential", "commercial", "industrial", "retail", "farmland", "forest",
        "cemetery", "recreation_ground", "grass", "allotments"
    ],
    "sport": [
        "soccer", "basketball", "tennis", "swimming", "athletics", "golf", "ski",
        "climbing", "volleyball", "table_tennis", "skateboard"
    ]
}

# MINIMUM — щоденні базові
LEVEL_MINIMUM: LevelSpec = {
    "education": {
        # ясла/садки/школи як справжні POI
        "amenity": ["childcare", "kindergarten", "school"],
        "__fallback__": {"building": ["school", "kindergarten"]},
    },

    "health": {
        # сучасна схема (часто без amenity=*)
        "healthcare": ["clinic", "doctor", "dentist", "pharmacy", "general_practice"],
        # дублюємо класичні amenity на випадок змішаного тегування
        "amenity": ["clinic", "doctors", "dentist", "pharmacy"],
        "__fallback__": {"building": ["clinic"]},
    },

    "culture": {
        # бібліотека як базовий доступ до знань
        "amenity": ["library"],
        "__fallback__": {"building": ["library"]},
    },

    "greens_sport": {
        # дворові/щоденні активності
        "leisure": ["park", "playground", "pitch"],
    },

    "shopping_services": {
        # «food» деталізовано: супермаркет, дрібна бакалія, хліб, овочі/фрукти
        "shop": ["supermarket", "convenience", "bakery", "greengrocer"],
        # базові сервіси
        "amenity": ["bank", "atm", "post_office", "marketplace"],
        "__fallback__": {"building": ["supermarket"]},
    },

    "transport": {
        # будь-яка PT-зупинка поблизу (10–15 хв пішки)
        "highway": ["bus_stop"],
        "public_transport": ["platform", "stop_position"],
        "railway": ["tram_stop"],
    },

    # Нові блоки для базової доступності
    "civic": {
        # безпека, адміністрація, базова інфраструктура комфорту
        "amenity": ["police", "fire_station", "townhall", "courthouse", "embassy", "toilets", "recycling", "shelter"],
        "__fallback__": {"building": ["civic"]},
    },

    "food": {
        # мінімальний набір місць харчування/кафе
        "amenity": ["cafe", "fast_food"],
        "__fallback__": {"building": ["retail"]},
    },
}

# DELTA_MEDIUM
DELTA_MEDIUM: LevelSpec = {
    "education": {
        # середня/вища освіта як «не щоденна, але близька» опція
        "amenity": ["college", "university"],
        "__fallback__": {"building": ["college", "university"]},
    },

    "culture": {
        # спільнотні простори та культові споруди
        "amenity": ["community_centre", "place_of_worship"],
        "tourism": ["gallery"],
    },

    "greens_sport": {
        # розважальні/організовані активності поруч
        "leisure": ["garden", "swimming_pool", "sports_centre"],
        "__fallback__": {"building": ["sports_hall"]},
    },

    "shopping_services": {
        # щоденні «дрібні» спеціалізації + заклади харчування середнього рівня
        "shop": ["butcher", "confectionery"],
        "amenity": ["restaurant", "bar", "pub", "ice_cream"],
        "__fallback__": {"building": ["retail"]},
    },

    "transport": {
        # розширення транспорту (велопаркування / зупинки більших типів)
        "amenity": ["bicycle_parking", "bus_station"],
        "railway": ["tram_stop"],
        "public_transport": ["platform"],
    },

    "work_services": {
        # офіси/робочі сервіси — для аналізу доступності праці/послуг
        "office": ["company", "government", "ngo", "insurance", "lawyer"],
        "__fallback__": {"building": ["commercial"]},
    },
}

# DELTA_MAXIMUM — лише додаткові теги (видалені дублікати вручну)
DELTA_MAXIMUM: LevelSpec = {
    "education": {
        "amenity": [],
        "__fallback__": {},
    },

    "health": {
        # стаціонар/довготривале перебування + ширші медпослуги
        "healthcare": ["hospital", "rehabilitation", "blood_donation"],
        "amenity": ["hospital", "nursing_home"],
        "__fallback__": {"building": ["hospital"]},
    },

    "culture": {
        # великі культурні локації
        "amenity": ["theatre", "cinema"],
        "tourism": ["museum", "art_gallery", "attraction"],
        "__fallback__": {"building": ["theatre", "cinema", "museum", "arts_centre"]},
    },

    "greens_sport": {
        # великі об’єкти спорту/дозвілля
        "leisure": ["stadium", "fitness_centre", "golf_course", "track", "ice_rink", "dog_park"],
        "__fallback__": {"building": ["stadium"]},
    },

    "shopping_services": {
        # широка номенклатура непродовольчих покупок + mobility/repair
        "shop": [
            "clothes", "shoes", "electronics", "hardware", "doityourself",
            "sports", "books", "stationery", "toys", "pet", "car_repair", "tyres", "motorcycle"
        ],
        "__fallback__": {"building": ["retail"]},
    },

    "transport": {
        # повна мультимодальна картинка
        "amenity": ["taxi", "parking", "car_sharing", "bicycle_rental", "charging_station"],
        "public_transport": ["platform"],
        "railway": ["tram_stop"],
        "__fallback__": {"building": ["parking", "train_station"]},
    },

    "tourism": {
        "tourism": ["hotel", "guest_house", "hostel", "motel"],
        "__fallback__": {"building": ["hotel"]},
    },
}


# Хелпери мерджу/DIFF/валідації
def _union_preserve_order(base: List[str], add: List[str]) -> List[str]:
    seen = set(base)
    out = list(base)
    for x in add:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def deep_merge_levels(base: Mapping[str, Mapping[str, List[str]]],
                      delta: Mapping[str, Mapping[str, List[str]]]
                      ) -> Dict[str, Dict[str, List[str]]]:
    merged: Dict[str, Dict[str, List[str]]] = {cat: {k: list(v) for k, v in sub.items()}
                                               for cat, sub in base.items()}
    for cat, sub in delta.items():
        if cat not in merged:
            merged[cat] = {}
        for osm_key, tags in sub.items():
            if osm_key not in merged[cat]:
                merged[cat][osm_key] = []
            merged[cat][osm_key] = _union_preserve_order(merged[cat][osm_key], list(tags))
    return merged


LEVEL_MEDIUM = deep_merge_levels(LEVEL_MINIMUM, DELTA_MEDIUM)
LEVEL_MAXIMUM = deep_merge_levels(LEVEL_MEDIUM, DELTA_MAXIMUM)

LEVELS_QUERIES: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "minimum": LEVEL_MINIMUM,
    "medium": LEVEL_MEDIUM,
    "maximum": LEVEL_MAXIMUM,
}
