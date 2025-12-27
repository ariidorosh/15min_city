# paths.py
from __future__ import annotations

import os
from typing import List

# Єдина точка правди для всіх шляхів даних
DATA_ROOT: str = os.getenv("DATA_ROOT", "data")

DIR_POI_ALL: str = os.path.join(DATA_ROOT, "poi", "all")
DIR_POI_LEVELS: str = os.path.join(DATA_ROOT, "poi", "levels")
DIR_BOUNDARIES: str = os.path.join(DATA_ROOT, "boundaries")
DIR_GRAPHS: str = os.path.join(DATA_ROOT, "graphs")
DIR_CACHE: str = os.path.join(DATA_ROOT, "cache")

# Службові директорії всередині data/
DIR_META: str = os.path.join(DATA_ROOT, "meta")

# Не в data/, бо це швидше “артефакти” для UI/експорту
DIR_OUTPUTS: str = os.getenv("OUTPUTS_DIR", "outputs")

# Додаткові місця, де може лежати кеш (Colab/перенос проєкту)
EXTRA_SEARCH_DIRS: List[str] = [os.getcwd(), "/mnt/data"]


def ensure_data_dirs() -> None:
    """Гарантує, що всі потрібні директорії існують."""
    for d in (
        DIR_POI_ALL,
        DIR_POI_LEVELS,
        DIR_BOUNDARIES,
        DIR_GRAPHS,
        DIR_CACHE,
        DIR_META,
        DIR_OUTPUTS,
    ):
        os.makedirs(d, exist_ok=True)