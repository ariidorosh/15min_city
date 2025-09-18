import os
import glob
import shutil
import re
from typing import Optional, Tuple, Dict, List

import osmnx as ox
from logger_config import logger

GRAPH_DIR = os.path.join("data", "graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)


def _safe_name(city_name: str) -> str:
    return city_name.lower().replace(",", "").replace(" ", "_")


def _norm_name(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-zа-яіїєґ0-9]+", "_", s)
    return s.strip("_")


def _locate_graph_file_debug(expected_path: str, place: str, required_tokens: Optional[List[str]] = None) -> Tuple[Optional[str], Dict]:
    info: Dict[str, object] = {
        "expected_path": expected_path,
        "used_path": None,
        "action": "not_found",
        "candidates": []
    }
    if os.path.exists(expected_path):
        info["used_path"] = expected_path
        info["action"] = "exists"
        return expected_path, info

    candidates = glob.glob(os.path.join(GRAPH_DIR, "*.graphml")) + \
                 glob.glob(os.path.join(os.getcwd(), "*.graphml")) + \
                 glob.glob("/mnt/data/*.graphml")
    place_tokens = re.findall(r"[\w\u0400-\u04FF]+", place.lower())
    required = [t for t in (required_tokens or []) if t]

    scored: List[Tuple[int, str]] = []
    for p in candidates:
        fname = _norm_name(os.path.basename(p))
        score = sum(1 for t in place_tokens if t and t in fname)
        # --- строгий фільтр: усі required-токени мають бути присутні
        if required and not all(t in fname for t in required):
            continue
        scored.append((score, p))

    scored.sort(key=lambda x: (-x[0], x[1]))
    info["candidates"] = [p for _, p in scored]

    if not scored or scored[0][0] <= 0:
        return None, info

    best = scored[0][1]
    try:
        os.makedirs(os.path.dirname(expected_path), exist_ok=True)
        shutil.copy2(best, expected_path)
        logger.info("Deployed graph cache %s -> %s", best, expected_path)
        info["used_path"] = expected_path
        info["action"] = "copied"
        return expected_path, info
    except Exception as e:
        logger.warning("Failed to copy graph %s -> %s: %s", best, expected_path, e)
        info["used_path"] = best
        info["action"] = "fallback_read"
        return best, info


def get_city_graph_with_source(city_name: str, network_type: str = "walk", required_tokens: Optional[List[str]] = None):
    safe = _safe_name(city_name)
    graph_path = os.path.join(GRAPH_DIR, f"{safe}.graphml")

    located, info = _locate_graph_file_debug(graph_path, city_name, required_tokens=required_tokens)
    if located and os.path.exists(located):
        try:
            logger.info("Завантаження графа з %s", located)
            G = ox.load_graphml(located)
            return G, info
        except Exception as e:
            logger.warning("load_graphml failed for %s: %s", located, e)

    # кеш не спрацював — тягнемо з OSM
    logger.info("Завантаження мапи міста з OpenStreetMap: %s", city_name)
    G = ox.graph_from_place(city_name, network_type=network_type, simplify=True)
    try:
        ox.save_graphml(G, graph_path)
        logger.info("Граф збережено: %s", graph_path)
    except Exception as e:
        logger.warning("Не вдалося зберегти graphml %s: %s", graph_path, e)

    info.update({"used_path": graph_path, "action": "downloaded"})
    return G, info


# Сумісна стара функція (без строгих токенів)
def get_city_graph(city_name: str, network_type: str = "walk"):
    safe = _safe_name(city_name)
    graph_path = os.path.join(GRAPH_DIR, f"{safe}.graphml")
    located = graph_path if os.path.exists(graph_path) else None
    if located:
        try:
            logger.info("Завантаження графа з %s", located)
            return ox.load_graphml(located)
        except Exception:
            pass
    logger.info("Завантаження мапи міста з OpenStreetMap: %s", city_name)
    G = ox.graph_from_place(city_name, network_type=network_type, simplify=True)
    try:
        ox.save_graphml(G, graph_path)
        logger.info("Граф збережено: %s", graph_path)
    except Exception:
        pass
    return G
