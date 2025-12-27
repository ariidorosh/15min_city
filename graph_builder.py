# graph_builder.py
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import networkx as nx
import osmnx as ox

from cache_manager import locate_and_deploy_cached_file_debug
from logger_config import logger
from osmnx_client import ensure_osmnx_settings
from paths import DIR_GRAPHS, ensure_data_dirs
from utils import safe_name


def _graph_path(city_name: str) -> str:
    safe = safe_name(city_name)
    return os.path.join(DIR_GRAPHS, f"{safe}.graphml")


def load_city_graph_from_cache_debug(
    city_name: str,
    *,
    required_tokens: Optional[List[str]] = None,
) -> Tuple[Optional[nx.MultiDiGraph], Dict[str, object]]:
    """
    Повертає: (G або None, info)

    info ключі (стабільно):
      - place
      - source: "cache"
      - cache_action: "exists" | "copied" | "fallback_read" | "not_found"
      - expected_path
      - used_path
      - candidates
    """
    ensure_data_dirs()
    expected_path = _graph_path(city_name)

    located, cinfo = locate_and_deploy_cached_file_debug(
        expected_path=expected_path,
        suffix=".graphml",
        place=city_name,
        required_tokens=required_tokens,
    )

    cache_action = cinfo.get("action") or "not_found"
    used_path = cinfo.get("used_path") or located

    info: Dict[str, object] = {
        "place": city_name,
        "source": "cache",
        "cache_action": cache_action,
        "expected_path": cinfo.get("expected_path", expected_path),
        "used_path": used_path,
        "candidates": cinfo.get("candidates", []),
    }

    if not used_path or not os.path.exists(used_path):
        return None, info

    try:
        logger.info("Завантаження графа з кешу: %s", used_path)
        G = ox.load_graphml(used_path)
        return G, info
    except Exception as e:
        logger.warning("load_graphml failed for %s: %s", used_path, e)
        # Файл існує, але битий/нечитабельний -> UI потім піде в OSM
        return None, info


def get_city_graph_with_source(
    city_name: str,
    *,
    network_type: str = "walk",
    required_tokens: Optional[List[str]] = None,
    force_refresh: bool = False,
    simplify: bool = True,
) -> Tuple[nx.MultiDiGraph, Dict[str, object]]:
    """
    Повертає (G, info).

    info ключі (основні):
      - place
      - source: "cache" | "osm"
      - cache_action (якщо була спроба кешу)
      - expected_path
      - used_path
      - candidates
      - saved: bool (чи вдалося записати graphml при source="osm")
      - force_refresh
    """
    ensure_data_dirs()
    expected = _graph_path(city_name)

    cache_info: Dict[str, object] = {}
    if not force_refresh:
        G_cached, cache_info = load_city_graph_from_cache_debug(city_name, required_tokens=required_tokens)
        if G_cached is not None:
            return G_cached, cache_info

    ensure_osmnx_settings()
    logger.info("Завантаження графа з OpenStreetMap: %s (network_type=%s)", city_name, network_type)

    G = ox.graph_from_place(city_name, network_type=network_type, simplify=simplify)

    saved = False
    used_path: Optional[str] = None
    try:
        ox.save_graphml(G, expected)
        saved = True
        used_path = expected
        logger.info("Граф збережено: %s", expected)
    except Exception as e:
        logger.warning("Не вдалося зберегти graphml %s: %s", expected, e)

    info: Dict[str, object] = {
        "place": city_name,
        "source": "osm",
        "expected_path": expected,
        "used_path": used_path,
        "candidates": cache_info.get("candidates", []),
        "cache_action": cache_info.get("cache_action") or cache_info.get("action") or "not_found",
        "saved": saved,
        "force_refresh": force_refresh,
    }

    return G, info


def get_city_graph(city_name: str, network_type: str = "walk") -> nx.MultiDiGraph:
    """Легасі: повертає тільки граф."""
    G, _ = get_city_graph_with_source(city_name=city_name, network_type=network_type)
    return G
