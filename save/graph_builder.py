# graph_builder.py
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import osmnx as ox

from logger_config import logger
from paths import DIR_GRAPHS, ensure_data_dirs
from utils import safe_name
from cache_manager import locate_and_deploy_cached_file_debug
from osmnx_client import ensure_osmnx_settings

from types import SourceInfo


def _graph_path(city_name: str) -> str:
    safe = safe_name(city_name)
    return os.path.join(DIR_GRAPHS, f"{safe}.graphml")


def load_city_graph_from_cache_debug(
    city_name: str,
    required_tokens: Optional[List[str]] = None,
) -> Tuple[Optional[ox.graph.graph_from_place], SourceInfo]:
    """
    Пробує знайти й завантажити graphml з кешу.
    Повертає: (G або None, SourceInfo)
    """
    ensure_data_dirs()
    expected_path = _graph_path(city_name)

    located, cache_info = locate_and_deploy_cached_file_debug(
        expected_path=expected_path,
        suffix=".graphml",
        place=city_name,
        required_tokens=required_tokens,
    )

    info: SourceInfo = {
        "place": city_name,
        "expected_path": expected_path,
        "used_path": cache_info.get("used_path"),
        "candidates": cache_info.get("candidates", []),
        "cache_action": cache_info.get("action"),
        "source": "cache",
    }

    if not located or not os.path.exists(located):
        return None, info

    try:
        logger.info("Завантаження графа з кешу: %s", located)
        G = ox.load_graphml(located)
        return G, info
    except Exception as e:
        logger.warning("load_graphml failed for %s: %s", located, e)
        # файл є, але не читається → fallback на OSM
        return None, info


def get_city_graph_with_source(
    city_name: str,
    network_type: str = "walk",
    required_tokens: Optional[List[str]] = None,
    force_refresh: bool = False,
    simplify: bool = True,
) -> Tuple[ox.graph.graph_from_place, SourceInfo]:
    """
    Повертає (G, SourceInfo)
    """
    ensure_data_dirs()
    graph_path = _graph_path(city_name)

    # ---------- пробуємо кеш ----------
    if not force_refresh:
        G_cached, info = load_city_graph_from_cache_debug(
            city_name,
            required_tokens=required_tokens,
        )
        if G_cached is not None:
            return G_cached, info

    # ---------- тягнемо з OSM ----------
    ensure_osmnx_settings()
    logger.info(
        "Завантаження графа з OpenStreetMap: %s (network_type=%s)",
        city_name,
        network_type,
    )

    G = ox.graph_from_place(
        city_name,
        network_type=network_type,
        simplify=simplify,
    )

    try:
        ox.save_graphml(G, graph_path)
        logger.info("Граф збережено: %s", graph_path)
    except Exception as e:
        logger.warning("Не вдалося зберегти graphml %s: %s", graph_path, e)

    info: SourceInfo = {
        "place": city_name,
        "source": "osm",
        "expected_path": graph_path,
        "used_path": graph_path,
        "candidates": [],
    }

    return G, info


def get_city_graph(city_name: str, network_type: str = "walk"):
    """
    Сумісна стара функція: повертає тільки граф.
    """
    G, _ = get_city_graph_with_source(
        city_name,
        network_type=network_type,
    )
    return G
