# app_types.py
from typing import List, Literal, Tuple, TypedDict

LatLon = Tuple[float, float]
Algorithm = Literal["dijkstra", "astar"]

SourceType = Literal["cache", "osm"]
CacheAction = Literal["exists", "copied", "fallback_read", "not_found"]


class SourceInfo(TypedDict, total=False):
    place: str
    source: SourceType
    cache_action: CacheAction
    expected_path: str
    used_path: str
    candidates: List[str]
