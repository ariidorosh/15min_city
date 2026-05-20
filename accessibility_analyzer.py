from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import geopandas as gpd
from shapely.geometry import MultiPolygon, Point, Polygon, box

from app_types import LatLon
from config import LEVELS_QUERIES
from logger_config import logger
from path_finder import IsochroneResult, compute_isochrone


AccessibilityStatus = Literal["poor", "medium", "good"]


DEFAULT_GROUP_WEIGHTS: Dict[str, float] = {
    "health": 1.40,
    "education": 1.20,
    "shopping_services": 1.20,
    "transport": 1.10,
    "greens_sport": 1.00,
    "civic": 0.95,
    "food": 0.90,
    "culture": 0.75,
    "work_services": 0.70,
    "tourism": 0.50,
}

DEFAULT_STATUS_THRESHOLDS: Dict[str, float] = {
    "good": 82.0,
    "medium": 60.0,
}

DEFAULT_LABEL_COLUMNS: Tuple[str, ...] = (
    "name",
    "brand",
    "operator",
    "poi_value",
    "amenity",
    "shop",
    "leisure",
    "tourism",
    "healthcare",
    "office",
    "sport",
    "building",
)

CRITICAL_GROUPS = {
    "health",
    "education",
    "shopping_services",
    "transport",
}

COVERAGE_RATIO_EXPONENT = 1.25
CRITICAL_GROUP_MISS_PENALTY = 0.10
MIN_CRITICAL_PENALTY = 0.72

DENSITY_PENALTY_MIN = 0.88
DENSITY_PENALTY_MAX = 1.00


@dataclass(frozen=True)
class GroupAccessibilityResult:
    group_name: str
    weight: float
    present: bool
    found_count: int
    matched_tags: List[str]
    sample_labels: List[str]
    score_contribution: float


@dataclass(frozen=True)
class AccessibilityEvaluation:
    center_latlon: LatLon
    level: str
    minutes: float
    cutoff_m: float
    walk_speed_kmh: float
    total_groups: int
    covered_groups: int
    missing_groups: List[str]
    inside_poi_count: int
    coverage_ratio: float
    score_100: float
    status: AccessibilityStatus
    group_results: List[GroupAccessibilityResult]
    isochrone: IsochroneResult


@dataclass(frozen=True)
class AccessibilityGridCell:
    center_latlon: LatLon
    geometry: Polygon
    score_100: float
    status: AccessibilityStatus
    covered_groups: int
    total_groups: int
    missing_groups: List[str]
    inside_poi_count: int
    level: str
    minutes: float


@dataclass(frozen=True)
class AccessibilityGridResult:
    level: str
    minutes: float
    walk_speed_kmh: float
    step_m: float
    total_cells: int
    successful_cells: int
    failed_cells: int
    cells: List[AccessibilityGridCell]
    area_bounds_latlon: Tuple[float, float, float, float]  # (min_lat, min_lon, max_lat, max_lon)
    score_min: Optional[float]
    score_max: Optional[float]
    score_mean: Optional[float]


class AccessibilityAnalyzerError(Exception):
    pass


# ============================================================
# Public API
# ============================================================
def evaluate_accessibility(
    G,
    gdf_all_poi: gpd.GeoDataFrame,
    *,
    center: LatLon,
    level: str = "medium",
    minutes: float = 15.0,
    walk_speed_kmh: float = 4.8,
    weight: str = "length",
    group_weights: Optional[Mapping[str, float]] = None,
    status_thresholds: Optional[Mapping[str, float]] = None,
    label_columns: Sequence[str] = DEFAULT_LABEL_COLUMNS,
    sample_labels_per_group: int = 5,
) -> AccessibilityEvaluation:
    """
    Рахує доступність для однієї точки:
      1) будує ізохрону;
      2) бере POI всередині;
      3) перевіряє покриття груп для рівня minimum/medium/maximum;
      4) повертає score 0..100, missing groups і деталізацію по групах.
    """
    if gdf_all_poi is None:
        raise AccessibilityAnalyzerError("gdf_all_poi=None — немає POI для аналізу")
    if getattr(gdf_all_poi, "empty", True):
        raise AccessibilityAnalyzerError("POI порожні — немає що аналізувати")
    if "geometry" not in gdf_all_poi.columns:
        raise AccessibilityAnalyzerError("У GeoDataFrame немає колонки geometry")

    level_norm = _normalize_level(level)
    minutes = float(minutes)
    walk_speed_kmh = float(walk_speed_kmh)
    if minutes <= 0:
        raise AccessibilityAnalyzerError("minutes має бути > 0")
    if walk_speed_kmh <= 0:
        raise AccessibilityAnalyzerError("walk_speed_kmh має бути > 0")

    cutoff_m = _minutes_to_cutoff_m(minutes, walk_speed_kmh)
    query_spec = LEVELS_QUERIES.get(level_norm)
    if not query_spec:
        raise AccessibilityAnalyzerError(f"Невідомий рівень доступності: {level}")

    iso = compute_isochrone(
        G,
        center_latlon=center,
        cutoff=cutoff_m,
        weight=weight,
        use_undirected=True,
        snap_mode="edge",
        build_polygon=True,
        polygon_buffer_m=15.0,
    )

    inside_poi = _filter_poi_inside_isochrone(
        gdf_all_poi,
        polygon=iso.polygon,
        center=center,
        cutoff_m=cutoff_m,
    )

    base_weights = dict(DEFAULT_GROUP_WEIGHTS)
    if group_weights:
        base_weights.update({str(k): float(v) for k, v in group_weights.items()})

    thresholds = dict(DEFAULT_STATUS_THRESHOLDS)
    if status_thresholds:
        thresholds.update({str(k): float(v) for k, v in status_thresholds.items()})

    group_results: List[GroupAccessibilityResult] = []
    total_weight = 0.0
    covered_weight = 0.0

    for group_name, raw_group_spec in query_spec.items():
        key_to_tags = _flatten_group_spec(raw_group_spec)
        group_matches = _match_group_poi(inside_poi, key_to_tags)

        weight_value = float(base_weights.get(group_name, 1.0))
        present = not group_matches.empty
        found_count = int(len(group_matches))
        matched_tags = _extract_matched_tags(group_matches)
        sample_labels = _extract_sample_labels(
            group_matches,
            label_columns=label_columns,
            limit=sample_labels_per_group,
        )
        contribution = weight_value if present else 0.0

        total_weight += weight_value
        covered_weight += contribution

        group_results.append(
            GroupAccessibilityResult(
                group_name=str(group_name),
                weight=weight_value,
                present=present,
                found_count=found_count,
                matched_tags=matched_tags,
                sample_labels=sample_labels,
                score_contribution=float(contribution),
            )
        )

    total_groups = len(group_results)
    covered_groups = sum(1 for g in group_results if g.present)
    missing_groups = [g.group_name for g in group_results if not g.present]

    coverage_ratio = (covered_groups / total_groups) if total_groups else 0.0
    base_score_100 = (covered_weight / total_weight * 100.0) if total_weight > 0 else 0.0

    missing_critical_count = sum(
        1
        for g in group_results
        if (not g.present) and (g.group_name in CRITICAL_GROUPS)
    )

    coverage_penalty = coverage_ratio ** COVERAGE_RATIO_EXPONENT

    critical_penalty = 1.0 - (missing_critical_count * CRITICAL_GROUP_MISS_PENALTY)
    critical_penalty = max(MIN_CRITICAL_PENALTY, critical_penalty)

    density_target = max(1, total_groups)
    density_ratio = min(1.0, float(len(inside_poi)) / float(density_target))
    density_penalty = DENSITY_PENALTY_MIN + (DENSITY_PENALTY_MAX - DENSITY_PENALTY_MIN) * density_ratio

    score_100 = base_score_100 * coverage_penalty * critical_penalty * density_penalty
    score_100 = max(0.0, min(100.0, float(score_100)))

    status = _score_to_status(score_100, thresholds)

    logger.info(
        "Accessibility evaluated: level=%s center=%s minutes=%.1f inside_poi=%d covered=%d/%d base=%.1f final=%.1f status=%s",
        level_norm,
        center,
        minutes,
        len(inside_poi),
        covered_groups,
        total_groups,
        base_score_100,
        score_100,
        status,
    )

    return AccessibilityEvaluation(
        center_latlon=(float(center[0]), float(center[1])),
        level=level_norm,
        minutes=float(minutes),
        cutoff_m=float(cutoff_m),
        walk_speed_kmh=float(walk_speed_kmh),
        total_groups=int(total_groups),
        covered_groups=int(covered_groups),
        missing_groups=[str(x) for x in missing_groups],
        inside_poi_count=int(len(inside_poi)),
        coverage_ratio=float(coverage_ratio),
        score_100=float(score_100),
        status=status,
        group_results=group_results,
        isochrone=iso,
    )


def evaluate_accessibility_grid(
    G,
    gdf_all_poi: gpd.GeoDataFrame,
    *,
    level: str = "medium",
    minutes: float = 15.0,
    walk_speed_kmh: float = 4.8,
    step_m: float = 600.0,
    max_cells: int = 160,
    area_geometry: Optional[Union[gpd.GeoDataFrame, Polygon, MultiPolygon]] = None,
    weight: str = "length",
) -> AccessibilityGridResult:
    """
    Будує міську карту доступності:
      1) беремо полігон аналізу (boundary або fallback із графа),
      2) розбиваємо на сітку,
      3) для центру кожної клітинки рахуємо evaluate_accessibility(...),
      4) повертаємо набір клітинок зі score/status.
    """
    level_norm = _normalize_level(level)

    if step_m <= 0:
        raise AccessibilityAnalyzerError("step_m має бути > 0")
    if max_cells <= 0:
        raise AccessibilityAnalyzerError("max_cells має бути > 0")

    analysis_poly_wgs = _resolve_analysis_polygon(G, area_geometry)
    analysis_poly_proj = gpd.GeoSeries([analysis_poly_wgs], crs=4326).to_crs(epsg=3857).iloc[0]

    grid_cells_proj = _generate_grid_cells(analysis_poly_proj, step_m=float(step_m))
    if not grid_cells_proj:
        raise AccessibilityAnalyzerError("Не вдалося згенерувати жодної клітинки для аналізу")

    if len(grid_cells_proj) > int(max_cells):
        grid_cells_proj = _sample_grid_cells(grid_cells_proj, int(max_cells))

    cells: List[AccessibilityGridCell] = []
    failed_cells = 0

    for idx, cell_proj in enumerate(grid_cells_proj, start=1):
        try:
            cell_wgs = gpd.GeoSeries([cell_proj], crs=3857).to_crs(epsg=4326).iloc[0]
            center_proj = cell_proj.centroid
            center_wgs = gpd.GeoSeries([center_proj], crs=3857).to_crs(epsg=4326).iloc[0]
            center = (float(center_wgs.y), float(center_wgs.x))

            ev = evaluate_accessibility(
                G,
                gdf_all_poi,
                center=center,
                level=level_norm,
                minutes=float(minutes),
                walk_speed_kmh=float(walk_speed_kmh),
                weight=weight,
            )

            cells.append(
                AccessibilityGridCell(
                    center_latlon=center,
                    geometry=cell_wgs,
                    score_100=float(ev.score_100),
                    status=ev.status,
                    covered_groups=int(ev.covered_groups),
                    total_groups=int(ev.total_groups),
                    missing_groups=list(ev.missing_groups),
                    inside_poi_count=int(ev.inside_poi_count),
                    level=level_norm,
                    minutes=float(minutes),
                )
            )
        except Exception as e:
            failed_cells += 1
            logger.warning("Accessibility grid cell failed (%d/%d): %s", idx, len(grid_cells_proj), e)

    all_scores = [c.score_100 for c in cells]

    informative_cells = [
        c for c in cells
        if (c.inside_poi_count > 0 or c.covered_groups > 0)
    ]

    summary_cells = informative_cells if informative_cells else cells
    scores = [c.score_100 for c in summary_cells]

    result = AccessibilityGridResult(
        level=level_norm,
        minutes=float(minutes),
        walk_speed_kmh=float(walk_speed_kmh),
        step_m=float(step_m),
        total_cells=int(len(grid_cells_proj)),
        successful_cells=int(len(cells)),
        failed_cells=int(failed_cells),
        cells=cells,
        area_bounds_latlon=_polygon_bounds_latlon(analysis_poly_wgs),
        score_min=min(all_scores) if all_scores else None,
        score_max=max(all_scores) if all_scores else None,
        score_mean=(sum(scores) / len(scores)) if scores else None,
    )

    logger.info(
        "Accessibility grid built: level=%s minutes=%.1f step=%.0fm cells=%d ok=%d failed=%d informative=%d mean=%s",
        result.level,
        result.minutes,
        result.step_m,
        result.total_cells,
        result.successful_cells,
        result.failed_cells,
        len(informative_cells),
        f"{result.score_mean:.1f}" if result.score_mean is not None else "—",
    )
    return result


# ============================================================
# Helpers
# ============================================================
def _normalize_level(level: str) -> str:
    x = (level or "medium").strip().lower()
    if x.startswith("min"):
        return "minimum"
    if x.startswith("max"):
        return "maximum"
    return "medium"


def _minutes_to_cutoff_m(minutes: float, walk_speed_kmh: float) -> float:
    return float(walk_speed_kmh) * 1000.0 * (float(minutes) / 60.0)


def _flatten_group_spec(raw_group_spec: Mapping[str, object]) -> Dict[str, List[str]]:
    """
    Перетворює group spec у просту форму {osm_key: [tag1, tag2, ...]}.

    Підтримує і нормальний __fallback__={"building": [...]} і випадок,
    коли __fallback__ уже зіпсований merge-ом (наприклад ['building']).
    У другому випадку просто ігноруємо такий fallback, бо теги втрачені.
    """
    out: Dict[str, List[str]] = {}

    for osm_key, raw_value in dict(raw_group_spec).items():
        if osm_key == "__fallback__":
            if isinstance(raw_value, Mapping):
                for fb_key, fb_tags in raw_value.items():
                    tags_list = _to_clean_str_list(fb_tags)
                    if tags_list:
                        out[fb_key] = _union_preserve_order(out.get(fb_key, []), tags_list)
            continue

        tags_list = _to_clean_str_list(raw_value)
        if tags_list:
            out[osm_key] = _union_preserve_order(out.get(osm_key, []), tags_list)

    return out


def _to_clean_str_list(value: object) -> List[str]:
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, Iterable):
        out: List[str] = []
        for x in value:
            if isinstance(x, str):
                s = x.strip()
                if s:
                    out.append(s)
        return out
    return []


def _union_preserve_order(base: List[str], add: List[str]) -> List[str]:
    seen = set(base)
    out = list(base)
    for x in add:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _filter_poi_inside_isochrone(
    gdf: gpd.GeoDataFrame,
    *,
    polygon,
    center: LatLon,
    cutoff_m: float,
) -> gpd.GeoDataFrame:
    if gdf is None or getattr(gdf, "empty", True):
        return gdf.iloc[0:0].copy()

    work = gdf.copy()

    if polygon is not None and not getattr(polygon, "is_empty", True):
        mask = work["geometry"].apply(lambda geom: _geom_inside_polygon(geom, polygon))
        return work[mask].copy()

    mask = work["geometry"].apply(lambda geom: _geom_within_radius(geom, center, cutoff_m))
    return work[mask].copy()


def _geom_inside_polygon(geom, polygon) -> bool:
    if geom is None:
        return False
    try:
        if geom.is_empty:
            return False
    except Exception:
        return False

    try:
        probe = geom if isinstance(geom, Point) else geom.representative_point()
        return bool(polygon.covers(probe))
    except Exception:
        return False


def _geom_within_radius(geom, center: LatLon, cutoff_m: float) -> bool:
    if geom is None:
        return False
    try:
        if geom.is_empty:
            return False
    except Exception:
        return False

    try:
        probe = geom if isinstance(geom, Point) else geom.representative_point()
        latlon = (float(probe.y), float(probe.x))
        return _haversine_m(center, latlon) <= float(cutoff_m)
    except Exception:
        return False


def _match_group_poi(gdf: gpd.GeoDataFrame, key_to_tags: Mapping[str, Sequence[str]]) -> gpd.GeoDataFrame:
    if gdf is None or getattr(gdf, "empty", True):
        return gdf.iloc[0:0].copy()
    if not key_to_tags:
        return gdf.iloc[0:0].copy()

    mask = None

    # Формат пайплайна: poi_key / poi_value
    if ("poi_key" in gdf.columns) and ("poi_value" in gdf.columns):
        for osm_key, tags in key_to_tags.items():
            if not tags:
                continue
            part = (
                gdf["poi_key"].astype(str).str.lower().eq(str(osm_key).lower())
                & gdf["poi_value"].astype(str).str.lower().isin([str(t).lower() for t in tags])
            )
            mask = part if mask is None else (mask | part)
        if mask is not None:
            return gdf[mask].copy()

    # Wide-format fallback: amenity/shop/leisure/... як окремі колонки
    for osm_key, tags in key_to_tags.items():
        if osm_key not in gdf.columns or not tags:
            continue
        part = gdf[osm_key].astype(str).str.lower().isin([str(t).lower() for t in tags])
        mask = part if mask is None else (mask | part)

    if mask is None:
        return gdf.iloc[0:0].copy()
    return gdf[mask].copy()


def _extract_matched_tags(gdf: gpd.GeoDataFrame, limit: int = 8) -> List[str]:
    if gdf is None or getattr(gdf, "empty", True):
        return []

    seen = set()
    out: List[str] = []

    if ("poi_key" in gdf.columns) and ("poi_value" in gdf.columns):
        for _, row in gdf.iterrows():
            key = str(row.get("poi_key") or "").strip()
            val = str(row.get("poi_value") or "").strip()
            if not key or not val:
                continue
            item = f"{key}={val}"
            if item not in seen:
                out.append(item)
                seen.add(item)
            if len(out) >= limit:
                break
        return out

    for col in ("amenity", "shop", "leisure", "tourism", "healthcare", "office", "sport", "building"):
        if col not in gdf.columns:
            continue
        values = gdf[col].dropna().astype(str)
        for val in values:
            item = f"{col}={val}"
            if item not in seen:
                out.append(item)
                seen.add(item)
            if len(out) >= limit:
                return out

    return out


def _extract_sample_labels(
    gdf: gpd.GeoDataFrame,
    *,
    label_columns: Sequence[str],
    limit: int,
) -> List[str]:
    if gdf is None or getattr(gdf, "empty", True):
        return []

    limit = max(1, int(limit))
    out: List[str] = []
    seen = set()

    for _, row in gdf.iterrows():
        label = _row_label(row, label_columns=label_columns)
        if not label or label in seen:
            continue
        out.append(label)
        seen.add(label)
        if len(out) >= limit:
            break

    return out


def _row_label(row, *, label_columns: Sequence[str]) -> str:
    for col in label_columns:
        if col in row.index:
            val = row.get(col)
            if val is None:
                continue
            text = str(val).strip()
            if text and text.lower() != "nan":
                return text
    return "POI"


def _score_to_status(score_100: float, thresholds: Mapping[str, float]) -> AccessibilityStatus:
    good_thr = float(thresholds.get("good", 82.0))
    medium_thr = float(thresholds.get("medium", 60.0))

    if score_100 >= good_thr:
        return "good"
    if score_100 >= medium_thr:
        return "medium"
    return "poor"


def _resolve_analysis_polygon(
    G,
    area_geometry: Optional[Union[gpd.GeoDataFrame, Polygon, MultiPolygon]],
):
    if isinstance(area_geometry, gpd.GeoDataFrame):
        if area_geometry.empty:
            return _graph_analysis_polygon(G)

        gdf = area_geometry
        try:
            if getattr(gdf, "crs", None) is None:
                gdf = gdf.set_crs(epsg=4326, allow_override=True)
            else:
                gdf = gdf.to_crs(epsg=4326)
        except Exception:
            pass

        geom = gdf.geometry.unary_union
        if geom is None or getattr(geom, "is_empty", True):
            return _graph_analysis_polygon(G)
        return geom

    if area_geometry is not None:
        return area_geometry

    return _graph_analysis_polygon(G)


def _graph_analysis_polygon(G):
    pts = []
    for _, data in G.nodes(data=True):
        try:
            pts.append(Point(float(data["x"]), float(data["y"])))
        except Exception:
            continue

    if not pts:
        raise AccessibilityAnalyzerError("Не вдалося побудувати полігон аналізу з графа")

    gs = gpd.GeoSeries(pts, crs=4326).to_crs(epsg=3857)
    poly = gs.unary_union.convex_hull.buffer(150.0)
    return gpd.GeoSeries([poly], crs=3857).to_crs(epsg=4326).iloc[0]


def _generate_grid_cells(poly_proj, *, step_m: float):
    minx, miny, maxx, maxy = poly_proj.bounds
    cells = []

    x = float(minx)
    while x < float(maxx):
        y = float(miny)
        while y < float(maxy):
            cell = box(x, y, min(x + step_m, maxx), min(y + step_m, maxy))
            try:
                if poly_proj.intersects(cell):
                    cells.append(cell)
            except Exception:
                pass
            y += step_m
        x += step_m

    return cells


def _sample_grid_cells(cells, max_cells: int):
    if len(cells) <= max_cells:
        return list(cells)

    step = max(1, int(len(cells) / max_cells))
    sampled = list(cells[::step])

    if len(sampled) > max_cells:
        sampled = sampled[:max_cells]

    return sampled


def _polygon_bounds_latlon(poly) -> Tuple[float, float, float, float]:
    minx, miny, maxx, maxy = poly.bounds
    return (float(miny), float(minx), float(maxy), float(maxx))


def _haversine_m(a: LatLon, b: LatLon) -> float:
    import math

    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    s = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(min(1.0, math.sqrt(s)))
    return 6371000.0 * c