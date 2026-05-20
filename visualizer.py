from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import folium
import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from folium.plugins import MarkerCluster
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from PyQt5.QtCore import QFile, QIODevice, QThread, pyqtSignal

from api import build_accessibility_evaluation, build_accessibility_grid, load_city_boundary
from accessibility_analyzer import DEFAULT_STATUS_THRESHOLDS
from config import LEVELS_QUERIES
from logger_config import logger
from path_finder import (
    PathfinderError,
    compute_isochrone,
    find_shortest_path,
    find_shortest_path_multi,
    find_shortest_path_via_poi_category,
    snap_to_graph,
)
from paths import DIR_OUTPUTS
from utils import extract_required_tokens

LatLon = Tuple[float, float]

# -------------------- Налаштування рендера --------------------

DEFAULT_CENTER: LatLon = (49.0, 24.0)

MAX_POI_PER_CATEGORY = 6000

ISO_EDGE_BUFFER_M_MIN = 35.0
ISO_EDGE_BUFFER_M_DEFAULT = 55.0
ISO_POLY_SIMPLIFY_M = 8.0
ISO_MAX_POI_IN_ISO_LAYER = 1200

LABEL_MAP = {
    "education": "Освіта",
    "health": "Медицина",
    "culture": "Культура",
    "greens_sport": "Зелена інфра / Спорт",
    "shopping_services": "Покупки / Сервіси",
    "transport": "Громадський транспорт",
    "civic": "Громадські сервіси / Безпека",
    "food": "Заклади харчування",
    "work_services": "Робота / Послуги",
    "tourism": "Туризм / Готелі",
}

_INJECT_MARKER = "injected: qtwebchannel + map_bridge.js"


class MapRenderWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(
        self,
        G,
        gdf_edges,
        gdf_all_poi,
        safe_city_name: str,
        city_name: str,
        level_name: str,
        route_start_latlon: Optional[LatLon] = None,
        route_end_latlon: Optional[LatLon] = None,
        route_algorithm: str = "dijkstra",
        route_via_category: Optional[str] = None,
        route_stops: Optional[List[LatLon]] = None,
        enable_click_pick: bool = True,
        isochrone_center_latlon: Optional[LatLon] = None,
        isochrone_minutes: Optional[List[int]] = None,
        isochrone_walk_speed_kmh: float = 4.8,
        isochrone_buffer_m: float = 0.0,
        accessibility_center_latlon: Optional[LatLon] = None,
        accessibility_minutes: int = 15,
        accessibility_walk_speed_kmh: float = 4.8,
        accessibility_citywide_enabled: bool = False,
        accessibility_grid_minutes: Optional[List[int]] = None,
        accessibility_grid_step_m: float = 320.0,
        accessibility_grid_max_cells: int = 3200,
    ):
        super().__init__()
        self.G = G
        self.gdf_edges = gdf_edges
        self.gdf_all_poi = gdf_all_poi
        self.safe_city_name = safe_city_name
        self.city_name = city_name
        self.level_name = level_name

        self.route_start_latlon = route_start_latlon
        self.route_end_latlon = route_end_latlon
        self.route_algorithm = (route_algorithm or "dijkstra").strip()
        self.route_via_category = (route_via_category or "none").strip()
        self.route_stops = list(route_stops or [])
        self.enable_click_pick = bool(enable_click_pick)

        self.isochrone_center_latlon = isochrone_center_latlon
        self.isochrone_minutes = list(isochrone_minutes or [])
        self.isochrone_walk_speed_kmh = float(isochrone_walk_speed_kmh)
        self.isochrone_buffer_m = float(isochrone_buffer_m)

        self.accessibility_center_latlon = accessibility_center_latlon
        self.accessibility_minutes = int(accessibility_minutes or 15)
        self.accessibility_walk_speed_kmh = float(accessibility_walk_speed_kmh)
        self.accessibility_citywide_enabled = bool(accessibility_citywide_enabled)
        self.accessibility_grid_minutes = list(accessibility_grid_minutes or [])
        self.accessibility_grid_step_m = float(accessibility_grid_step_m)
        self.accessibility_grid_max_cells = int(accessibility_grid_max_cells)

    # -------------------- helpers: geo/poi --------------------

    @staticmethod
    def _ensure_outputs_dir() -> None:
        try:
            os.makedirs(DIR_OUTPUTS, exist_ok=True)
        except Exception as e:
            logger.warning("DIR_OUTPUTS не створено (%s): %s", DIR_OUTPUTS, e)

    @staticmethod
    def _ensure_gdf_crs(gdf, epsg: int = 4326):
        if gdf is None:
            return gdf
        try:
            if getattr(gdf, "crs", None) is None:
                gdf = gdf.set_crs(epsg=epsg, allow_override=True)
        except Exception:
            pass
        return gdf

    def _compute_center_latlon(self) -> LatLon:
        try:
            if self.gdf_edges is not None and not self.gdf_edges.empty and "geometry" in self.gdf_edges.columns:
                minx, miny, maxx, maxy = self.gdf_edges.total_bounds
                cx = (minx + maxx) / 2.0
                cy = (miny + maxy) / 2.0
                return (cy, cx)
        except Exception:
            pass

        try:
            b_gdf, _ = load_city_boundary(
                self.city_name,
                required_tokens=extract_required_tokens(self.city_name),
            )
            if b_gdf is not None and not b_gdf.empty and "geometry" in b_gdf.columns:
                p = b_gdf.geometry.unary_union.representative_point()
                return (float(p.y), float(p.x))
        except Exception:
            pass

        return DEFAULT_CENTER

    def _fit_map_to_city(self, m: folium.Map) -> None:
        try:
            b_gdf, _ = load_city_boundary(
                self.city_name,
                required_tokens=extract_required_tokens(self.city_name),
            )
            if b_gdf is not None and not b_gdf.empty:
                b_gdf = self._ensure_gdf_crs(b_gdf)
                minx, miny, maxx, maxy = b_gdf.total_bounds
                m.fit_bounds([[miny, minx], [maxy, maxx]])
                return
        except Exception:
            pass

        try:
            if self.gdf_edges is not None and not self.gdf_edges.empty:
                gdf = self._ensure_gdf_crs(self.gdf_edges)
                minx, miny, maxx, maxy = gdf.total_bounds
                m.fit_bounds([[miny, minx], [maxy, maxx]])
        except Exception:
            pass

    def _get_city_clip_geom_3857(self):
        area_gdf = None

        try:
            b_gdf, _ = load_city_boundary(
                self.city_name,
                required_tokens=extract_required_tokens(self.city_name),
            )
            if b_gdf is not None and not b_gdf.empty:
                area_gdf = self._ensure_gdf_crs(b_gdf)
        except Exception:
            area_gdf = None

        if area_gdf is not None and not area_gdf.empty:
            try:
                clip_geom_3857 = area_gdf.to_crs(3857).geometry.unary_union
                if clip_geom_3857 is not None and not clip_geom_3857.is_empty:
                    return area_gdf, clip_geom_3857
            except Exception:
                pass

        try:
            if self.gdf_edges is not None and not self.gdf_edges.empty:
                edges = self._ensure_gdf_crs(self.gdf_edges).to_crs(3857)
                hull = edges.geometry.unary_union.convex_hull.buffer(250.0)
                if hull is not None and not hull.is_empty:
                    fallback_gdf = gpd.GeoDataFrame(geometry=[hull], crs=3857).to_crs(4326)
                    return fallback_gdf, hull
        except Exception:
            pass

        return None, None

    def _resolve_citywide_grid_params(self, clip_geom_3857):
        area_m2 = max(float(clip_geom_3857.area), 1.0)
        minx, miny, maxx, maxy = clip_geom_3857.bounds

        width_m = max(float(maxx - minx), 1.0)
        height_m = max(float(maxy - miny), 1.0)
        longest_side_m = max(width_m, height_m)

        wanted_cells = int(min(max(self.accessibility_grid_max_cells, 1800), 6000))

        step_from_area = float(np.sqrt(area_m2 / wanted_cells))
        step_from_side = float(longest_side_m / 58.0)

        step_m = max(step_from_area, step_from_side)
        step_m = min(max(step_m, 90.0), 850.0)

        est_cells_bbox = int(np.ceil(width_m / step_m) * np.ceil(height_m / step_m))

        dynamic_max_cells = max(
            self.accessibility_grid_max_cells,
            min(9000, int(est_cells_bbox * 1.35) + 200)
        )

        surface_step_m = min(max(step_m / 2.6, 30.0), 220.0)

        return step_m, dynamic_max_cells, surface_step_m

    def _resolve_growth_cell_size_m(self, clip_geom_3857) -> float:
        """
        Розмір дрібної клітинки для "розростання" зон.
        Чим менше місто — тим дрібніше, чим більше — тим грубіше.
        """
        minx, miny, maxx, maxy = clip_geom_3857.bounds
        width_m = max(float(maxx - minx), 1.0)
        height_m = max(float(maxy - miny), 1.0)
        longest_side_m = max(width_m, height_m)

        cell_m = longest_side_m / 140.0
        cell_m = min(max(cell_m, 35.0), 140.0)
        return float(cell_m)

    def _assign_status_by_nearest_point(
        self,
        clip_geom_3857,
        xs: np.ndarray,
        ys: np.ndarray,
        statuses: List[str],
        *,
        cell_size_m: float,
        chunk_size: int = 2500,
    ):
        """
        Будує дрібну сітку по місту і кожній клітинці дає статус найближчої точки.
        Це і є "розростання" точок своїм кольором.
        """
        minx, miny, maxx, maxy = clip_geom_3857.bounds

        x_edges = np.arange(minx, maxx + cell_size_m, cell_size_m, dtype=float)
        y_edges = np.arange(miny, maxy + cell_size_m, cell_size_m, dtype=float)

        if len(x_edges) < 2 or len(y_edges) < 2:
            return []

        centers = []
        cells = []

        for ix in range(len(x_edges) - 1):
            x0 = float(x_edges[ix])
            x1 = float(x_edges[ix + 1])

            for iy in range(len(y_edges) - 1):
                y0 = float(y_edges[iy])
                y1 = float(y_edges[iy + 1])

                cell = Polygon([
                    (x0, y0),
                    (x1, y0),
                    (x1, y1),
                    (x0, y1),
                ])

                try:
                    if not clip_geom_3857.intersects(cell):
                        continue

                    clipped = cell.intersection(clip_geom_3857)
                    if clipped.is_empty:
                        continue

                    rp = clipped.representative_point()
                    centers.append((float(rp.x), float(rp.y)))
                    cells.append(clipped)
                except Exception:
                    continue

        if not centers:
            return []

        pts = np.column_stack([xs.astype(float), ys.astype(float)])
        q = np.array(centers, dtype=float)

        nearest_statuses: List[str] = []
        for i in range(0, len(q), chunk_size):
            qq = q[i:i + chunk_size]

            dx = qq[:, None, 0] - pts[None, :, 0]
            dy = qq[:, None, 1] - pts[None, :, 1]
            dist2 = dx * dx + dy * dy

            nearest_idx = np.argmin(dist2, axis=1)
            nearest_statuses.extend([statuses[int(j)] for j in nearest_idx])

        out = []
        for geom, status in zip(cells, nearest_statuses):
            out.append((geom, status))

        return out

    def _merge_growth_cells_by_status(self, grown_cells):
        """
        grown_cells: list[(geom_3857, status_name)]
        Повертає об'єднані полігони по статусах.
        """
        by_status: Dict[str, List[Any]] = {
            "good": [],
            "medium": [],
            "poor": [],
        }

        for geom, status in grown_cells:
            if geom is None or getattr(geom, "is_empty", True):
                continue
            s = str(status or "poor").strip().lower()
            if s not in by_status:
                s = "poor"
            by_status[s].append(geom)

        merged = []
        for status_name, geoms in by_status.items():
            if not geoms:
                continue
            try:
                union_geom = unary_union(geoms)
                if union_geom is None or getattr(union_geom, "is_empty", True):
                    continue
                merged.append((status_name, union_geom))
            except Exception:
                continue

        return merged

    @staticmethod
    def _subset_for_tags(gdf_all, tags_dict):
        if gdf_all is None or getattr(gdf_all, "empty", True):
            return gdf_all

        if "poi_key" in gdf_all.columns and "poi_value" in gdf_all.columns:
            mask_total = None
            for key, vals in (tags_dict or {}).items():
                try:
                    vals_norm = [str(v) for v in (vals or [])]
                    m = (gdf_all["poi_key"] == key) & (gdf_all["poi_value"].astype(str).isin(vals_norm))
                    mask_total = m if mask_total is None else (mask_total | m)
                except Exception:
                    continue
            return gdf_all[mask_total] if mask_total is not None else gdf_all.iloc[0:0]

        mask_total = None
        for key, vals in (tags_dict or {}).items():
            if key not in gdf_all.columns:
                continue
            try:
                vals_norm = [str(v) for v in (vals or [])]
                m = gdf_all[key].astype(str).isin(vals_norm)
                mask_total = m if mask_total is None else (mask_total | m)
            except Exception:
                continue

        return gdf_all[mask_total] if mask_total is not None else gdf_all.iloc[0:0]

    def _poi_for_current_level(self):
        if self.gdf_all_poi is None or getattr(self.gdf_all_poi, "empty", True):
            return self.gdf_all_poi

        level_cfg = LEVELS_QUERIES.get(self.level_name, {}) or {}
        if not level_cfg:
            return self.gdf_all_poi.iloc[0:0]

        parts = []
        for category, tags_dict in level_cfg.items():
            g = self._subset_for_tags(self.gdf_all_poi, tags_dict)
            if g is None or g.empty:
                continue
            g = g.copy()
            g["__lvl_category__"] = category
            parts.append(g)

        if not parts:
            return self.gdf_all_poi.iloc[0:0]

        try:
            gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=getattr(parts[0], "crs", None))
        except Exception:
            gdf = parts[0].copy()
        gdf = self._ensure_gdf_crs(gdf)

        try:
            if "poi_key" in gdf.columns and "poi_value" in gdf.columns:
                gdf = gdf[~((gdf["poi_key"] == "building") & (gdf["poi_value"].astype(str) == "house"))]
        except Exception:
            pass

        try:
            if "building" in gdf.columns:
                gdf = gdf[gdf["building"].astype(str) != "house"]
        except Exception:
            pass

        for key in ("osmid", "element_id", "id"):
            if key in gdf.columns:
                try:
                    gdf = gdf.drop_duplicates(subset=[key])
                except Exception:
                    pass
                break

        return gdf

    @staticmethod
    def _geom_to_point(geom) -> Optional[Point]:
        if geom is None:
            return None
        try:
            if geom.is_empty:
                return None
        except Exception:
            pass

        try:
            if isinstance(geom, Point):
                return geom
            return geom.representative_point()
        except Exception:
            return None

    # -------------------- color helpers --------------------

    @staticmethod
    def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
        color = color.lstrip("#")
        return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        r, g, b = rgb
        return f"#{r:02x}{g:02x}{b:02x}"

    @classmethod
    def _blend_hex(cls, c1: str, c2: str, t: float) -> str:
        t = max(0.0, min(1.0, float(t)))
        r1, g1, b1 = cls._hex_to_rgb(c1)
        r2, g2, b2 = cls._hex_to_rgb(c2)
        rgb = (
            int(round(r1 + (r2 - r1) * t)),
            int(round(g1 + (g2 - g1) * t)),
            int(round(b1 + (b2 - b1) * t)),
        )
        return cls._rgb_to_hex(rgb)

    def _status_thresholds(self) -> Tuple[float, float]:
        medium_thr = float(DEFAULT_STATUS_THRESHOLDS.get("medium", 55.0))
        good_thr = float(DEFAULT_STATUS_THRESHOLDS.get("good", 80.0))
        if medium_thr >= good_thr:
            medium_thr, good_thr = 55.0, 80.0
        return medium_thr, good_thr

    def _score_to_status_name(self, score: float) -> str:
        medium_thr, good_thr = self._status_thresholds()
        s = float(score)
        if s >= good_thr:
            return "good"
        if s >= medium_thr:
            return "medium"
        return "poor"

    def _status_fill_color(self, status_or_score) -> str:
        if isinstance(status_or_score, (int, float)):
            status = self._score_to_status_name(float(status_or_score))
        else:
            status = str(status_or_score or "").strip().lower()

        palette = {
            "good": "#5BAE68",
            "medium": "#E7C65B",
            "poor": "#D96B63",
        }
        return palette.get(status, "#D96B63")

    def _status_border_color(self, status_or_score) -> str:
        fill = self._status_fill_color(status_or_score)
        return self._blend_hex(fill, "#1f2937", 0.38)

    def _score_fill_color(self, score: float) -> str:
        return self._status_fill_color(score)

    def _score_border_color(self, score: float) -> str:
        return self._status_border_color(score)

    def _contour_levels(self) -> List[float]:
        medium_thr, good_thr = self._status_thresholds()
        return [0.0, medium_thr, good_thr, 100.0]

    def _idw_interpolate_grid(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        scores: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        *,
        power: float = 2.0,
        chunk_size: int = 1800,
    ) -> np.ndarray:
        pts = np.column_stack([xs, ys]).astype(float)
        vals = scores.astype(float)

        query = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(float)
        out = np.empty(len(query), dtype=float)

        for i in range(0, len(query), chunk_size):
            q = query[i:i + chunk_size]

            dx = q[:, None, 0] - pts[None, :, 0]
            dy = q[:, None, 1] - pts[None, :, 1]
            dist2 = dx * dx + dy * dy

            exact_mask = dist2 < 1.0
            safe_dist2 = np.maximum(dist2, 1.0)

            weights = 1.0 / np.power(safe_dist2, power / 2.0)
            weighted = (weights * vals[None, :]).sum(axis=1)
            weights_sum = weights.sum(axis=1)

            pred = weighted / np.maximum(weights_sum, 1e-12)

            if exact_mask.any():
                rows = np.where(exact_mask.any(axis=1))[0]
                for r in rows:
                    c = int(np.argmax(exact_mask[r]))
                    pred[r] = vals[c]

            out[i:i + chunk_size] = pred

        return out.reshape(grid_x.shape)

    def _build_contour_band_geometries(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z_masked,
        *,
        levels: List[float],
        clip_geom,
    ) -> List[Tuple[float, float, object]]:
        fig, ax = plt.subplots(figsize=(6, 6))
        try:
            cs = ax.contourf(X, Y, Z_masked, levels=levels, antialiased=True)
            band_geoms: List[Tuple[float, float, object]] = []

            collections = getattr(cs, "collections", None)
            if collections is None:
                return []

            for idx, collection in enumerate(collections):
                polys = []

                for path in collection.get_paths():
                    rings = path.to_polygons()
                    if not rings:
                        continue

                    outer = rings[0]
                    if len(outer) < 3:
                        continue

                    holes = [ring for ring in rings[1:] if len(ring) >= 3]

                    try:
                        poly = Polygon(outer, holes)
                    except Exception:
                        continue

                    if not poly.is_valid:
                        try:
                            poly = poly.buffer(0)
                        except Exception:
                            continue

                    if poly.is_empty:
                        continue

                    polys.append(poly)

                if not polys:
                    continue

                geom = unary_union(polys)

                if clip_geom is not None and not getattr(clip_geom, "is_empty", True):
                    try:
                        geom = geom.intersection(clip_geom)
                    except Exception:
                        pass

                if geom.is_empty:
                    continue

                band_geoms.append((float(levels[idx]), float(levels[idx + 1]), geom))

            return band_geoms
        finally:
            plt.close(fig)

    # -------------------- helpers: folium layers --------------------

    def _add_streets_layer(self, m: folium.Map, *, show: bool) -> None:
        try:
            if self.gdf_edges is None or self.gdf_edges.empty:
                return
            gdf = self._ensure_gdf_crs(self.gdf_edges)

            folium.GeoJson(
                data=gdf[["geometry"]].to_json(),
                name="Вулиці",
                show=bool(show),
                style_function=lambda _: {
                    "color": "#4b5563",
                    "weight": 1.6,
                    "opacity": 0.30,
                },
            ).add_to(m)
        except Exception as e:
            logger.warning("Не вдалося додати шар 'Вулиці': %s", e)

    def _add_boundary_layer(self, m: folium.Map) -> None:
        try:
            b_gdf, binfo = load_city_boundary(
                self.city_name,
                required_tokens=extract_required_tokens(self.city_name),
            )
            if b_gdf is None:
                b_gdf = ox.geocode_to_gdf(self.city_name).to_crs(4326)
                self.status.emit("Boundary: кеш не знайдено — взято з OSM")
            else:
                used = binfo.get("used_path") or binfo.get("expected_path") or "—"
                act = binfo.get("cache_action")
                extra = f" ({act})" if act else ""
                self.status.emit(f"Boundary: кеш → {used}{extra}")

            if b_gdf is None or b_gdf.empty:
                return

            b_gdf = self._ensure_gdf_crs(b_gdf)

            folium.GeoJson(
                data=b_gdf[["geometry"]].to_json(),
                name="Кордон міста",
                show=True,
                style_function=lambda _: {
                    "color": "#b45353",
                    "weight": 2.2,
                    "fill": False,
                    "opacity": 0.78,
                    "dashArray": "5,6",
                },
            ).add_to(m)
        except Exception as e:
            logger.warning("Boundary: не вдалося отримати/намалювати для '%s': %s", self.city_name, e)

    @staticmethod
    def _fmt_ll(ll: LatLon) -> str:
        return f"{ll[0]:.6f},{ll[1]:.6f}"

    # -------------------- Маршрут --------------------

    @staticmethod
    def _detour_minutes_from_m(detour_m: float, speed_kmh: float) -> float:
        m_per_min = float(speed_kmh) * 1000.0 / 60.0
        if m_per_min <= 0:
            return 0.0
        return float(detour_m) / float(m_per_min)

    def _add_route_layer(self, m: folium.Map) -> None:
        if not (self.route_start_latlon and self.route_end_latlon):
            return

        fg = folium.FeatureGroup(name="Маршрут", show=True)
        fg.add_to(m)

        try:
            Gwork = self.G.to_undirected()

            snap_s = snap_to_graph(Gwork, self.route_start_latlon, mode="edge")
            snap_e = snap_to_graph(Gwork, self.route_end_latlon, mode="edge")

            self.status.emit(
                "SNAP START: "
                f"клік {self._fmt_ll(snap_s.input_latlon)} → "
                f"дорога {self._fmt_ll(snap_s.snapped_latlon)} "
                f"({snap_s.dist_to_snapped_m:.0f} м)"
            )
            self.status.emit(
                "SNAP END:   "
                f"клік {self._fmt_ll(snap_e.input_latlon)} → "
                f"дорога {self._fmt_ll(snap_e.snapped_latlon)} "
                f"({snap_e.dist_to_snapped_m:.0f} м)"
            )

            folium.Marker(
                location=list(snap_s.input_latlon),
                tooltip="Старт (клік/будинок)",
                popup=f"Старт (клік): {self._fmt_ll(snap_s.input_latlon)}",
            ).add_to(fg)

            folium.Marker(
                location=list(snap_e.input_latlon),
                tooltip="Фініш (клік/будинок)",
                popup=f"Фініш (клік): {self._fmt_ll(snap_e.input_latlon)}",
            ).add_to(fg)

            folium.CircleMarker(
                location=list(snap_s.snapped_latlon),
                radius=6,
                weight=2,
                tooltip="Старт (точка на дорозі)",
                popup=(
                    "Старт (на дорозі): "
                    f"{self._fmt_ll(snap_s.snapped_latlon)}<br>"
                    f"від кліку: ~{snap_s.dist_to_snapped_m:.0f} м<br>"
                    f"edge: {snap_s.edge_u}-{snap_s.edge_v} (key={snap_s.edge_key})"
                ),
            ).add_to(fg)

            folium.CircleMarker(
                location=list(snap_e.snapped_latlon),
                radius=6,
                weight=2,
                tooltip="Фініш (точка на дорозі)",
                popup=(
                    "Фініш (на дорозі): "
                    f"{self._fmt_ll(snap_e.snapped_latlon)}<br>"
                    f"від кліку: ~{snap_e.dist_to_snapped_m:.0f} м<br>"
                    f"edge: {snap_e.edge_u}-{snap_e.edge_v} (key={snap_e.edge_key})"
                ),
            ).add_to(fg)

            folium.PolyLine(
                locations=[snap_s.input_latlon, snap_s.snapped_latlon],
                weight=2,
                opacity=0.75,
                dash_array="6,6",
                tooltip="Старт: клік → дорога",
            ).add_to(fg)

            folium.PolyLine(
                locations=[snap_e.input_latlon, snap_e.snapped_latlon],
                weight=2,
                opacity=0.75,
                dash_array="6,6",
                tooltip="Фініш: клік → дорога",
            ).add_to(fg)

        except Exception as e:
            logger.warning("Route SNAP debug: не вдалося порахувати/намалювати snap: %s", e)
            folium.Marker(location=list(self.route_start_latlon), tooltip="Старт").add_to(fg)
            folium.Marker(location=list(self.route_end_latlon), tooltip="Фініш").add_to(fg)

        if self.route_stops:
            for i, st in enumerate(self.route_stops, start=1):
                folium.Marker(
                    location=[st[0], st[1]],
                    tooltip=f"Stop #{i}",
                    popup=f"Stop #{i}: {self._fmt_ll(st)}",
                    icon=folium.Icon(icon="info-sign"),
                ).add_to(fg)

        via = (self.route_via_category or "none").strip()
        has_multistops = bool(self.route_stops)

        self.status.emit(
            "Маршрут: "
            f"START {self._fmt_ll(self.route_start_latlon)} "
            f"→ END {self._fmt_ll(self.route_end_latlon)} "
            f"| alg={self.route_algorithm} "
            f"| stops={len(self.route_stops)} "
            f"| via={via}"
        )

        try:
            self.status.emit("Маршрут: обчислення…")

            result_coords = None
            result_len_m = 0.0

            if has_multistops:
                points = [self.route_start_latlon] + list(self.route_stops) + [self.route_end_latlon]
                mp = find_shortest_path_multi(
                    self.G,
                    points=points,
                    algorithm=self.route_algorithm,
                    weight="length",
                    use_undirected=True,
                    snap_mode="edge",
                )
                result_coords = mp.coords
                result_len_m = float(mp.length_m)
                self.status.emit(f"Маршрут multi-stop: сегментів={len(mp.segments)}, довжина~{result_len_m:.0f} м")
            else:
                if via and via.lower() != "none":
                    detour_minutes = 10.0
                    speed_kmh = float(self.isochrone_walk_speed_kmh or 4.8)
                    detour_limit_m = detour_minutes * (speed_kmh * 1000.0 / 60.0)

                    try:
                        r, sel = find_shortest_path_via_poi_category(
                            self.G,
                            self.gdf_all_poi,
                            self.route_start_latlon,
                            self.route_end_latlon,
                            via_category=via,
                            algorithm=self.route_algorithm,
                            weight="length",
                            use_undirected=True,
                            snap_mode="edge",
                            prefilter_max=400,
                            max_candidates_to_check=25,
                            detour_limit_m=detour_limit_m,
                            route_sample_max_points=250,
                        )

                        folium.Marker(
                            location=[sel.poi_latlon[0], sel.poi_latlon[1]],
                            tooltip=f"Зупинка (POI): {sel.category}",
                            popup=(
                                f"<b>Зупинка (POI):</b> {sel.category}<br>"
                                f"<b>Обрано:</b> {sel.label}<br>"
                                f"<b>Start→POI:</b> ~{sel.dist_start_to_poi_m:.0f} м<br>"
                                f"<b>POI→End:</b> ~{sel.dist_poi_to_end_m:.0f} м<br>"
                                f"<b>Разом:</b> ~{sel.total_m:.0f} м"
                            ),
                            icon=folium.Icon(icon="flag"),
                        ).add_to(fg)

                        result_coords = r.coords
                        result_len_m = float(r.length_m)

                    except PathfinderError as e:
                        r0 = find_shortest_path(
                            self.G,
                            self.route_start_latlon,
                            self.route_end_latlon,
                            algorithm=self.route_algorithm,
                            weight="length",
                            use_undirected=True,
                            snap_mode="edge",
                        )
                        result_coords = r0.coords
                        result_len_m = float(r0.length_m)

                        code = getattr(e, "code", "") or ""
                        ctx = getattr(e, "context", None)
                        ctxd: Dict[str, Any] = {}
                        try:
                            ctxd = ctx.to_dict() if ctx is not None else {}
                        except Exception:
                            ctxd = {}

                        if code == "NO_POI_FOR_CATEGORY":
                            self.status.emit(
                                f"Маршрут через '{via}': POI не знайдено. Показую найкоротший маршрут."
                            )
                        elif code == "VIA_DETOUR_LIMIT":
                            best_detour_m = float(ctxd.get("best_detour_m", 0.0) or 0.0)
                            best_detour_min = self._detour_minutes_from_m(best_detour_m, speed_kmh)
                            self.status.emit(
                                f"Маршрут через '{via}' не вкладається в +{int(detour_minutes)} хв. "
                                f"Найкраще можливе відхилення ~{best_detour_min:.1f} хв. "
                                f"Показую найкоротший маршрут."
                            )
                        else:
                            self.status.emit(
                                f"Маршрут через '{via}' не вдалося побудувати. Показую найкоротший маршрут. ({e})"
                            )

                else:
                    r = find_shortest_path(
                        self.G,
                        self.route_start_latlon,
                        self.route_end_latlon,
                        algorithm=self.route_algorithm,
                        weight="length",
                        use_undirected=True,
                        snap_mode="edge",
                    )
                    result_coords = r.coords
                    result_len_m = float(r.length_m)

            if result_coords:
                folium.PolyLine(
                    locations=result_coords,
                    weight=5,
                    opacity=0.9,
                    tooltip=f"Довжина: {result_len_m:.0f} м",
                ).add_to(fg)

            self.status.emit(f"Маршрут: готово (довжина ~ {result_len_m:.0f} м)")

        except PathfinderError as e:
            logger.warning("Маршрут не побудовано: %s", e)
            self.status.emit(f"Маршрут: не побудовано — {e}")
        except Exception as e:
            logger.exception("Маршрут: помилка: %s", e)
            self.status.emit(f"Маршрут: помилка — {e}")

    # -------------------- Оцінка доступності --------------------

    @staticmethod
    def _accessibility_status_label(status: str) -> str:
        status_norm = (status or "").strip().lower()
        if status_norm == "good":
            return "Добра"
        if status_norm == "medium":
            return "Середня"
        return "Слабка"

    def _group_label(self, name: str) -> str:
        return LABEL_MAP.get(name, str(name).replace("_", " ").strip().title())

    def _add_accessibility_layer(self, m: folium.Map) -> None:
        if not self.accessibility_center_latlon:
            return

        center = self.accessibility_center_latlon
        minutes = max(1, int(self.accessibility_minutes or 15))

        fg = folium.FeatureGroup(name=f"Оцінка доступності ({minutes} хв)", show=True)
        fg.add_to(m)

        try:
            self.status.emit(
                f"Оцінка доступності: аналізую точку {self._fmt_ll(center)} | "
                f"{minutes} хв | рівень={self.level_name}"
            )

            result = build_accessibility_evaluation(
                self.G,
                self.gdf_all_poi,
                center=center,
                level=self.level_name,
                minutes=float(minutes),
                walk_speed_kmh=float(self.accessibility_walk_speed_kmh),
                weight="length",
            )

            fill_color = self._status_fill_color(result.status)
            border_color = self._status_border_color(result.status)
            status_label = self._accessibility_status_label(result.status)

            try:
                snap = snap_to_graph(self.G.to_undirected(), center, mode="edge")
                snapped = snap.snapped_latlon
            except Exception:
                snapped = center

            covered_labels = [self._group_label(g.group_name) for g in result.group_results if g.present]
            missing_labels = [self._group_label(x) for x in result.missing_groups]

            covered_html = ", ".join(covered_labels) if covered_labels else "—"
            missing_html = ", ".join(missing_labels) if missing_labels else "—"

            popup_html = (
                f"<div style='min-width:280px'>"
                f"<h4 style='margin:0 0 8px 0;'>Оцінка 15-хв доступності</h4>"
                f"<b>Рівень:</b> {self.level_name}<br>"
                f"<b>Час:</b> {result.minutes:.0f} хв<br>"
                f"<b>Статус:</b> {status_label}<br>"
                f"<b>Score:</b> {result.score_100:.1f} / 100<br>"
                f"<b>Покрито груп:</b> {result.covered_groups} / {result.total_groups}<br>"
                f"<b>POI всередині ізохрони:</b> {result.inside_poi_count}<br><br>"
                f"<b>Є доступ:</b> {covered_html}<br><br>"
                f"<b>Бракує:</b> {missing_html}"
                f"</div>"
            )

            folium.Marker(
                location=[center[0], center[1]],
                tooltip=f"Оцінка доступності: {result.score_100:.1f}/100",
                popup=folium.Popup(popup_html, max_width=420),
                icon=folium.Icon(icon="info-sign"),
            ).add_to(fg)

            if snapped != center:
                folium.CircleMarker(
                    location=[snapped[0], snapped[1]],
                    radius=5,
                    weight=2,
                    color=border_color,
                    fill=True,
                    fill_color=fill_color,
                    fill_opacity=0.95,
                    tooltip="Центр аналізу (на дорозі)",
                    popup=(
                        f"<b>Snapped центр</b>: {self._fmt_ll(snapped)}<br>"
                        f"Вихідна точка: {self._fmt_ll(center)}"
                    ),
                ).add_to(fg)

            poly = result.isochrone.polygon
            if poly is not None:
                geojson = {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {
                                "minutes": float(result.minutes),
                                "score_100": float(result.score_100),
                                "status": str(result.status),
                            },
                            "geometry": poly.__geo_interface__,
                        }
                    ],
                }

                folium.GeoJson(
                    data=geojson,
                    show=True,
                    style_function=lambda _feat, fc=fill_color, bc=border_color: {
                        "color": bc,
                        "weight": 2.2,
                        "fill": True,
                        "fillColor": fc,
                        "fillOpacity": 0.24,
                        "opacity": 0.82,
                    },
                    highlight_function=lambda _feat: {
                        "color": "#1f2937",
                        "weight": 3.0,
                        "fillOpacity": 0.30,
                        "opacity": 0.95,
                    },
                    tooltip=(
                        f"{status_label}: {result.score_100:.1f}/100 | "
                        f"покрито {result.covered_groups}/{result.total_groups}"
                    ),
                ).add_to(fg)

            self.status.emit(
                f"Оцінка доступності: {status_label.lower()} | "
                f"score {result.score_100:.1f}/100 | "
                f"покрито {result.covered_groups}/{result.total_groups}"
            )

        except Exception as e:
            logger.exception("Accessibility layer failed: %s", e)
            self.status.emit(f"Оцінка доступності: помилка — {e}")

    def _add_accessibility_legend(self, m: folium.Map) -> None:
        medium_thr, good_thr = self._status_thresholds()

        poor_color = self._status_fill_color("poor")
        medium_color = self._status_fill_color("medium")
        good_color = self._status_fill_color("good")

        html = f"""
        <div style="
            position: fixed;
            bottom: 18px;
            left: 18px;
            z-index: 9999;
            background: rgba(255,255,255,0.95);
            border: 1px solid rgba(0,0,0,0.14);
            border-radius: 12px;
            padding: 12px 14px;
            font-size: 13px;
            min-width: 230px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.14);
            backdrop-filter: blur(4px);
        ">
          <div style="font-weight:700; margin-bottom:8px;">Карта доступності</div>

          <div style="display:flex; align-items:center; margin-bottom:6px;">
            <span style="display:inline-block;width:12px;height:12px;background:{good_color};border-radius:50%;margin-right:8px;"></span>
            <span>добра доступність (≥ {good_thr:.0f})</span>
          </div>

          <div style="display:flex; align-items:center; margin-bottom:6px;">
            <span style="display:inline-block;width:12px;height:12px;background:{medium_color};border-radius:50%;margin-right:8px;"></span>
            <span>середня доступність ({medium_thr:.0f}–{good_thr:.0f})</span>
          </div>

          <div style="display:flex; align-items:center;">
            <span style="display:inline-block;width:12px;height:12px;background:{poor_color};border-radius:50%;margin-right:8px;"></span>
            <span>слабка доступність (&lt; {medium_thr:.0f})</span>
          </div>
        </div>
        """
        try:
            m.get_root().html.add_child(folium.Element(html))
        except Exception:
            pass

    def _add_accessibility_summary_badge(
        self,
        m: folium.Map,
        *,
        minutes: int,
        mean_score: Optional[float],
        cells_ok: int,
        cells_total: int,
    ) -> None:
        mean_text = f"{mean_score:.1f}" if mean_score is not None else "—"
        html = f"""
        <div style="
            position: fixed;
            top: 18px;
            left: 70px;
            z-index: 9999;
            background: rgba(255,255,255,0.94);
            border: 1px solid rgba(0,0,0,0.12);
            border-radius: 12px;
            padding: 10px 12px;
            font-size: 13px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.14);
            backdrop-filter: blur(4px);
        ">
          <div style="font-weight:700; margin-bottom:4px;">{minutes} хв • {self.level_name}</div>
          <div>Середній score: <b>{mean_text}</b></div>
          <div>Точок аналізу: <b>{cells_ok}/{cells_total}</b></div>
        </div>
        """
        try:
            m.get_root().html.add_child(folium.Element(html))
        except Exception:
            pass

    def _add_accessibility_citywide_layers(self, m: folium.Map) -> None:
        if not self.accessibility_citywide_enabled:
            return

        minutes_list = sorted({int(x) for x in self.accessibility_grid_minutes if int(x) > 0})
        if not minutes_list:
            minutes_list = [15]

        area_gdf, clip_geom_3857 = self._get_city_clip_geom_3857()
        if clip_geom_3857 is None or getattr(clip_geom_3857, "is_empty", True):
            self.status.emit("Карта доступності міста: не вдалося отримати геометрію міста.")
            return

        dynamic_step_m, dynamic_max_cells, _surface_step_m = self._resolve_citywide_grid_params(clip_geom_3857)
        growth_cell_m = self._resolve_growth_cell_size_m(clip_geom_3857)

        self._add_accessibility_legend(m)

        for idx, minutes in enumerate(minutes_list, start=1):
            self.status.emit(
                f"Карта доступності міста: {minutes} хв | рівень={self.level_name} | "
                f"точки={dynamic_step_m:.0f} м | ріст зон={growth_cell_m:.0f} м"
            )

            try:
                result = build_accessibility_grid(
                    self.G,
                    self.gdf_all_poi,
                    level=self.level_name,
                    minutes=float(minutes),
                    walk_speed_kmh=float(self.accessibility_walk_speed_kmh),
                    step_m=float(dynamic_step_m),
                    max_cells=int(dynamic_max_cells),
                    area_geometry=area_gdf,
                    weight="length",
                )
            except Exception as e:
                logger.exception("Citywide accessibility grid failed (%s хв): %s", minutes, e)
                self.status.emit(f"Карта доступності міста {minutes} хв: помилка — {e}")
                continue

            samples: List[Tuple[float, float, Any]] = []
            for cell in result.cells:
                p = self._geom_to_point(cell.geometry)
                if p is None:
                    continue
                samples.append((float(p.y), float(p.x), cell))

            if len(samples) < 3:
                self.status.emit(f"Карта доступності міста {minutes} хв: замало точок аналізу.")
                continue

            pts_gdf = gpd.GeoDataFrame(
                {
                    "status_name": [
                        str(cell.status or self._score_to_status_name(cell.score_100)).strip().lower()
                        for _, _, cell in samples
                    ],
                    "score": [float(cell.score_100) for _, _, cell in samples],
                    "covered_text": [f"{cell.covered_groups}/{cell.total_groups}" for _, _, cell in samples],
                    "inside_poi_count": [int(cell.inside_poi_count) for _, _, cell in samples],
                    "missing_text": [
                        ", ".join(self._group_label(x) for x in cell.missing_groups) if cell.missing_groups else "—"
                        for _, _, cell in samples
                    ],
                },
                geometry=[Point(lon, lat) for lat, lon, _ in samples],
                crs=4326,
            ).to_crs(3857)

            xs = pts_gdf.geometry.x.to_numpy(dtype=float)
            ys = pts_gdf.geometry.y.to_numpy(dtype=float)
            statuses = pts_gdf["status_name"].astype(str).tolist()

            grown_cells = self._assign_status_by_nearest_point(
                clip_geom_3857,
                xs,
                ys,
                statuses,
                cell_size_m=float(growth_cell_m),
                chunk_size=2500,
            )

            if not grown_cells:
                self.status.emit(f"Карта доступності міста {minutes} хв: не вдалося виростити зони.")
                continue

            merged_zones = self._merge_growth_cells_by_status(grown_cells)
            if not merged_zones:
                self.status.emit(f"Карта доступності міста {minutes} хв: не вдалося об'єднати зони.")
                continue

            smooth_layer_name = f"Карта доступності {minutes} хв [{self.level_name}]"
            smooth_fg = folium.FeatureGroup(name=smooth_layer_name, show=(idx == len(minutes_list)))
            smooth_fg.add_to(m)

            features = []
            for status_name, geom_3857 in merged_zones:
                try:
                    geom_4326 = gpd.GeoSeries([geom_3857], crs=3857).to_crs(4326).iloc[0]
                except Exception:
                    continue

                if geom_4326.is_empty:
                    continue

                features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "status_name": status_name,
                            "status_label": self._accessibility_status_label(status_name),
                            "fill_color": self._status_fill_color(status_name),
                            "border_color": self._status_border_color(status_name),
                        },
                        "geometry": geom_4326.__geo_interface__,
                    }
                )

            if features:
                geojson = {
                    "type": "FeatureCollection",
                    "features": features,
                }

                folium.GeoJson(
                    data=geojson,
                    show=True,
                    style_function=lambda feat: {
                        "color": feat["properties"]["border_color"],
                        "weight": 1.25,
                        "fill": True,
                        "fillColor": feat["properties"]["fill_color"],
                        "fillOpacity": 0.30,
                        "opacity": 0.72,
                    },
                    highlight_function=lambda feat: {
                        "color": "#1f2937",
                        "weight": 1.9,
                        "fillColor": feat["properties"]["fill_color"],
                        "fillOpacity": 0.42,
                        "opacity": 0.92,
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=["status_label"],
                        aliases=["Статус:"],
                        labels=True,
                        sticky=True,
                    ),
                ).add_to(smooth_fg)

            detail_layer_name = f"Детальні точки {minutes} хв [{self.level_name}]"
            detail_fg = folium.FeatureGroup(name=detail_layer_name, show=False)
            detail_fg.add_to(m)

            detail_limit = 1200
            step = max(1, len(samples) // detail_limit)

            for i, (lat, lon, cell) in enumerate(samples):
                if i % step != 0:
                    continue

                status_name = str(cell.status or self._score_to_status_name(cell.score_100)).strip().lower()
                fill_color = self._status_fill_color(status_name)
                border_color = self._status_border_color(status_name)
                status_label = self._accessibility_status_label(status_name)
                missing_text = ", ".join(
                    self._group_label(x) for x in cell.missing_groups) if cell.missing_groups else "—"

                popup_html = (
                    f"<div style='min-width:260px'>"
                    f"<b>{minutes} хв • {self.level_name}</b><br>"
                    f"<b>Статус:</b> {status_label}<br>"
                    f"<b>Score:</b> {cell.score_100:.1f}/100<br>"
                    f"<b>Покрито груп:</b> {cell.covered_groups}/{cell.total_groups}<br>"
                    f"<b>POI всередині:</b> {cell.inside_poi_count}<br>"
                    f"<b>Бракує:</b> {missing_text}"
                    f"</div>"
                )

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=2.4,
                    color=border_color,
                    weight=0.8,
                    fill=True,
                    fill_color=fill_color,
                    fill_opacity=0.72,
                    opacity=0.75,
                    tooltip=f"Score: {cell.score_100:.1f}",
                    popup=folium.Popup(popup_html, max_width=380),
                ).add_to(detail_fg)

            if idx == len(minutes_list):
                self._add_accessibility_summary_badge(
                    m,
                    minutes=minutes,
                    mean_score=result.score_mean,
                    cells_ok=result.successful_cells,
                    cells_total=result.total_cells,
                )

            clipped_note = ""
            if result.total_cells >= int(dynamic_max_cells):
                clipped_note = " | авто-ліміт досягнуто"

            if result.score_mean is not None:
                self.status.emit(
                    f"Карта доступності {minutes} хв: точок {result.successful_cells}/{result.total_cells}, "
                    f"mean score = {result.score_mean:.1f}{clipped_note}"
                )
            else:
                self.status.emit(
                    f"Карта доступності {minutes} хв: точок {result.successful_cells}/{result.total_cells}{clipped_note}"
                )

            logger.info(
                "Citywide accessibility rendered (grown status zones): minutes=%s level=%s cells=%d/%d mean=%s point_step=%.1f grow_step=%.1f",
                minutes,
                self.level_name,
                result.successful_cells,
                result.total_cells,
                f"{result.score_mean:.1f}" if result.score_mean is not None else "—",
                dynamic_step_m,
                growth_cell_m,
            )

    # -------------------- Ізохрони --------------------

    @staticmethod
    def _minutes_to_distance_m(minutes: int, speed_kmh: float) -> float:
        return float(minutes) * (float(speed_kmh) * 1000.0 / 60.0)

    def _iso_color(self, minutes: int) -> str:
        if minutes == 5:
            return "#2b8cbe"
        if minutes == 10:
            return "#41ab5d"
        if minutes == 15:
            return "#fdae6b"
        return "#756bb1"

    def _reachable_subgraph_and_edges(
        self,
        center: LatLon,
        cutoff_m: float,
        *,
        use_undirected: bool = True,
    ):
        Gwork = self.G.to_undirected() if use_undirected else self.G

        snap = snap_to_graph(Gwork, center, mode="edge")
        snapped = snap.snapped_latlon

        origin = None
        try:
            u = snap.edge_u
            v = snap.edge_v

            if u is not None and u in Gwork.nodes:
                origin = u
            elif v is not None and v in Gwork.nodes:
                origin = v
            else:
                origin = next(iter(Gwork.nodes))
        except Exception:
            origin = next(iter(Gwork.nodes))

        try:
            subG = nx.ego_graph(Gwork, origin, radius=float(cutoff_m), distance="length")
        except Exception:
            dist = nx.single_source_dijkstra_path_length(Gwork, origin, cutoff=float(cutoff_m), weight="length")
            nodes = set(dist.keys())
            subG = Gwork.subgraph(nodes).copy()

        try:
            gdf_edges_iso = ox.utils_graph.graph_to_gdfs(subG, nodes=False, fill_edge_geometry=True)
        except Exception:
            gdf_edges_iso = None

        if gdf_edges_iso is not None:
            gdf_edges_iso = self._ensure_gdf_crs(gdf_edges_iso)

        return {
            "snap": snap,
            "snapped": snapped,
            "origin": origin,
            "subG": subG,
            "gdf_edges": gdf_edges_iso,
        }

    def _polygon_from_reachable_edges(
        self,
        gdf_edges_iso,
        *,
        extra_buffer_m: float,
    ):
        if gdf_edges_iso is None or getattr(gdf_edges_iso, "empty", True):
            return None

        try:
            gdf_edges_iso = self._ensure_gdf_crs(gdf_edges_iso)
            gdf_proj = gdf_edges_iso.to_crs(epsg=3857)

            buf = float(extra_buffer_m)
            if buf < ISO_EDGE_BUFFER_M_MIN:
                buf = ISO_EDGE_BUFFER_M_MIN

            geoms = [g for g in gdf_proj.geometry.values if g is not None]
            if not geoms:
                return None

            poly = unary_union(geoms).buffer(buf)

            try:
                poly = poly.buffer(0)
            except Exception:
                pass

            try:
                poly = poly.simplify(ISO_POLY_SIMPLIFY_M)
            except Exception:
                pass

            poly_wgs = gpd.GeoSeries([poly], crs=3857).to_crs(4326).iloc[0]
            return poly_wgs
        except Exception as e:
            logger.warning("Iso polygon from edges failed: %s", e)
            return None

    def _add_isochrone_layers(self, m: folium.Map) -> None:
        if not self.isochrone_center_latlon or not self.isochrone_minutes:
            return

        center = self.isochrone_center_latlon
        minutes_list = sorted({int(x) for x in self.isochrone_minutes if int(x) > 0})
        if not minutes_list:
            return

        try:
            fg_center = folium.FeatureGroup(name="Ізохрона — центр", show=True)
            fg_center.add_to(m)
            folium.Marker(
                location=[center[0], center[1]],
                tooltip="Центр ізохрони",
                popup=f"Центр: {self._fmt_ll(center)}",
                icon=folium.Icon(icon="home"),
            ).add_to(fg_center)
        except Exception:
            pass

        for minutes in minutes_list:
            cutoff_m = self._minutes_to_distance_m(minutes, self.isochrone_walk_speed_kmh)

            self.status.emit(
                f"Ізохрона: {minutes} хв ≈ {cutoff_m:.0f} м (speed={self.isochrone_walk_speed_kmh:.1f} км/год)."
            )

            color = self._iso_color(minutes)

            fg = folium.FeatureGroup(name=f"Ізохрона {minutes} хв", show=True)
            fg.add_to(m)

            info = self._reachable_subgraph_and_edges(center, cutoff_m, use_undirected=True)
            snap = info["snap"]
            snapped = info["snapped"]
            gdf_edges_iso = info["gdf_edges"]

            try:
                folium.CircleMarker(
                    location=[snapped[0], snapped[1]],
                    radius=5,
                    weight=2,
                    tooltip=f"Центр (на дорозі) для {minutes} хв",
                    popup=(
                        f"<b>Snapped центр</b>: {self._fmt_ll(snapped)}<br>"
                        f"від кліку: ~{snap.dist_to_snapped_m:.0f} м<br>"
                        f"edge: {snap.edge_u}-{snap.edge_v} (key={snap.edge_key})"
                    ),
                ).add_to(fg)
            except Exception:
                pass

            poly = None
            try:
                iso = compute_isochrone(
                    self.G,
                    center,
                    cutoff=cutoff_m,
                    weight="length",
                    use_undirected=True,
                    snap_mode="edge",
                    polygon_buffer_m=self.isochrone_buffer_m,
                )
                poly = iso.polygon
            except Exception:
                poly = None

            if poly is None:
                extra_buf = self.isochrone_buffer_m if self.isochrone_buffer_m > 0 else ISO_EDGE_BUFFER_M_DEFAULT
                poly = self._polygon_from_reachable_edges(gdf_edges_iso, extra_buffer_m=extra_buf)

            if poly is None:
                self.status.emit(f"Ізохрона {minutes} хв: не вдалось зібрати полігон (замало даних).")
            else:
                geojson = {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {
                                "minutes": minutes,
                                "cutoff_m": float(cutoff_m),
                                "speed_kmh": float(self.isochrone_walk_speed_kmh),
                            },
                            "geometry": poly.__geo_interface__,
                        }
                    ],
                }

                folium.GeoJson(
                    data=geojson,
                    show=True,
                    style_function=lambda _feat, c=color: {
                        "color": c,
                        "weight": 3,
                        "fill": True,
                        "fillColor": c,
                        "fillOpacity": 0.18,
                        "opacity": 0.82,
                    },
                    tooltip=f"Ізохрона {minutes} хв",
                ).add_to(fg)

            try:
                if gdf_edges_iso is not None and not gdf_edges_iso.empty:
                    gdf_draw = gdf_edges_iso
                    if len(gdf_draw) > 25000:
                        gdf_draw = gdf_draw.sample(n=25000, random_state=42)

                    folium.GeoJson(
                        data=gdf_draw[["geometry"]].to_json(),
                        show=True,
                        style_function=lambda _feat, c=color: {
                            "color": c,
                            "weight": 1.8,
                            "opacity": 0.70,
                        },
                    ).add_to(fg)
                    self.status.emit(f"Ізохрона {minutes} хв: ребер у підграфі = {len(gdf_edges_iso)}")
                else:
                    self.status.emit(f"Ізохрона {minutes} хв: підграф порожній.")
            except Exception as e:
                logger.warning("Iso roads layer failed: %s", e)

            try:
                if poly is not None:
                    gdf_poi = self._poi_for_current_level()
                    if gdf_poi is None or gdf_poi.empty:
                        self.status.emit(f"POI в ізохроні {minutes} хв: 0 (рівень '{self.level_name}')")
                    else:
                        if len(gdf_poi) > 15000:
                            gdf_poi = gdf_poi.sample(n=15000, random_state=42)

                        pts = gdf_poi.geometry.apply(self._geom_to_point)
                        pts_gs = gpd.GeoSeries(pts, crs=gdf_poi.crs)

                        mask = pts_gs.within(poly)
                        inside = gdf_poi[mask].copy()
                        if len(inside) > ISO_MAX_POI_IN_ISO_LAYER:
                            inside = inside.sample(n=ISO_MAX_POI_IN_ISO_LAYER, random_state=42)

                        fg_poi = folium.FeatureGroup(
                            name=f"POI в ізохроні {minutes} хв [{self.level_name}]",
                            show=False
                        )
                        fg_poi.add_to(m)
                        cluster = MarkerCluster().add_to(fg_poi)

                        for _, row in inside.iterrows():
                            p = self._geom_to_point(row.get("geometry"))
                            if p is None:
                                continue

                            name = str(row.get("name") or row.get("brand") or "POI")
                            poi_type = (
                                row.get("poi_value")
                                or row.get("amenity")
                                or row.get("shop")
                                or row.get("leisure")
                                or row.get("tourism")
                                or row.get("public_transport")
                                or row.get("railway")
                                or row.get("office")
                                or row.get("highway")
                                or "poi"
                            )

                            cat = row.get("__lvl_category__")
                            cat_label = LABEL_MAP.get(cat, str(cat)) if cat else ""

                            desc = f"<b>{name}</b><br>Тип: {poi_type}"
                            if cat_label:
                                desc += f"<br>Категорія: {cat_label}"

                            folium.Marker(
                                location=[float(p.y), float(p.x)],
                                tooltip=name,
                                popup=folium.Popup(desc, max_width=300),
                            ).add_to(cluster)

                        self.status.emit(f"POI в ізохроні {minutes} хв: {len(inside)} (без house)")
            except Exception as e:
                logger.warning("Iso POI layer failed: %s", e)

    # -------------------- POI --------------------

    def _add_poi_layers(self, m: folium.Map) -> None:
        level_cfg = LEVELS_QUERIES.get(self.level_name, {}) or {}
        if not level_cfg:
            return

        for category, tags_dict in level_cfg.items():
            try:
                gdf = self._subset_for_tags(self.gdf_all_poi, tags_dict)
                if gdf is None or gdf.empty:
                    continue

                gdf = self._ensure_gdf_crs(gdf)

                try:
                    if len(gdf) > MAX_POI_PER_CATEGORY:
                        gdf = gdf.sample(n=MAX_POI_PER_CATEGORY, random_state=42)
                except Exception:
                    try:
                        gdf = gdf.head(MAX_POI_PER_CATEGORY)
                    except Exception:
                        pass

                layer_name = f"POI — {LABEL_MAP.get(category, category)} [{self.level_name}]"
                fg = folium.FeatureGroup(name=layer_name, show=False)
                cluster = MarkerCluster().add_to(fg)

                for _, row in gdf.iterrows():
                    geom = self._geom_to_point(row.get("geometry"))
                    if geom is None:
                        continue

                    name = str(row.get("name") or row.get("brand") or "Без назви")
                    poi_type = (
                        row.get("poi_value")
                        or row.get("amenity")
                        or row.get("shop")
                        or row.get("leisure")
                        or row.get("tourism")
                        or row.get("public_transport")
                        or row.get("railway")
                        or "poi"
                    )

                    addr = ", ".join(filter(None, [row.get("addr:street"), row.get("addr:housenumber")]))
                    desc = f"<b>{name}</b><br>Тип: {poi_type}"
                    if addr:
                        desc += f"<br>Адреса: {addr}"

                    folium.Marker(
                        location=[float(geom.y), float(geom.x)],
                        tooltip=name,
                        popup=folium.Popup(desc, max_width=300),
                    ).add_to(cluster)

                fg.add_to(m)
            except Exception as e:
                logger.warning("POI '%s': не вдалося відрендерити: %s", category, e)

    # -------------------- QtWebChannel click-pick injection --------------------

    @staticmethod
    def _read_qwebchannel_js() -> str:
        qf = QFile(":/qtwebchannel/qwebchannel.js")
        if not qf.open(QIODevice.ReadOnly):
            logger.warning("InjectJS: не можу відкрити :/qtwebchannel/qwebchannel.js")
            return ""
        try:
            return bytes(qf.readAll()).decode("utf-8", errors="ignore")
        finally:
            try:
                qf.close()
            except Exception:
                pass

    def _inject_click_bridge_js(self, map_file: str) -> None:
        if not self.enable_click_pick:
            return

        try:
            with open(map_file, "r", encoding="utf-8") as f:
                html = f.read()

            if _INJECT_MARKER in html:
                return

            m = re.search(r"var\s+(map_[A-Za-z0-9_]+)\s*=\s*L\.map", html)
            if not m:
                logger.warning("InjectJS: не знайшов змінну карти у HTML")
                return

            map_var = m.group(1)

            bridge_path = os.path.join(os.path.dirname(__file__), "map_bridge.js")
            try:
                with open(bridge_path, "r", encoding="utf-8") as f:
                    bridge_js = f.read()
            except Exception as e:
                logger.error("InjectJS: не вдалося прочитати map_bridge.js (%s): %s", bridge_path, e)
                return

            qwebchannel_js = self._read_qwebchannel_js()
            if qwebchannel_js:
                qwebchannel_tag = f"<script>\n{qwebchannel_js}\n</script>"
            else:
                qwebchannel_tag = '<script src="qrc:///qtwebchannel/qwebchannel.js"></script>'

            inject = f"""
<!-- {_INJECT_MARKER} -->
{qwebchannel_tag}

<script>
  window.__folium_map__ = {map_var};
</script>

<script>
{bridge_js}
</script>
"""

            if "</body>" in html:
                html = html.replace("</body>", inject + "\n</body>")
            else:
                html += inject

            with open(map_file, "w", encoding="utf-8") as f:
                f.write(html)

            logger.info("InjectJS: підключено map_bridge.js → %s", map_file)

        except Exception as e:
            logger.exception("InjectJS помилка: %s", e)

    # -------------------- main thread work --------------------

    def run(self):
        try:
            self._ensure_outputs_dir()

            self.progress.emit(5)
            self.status.emit(f"Рендер рівня '{self.level_name}'…")

            center_latlon = self._compute_center_latlon()

            m = folium.Map(
                location=[center_latlon[0], center_latlon[1]],
                zoom_start=13,
                tiles="cartodbpositron",
                control_scale=True,
            )

            self.progress.emit(20)

            self._add_route_layer(m)
            self.progress.emit(24)

            self._add_accessibility_citywide_layers(m)
            self.progress.emit(40)

            self._add_accessibility_layer(m)
            self.progress.emit(50)

            self._add_isochrone_layers(m)
            self.progress.emit(62)

            self._add_streets_layer(m, show=False)
            self.progress.emit(72)

            self._add_boundary_layer(m)
            self.progress.emit(80)

            self.status.emit("POI: рендер…")
            self._add_poi_layers(m)
            self.progress.emit(92)

            self._fit_map_to_city(m)

            folium.LayerControl(collapsed=False).add_to(m)

            ts = int(time.time() * 1000)
            map_file = os.path.join(DIR_OUTPUTS, f"{self.safe_city_name}_map_{ts}.html")
            m.save(map_file)

            self._inject_click_bridge_js(map_file)

            self.progress.emit(100)
            self.status.emit(f"Карта збережена: {map_file}")
            self.finished.emit(map_file)

        except Exception as e:
            logger.exception("MapRenderWorker помилка: %s", e)
            self.error.emit(f"Помилка рендеру карти: {e}")