# visualizer.py
from __future__ import annotations

import os
import re
import time
from typing import List, Optional, Tuple, Dict, Any

import folium
import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
from folium.plugins import MarkerCluster
from shapely.geometry import Point, LineString
from shapely.ops import unary_union

from PyQt5.QtCore import QThread, pyqtSignal, QFile, QIODevice

from api import load_city_boundary
from config import LEVELS_QUERIES
from logger_config import logger
from path_finder import (
    PathfinderError,
    snap_to_graph,
    find_shortest_path,
    find_shortest_path_multi,
    find_shortest_path_via_poi_category,
    compute_isochrone,
)
from paths import DIR_OUTPUTS
from utils import extract_required_tokens

LatLon = Tuple[float, float]

# -------------------- Налаштування рендера --------------------

DEFAULT_CENTER: LatLon = (49.0, 24.0)

MAX_POI_PER_CATEGORY = 6000

# Ізохрони: щоб полігон виглядав нормально навіть у “тонких” графах
ISO_EDGE_BUFFER_M_MIN = 35.0        # мінімальний буфер навколо reachable-ребер (м)
ISO_EDGE_BUFFER_M_DEFAULT = 55.0    # стандартний буфер (м)
ISO_POLY_SIMPLIFY_M = 8.0           # спрощення полігону в метрах
ISO_MAX_POI_IN_ISO_LAYER = 1200     # обмеження POI для шару “POI в ізохроні”

LABEL_MAP = {
    "education": "Освіта",
    "health": "Медицина",
    "culture": "Культура",
    "greens_sport": "Зелена інфра / Спорт",
    "shopping_services": "Покупки / Сервіси",
    "transport": "Громадський транспорт",
}

# Маркер, щоб не інжектити JS двічі
_INJECT_MARKER = "injected: qtwebchannel + map_bridge.js"


class MapRenderWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(str)  # map_file_path
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
        # -------- ІЗОХРОНИ --------
        isochrone_center_latlon: Optional[LatLon] = None,
        isochrone_minutes: Optional[List[int]] = None,
        isochrone_walk_speed_kmh: float = 4.8,
        isochrone_buffer_m: float = 0.0,
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

        # -------- ІЗОХРОНИ --------
        self.isochrone_center_latlon = isochrone_center_latlon
        self.isochrone_minutes = list(isochrone_minutes or [])
        self.isochrone_walk_speed_kmh = float(isochrone_walk_speed_kmh)
        self.isochrone_buffer_m = float(isochrone_buffer_m)

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
        """
        POI тільки з LEVELS_QUERIES[self.level_name], без шуму.
        Окремо вирізаємо building=house, щоб не засмічувало POI в ізохроні.
        """
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
                style_function=lambda _: {"color": "#4a4a4a", "weight": 2, "opacity": 0.55},
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
                style_function=lambda _: {"color": "#d9534f", "weight": 3, "fill": False, "opacity": 0.9},
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

        # 1) SNAP debug (start/end)
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
                opacity=0.85,
                dash_array="6,6",
                tooltip="Старт: клік → дорога",
            ).add_to(fg)

            folium.PolyLine(
                locations=[snap_e.input_latlon, snap_e.snapped_latlon],
                weight=2,
                opacity=0.85,
                dash_array="6,6",
                tooltip="Фініш: клік → дорога",
            ).add_to(fg)

        except Exception as e:
            logger.warning("Route SNAP debug: не вдалося порахувати/намалювати snap: %s", e)
            folium.Marker(location=list(self.route_start_latlon), tooltip="Старт").add_to(fg)
            folium.Marker(location=list(self.route_end_latlon), tooltip="Фініш").add_to(fg)

        # Stop markers
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
                    algorithm=self.route_algorithm,  # type: ignore
                    weight="length",
                    use_undirected=True,
                    snap_mode="edge",
                )
                result_coords = mp.coords
                result_len_m = float(mp.length_m)
                self.status.emit(f"Маршрут multi-stop: сегментів={len(mp.segments)}, довжина~{result_len_m:.0f} м")
            else:
                if via and via.lower() != "none":
                    # Ліміт: +10 хв від найкоротшого (у метрах через walking speed)
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
                            algorithm=self.route_algorithm,  # type: ignore
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
                        # Fallback: показуємо найкоротший маршрут + “людське” пояснення
                        r0 = find_shortest_path(
                            self.G,
                            self.route_start_latlon,
                            self.route_end_latlon,
                            algorithm=self.route_algorithm,  # type: ignore
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
                        algorithm=self.route_algorithm,  # type: ignore
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

            poly_wgs = gpd.GeoSeries([poly], crs=3857).to_crs(epsg=4326).iloc[0]
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
                    name=f"Ізохрона {minutes} хв (полігон)",
                    show=True,
                    style_function=lambda _feat, c=color: {
                        "color": c,
                        "weight": 3,
                        "fill": True,
                        "fillColor": c,
                        "fillOpacity": 0.22,
                        "opacity": 0.95,
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
                        name=f"Ізохрона {minutes} хв — дороги",
                        show=True,
                        style_function=lambda _feat, c=color: {
                            "color": c,
                            "weight": 2,
                            "opacity": 0.85,
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
            self.progress.emit(35)

            self._add_isochrone_layers(m)
            self.progress.emit(50)

            self._add_streets_layer(m, show=False)
            self.progress.emit(65)

            self._add_boundary_layer(m)
            self.progress.emit(75)

            self.status.emit("POI: рендер…")
            self._add_poi_layers(m)
            self.progress.emit(92)

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
