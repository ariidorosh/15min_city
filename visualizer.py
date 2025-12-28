# visualizer.py
from __future__ import annotations

import os
import re
import time
from typing import List, Optional, Tuple

import folium
import osmnx as ox
from folium.plugins import MarkerCluster
from shapely.geometry import Point

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
)
from paths import DIR_OUTPUTS
from utils import extract_required_tokens

LatLon = Tuple[float, float]

# -------------------- Налаштування рендера --------------------

DEFAULT_CENTER: LatLon = (49.0, 24.0)

MAX_POI_PER_CATEGORY = 6000

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

    # -------------------- helpers: geo/poi --------------------

    @staticmethod
    def _ensure_outputs_dir() -> None:
        try:
            os.makedirs(DIR_OUTPUTS, exist_ok=True)
        except Exception as e:
            logger.warning("DIR_OUTPUTS не створено (%s): %s", DIR_OUTPUTS, e)

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

            folium.GeoJson(
                data=self.gdf_edges[["geometry"]].to_json(),
                name="Вулиці",
                show=bool(show),
                style_function=lambda _: {"color": "#4a4a4a", "weight": 2, "opacity": 0.7},
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

    def _add_route_layer(self, m: folium.Map) -> None:
        """
        Рендер маршруту:
          - simple: start->end
          - via category (1 stop): start->best_poi->end
          - multistop: start->stop1->...->end
        + SNAP debug “клік → точка на дорозі”
        """
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

        # 1.1) Stop markers (якщо вони є)
        if self.route_stops:
            for i, st in enumerate(self.route_stops, start=1):
                folium.Marker(
                    location=[st[0], st[1]],
                    tooltip=f"Stop #{i}",
                    popup=f"Stop #{i}: {self._fmt_ll(st)}",
                    icon=folium.Icon(icon="info-sign"),
                ).add_to(fg)

        # 2) Сам маршрут
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
            result_start_node = None
            result_end_node = None

            # Пріоритет: якщо є stop'и — робимо multistop
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

                # для дебагу “вузли”
                if mp.segments:
                    result_start_node = mp.segments[0].start_node
                    result_end_node = mp.segments[-1].end_node

                self.status.emit(f"Маршрут multi-stop: сегментів={len(mp.segments)}, довжина~{result_len_m:.0f} м")

            # Якщо stop'ів нема — тоді або via-category, або звичайний
            else:
                if via and via.lower() != "none":
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
                    result_start_node = r.start_node
                    result_end_node = r.end_node
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
                    result_start_node = r.start_node
                    result_end_node = r.end_node

            # Полілінія маршруту
            if result_coords:
                folium.PolyLine(
                    locations=result_coords,
                    weight=5,
                    opacity=0.9,
                    tooltip=f"Довжина: {result_len_m:.0f} м",
                ).add_to(fg)

            # Маркери вузлів графа, які обрав алгоритм (якщо є)
            if result_start_node is not None and result_end_node is not None:
                try:
                    n1 = int(result_start_node)
                    n2 = int(result_end_node)

                    d1 = self.G.nodes[n1]
                    d2 = self.G.nodes[n2]

                    n1_ll = (float(d1["y"]), float(d1["x"]))
                    n2_ll = (float(d2["y"]), float(d2["x"]))

                    folium.CircleMarker(
                        location=list(n1_ll),
                        radius=5,
                        weight=2,
                        tooltip=f"Старт (вузол графа {n1})",
                        popup=f"Старт вузол: {n1}<br>{self._fmt_ll(n1_ll)}",
                    ).add_to(fg)

                    folium.CircleMarker(
                        location=list(n2_ll),
                        radius=5,
                        weight=2,
                        tooltip=f"Фініш (вузол графа {n2})",
                        popup=f"Фініш вузол: {n2}<br>{self._fmt_ll(n2_ll)}",
                    ).add_to(fg)

                    self.status.emit(f"Маршрут: вузли графа START={n1}, END={n2}")
                except Exception:
                    pass

            self.status.emit(f"Маршрут: готово (довжина ~ {result_len_m:.0f} м)")

        except PathfinderError as e:
            logger.warning("Маршрут не побудовано: %s", e)
            self.status.emit(f"Маршрут: не побудовано — {e}")
        except Exception as e:
            logger.exception("Маршрут: помилка: %s", e)
            self.status.emit(f"Маршрут: помилка — {e}")

    def _add_poi_layers(self, m: folium.Map) -> None:
        level_cfg = LEVELS_QUERIES.get(self.level_name, {}) or {}
        if not level_cfg:
            return

        for category, tags_dict in level_cfg.items():
            try:
                gdf = self._subset_for_tags(self.gdf_all_poi, tags_dict)
                if gdf is None or gdf.empty:
                    continue

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
        """
        Надійно читає qwebchannel.js з Qt resources і повертає як текст.
        Якщо не вдалось — повертає "".
        """
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
        """
        Підключає QtWebChannel + map_bridge.js (вмістом)
        і передає folium-map у window.__folium_map__.

        Важливо:
        - qwebchannel.js інлайнимо прямо в HTML, щоб не залежати від qrc:/// у file:// режимі.
        """
        if not self.enable_click_pick:
            return

        try:
            with open(map_file, "r", encoding="utf-8") as f:
                html = f.read()

            # якщо вже інжектовано — вдруге не ліземо
            if _INJECT_MARKER in html:
                return

            # шукаємо змінну folium-карти (map_xxxxx)
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
                # fallback: якщо ресурс недоступний
                qwebchannel_tag = '<script src="qrc:///qtwebchannel/qwebchannel.js"></script>'

            inject = f"""
<!-- {_INJECT_MARKER} -->
{qwebchannel_tag}

<script>
  // передаємо folium map у JS
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

            show_streets = bool(self.route_start_latlon and self.route_end_latlon)
            self._add_streets_layer(m, show=show_streets)
            self.progress.emit(50)

            self._add_boundary_layer(m)
            self.progress.emit(60)

            self.status.emit("POI: рендер…")
            self._add_poi_layers(m)
            self.progress.emit(85)

            folium.LayerControl(collapsed=False).add_to(m)

            ts = int(time.time() * 1000)
            map_file = os.path.join(DIR_OUTPUTS, f"{self.safe_city_name}_map_{ts}.html")
            m.save(map_file)

            # інжектимо міст для кліків (якщо увімкнено)
            self._inject_click_bridge_js(map_file)

            self.progress.emit(100)
            self.status.emit(f"Карта збережена: {map_file}")
            self.finished.emit(map_file)

        except Exception as e:
            logger.exception("MapRenderWorker помилка: %s", e)
            self.error.emit(f"Помилка рендеру карти: {e}")
