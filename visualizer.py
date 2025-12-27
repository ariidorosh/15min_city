# visualizer.py
from __future__ import annotations

import os
import re
import time
from typing import Optional, Tuple

import folium
import osmnx as ox
from folium.plugins import MarkerCluster
from shapely.geometry import Point

from PyQt5.QtCore import QThread, pyqtSignal

from api import load_city_boundary
from config import LEVELS_QUERIES
from logger_config import logger
from path_finder import (
    PathfinderError,
    snap_to_graph,
    find_shortest_path,
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
        self.route_via_category = (route_via_category or "none").strip().lower()
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

    def _add_route_layer(self, m: folium.Map) -> None:
        """
        Рендер маршруту + дебаг “клік → точка на дорозі”.
        """
        if not (self.route_start_latlon and self.route_end_latlon):
            return

        fg = folium.FeatureGroup(name="Маршрут", show=True)
        fg.add_to(m)

        # 1) SNAP дебаг (клік → дорога)
        try:
            Gwork = self.G.to_undirected()

            snap_s = snap_to_graph(Gwork, self.route_start_latlon, mode="edge")
            snap_e = snap_to_graph(Gwork, self.route_end_latlon, mode="edge")

            self.status.emit(
                "SNAP START: "
                f"клік {snap_s.input_latlon[0]:.6f},{snap_s.input_latlon[1]:.6f} → "
                f"дорога {snap_s.snapped_latlon[0]:.6f},{snap_s.snapped_latlon[1]:.6f} "
                f"({snap_s.dist_to_snapped_m:.0f} м)"
            )
            self.status.emit(
                "SNAP END:   "
                f"клік {snap_e.input_latlon[0]:.6f},{snap_e.input_latlon[1]:.6f} → "
                f"дорога {snap_e.snapped_latlon[0]:.6f},{snap_e.snapped_latlon[1]:.6f} "
                f"({snap_e.dist_to_snapped_m:.0f} м)"
            )

            folium.Marker(
                location=list(snap_s.input_latlon),
                tooltip="Старт (клік/будинок)",
                popup=f"Старт (клік): {snap_s.input_latlon[0]:.6f},{snap_s.input_latlon[1]:.6f}",
            ).add_to(fg)

            folium.Marker(
                location=list(snap_e.input_latlon),
                tooltip="Фініш (клік/будинок)",
                popup=f"Фініш (клік): {snap_e.input_latlon[0]:.6f},{snap_e.input_latlon[1]:.6f}",
            ).add_to(fg)

            folium.CircleMarker(
                location=list(snap_s.snapped_latlon),
                radius=6,
                weight=2,
                tooltip="Старт (точка на дорозі)",
                popup=(
                    "Старт (на дорозі): "
                    f"{snap_s.snapped_latlon[0]:.6f},{snap_s.snapped_latlon[1]:.6f}<br>"
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
                    f"{snap_e.snapped_latlon[0]:.6f},{snap_e.snapped_latlon[1]:.6f}<br>"
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

        # 2) Сам маршрут
        via = (self.route_via_category or "none").strip().lower()
        self.status.emit(
            f"Маршрут: START {self.route_start_latlon[0]:.6f},{self.route_start_latlon[1]:.6f} "
            f"→ END {self.route_end_latlon[0]:.6f},{self.route_end_latlon[1]:.6f} | alg={self.route_algorithm} | via={via}"
        )

        try:
            self.status.emit("Маршрут: обчислення…")

            if via and via != "none":
                result, sel = find_shortest_path_via_poi_category(
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
                )

                folium.Marker(
                    location=[sel.poi_latlon[0], sel.poi_latlon[1]],
                    tooltip=f"Зупинка: {sel.category}",
                    popup=(
                        f"<b>Зупинка:</b> {sel.category}<br>"
                        f"<b>Обрано:</b> {sel.label}<br>"
                        f"<b>Start→POI:</b> ~{sel.dist_start_to_poi_m:.0f} м<br>"
                        f"<b>POI→End:</b> ~{sel.dist_poi_to_end_m:.0f} м<br>"
                        f"<b>Разом:</b> ~{sel.total_m:.0f} м"
                    ),
                ).add_to(fg)

            else:
                result = find_shortest_path(
                    self.G,
                    self.route_start_latlon,
                    self.route_end_latlon,
                    algorithm=self.route_algorithm,
                    weight="length",
                    use_undirected=True,
                    snap_mode="edge",
                )

            if getattr(result, "coords", None):
                folium.PolyLine(
                    locations=result.coords,
                    weight=5,
                    opacity=0.9,
                    tooltip=f"Довжина: {getattr(result, 'length_m', 0.0):.0f} м",
                ).add_to(fg)

            # Маркери вузлів графа, які обрав алгоритм
            try:
                n1 = int(getattr(result, "start_node", 0))
                n2 = int(getattr(result, "end_node", 0))

                d1 = self.G.nodes[n1]
                d2 = self.G.nodes[n2]

                n1_ll = (float(d1["y"]), float(d1["x"]))
                n2_ll = (float(d2["y"]), float(d2["x"]))

                folium.CircleMarker(
                    location=list(n1_ll),
                    radius=5,
                    weight=2,
                    tooltip=f"Старт (вузол графа {n1})",
                    popup=f"Старт вузол: {n1}<br>{n1_ll[0]:.6f},{n1_ll[1]:.6f}",
                ).add_to(fg)

                folium.CircleMarker(
                    location=list(n2_ll),
                    radius=5,
                    weight=2,
                    tooltip=f"Фініш (вузол графа {n2})",
                    popup=f"Фініш вузол: {n2}<br>{n2_ll[0]:.6f},{n2_ll[1]:.6f}",
                ).add_to(fg)

                self.status.emit(f"Маршрут: вузли графа START={n1}, END={n2}")
            except Exception:
                pass

            self.status.emit(f"Маршрут: готово (довжина ~ {getattr(result, 'length_m', 0.0):.0f} м)")

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

    def _inject_click_bridge_js(self, map_file: str) -> None:
        if not self.enable_click_pick:
            return

        try:
            with open(map_file, "r", encoding="utf-8") as f:
                html = f.read()

            if "WebChannel bridge ready" in html and "window.__pick_target" in html:
                return

            m = re.search(r"var\s+(map_[A-Za-z0-9_]+)\s*=\s*L\.map", html)
            if not m:
                logger.warning("InjectJS: не знайшов змінну карти у HTML")
                return

            map_var = m.group(1)

            js = f"""
<!-- injected: qtwebchannel click-pick -->
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<script>
(function() {{
  function initBridge() {{
    if (typeof qt === 'undefined') {{
      console.log('qt is undefined (no webchannel?)');
      return;
    }}

    new QWebChannel(qt.webChannelTransport, function(channel) {{
      window.bridge = channel.objects.bridge;
      console.log('WebChannel bridge ready');
    }});

    var map = {map_var};
    window.__pick_target = window.__pick_target || '';

    function setMarker(kind, lat, lon) {{
      try {{
        if (kind === 'start') {{
          if (window.__startMarker) map.removeLayer(window.__startMarker);
          window.__startMarker = L.marker([lat, lon]).addTo(map).bindPopup('Start');
        }} else if (kind === 'end') {{
          if (window.__endMarker) map.removeLayer(window.__endMarker);
          window.__endMarker = L.marker([lat, lon]).addTo(map).bindPopup('End');
        }}
      }} catch (e) {{
        console.log('marker error', e);
      }}
    }}

    map.on('click', function(e) {{
      var lat = e.latlng.lat;
      var lon = e.latlng.lng;

      var tgt = window.__pick_target || '';
      if (!tgt) return;

      setMarker(tgt, lat, lon);

      if (window.bridge && window.bridge.map_clicked) {{
        window.bridge.map_clicked(lat, lon);
      }}
    }});
  }}

  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', initBridge);
  }} else {{
    initBridge();
  }}
}})();
</script>
"""
            if "</body>" in html:
                html = html.replace("</body>", js + "\n</body>")
            else:
                html += js

            with open(map_file, "w", encoding="utf-8") as f:
                f.write(html)

            logger.info("InjectJS: вставив webchannel click handler у %s", map_file)

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

            self._inject_click_bridge_js(map_file)

            self.progress.emit(100)
            self.status.emit(f"Карта збережена: {map_file}")
            self.finished.emit(map_file)

        except Exception as e:
            logger.exception("MapRenderWorker помилка: %s", e)
            self.error.emit(f"Помилка рендеру карти: {e}")
