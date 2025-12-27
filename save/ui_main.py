import os
import re
import glob
import json
import math
import time


from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QScrollArea, QComboBox, QLineEdit, QToolButton, QSizePolicy,
    QProgressBar, QMessageBox
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt, QUrl, QThread, pyqtSignal, QObject, pyqtSlot

# QtWebChannel може бути не встановлений у деяких збірках.
WEBCHANNEL_AVAILABLE = True
try:
    from PyQt5.QtWebChannel import QWebChannel
except Exception:
    WEBCHANNEL_AVAILABLE = False

import folium
import osmnx as ox
from shapely.geometry import Point
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

from api import load_city_graph, load_poi, load_city_boundary, build_route

from folium.plugins import MarkerCluster
from config import LEVELS_QUERIES
from logger_config import logger

from path_finder import PathfinderError


DATA_ROOT = "data"
DIR_POI_ALL = os.path.join(DATA_ROOT, "poi", "all")
DIR_BOUNDARIES = os.path.join(DATA_ROOT, "boundaries")
DIR_GRAPHS = os.path.join(DATA_ROOT, "graphs")

META_DIR = os.path.join("data", "meta")
os.makedirs(META_DIR, exist_ok=True)
CACHE_FILE = os.path.join(META_DIR, "cities.json")
os.makedirs("outputs", exist_ok=True)

COUNTRY_MAP = {
    "Україна": ("ua", "uk"),
    "Польща": ("pl", "pl"),
}


# ----------------------------
# Утиліти
# ----------------------------
def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-zа-яіїєґ0-9]+", "_", s)
    return s.strip("_")


def _extract_required_tokens(full_place: str):
    parts = [p.strip() for p in full_place.split(",")]
    city_tok = _norm(parts[0]) if parts else ""
    region_tok = ""
    for p in parts[1:]:
        if re.search(r"область|oblast|voivodeship|місто київ|city of kyiv", p, flags=re.I):
            region_tok = _norm(p)
            break
    toks = [t for t in (city_tok, region_tok) if t]
    return toks


def load_cached_cities():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("cities", [])
        except Exception:
            return []
    return []


def save_cached_cities(cities):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({"cities": cities}, f, ensure_ascii=False, indent=2)


def _unsafename_to_place(safe_base: str) -> str:
    return safe_base.replace("_", " ").strip()


def discover_cached_cities() -> list:
    found = set()
    for p in glob.glob(os.path.join(DIR_POI_ALL, "*_poi_all.geojson")):
        base = os.path.basename(p)
        place_safe = re.sub(r"_poi_all\.geojson$", "", base)
        found.add(_unsafename_to_place(place_safe))
    for p in glob.glob(os.path.join(DIR_BOUNDARIES, "*_boundary.geojson")):
        base = os.path.basename(p)
        place_safe = re.sub(r"_boundary\.geojson$", "", base)
        found.add(_unsafename_to_place(place_safe))
    for p in glob.glob(os.path.join(DIR_GRAPHS, "*.graphml")):
        base = os.path.basename(p)
        place_safe = re.sub(r"\.graphml$", "", base)
        found.add(_unsafename_to_place(place_safe))
    return sorted({c for c in found if c})


def parse_latlon_text(text: str, name: str) -> tuple:
    """
    Приймає "lat, lon" або "lat lon".
    Повертає (lat, lon) як float.
    """
    s = (text or "").strip()
    if not s:
        raise ValueError(f"{name}: порожнє значення")

    parts = re.split(r"[,\s]+", s)
    parts = [p for p in parts if p]
    if len(parts) != 2:
        raise ValueError(f"{name}: формат має бути 'lat, lon' (наприклад 49.8397, 24.0297)")

    lat = float(parts[0])
    lon = float(parts[1])

    if math.isnan(lat) or math.isnan(lon):
        raise ValueError(f"{name}: координати не можуть бути NaN")

    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        raise ValueError(f"{name}: координати поза діапазоном (-90..90, -180..180)")

    return (lat, lon)


# ----------------------------
# WebChannel bridge (клік по карті -> Python)
# ----------------------------
class MapBridge(QObject):
    picked = pyqtSignal(float, float)  # (lat, lon)

    @pyqtSlot(float, float)
    def map_clicked(self, lat, lon):
        self.picked.emit(float(lat), float(lon))


# ----------------------------
# Workers
# ----------------------------
class CitySearchWorker(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, country_code: str, language: str, query: str):
        super().__init__()
        self.country_code = country_code
        self.language = language
        self.query = query

    def run(self):
        try:
            self.status.emit(f"Пошук міст у країні [{self.country_code}]…")
            geolocator = Nominatim(user_agent="15min_city_app", timeout=10)
            results = geolocator.geocode(
                self.query,
                exactly_one=False,
                limit=10,
                addressdetails=True,
                country_codes=self.country_code,
                language=self.language
            )
            items = [r.address for r in results or []]
            if not items:
                self.error.emit("Не знайдено жодного міста у вибраній країні")
                logger.info("CitySearch: 0 результатів для '%s' (cc=%s)", self.query, self.country_code)
            else:
                self.status.emit(f"Знайдено {len(items)} міст(а) у [{self.country_code}]")
                logger.info("CitySearch: знайдено %d результатів для '%s' (cc=%s)", len(items), self.query,
                            self.country_code)
                self.finished.emit(items)
        except (GeocoderTimedOut, GeocoderUnavailable):
            logger.exception("CitySearch: сервіс геокодування тимчасово недоступний")
            self.error.emit("Сервіс геокодування тимчасово недоступний. Спробуйте ще раз.")
        except Exception as e:
            logger.exception("CitySearch помилка для '%s': %s", self.query, e)
            self.error.emit(f"Помилка пошуку: {e}")


class AddressSearchWorker(QThread):
    finished = pyqtSignal(list)  # list of dicts: {label, lat, lon}
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, country_code: str, language: str, city_name: str, query: str):
        super().__init__()
        self.country_code = country_code
        self.language = language
        self.city_name = city_name
        self.query = query

    def run(self):
        try:
            q = (self.query or "").strip()
            if not q:
                self.error.emit("Введіть вулицю/будинок для пошуку.")
                return

            full_query = f"{q}, {self.city_name}"
            self.status.emit("Адреса: шукаю варіанти…")
            geolocator = Nominatim(user_agent="15min_city_app", timeout=10)
            results = geolocator.geocode(
                full_query,
                exactly_one=False,
                limit=8,
                addressdetails=True,
                country_codes=self.country_code,
                language=self.language
            )

            out = []
            for r in results or []:
                try:
                    out.append({
                        "label": r.address,
                        "lat": float(r.latitude),
                        "lon": float(r.longitude),
                    })
                except Exception:
                    continue

            if not out:
                self.error.emit("За цією адресою нічого не знайдено. Спробуйте інший формат (наприклад 'Shevchenka 10').")
                return

            self.status.emit(f"Адреса: знайдено {len(out)} варіант(ів)")
            self.finished.emit(out)

        except (GeocoderTimedOut, GeocoderUnavailable):
            self.error.emit("Сервіс геокодування тимчасово недоступний. Спробуйте ще раз.")
        except Exception as e:
            logger.exception("AddressSearchWorker помилка: %s", e)
            self.error.emit(f"Помилка пошуку адреси: {e}")


class GraphWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(object, object, str, str)  # (G, gdf_edges, safe_city_name, city_name)
    error = pyqtSignal(str)

    def __init__(self, city_name: str, required_tokens: list):
        super().__init__()
        self.city_name = city_name
        self.required_tokens = required_tokens

    def run(self):
        try:
            self.progress.emit(10)
            self.status.emit("Граф доріжок: пошук кешу…")

            G, info = load_city_graph(
                self.city_name,
                network_type="walk",
                required_tokens=self.required_tokens,
                force_refresh=False,
            )

            source = info.get("source")
            cache_action = info.get("cache_action")
            used = info.get("used_path") or "—"
            expected = info.get("expected_path") or "—"
            candidates = info.get("candidates", []) or []

            if source == "cache":
                msg = f"Граф: кеш → {used}"
                if cache_action == "copied":
                    msg += f" (скопійовано у {expected})"
                elif cache_action == "fallback_read":
                    msg += " (читання напряму без копії)"
                elif cache_action and cache_action != "exists":
                    msg += f" (cache_action={cache_action})"
            else:
                cand_str = ", ".join(candidates[:3]) or "—"
                msg = f"Граф: OSM → {used} (кеш: {expected}, кандидати: {cand_str})"

            self.status.emit(msg)
            logger.info("GraphWorker source: %s", msg)

            self.progress.emit(70)
            gdf_edges = ox.utils_graph.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
            self.progress.emit(100)

            safe_city_name = self.city_name.lower().replace(",", "").replace(" ", "_")
            self.finished.emit(G, gdf_edges, safe_city_name, self.city_name)
        except Exception as e:
            logger.exception("GraphWorker помилка: %s", e)
            self.error.emit(f"Помилка побудови графа: {e}")


class AllPOIWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(object)  # gdf_all_poi
    error = pyqtSignal(str)

    def __init__(self, city_name: str, required_tokens: list):
        super().__init__()
        self.city_name = city_name
        self.required_tokens = required_tokens

    def run(self):
        try:
            self.progress.emit(10)
            self.status.emit("POI: перевіряю кеш…")

            gdf, info = load_poi(
                self.city_name,
                required_tokens=self.required_tokens,
                force_refresh=False,
                split_by_topkey=False,
            )

            source = info.get("source")
            cache_action = info.get("cache_action")
            used = info.get("used_path") or "—"
            expected = info.get("expected_path") or "—"
            candidates = info.get("candidates", []) or []

            if source == "cache":
                self.status.emit(f"POI: кеш → {used} (записів: {len(gdf)})")
            else:
                cand_str = ", ".join(candidates[:3]) or "—"
                extra = f" (cache_action={cache_action})" if cache_action else ""
                self.status.emit(
                    f"POI: OSM → {used}{extra}; кеш: {expected}; кандидати: {cand_str} (записів: {len(gdf)})"
                )

            self.progress.emit(100)
            self.finished.emit(gdf)
        except Exception as e:
            logger.exception("AllPOIWorker помилка: %s", e)
            self.error.emit(f"Помилка завантаження POI: {e}")


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
        route_start_latlon=None,
        route_end_latlon=None,
        route_algorithm: str = "dijkstra",
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
        self.route_algorithm = route_algorithm
        self.enable_click_pick = enable_click_pick and WEBCHANNEL_AVAILABLE

    @staticmethod
    def _subset_for_tags(gdf_all, tags_dict):
        if gdf_all is None or gdf_all.empty:
            return gdf_all
        if "poi_key" not in gdf_all.columns or "poi_value" not in gdf_all.columns:
            mask_total = None
            for key, vals in tags_dict.items():
                if key in gdf_all.columns:
                    m = gdf_all[key].astype(str).isin([str(v) for v in vals])
                    mask_total = m if mask_total is None else (mask_total | m)
            return gdf_all[mask_total] if mask_total is not None else gdf_all.iloc[0:0]

        mask_total = None
        for key, vals in tags_dict.items():
            m = (gdf_all["poi_key"] == key) & (gdf_all["poi_value"].isin([str(v) for v in vals]))
            mask_total = m if mask_total is None else (mask_total | m)
        return gdf_all[mask_total] if mask_total is not None else gdf_all.iloc[0:0]

    def _inject_click_bridge_js(self, map_file: str) -> None:
        """
        Додає JS у збережений folium HTML:
        - підключає qwebchannel.js
        - ловить кліки по карті
        - відправляє координати в Python через bridge.map_clicked(lat, lon)
        """
        if not self.enable_click_pick:
            return

        try:
            with open(map_file, "r", encoding="utf-8") as f:
                html = f.read()

            # знайдемо ім'я змінної карти folium: var map_xxxxx = L.map(...)
            m = re.search(r"var\s+(map_[A-Za-z0-9_]+)\s*=\s*L\.map", html)
            if not m:
                logger.warning("InjectJS: не знайшов змінну карти у HTML")
                return
            map_var = m.group(1)

            js = f"""
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
    window.__pick_target = window.__pick_target || ''; // 'start' або 'end' або ''

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
      if (!tgt) {{
        // якщо не вибрано режим — просто ігноруємо (можна змінити логіку)
        return;
      }}

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

    def run(self):
        try:
            self.progress.emit(10)
            self.status.emit(f"Рендер рівня '{self.level_name}'…")

            # Центр карти
            try:
                center = self.gdf_edges.geometry.unary_union.centroid
                center_latlon = [center.y, center.x]
            except Exception:
                self.status.emit("Попередження: не вдалося визначити центр по ребрах — спробую по boundary")
                b_gdf, _ = load_city_boundary(
                    self.city_name, required_tokens=_extract_required_tokens(self.city_name)
                )
                if b_gdf is not None and not b_gdf.empty:
                    c = b_gdf.geometry.unary_union.representative_point()
                    center_latlon = [c.y, c.x]
                else:
                    center_latlon = [49.0, 24.0]

            m = folium.Map(location=center_latlon, zoom_start=13,
                           tiles="cartodbpositron", control_scale=True)
            self.progress.emit(25)

            # Маршрут (якщо задано)
            if self.route_start_latlon and self.route_end_latlon:
                try:
                    self.status.emit("Маршрут: обчислення…")

                    result = build_route(
                        self.G,
                        start=self.route_start_latlon,
                        end=self.route_end_latlon,
                        algorithm=self.route_algorithm,
                        weight="length",
                    )

                    folium.Marker(location=list(self.route_start_latlon), tooltip="Start", popup="Start").add_to(m)
                    folium.Marker(location=list(self.route_end_latlon), tooltip="End", popup="End").add_to(m)

                    folium.PolyLine(
                        locations=result.coords,
                        weight=5,
                        opacity=0.9,
                        tooltip=f"Route length: {result.length_m:.0f} m"
                    ).add_to(m)

                    self.status.emit(f"Маршрут: готово (довжина ~ {result.length_m:.0f} м)")
                except PathfinderError as e:
                    logger.warning("Маршрут не побудовано: %s", e)
                    self.status.emit("Маршрут: не побудовано — див. лог")
                except Exception as e:
                    logger.exception("Маршрут помилка: %s", e)
                    self.status.emit("Маршрут: помилка — див. лог")

            self.progress.emit(35)

            # Вулиці
            folium.GeoJson(
                data=self.gdf_edges[["geometry"]].to_json(),
                name="Вулиці",
                show=False,
                style_function=lambda x: {"color": "#4a4a4a", "weight": 2, "opacity": 0.7}
            ).add_to(m)

            # Кордон
            try:
                b_gdf, binfo = load_city_boundary(
                    self.city_name,
                    required_tokens=_extract_required_tokens(self.city_name),
                )

                if b_gdf is not None and not b_gdf.empty:
                    used = binfo.get("used_path") or "—"
                    expected = binfo.get("expected_path") or "—"
                    cache_action = binfo.get("cache_action") or "exists"
                    self.status.emit(f"Boundary: кеш → {used} (cache_action={cache_action}, expected={expected})")
                else:
                    b_gdf = ox.geocode_to_gdf(self.city_name).to_crs(4326)
                    self.status.emit("Boundary: кеш не знайдено — взято з OSM")

                if b_gdf is not None and not b_gdf.empty:
                    folium.GeoJson(
                        data=b_gdf[["geometry"]].to_json(),
                        name="Кордон міста",
                        show=True,
                        style_function=lambda x: {"color": "#d9534f", "weight": 3, "fill": False, "opacity": 0.9}
                    ).add_to(m)
            except Exception:
                logger.warning("MapRenderWorker: не вдалося отримати boundary для '%s'", self.city_name)

            self.progress.emit(50)

            label_map = {
                "education": "Освіта",
                "health": "Медицина",
                "culture": "Культура",
                "greens_sport": "Зелена інфра / Спорт",
                "shopping_services": "Покупки / Сервіси",
                "transport": "Громадський транспорт",
            }

            level_cfg = LEVELS_QUERIES.get(self.level_name, {})
            for category, tags_dict in level_cfg.items():
                try:
                    gdf = self._subset_for_tags(self.gdf_all_poi, tags_dict)
                    if gdf is None or gdf.empty:
                        continue

                    fg = folium.FeatureGroup(
                        name=f"POI — {label_map.get(category, category)} [{self.level_name}]",
                        show=False
                    )
                    cluster = MarkerCluster().add_to(fg)

                    for _, row in gdf.iterrows():
                        geom = row.geometry
                        if geom is None or geom.is_empty:
                            continue
                        try:
                            if not isinstance(geom, Point):
                                geom = geom.representative_point()
                        except Exception:
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
                            location=[geom.y, geom.x],
                            tooltip=name,
                            popup=folium.Popup(desc, max_width=300),
                        ).add_to(cluster)

                    fg.add_to(m)
                except Exception as e:
                    logger.warning("Локальна фільтрація POI '%s' не вдалася: %s", category, e)

            folium.LayerControl(collapsed=False).add_to(m)

            ts = int(time.time() * 1000)
            map_file = os.path.join("outputs", f"{self.safe_city_name}_map_{ts}.html")
            m.save(map_file)

            # NEW: кліки по карті -> Python
            self._inject_click_bridge_js(map_file)

            self.progress.emit(100)
            self.status.emit(f"Карта збережена: {map_file}")
            self.finished.emit(map_file)

        except Exception as e:
            logger.exception("MapRenderWorker помилка: %s", e)
            self.error.emit(f"Помилка рендеру карти: {e}")


# ----------------------------
# MainWindow
# ----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("15-Minute City Map Viewer")

        screen_height = QApplication.primaryScreen().size().height()
        initial_height = int(screen_height * 0.9)
        initial_width = int(initial_height * (1831 / 2048) * 1.3)
        self.setGeometry(100, 100, initial_width, initial_height)

        saved = load_cached_cities()
        discovered = discover_cached_cities()
        self.cached_cities = sorted({*saved, *discovered})

        self.selected_country = None

        container = QWidget()
        self.main_layout = QHBoxLayout(container)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setAlignment(Qt.AlignTop)

        # --- UI: країна/місто ---
        self.left_layout.addWidget(QLabel("Країна:"))
        self.country_select = QComboBox()
        self.country_select.addItems(["Польща", "Україна"])
        self.country_select.currentIndexChanged.connect(self.country_selected)
        self.left_layout.addWidget(self.country_select)

        self.country_confirm = QLabel("Оберіть країну для пошуку міст")
        self.left_layout.addWidget(self.country_confirm)

        self.left_layout.addWidget(QLabel("Пошук міста:"))
        self.city_input = QLineEdit()
        self.city_input.setPlaceholderText("Введіть назву міста")
        self.city_button = QToolButton(self.city_input)
        self.city_button.setText("↓")
        self.city_button.clicked.connect(self.search_city)
        self.city_input.setTextMargins(0, 0, 30, 0)
        self.city_input.resizeEvent = self._resize_city_button
        self.left_layout.addWidget(self.city_input)

        self.found_cities = QComboBox()
        self.left_layout.addWidget(self.found_cities)

        self.left_layout.addWidget(QLabel("Попередні міста (кеш знайдено автоматично):"))
        self.prev_cities = QComboBox()
        self.prev_cities.addItems(self.cached_cities)
        self.left_layout.addWidget(self.prev_cities)

        # --- рівень POI ---
        self.left_layout.addWidget(QLabel("Рівень доступності (POI):"))
        self.level_select = QComboBox()
        self.level_select.addItems(["medium (середній)", "minimum (база)", "maximum (макс)"])
        self.level_select.setCurrentIndex(0)
        self.left_layout.addWidget(self.level_select)

        # --- Маршрут: режим ---
        self.left_layout.addWidget(QLabel("Маршрут: режим вводу"))
        self.route_mode = QComboBox()
        self.route_mode.addItems(["Координати", "Адреса", "Клік по карті"])
        self.route_mode.currentIndexChanged.connect(self._update_route_mode_ui)
        self.left_layout.addWidget(self.route_mode)

        # --- Блок координат ---
        self.route_start = QLineEdit()
        self.route_start.setPlaceholderText("Старт (lat, lon) напр. 49.8397, 24.0297")
        self.route_end = QLineEdit()
        self.route_end.setPlaceholderText("Фініш (lat, lon) напр. 49.8500, 24.0200")

        self.coords_widgets = [self.route_start, self.route_end]
        for w in self.coords_widgets:
            self.left_layout.addWidget(w)

        # --- Блок адрес ---
        self.addr_start_input = QLineEdit()
        self.addr_start_input.setPlaceholderText("Старт: вулиця + будинок (напр. Shevchenka 10)")
        self.addr_start_btn = QPushButton("Знайти старт")
        self.addr_start_results = QComboBox()

        self.addr_end_input = QLineEdit()
        self.addr_end_input.setPlaceholderText("Фініш: вулиця + будинок (напр. Bandery 12)")
        self.addr_end_btn = QPushButton("Знайти фініш")
        self.addr_end_results = QComboBox()

        self.addr_start_btn.clicked.connect(lambda: self._search_address(is_start=True))
        self.addr_end_btn.clicked.connect(lambda: self._search_address(is_start=False))

        self.addr_widgets = [
            self.addr_start_input, self.addr_start_btn, self.addr_start_results,
            self.addr_end_input, self.addr_end_btn, self.addr_end_results
        ]
        for w in self.addr_widgets:
            self.left_layout.addWidget(w)

        # --- Блок кліків ---
        self.btn_pick_start = QPushButton("Вибрати старт на карті")
        self.btn_pick_end = QPushButton("Вибрати фініш на карті")
        self.btn_pick_start.clicked.connect(lambda: self._set_pick_target("start"))
        self.btn_pick_end.clicked.connect(lambda: self._set_pick_target("end"))

        if not WEBCHANNEL_AVAILABLE:
            self.btn_pick_start.setEnabled(False)
            self.btn_pick_end.setEnabled(False)

        self.click_widgets = [self.btn_pick_start, self.btn_pick_end]
        for w in self.click_widgets:
            self.left_layout.addWidget(w)

        # --- алгоритм ---
        self.left_layout.addWidget(QLabel("Алгоритм маршруту:"))
        self.route_alg = QComboBox()
        self.route_alg.addItems(["dijkstra", "astar"])
        self.left_layout.addWidget(self.route_alg)

        self.btn_route = QPushButton("Побудувати маршрут")
        self.btn_route.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_route.setMinimumHeight(40)
        self.left_layout.addWidget(self.btn_route)
        self.btn_route.clicked.connect(self.build_route_only)

        # --- прогрес / кнопки ---
        self.progress_label = QLabel("")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.left_layout.addWidget(self.progress_label)
        self.left_layout.addWidget(self.progress_bar)

        self.btn_build = QPushButton("Побудувати карту")
        self.btn_clear_cache = QPushButton("Очистити кеш міст")
        for btn in (self.btn_build, self.btn_clear_cache):
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setMinimumHeight(40)
        self.left_layout.addWidget(self.btn_build)
        self.left_layout.addWidget(self.btn_clear_cache)

        self.btn_build.clicked.connect(self.build_map)
        self.btn_clear_cache.clicked.connect(self.clear_city_cache)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(self.left_panel)

        self.web_view = QWebEngineView()
        self.web_view.loadFinished.connect(self._on_web_load_finished)

        self.main_layout.addWidget(left_scroll, 1)
        self.main_layout.addWidget(self.web_view, 2)
        self.setCentralWidget(container)

        # --- WebChannel init ---
        self._bridge = None
        self._channel = None
        if WEBCHANNEL_AVAILABLE:
            self._bridge = MapBridge()
            self._bridge.picked.connect(self._on_map_picked)
            self._channel = QWebChannel(self.web_view.page())
            self._channel.registerObject("bridge", self._bridge)
            self.web_view.page().setWebChannel(self._channel)

        # --- state ---
        self._threads = []
        self._last_G = None
        self._last_gdf_edges = None
        self._last_gdf_all_poi = None
        self._last_safe_city = None
        self._last_city = None

        self._pick_target = ""  # 'start' або 'end'
        self._picked_start = None  # (lat, lon)
        self._picked_end = None    # (lat, lon)

        self._update_route_mode_ui()

    # ---------------- UI helpers ----------------
    def _set_status(self, msg: str):
        self.progress_label.setText(msg)

    def resizeEvent(self, event):
        panel_width = self.width() // 3
        self.left_panel.setMinimumWidth(panel_width)
        self.left_panel.setMaximumWidth(panel_width)
        super().resizeEvent(event)

    def _resize_city_button(self, event):
        self.city_button.move(self.city_input.width() - 25, 1)
        self.city_button.resize(25, self.city_input.height())
        QLineEdit.resizeEvent(self.city_input, event)

    def country_selected(self):
        self.selected_country = self.country_select.currentText()
        self.country_confirm.setText(f"Пошук міст буде виконуватись у: {self.selected_country}")

    def _current_level(self) -> str:
        text = self.level_select.currentText()
        if text.startswith("minimum"):
            return "minimum"
        if text.startswith("maximum"):
            return "maximum"
        return "medium"

    def _current_country_params(self):
        if self.selected_country in COUNTRY_MAP:
            return COUNTRY_MAP[self.selected_country]
        return ("ua", "uk")

    def _update_route_mode_ui(self):
        mode = self.route_mode.currentText()

        show_coords = (mode == "Координати")
        show_addr = (mode == "Адреса")
        show_click = (mode == "Клік по карті")

        for w in self.coords_widgets:
            w.setVisible(show_coords)

        for w in self.addr_widgets:
            w.setVisible(show_addr)

        for w in self.click_widgets:
            w.setVisible(show_click)

        if show_click and not WEBCHANNEL_AVAILABLE:
            self._set_status("Клік по карті недоступний: немає QtWebChannel у вашому оточенні.")

    def _on_web_load_finished(self, ok: bool):
        if not ok:
            return
        # застосувати поточну ціль кліку (start/end) у JS
        if WEBCHANNEL_AVAILABLE:
            tgt = self._pick_target or ""
            js = f"window.__pick_target = {json.dumps(tgt)};"
            self.web_view.page().runJavaScript(js)

    def _set_pick_target(self, target: str):
        if not WEBCHANNEL_AVAILABLE:
            QMessageBox.warning(self, "Клік по карті", "QtWebChannel недоступний у вашому середовищі.")
            return
        if target not in ("start", "end"):
            target = ""
        self._pick_target = target
        self._set_status(f"Клік по карті: вибери {('СТАРТ' if target=='start' else 'ФІНІШ')} на мапі")
        self.web_view.page().runJavaScript(f"window.__pick_target = {json.dumps(target)};")

    def _on_map_picked(self, lat: float, lon: float):
        # Якщо користувач не натиснув "вибрати старт/фініш" — нічого не робимо
        if not self._pick_target:
            return

        if self._pick_target == "start":
            self._picked_start = (lat, lon)
            self.route_start.setText(f"{lat:.6f}, {lon:.6f}")
            self._set_status("Старт обрано. Тепер вибери фініш або будуй маршрут.")
        elif self._pick_target == "end":
            self._picked_end = (lat, lon)
            self.route_end.setText(f"{lat:.6f}, {lon:.6f}")
            self._set_status("Фініш обрано. Можеш будувати маршрут.")

    # ---------------- City search ----------------
    def search_city(self):
        if not self.selected_country:
            QMessageBox.information(self, "Увага", "Спочатку оберіть країну.")
            return
        query = self.city_input.text().strip()
        if not query:
            QMessageBox.information(self, "Увага", "Введіть частину назви міста.")
            return

        self.found_cities.clear()
        country_code, language = self._current_country_params()
        worker = CitySearchWorker(country_code, language, query)
        worker.status.connect(self._set_status)
        worker.finished.connect(self._on_search_finished)
        worker.error.connect(self._on_search_error)
        worker.finished.connect(lambda *_: self._cleanup_thread(worker))
        worker.error.connect(lambda *_: self._cleanup_thread(worker))
        self._threads.append(worker)
        worker.start()

    def _on_search_finished(self, items: list):
        self.progress_label.setText("")
        self.found_cities.clear()
        self.found_cities.addItems(items)
        logger.info("UI: пошук завершено, знайдено %d позицій", len(items))

    def _on_search_error(self, message: str):
        self.progress_label.setText("")
        self.found_cities.clear()
        self.found_cities.addItem("Не знайдено")
        logger.warning("UI: помилка пошуку: %s", message)
        QMessageBox.warning(self, "Пошук", message)

    def _current_city_for_address(self) -> str:
        city = self.found_cities.currentText()
        if city == "Не знайдено" or not city:
            city = self.prev_cities.currentText()
        return city or ""

    def _search_address(self, is_start: bool):
        city = self._current_city_for_address()
        if not city:
            QMessageBox.information(self, "Адреса", "Спочатку обери/побудуй місто.")
            return

        q = (self.addr_start_input.text() if is_start else self.addr_end_input.text()).strip()
        if not q:
            QMessageBox.information(self, "Адреса", "Введи вулицю і номер будинку.")
            return

        country_code, language = self._current_country_params()
        worker = AddressSearchWorker(country_code, language, city, q)
        worker.status.connect(self._set_status)
        worker.finished.connect(lambda items: self._on_address_found(items, is_start))
        worker.error.connect(lambda msg: self._on_address_error(msg, is_start))
        worker.finished.connect(lambda *_: self._cleanup_thread(worker))
        worker.error.connect(lambda *_: self._cleanup_thread(worker))
        self._threads.append(worker)
        worker.start()

    def _on_address_found(self, items: list, is_start: bool):
        combo = self.addr_start_results if is_start else self.addr_end_results
        combo.clear()
        for it in items:
            label = it.get("label", "—")
            lat = float(it.get("lat"))
            lon = float(it.get("lon"))
            combo.addItem(label, (lat, lon))
        combo.setCurrentIndex(0)
        self._set_status("Адреса: вибери правильний варіант зі списку")

    def _on_address_error(self, msg: str, is_start: bool):
        logger.warning("Address error: %s", msg)
        QMessageBox.warning(self, "Адреса", msg)
        self._set_status("")

    # ---------------- Build map / route ----------------
    def build_map(self):
        city = self._current_city_for_address()
        if not city:
            QMessageBox.information(self, "Увага", "Спочатку оберіть або знайдіть місто.")
            return

        if city not in self.cached_cities:
            self.cached_cities.append(city)
            save_cached_cities(self.cached_cities)
            self.prev_cities.addItem(city)

        required_tokens = _extract_required_tokens(city)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_build.setEnabled(False)
        self.btn_clear_cache.setEnabled(False)
        self.btn_route.setEnabled(False)

        self._picked_start = None
        self._picked_end = None
        self._pick_target = ""
        if WEBCHANNEL_AVAILABLE:
            self.web_view.page().runJavaScript("window.__pick_target = '';")

        worker = GraphWorker(city, required_tokens)
        worker.status.connect(self._set_status)
        worker.progress.connect(self.progress_bar.setValue)
        worker.finished.connect(self.on_graph_ready)
        worker.error.connect(self._on_build_error)
        worker.finished.connect(lambda *_: self._cleanup_thread(worker))
        worker.error.connect(lambda *_: self._cleanup_thread(worker))
        self._threads.append(worker)
        worker.start()

    def on_graph_ready(self, G, gdf_edges, safe_city_name: str, city_name: str):
        self._last_G = G
        self._last_gdf_edges = gdf_edges
        self._last_safe_city = safe_city_name
        self._last_city = city_name

        self.progress_bar.setValue(0)
        poi_worker = AllPOIWorker(city_name, _extract_required_tokens(city_name))
        poi_worker.status.connect(self._set_status)
        poi_worker.progress.connect(self.progress_bar.setValue)
        poi_worker.finished.connect(self.on_all_poi_ready)
        poi_worker.error.connect(self._on_build_error)
        poi_worker.finished.connect(lambda *_: self._cleanup_thread(poi_worker))
        poi_worker.error.connect(lambda *_: self._cleanup_thread(poi_worker))
        self._threads.append(poi_worker)
        poi_worker.start()

    def on_all_poi_ready(self, gdf_all_poi):
        self._last_gdf_all_poi = gdf_all_poi

        self.progress_bar.setValue(0)
        level = self._current_level()

        mworker = MapRenderWorker(
            self._last_G,
            self._last_gdf_edges,
            self._last_gdf_all_poi,
            self._last_safe_city,
            self._last_city,
            level,
            route_start_latlon=None,
            route_end_latlon=None,
            route_algorithm="dijkstra",
            enable_click_pick=True
        )
        mworker.status.connect(self._set_status)
        mworker.progress.connect(self.progress_bar.setValue)
        mworker.finished.connect(self._on_map_ready)
        mworker.error.connect(self._on_build_error)
        mworker.finished.connect(lambda *_: self._cleanup_thread(mworker))
        mworker.error.connect(lambda *_: self._cleanup_thread(mworker))
        self._threads.append(mworker)
        mworker.start()

    def _get_route_points_by_mode(self):
        mode = self.route_mode.currentText()

        if mode == "Координати":
            start_latlon = parse_latlon_text(self.route_start.text(), "Старт")
            end_latlon = parse_latlon_text(self.route_end.text(), "Фініш")
            return start_latlon, end_latlon

        if mode == "Адреса":
            if self.addr_start_results.count() == 0 or self.addr_end_results.count() == 0:
                raise ValueError("Спочатку знайди старт і фініш через кнопки пошуку адреси.")
            start = self.addr_start_results.currentData()
            end = self.addr_end_results.currentData()
            if not start or not end:
                raise ValueError("Вибери варіант адреси зі списку (старт і фініш).")
            return tuple(start), tuple(end)

        if mode == "Клік по карті":
            if not self._picked_start or not self._picked_end:
                raise ValueError("Спочатку вибери старт і фініш на карті кнопками 'Вибрати ... на карті'.")
            return self._picked_start, self._picked_end

        raise ValueError("Невідомий режим маршруту.")

    def build_route_only(self):
        if self._last_G is None or self._last_gdf_edges is None or self._last_city is None or self._last_safe_city is None:
            QMessageBox.information(self, "Маршрут", "Спочатку побудуй карту для міста (кнопка 'Побудувати карту').")
            return
        if self._last_gdf_all_poi is None:
            QMessageBox.information(self, "Маршрут", "POI ще не завантажились. Спробуй ще раз через пару секунд.")
            return

        try:
            start_latlon, end_latlon = self._get_route_points_by_mode()
        except Exception as e:
            QMessageBox.warning(self, "Маршрут", str(e))
            return

        alg = (self.route_alg.currentText() or "dijkstra").strip()
        level = self._current_level()

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_build.setEnabled(False)
        self.btn_clear_cache.setEnabled(False)
        self.btn_route.setEnabled(False)

        mworker = MapRenderWorker(
            self._last_G,
            self._last_gdf_edges,
            self._last_gdf_all_poi,
            self._last_safe_city,
            self._last_city,
            level,
            route_start_latlon=start_latlon,
            route_end_latlon=end_latlon,
            route_algorithm=alg,
            enable_click_pick=True
        )
        mworker.status.connect(self._set_status)
        mworker.progress.connect(self.progress_bar.setValue)
        mworker.finished.connect(self._on_map_ready)
        mworker.error.connect(self._on_build_error)
        mworker.finished.connect(lambda *_: self._cleanup_thread(mworker))
        mworker.error.connect(lambda *_: self._cleanup_thread(mworker))
        self._threads.append(mworker)
        mworker.start()

    def _on_map_ready(self, map_file_path: str):
        self.progress_label.setText("")
        self.progress_bar.setVisible(False)
        self.btn_build.setEnabled(True)
        self.btn_clear_cache.setEnabled(True)
        self.btn_route.setEnabled(True)

        logger.info("UI: завантажуємо карту в web_view: %s", map_file_path)
        self.web_view.load(QUrl.fromLocalFile(os.path.abspath(map_file_path)))

    def _on_build_error(self, message: str):
        self.progress_label.setText("")
        self.progress_bar.setVisible(False)
        self.btn_build.setEnabled(True)
        self.btn_clear_cache.setEnabled(True)
        self.btn_route.setEnabled(True)

        logger.error("UI: build error: %s", message)
        QMessageBox.critical(self, "Помилка", message)

    def _cleanup_thread(self, thread_obj: QThread):
        try:
            self._threads.remove(thread_obj)
        except ValueError:
            pass

    def clear_city_cache(self):
        self.cached_cities = []
        save_cached_cities(self.cached_cities)
        self.prev_cities.clear()
        logger.info("UI: кеш міст очищено")
        QMessageBox.information(self, "Кеш", "Список збережених міст очищено.")
