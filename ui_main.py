import os
import re
import glob
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QScrollArea, QComboBox, QLineEdit, QToolButton, QSizePolicy,
    QProgressBar, QMessageBox
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt, QUrl, QThread, pyqtSignal

import folium
import osmnx as ox
from shapely.geometry import Point
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

from graph_builder import get_city_graph_with_source
from folium.plugins import MarkerCluster
from config import LEVELS_QUERIES
from logger_config import logger

from poi_extractor import get_poi_with_source, load_boundary_from_cache_debug

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
# Утиліти нормалізації/токенів (місто + область)
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


class GraphWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(object, str, str)  # (gdf_edges, safe_city_name, city_name)
    error = pyqtSignal(str)

    def __init__(self, city_name: str, required_tokens: list):
        super().__init__()
        self.city_name = city_name
        self.required_tokens = required_tokens

    def run(self):
        try:
            self.progress.emit(10)
            self.status.emit("Граф доріжок: пошук кешу…")
            G, info = get_city_graph_with_source(
                self.city_name, network_type="walk", required_tokens=self.required_tokens
            )
            action = info.get("action")
            used = info.get("used_path")
            expected = info.get("expected_path")
            candidates = info.get("candidates", [])
            if action in ("exists", "copied", "fallback_read"):
                msg = f"Граф: взято з кешу → {used}"
                if action == "copied":
                    msg += f" (скопійовано в {expected})"
                if action == "fallback_read":
                    msg += f" (читаю напряму без копії)"
            else:
                msg = f"Граф: кеш не знайдено. Шукали: {expected}. Кандидати: {', '.join(candidates[:3]) or '—'}. Завантажено з OSM."
            self.status.emit(msg)
            logger.info("GraphWorker source: %s", msg)

            self.progress.emit(70)
            gdf_edges = ox.utils_graph.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
            self.progress.emit(100)
            safe_city_name = self.city_name.lower().replace(",", "").replace(" ", "_")
            self.finished.emit(gdf_edges, safe_city_name, self.city_name)
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
            gdf, info = get_poi_with_source(
                self.city_name, split_by_topkey=False, force_refresh=False, required_tokens=self.required_tokens
            )
            action = info.get("action")
            used = info.get("used_path")
            expected = info.get("expected_path")
            candidates = info.get("candidates", [])
            if action == "cache":
                self.status.emit(f"POI: кеш знайдено → {used} (записів: {len(gdf)})")
            elif action == "osm":
                cand_str = ", ".join(candidates[:3]) or "—"
                self.status.emit(
                    f"POI: кеш не знайдено. Шукали: {expected}. Кандидати: {cand_str}. Тягну з OSM → зберіг у {used} (записів: {len(gdf)})")
            else:
                self.status.emit(f"POI: джерело: {action or 'невідомо'} → {used or '—'} (записів: {len(gdf)})")
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

    def __init__(self, gdf_edges, gdf_all_poi, safe_city_name: str, city_name: str, level_name: str):
        super().__init__()
        self.gdf_edges = gdf_edges
        self.gdf_all_poi = gdf_all_poi
        self.safe_city_name = safe_city_name
        self.city_name = city_name
        self.level_name = level_name

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

    def run(self):
        try:
            self.progress.emit(10)
            self.status.emit(f"Рендер рівня '{self.level_name}'…")

            # Центр
            try:
                center = self.gdf_edges.geometry.unary_union.centroid
                center_latlon = [center.y, center.x]
            except Exception:
                self.status.emit("Попередження: не вдалося визначити центр по ребрах — спробую по boundary")
                b_gdf, _ = load_boundary_from_cache_debug(
                    self.city_name, required_tokens=_extract_required_tokens(self.city_name)
                    # опціонально передаємо токени
                )
                if b_gdf is not None and not b_gdf.empty:
                    c = b_gdf.geometry.unary_union.representative_point()
                    center_latlon = [c.y, c.x]
                else:
                    center_latlon = [49.0, 24.0]  # запасний центр

            m = folium.Map(location=center_latlon, zoom_start=13,
                           tiles="cartodbpositron", control_scale=True)
            self.progress.emit(35)

            # Вулиці (видимі одразу)
            folium.GeoJson(
                data=self.gdf_edges[["geometry"]].to_json(),
                name="Вулиці",
                show=False,
                style_function=lambda x: {"color": "#4a4a4a", "weight": 2, "opacity": 0.7}
            ).add_to(m)

            # Кордон (кеш/OSM) — мʼякий skip, якщо немає
            try:
                b_gdf, binfo = load_boundary_from_cache_debug(
                    self.city_name, required_tokens=_extract_required_tokens(self.city_name)  # опціонально
                )
                if b_gdf is not None:
                    self.status.emit(
                        f"Boundary: кеш {'є' if binfo.get('found') else 'нема'}; "
                        f"очікував {binfo.get('expected_path')}, використав {binfo.get('used_path') or 'OSM'}"
                    )
                if b_gdf is None:
                    b_gdf = ox.geocode_to_gdf(self.city_name).to_crs(4326)
                    self.status.emit("Boundary: кеш не знайдено — взято з OSM")
                if b_gdf is not None and not b_gdf.empty:
                    folium.GeoJson(
                        data=b_gdf[["geometry"]].to_json(),
                        name="Кордон міста",
                        show=True,
                        style_function=lambda x: {"color": "#d9534f", "weight": 3, "fill": False, "opacity": 0.9}
                    ).add_to(m)
                else:
                    self.status.emit("Boundary: відсутній — пропускаю шар")
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
                        self.status.emit(f"POI: '{category}' — порожньо")
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

            map_file = os.path.join("outputs", f"{self.safe_city_name}_map.html")
            m.save(map_file)
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

        self.left_layout.addWidget(QLabel("Рівень доступності (POI):"))
        self.level_select = QComboBox()
        self.level_select.addItems(["medium (середній)", "minimum (база)", "maximum (макс)"])
        self.level_select.setCurrentIndex(0)
        self.left_layout.addWidget(self.level_select)

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

        self.main_layout.addWidget(left_scroll, 1)
        self.main_layout.addWidget(self.web_view, 2)
        self.setCentralWidget(container)

        self._threads = []
        self._last_gdf_edges = None
        self._last_safe_city = None
        self._last_city = None

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

    def build_map(self):
        city = self.found_cities.currentText()
        if city == "Не знайдено" or not city:
            city = self.prev_cities.currentText()
        if not city:
            QMessageBox.information(self, "Увага", "Спочатку оберіть або знайдіть місто.")
            return

        if city not in self.cached_cities:
            self.cached_cities.append(city)
            save_cached_cities(self.cached_cities)
            self.prev_cities.addItem(city)

        # строгі токени для кешу
        required_tokens = _extract_required_tokens(city)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_build.setEnabled(False)
        self.btn_clear_cache.setEnabled(False)
        logger.info("UI: починаємо build_map для '%s'", city)

        worker = GraphWorker(city, required_tokens)
        worker.status.connect(self._set_status)
        worker.progress.connect(self.progress_bar.setValue)
        worker.finished.connect(self.on_graph_ready)
        worker.error.connect(self._on_build_error)
        worker.finished.connect(lambda *_: self._cleanup_thread(worker))
        worker.error.connect(lambda *_: self._cleanup_thread(worker))
        self._threads.append(worker)
        worker.start()

    def on_graph_ready(self, gdf_edges, safe_city_name: str, city_name: str):
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
        self.progress_bar.setValue(0)
        level = self._current_level()
        mworker = MapRenderWorker(self._last_gdf_edges, gdf_all_poi, self._last_safe_city, self._last_city, level)
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
        logger.info("UI: завантажуємо карту в web_view: %s", map_file_path)
        self.web_view.load(QUrl.fromLocalFile(os.path.abspath(map_file_path)))

    def _on_build_error(self, message: str):
        self.progress_label.setText("")
        self.progress_bar.setVisible(False)
        self.btn_build.setEnabled(True)
        self.btn_clear_cache.setEnabled(True)
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
