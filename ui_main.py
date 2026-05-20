import os
import json

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QScrollArea, QComboBox, QLineEdit, QToolButton, QSizePolicy,
    QProgressBar, QMessageBox
)
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
from PyQt5.QtCore import Qt, QUrl, QThread, pyqtSignal, QObject, pyqtSlot

WEBCHANNEL_AVAILABLE = True
try:
    from PyQt5.QtWebChannel import QWebChannel
except Exception:
    WEBCHANNEL_AVAILABLE = False

import osmnx as ox
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

from api import load_city_graph, load_poi
from logger_config import logger
from paths import ensure_data_dirs
from utils import safe_name, parse_latlon_text, extract_required_tokens
from cache_manager import load_cached_cities, save_cached_cities, discover_cached_cities
from visualizer import MapRenderWorker


ensure_data_dirs()

COUNTRY_MAP = {
    "Україна": ("ua", "uk"),
    "Польща": ("pl", "pl"),
}


class LoggingWebPage(QWebEnginePage):
    """
    Виводить console.log/console.error з JS у ваш logger.
    Дуже допомагає побачити, на чому саме стопориться map_bridge.js
    """
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        try:
            logger.info("JS console [%s] %s:%s %s", level, sourceID, lineNumber, message)
        except Exception:
            pass


class MapBridge(QObject):
    picked = pyqtSignal(float, float, str, str)

    @pyqtSlot(float, float, str, str, name="map_clicked")
    def map_clicked(self, lat, lon, target, category):
        self.picked.emit(
            float(lat),
            float(lon),
            str(target or ""),
            str(category or "")
        )


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
                logger.info(
                    "CitySearch: знайдено %d результатів для '%s' (cc=%s)",
                    len(items), self.query, self.country_code
                )
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
                self.error.emit(
                    "За цією адресою нічого не знайдено. "
                    "Спробуйте інший формат (наприклад 'Shevchenka 10')."
                )
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
            used = info.get("used_path")
            expected = info.get("expected_path")
            candidates = info.get("candidates", []) or []
            cache_action = info.get("cache_action")

            if source == "cache":
                msg = f"Граф: кеш → {used or '—'}"
                if cache_action == "copied" and expected:
                    msg += f" (скопійовано в {expected})"
                elif cache_action == "fallback_read":
                    msg += " (читання напряму без копії)"
            else:
                cand_str = ", ".join(candidates[:3]) or "—"
                msg = (
                    f"Граф: кеш не знайдено. Шукали: {expected or '—'}. "
                    f"Кандидати: {cand_str}. Завантажено з OSM."
                )
                if used:
                    msg += f" Збережено → {used}"
                else:
                    msg += " (не вдалося зберегти graphml)"

            self.status.emit(msg)
            logger.info("GraphWorker source: %s", msg)

            self.progress.emit(70)
            gdf_edges = ox.utils_graph.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
            self.progress.emit(100)

            safe_city_name = safe_name(self.city_name)
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
            used = info.get("used_path")
            expected = info.get("expected_path")
            candidates = info.get("candidates", []) or []

            if source == "cache":
                self.status.emit(f"POI: кеш → {used or '—'} (записів: {len(gdf)})")
            else:
                cand_str = ", ".join(candidates[:3]) or "—"
                self.status.emit(
                    f"POI: кеш не знайдено. Шукали: {expected or '—'}. "
                    f"Кандидати: {cand_str}. Тягну з OSM → {used or '—'} (записів: {len(gdf)})"
                )

            self.progress.emit(100)
            self.finished.emit(gdf)
        except Exception as e:
            logger.exception("AllPOIWorker помилка: %s", e)
            self.error.emit(f"Помилка завантаження POI: {e}")


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

        # ============================================================
        # 1) Країна
        # ============================================================
        self.left_layout.addWidget(QLabel("Країна:"))
        self.country_select = QComboBox()
        self.country_select.addItems(["Польща", "Україна"])
        self.country_select.currentIndexChanged.connect(self.country_selected)
        self.left_layout.addWidget(self.country_select)

        self.country_confirm = QLabel("Оберіть країну для пошуку міст")
        self.left_layout.addWidget(self.country_confirm)

        # ============================================================
        # 2) Пошук міста + попередні міста
        # ============================================================
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

        # ============================================================
        # 3) Рівень доступності
        # ============================================================
        self.left_layout.addWidget(QLabel("Рівень доступності (POI):"))
        self.level_select = QComboBox()
        self.level_select.addItems(["medium (середній)", "minimum (база)", "maximum (макс)"])
        self.level_select.setCurrentIndex(0)
        self.left_layout.addWidget(self.level_select)

        # ============================================================
        # 4) Побудувати карту
        # ============================================================
        self.btn_build = QPushButton("Побудувати карту")
        self.btn_build.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_build.setMinimumHeight(40)
        self.left_layout.addWidget(self.btn_build)
        self.btn_build.clicked.connect(self.build_map)

        # ============================================================
        # 5) Маршрути + алгоритм
        # ============================================================
        self.left_layout.addWidget(QLabel("Маршрут: режим вводу"))
        self.route_mode = QComboBox()
        self.route_mode.addItems(["Координати", "Адреса", "Клік по карті"])
        self.route_mode.currentIndexChanged.connect(self._update_route_mode_ui)
        self.left_layout.addWidget(self.route_mode)

        self.route_start = QLineEdit()
        self.route_start.setPlaceholderText("Старт (lat, lon) напр. 49.8397, 24.0297")
        self.route_end = QLineEdit()
        self.route_end.setPlaceholderText("Фініш (lat, lon) напр. 49.8500, 24.0200")
        self.coords_widgets = [self.route_start, self.route_end]
        for w in self.coords_widgets:
            self.left_layout.addWidget(w)

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

        # --- click picking buttons ---
        self.btn_pick_start = QPushButton("Вибрати старт на карті")
        self.btn_pick_end = QPushButton("Вибрати фініш на карті")
        self.btn_pick_stop = QPushButton("Додати stop на карті")
        self.btn_clear_stops = QPushButton("Очистити stops")

        self.btn_pick_start.clicked.connect(lambda: self._set_pick_target("start"))
        self.btn_pick_end.clicked.connect(lambda: self._set_pick_target("end"))
        self.btn_pick_stop.clicked.connect(lambda: self._set_pick_target("stop"))
        self.btn_clear_stops.clicked.connect(self._clear_stops)

        if not WEBCHANNEL_AVAILABLE:
            self.btn_pick_start.setEnabled(False)
            self.btn_pick_end.setEnabled(False)
            self.btn_pick_stop.setEnabled(False)
            self.btn_clear_stops.setEnabled(False)

        self.click_widgets = [self.btn_pick_start, self.btn_pick_end, self.btn_pick_stop, self.btn_clear_stops]
        for w in self.click_widgets:
            self.left_layout.addWidget(w)

        self.left_layout.addWidget(QLabel("Алгоритм маршруту:"))
        self.route_alg = QComboBox()
        self.route_alg.addItems(["dijkstra", "astar"])
        self.left_layout.addWidget(self.route_alg)

        self.left_layout.addWidget(QLabel("Проміжна зупинка (категорія):"))
        self.via_category = QComboBox()
        self.via_category.addItems(["Нема", "Парк"])
        self.left_layout.addWidget(self.via_category)

        self.btn_route = QPushButton("Побудувати маршрут")
        self.btn_route.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_route.setMinimumHeight(40)
        self.left_layout.addWidget(self.btn_route)
        self.btn_route.clicked.connect(self.build_route_only)

        # ============================================================
        # 5.1) Ізохрони
        # ============================================================
        self.left_layout.addWidget(QLabel("Ізохрони / карта доступності:"))

        self.iso_select = QComboBox()
        self.iso_select.addItems(["5 хв", "10 хв", "15 хв", "5/10/15"])
        self.iso_select.setCurrentIndex(3)
        self.left_layout.addWidget(self.iso_select)

        self.btn_isochrone = QPushButton("Побудувати ізохрону")
        self.btn_isochrone.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_isochrone.setMinimumHeight(40)
        self.left_layout.addWidget(self.btn_isochrone)
        self.btn_isochrone.clicked.connect(self.build_isochrone_only)

        self.btn_accessibility = QPushButton("Карта доступності міста")
        self.btn_accessibility.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_accessibility.setMinimumHeight(40)
        self.left_layout.addWidget(self.btn_accessibility)
        self.btn_accessibility.clicked.connect(self.build_accessibility_only)

        # ============================================================
        # Прогрес / статус
        # ============================================================
        self.progress_label = QLabel("")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.left_layout.addWidget(self.progress_label)
        self.left_layout.addWidget(self.progress_bar)

        # ============================================================
        # 6) В самому низу: очистити кеш
        # ============================================================
        self.btn_clear_cache = QPushButton("Очистити кеш міст")
        self.btn_clear_cache.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_clear_cache.setMinimumHeight(40)
        self.left_layout.addWidget(self.btn_clear_cache)
        self.btn_clear_cache.clicked.connect(self.clear_city_cache)

        # ------------------------------------------------------------
        # Ліва панель у скролі + WebView праворуч
        # ------------------------------------------------------------
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(self.left_panel)

        self.web_view = QWebEngineView()
        self.web_view.setPage(LoggingWebPage(self.web_view))
        self.web_view.loadFinished.connect(self._on_web_load_finished)

        self.main_layout.addWidget(left_scroll, 1)
        self.main_layout.addWidget(self.web_view, 2)
        self.setCentralWidget(container)

        # ------------------------------------------------------------
        # WebChannel (клік по карті) — підключаємо ПІСЛЯ setPage(...)
        # ------------------------------------------------------------
        self._bridge = None
        self._channel = None
        if WEBCHANNEL_AVAILABLE:
            self._bridge = MapBridge()
            self._bridge.picked.connect(self._on_map_picked)
            self._channel = QWebChannel(self.web_view.page())
            self._channel.registerObject("bridge", self._bridge)
            self.web_view.page().setWebChannel(self._channel)

        # ------------------------------------------------------------
        # Стан
        # ------------------------------------------------------------
        self._threads = []
        self._last_G = None
        self._last_gdf_edges = None
        self._last_gdf_all_poi = None
        self._last_safe_city = None
        self._last_city = None

        self._pick_target = ""
        self._picked_start = None
        self._picked_end = None
        self._picked_stops = []  # list[(lat, lon)]

        # щоб одразу було зрозуміло, яка країна вибрана за замовчуванням
        self.country_selected()
        self._update_route_mode_ui()

    # -------- helpers --------
    def _set_status(self, msg: str):
        self.progress_label.setText(msg)

    def _set_busy(self, busy: bool, *, show_progress: bool = True):
        self.btn_build.setEnabled(not busy)
        self.btn_clear_cache.setEnabled(not busy)
        self.btn_route.setEnabled(not busy)
        self.btn_isochrone.setEnabled(not busy)
        self.btn_accessibility.setEnabled(not busy)

        self.progress_bar.setVisible(show_progress and busy)
        if busy:
            self.progress_bar.setValue(0)

    def _cleanup_thread(self, thread_obj: QThread):
        try:
            self._threads.remove(thread_obj)
        except ValueError:
            pass
        try:
            thread_obj.deleteLater()
        except Exception:
            pass

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

    def _ui_via_category_value(self) -> str:
        """
        Переклад з UI у те, що очікує path_finder/visualizer.
        """
        t = (self.via_category.currentText() or "").strip().lower()
        if "нема" in t:
            return "none"
        if "парк" in t:
            return "park"
        return "none"

    def _update_route_mode_ui(self):
        mode = self.route_mode.currentText()

        show_coords = (mode in ("Координати", "Клік по карті"))
        show_addr = (mode == "Адреса")
        show_click = (mode == "Клік по карті")

        for w in self.coords_widgets:
            w.setVisible(show_coords)

        for w in self.addr_widgets:
            w.setVisible(show_addr)

        for w in self.click_widgets:
            w.setVisible(show_click)

        if mode == "Координати":
            self.route_start.setReadOnly(False)
            self.route_end.setReadOnly(False)
        else:
            self.route_start.setReadOnly(True)
            self.route_end.setReadOnly(True)

        if show_click and not WEBCHANNEL_AVAILABLE:
            self._set_status("Клік по карті недоступний: немає QtWebChannel у вашому оточенні.")

    def _on_web_load_finished(self, ok: bool):
        if not ok:
            return
        if WEBCHANNEL_AVAILABLE:
            tgt = self._pick_target or ""
            cat = self._ui_via_category_value()
            self.web_view.page().runJavaScript(f"window.__pick_target = {json.dumps(tgt)};")
            self.web_view.page().runJavaScript(f"window.__pick_category = {json.dumps(cat)};")

    def _set_pick_target(self, target: str):
        if not WEBCHANNEL_AVAILABLE:
            QMessageBox.warning(self, "Клік по карті", "QtWebChannel недоступний у вашому середовищі.")
            return
        if target not in ("start", "end", "stop"):
            target = ""
        self._pick_target = target

        if target == "start":
            self._set_status("Клік по карті: вибери СТАРТ на мапі")
        elif target == "end":
            self._set_status("Клік по карті: вибери ФІНІШ на мапі")
        elif target == "stop":
            self._set_status("Клік по карті: додавай STOP-и (можна кілька).")

        self.web_view.page().runJavaScript(f"window.__pick_target = {json.dumps(target)};")
        self.web_view.page().runJavaScript(f"window.__pick_category = {json.dumps(self._ui_via_category_value())};")

    def _on_map_picked(self, lat: float, lon: float, target: str, category: str):
        tgt = (target or self._pick_target or "").strip().lower()
        if not tgt:
            return

        if tgt == "start":
            self._picked_start = (lat, lon)
            self.route_start.setText(f"{lat:.6f}, {lon:.6f}")
            self._set_status("Старт обрано. Тепер вибери фініш / stop-и або будуй маршрут.")
            logger.info("UI: picked START on map: lat=%.6f lon=%.6f", lat, lon)

        elif tgt == "end":
            self._picked_end = (lat, lon)
            self.route_end.setText(f"{lat:.6f}, {lon:.6f}")
            self._set_status("Фініш обрано. Можеш будувати маршрут.")
            logger.info("UI: picked END on map: lat=%.6f lon=%.6f", lat, lon)

        elif tgt == "stop":
            self._picked_stops.append((lat, lon))
            self._set_status(f"Додано stop #{len(self._picked_stops)}: {lat:.6f}, {lon:.6f}")
            logger.info("UI: picked STOP #%d: lat=%.6f lon=%.6f", len(self._picked_stops), lat, lon)

    def _clear_stops(self):
        self._picked_stops = []
        self._set_status("Stops очищено.")
        if WEBCHANNEL_AVAILABLE:
            self.web_view.page().runJavaScript("""
                try {
                  if (window.__stopMarkers && window.__stopMarkers.length) {
                    for (var i=0;i<window.__stopMarkers.length;i++) { window.__stopMarkers[i].remove(); }
                  }
                  window.__stopMarkers = [];
                } catch(e) {}
            """)

    # -------- city search --------
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

    # -------- address search --------
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

        chosen = combo.currentData()
        if chosen:
            if is_start:
                self.route_start.setText(f"{chosen[0]:.6f}, {chosen[1]:.6f}")
            else:
                self.route_end.setText(f"{chosen[0]:.6f}, {chosen[1]:.6f}")

    def _on_address_error(self, msg: str, is_start: bool):
        logger.warning("Address error: %s", msg)
        QMessageBox.warning(self, "Адреса", msg)
        self._set_status("")

    # -------- build map / route / isochrone --------
    def build_map(self):
        city = self._current_city_for_address()
        if not city:
            QMessageBox.information(self, "Увага", "Спочатку оберіть або знайдіть місто.")
            return

        if city not in self.cached_cities:
            self.cached_cities.append(city)
            save_cached_cities(self.cached_cities)
            self.prev_cities.addItem(city)

        required_tokens = extract_required_tokens(city)

        self._set_busy(True)

        # скидаємо вибори для кліків
        self._picked_start = None
        self._picked_end = None
        self._picked_stops = []
        self._pick_target = ""
        if WEBCHANNEL_AVAILABLE:
            self.web_view.page().runJavaScript("window.__pick_target = '';")
            self.web_view.page().runJavaScript("window.__pick_category = 'none';")

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

        poi_worker = AllPOIWorker(city_name, extract_required_tokens(city_name))
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
            route_via_category="none",
            route_stops=[],
            enable_click_pick=WEBCHANNEL_AVAILABLE,
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
        if (
            self._last_G is None or self._last_gdf_edges is None
            or self._last_city is None or self._last_safe_city is None
        ):
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
        mode = self.route_mode.currentText()
        via = self._ui_via_category_value()

        logger.info(
            "UI: build route requested | mode=%s | alg=%s | start=%s | end=%s | stops=%d | via=%s",
            mode, alg, start_latlon, end_latlon, len(self._picked_stops), via
        )
        try:
            extra = ""
            if self._picked_stops:
                extra = f" | stops={len(self._picked_stops)}"
            elif via and via != "none":
                extra = f" | via={via}"

            self._set_status(
                f"Маршрут: {mode} | alg={alg}{extra} | "
                f"старт {start_latlon[0]:.6f},{start_latlon[1]:.6f} → "
                f"фініш {end_latlon[0]:.6f},{end_latlon[1]:.6f}"
            )
        except Exception:
            self._set_status(f"Маршрут: {mode} | alg={alg} | start={start_latlon} → end={end_latlon}")

        self._set_busy(True)

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
            route_via_category=via,
            route_stops=list(self._picked_stops),
            enable_click_pick=WEBCHANNEL_AVAILABLE,
        )
        mworker.status.connect(self._set_status)
        mworker.progress.connect(self.progress_bar.setValue)
        mworker.finished.connect(self._on_map_ready)
        mworker.error.connect(self._on_build_error)

        mworker.finished.connect(lambda *_: self._cleanup_thread(mworker))
        mworker.error.connect(lambda *_: self._cleanup_thread(mworker))

        self._threads.append(mworker)
        mworker.start()

    # -------- isochrone --------
    def _parse_iso_minutes(self) -> list:
        t = (self.iso_select.currentText() or "").strip()
        if t.startswith("5/10/15"):
            return [5, 10, 15]
        if t.startswith("5"):
            return [5]
        if t.startswith("10"):
            return [10]
        if t.startswith("15"):
            return [15]
        return [5, 10, 15]

    def _get_isochrone_center_by_mode(self):
        """
        Ізохрона будується від "Старту" у поточному режимі вводу:
          - Координати: беремо route_start
          - Адреса: беремо addr_start_results
          - Клік по карті: беремо _picked_start
        """
        mode = self.route_mode.currentText()

        if mode == "Координати":
            return parse_latlon_text(self.route_start.text(), "Центр ізохрони (Старт)")

        if mode == "Адреса":
            if self.addr_start_results.count() == 0:
                raise ValueError("Спочатку знайди старт через кнопку пошуку адреси.")
            start = self.addr_start_results.currentData()
            if not start:
                raise ValueError("Вибери варіант стартової адреси зі списку.")
            return tuple(start)

        if mode == "Клік по карті":
            if not self._picked_start:
                raise ValueError("Спочатку вибери старт на карті (кнопка 'Вибрати старт на карті').")
            return self._picked_start

        raise ValueError("Невідомий режим вводу.")

    def build_isochrone_only(self):
        if (
            self._last_G is None or self._last_gdf_edges is None
            or self._last_city is None or self._last_safe_city is None
        ):
            QMessageBox.information(self, "Ізохрона", "Спочатку побудуй карту для міста (кнопка 'Побудувати карту').")
            return
        if self._last_gdf_all_poi is None:
            QMessageBox.information(self, "Ізохрона", "POI ще не завантажились. Спробуй ще раз через пару секунд.")
            return

        try:
            center = self._get_isochrone_center_by_mode()
        except Exception as e:
            QMessageBox.warning(self, "Ізохрона", str(e))
            return

        minutes_list = self._parse_iso_minutes()
        level = self._current_level()

        logger.info("UI: build isochrone requested | center=%s | minutes=%s", center, minutes_list)
        try:
            self._set_status(f"Ізохрона: центр {center[0]:.6f},{center[1]:.6f} | хв={minutes_list}")
        except Exception:
            self._set_status(f"Ізохрона: center={center} | minutes={minutes_list}")

        self._set_busy(True)

        mworker = MapRenderWorker(
            self._last_G,
            self._last_gdf_edges,
            self._last_gdf_all_poi,
            self._last_safe_city,
            self._last_city,
            level,

            # маршрут не будуємо
            route_start_latlon=None,
            route_end_latlon=None,
            route_algorithm="dijkstra",
            route_via_category="none",
            route_stops=[],

            enable_click_pick=WEBCHANNEL_AVAILABLE,

            # ізохрони
            isochrone_center_latlon=center,
            isochrone_minutes=minutes_list,
            isochrone_walk_speed_kmh=4.8,
            isochrone_buffer_m=0.0,
        )

        mworker.status.connect(self._set_status)
        mworker.progress.connect(self.progress_bar.setValue)
        mworker.finished.connect(self._on_map_ready)
        mworker.error.connect(self._on_build_error)

        mworker.finished.connect(lambda *_: self._cleanup_thread(mworker))
        mworker.error.connect(lambda *_: self._cleanup_thread(mworker))

        self._threads.append(mworker)
        mworker.start()

    def build_accessibility_only(self):
        if (
            self._last_G is None or self._last_gdf_edges is None
            or self._last_city is None or self._last_safe_city is None
        ):
            QMessageBox.information(
                self,
                "Карта доступності міста",
                "Спочатку побудуй карту для міста (кнопка 'Побудувати карту')."
            )
            return
        if self._last_gdf_all_poi is None:
            QMessageBox.information(
                self,
                "Карта доступності міста",
                "POI ще не завантажились. Спробуй ще раз через пару секунд."
            )
            return

        level = self._current_level()
        minutes_list = self._parse_iso_minutes()

        # Якщо вибрано 5/10/15 — робимо грубшу сітку, щоб не вмерти по часу
        if len(minutes_list) > 1:
            grid_step_m = 750.0
            grid_max_cells = 110
        else:
            grid_step_m = 600.0
            grid_max_cells = 160

        logger.info(
            "UI: build citywide accessibility requested | minutes=%s | level=%s | step=%.0f | max_cells=%d",
            minutes_list, level, grid_step_m, grid_max_cells
        )

        self._set_status(
            f"Карта доступності міста: хв={minutes_list} | рівень={level} | "
            f"сітка={grid_step_m:.0f} м | max_cells={grid_max_cells}"
        )

        self._set_busy(True)

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
            route_via_category="none",
            route_stops=[],

            enable_click_pick=WEBCHANNEL_AVAILABLE,

            # локальна точка-оцінка не потрібна
            isochrone_center_latlon=None,
            isochrone_minutes=[],
            isochrone_walk_speed_kmh=4.8,
            isochrone_buffer_m=0.0,

            accessibility_center_latlon=None,
            accessibility_minutes=15,
            accessibility_walk_speed_kmh=4.8,

            accessibility_citywide_enabled=True,
            accessibility_grid_minutes=minutes_list,
            accessibility_grid_step_m=grid_step_m,
            accessibility_grid_max_cells=grid_max_cells,
        )

        mworker.status.connect(self._set_status)
        mworker.progress.connect(self.progress_bar.setValue)
        mworker.finished.connect(self._on_map_ready)
        mworker.error.connect(self._on_build_error)

        mworker.finished.connect(lambda *_: self._cleanup_thread(mworker))
        mworker.error.connect(lambda *_: self._cleanup_thread(mworker))

        self._threads.append(mworker)
        mworker.start()

    # -------- common callbacks --------
    def _on_map_ready(self, map_file_path: str):
        self.progress_label.setText("")
        self.progress_bar.setVisible(False)
        self._set_busy(False, show_progress=False)

        logger.info("UI: завантажуємо карту в web_view: %s", map_file_path)
        self.web_view.load(QUrl.fromLocalFile(os.path.abspath(map_file_path)))

    def _on_build_error(self, message: str):
        self.progress_label.setText("")
        self.progress_bar.setVisible(False)
        self._set_busy(False, show_progress=False)

        logger.error("UI: build error: %s", message)
        QMessageBox.critical(self, "Помилка", message)

    def clear_city_cache(self):
        self.cached_cities = []
        save_cached_cities(self.cached_cities)
        self.prev_cities.clear()
        logger.info("UI: кеш міст очищено")
        QMessageBox.information(self, "Кеш", "Список збережених міст очищено.")