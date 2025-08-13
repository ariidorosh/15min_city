import sys
import os
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
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import graph_builder
from folium.plugins import MarkerCluster
from poi_extractor import LEVELS_QUERIES, fetch_category

CACHE_FILE = os.path.join("cache", "cities.json")
os.makedirs("cache", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


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


class CitySearchWorker(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, country: str, query: str):
        super().__init__()
        self.country = country
        self.query = query

    def run(self):
        try:
            geolocator = Nominatim(user_agent="15min_city_app", timeout=10)
            results = geolocator.geocode(self.query, exactly_one=False, limit=10, addressdetails=True)
            items = []
            for r in results or []:
                country = (r.raw.get("address", {}) or {}).get("country", "")
                if self.country.lower() in country.lower():
                    items.append(r.address)
            if not items:
                self.error.emit("Не знайдено жодного міста у вибраній країні")
            else:
                self.finished.emit(items)
        except (GeocoderTimedOut, GeocoderUnavailable):
            self.error.emit("Сервіс геокодування тимчасово недоступний. Спробуйте ще раз.")
        except Exception as e:
            self.error.emit(f"Помилка пошуку: {e}")


class GraphWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object, str, str)  # (gdf_edges, safe_city_name, city_name)
    error = pyqtSignal(str)

    def __init__(self, city_name: str):
        super().__init__()
        self.city_name = city_name

    def run(self):
        try:
            self.progress.emit(10)
            G = graph_builder.get_city_graph(self.city_name, network_type="walk")
            self.progress.emit(70)
            gdf_edges = ox.utils_graph.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
            self.progress.emit(100)
            safe_city_name = self.city_name.lower().replace(",", "").replace(" ", "_")
            self.finished.emit(gdf_edges, safe_city_name, self.city_name)
        except Exception as e:
            self.error.emit(f"Помилка побудови графа: {e}")


class MapRenderWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, gdf_edges, safe_city_name: str, city_name: str, level_name: str):
        super().__init__()
        self.gdf_edges = gdf_edges
        self.safe_city_name = safe_city_name
        self.city_name = city_name
        self.level_name = level_name  # "minimum" | "medium" | "maximum"

    def run(self):
        try:
            self.progress.emit(10)
            center = self.gdf_edges.geometry.unary_union.centroid
            m = folium.Map(location=[center.y, center.x], zoom_start=13,
                           tiles="cartodbpositron", control_scale=True)
            self.progress.emit(35)

            # Вулиці — вимкнено за замовчуванням
            folium.GeoJson(
                data=self.gdf_edges[["geometry"]].to_json(),
                name="Вулиці",
                show=False,
                style_function=lambda x: {"color": "#4a4a4a", "weight": 2, "opacity": 0.7}
            ).add_to(m)

            # Кордон міста — увімкнено за замовчуванням
            try:
                boundary_gdf = ox.geocode_to_gdf(self.city_name).to_crs(4326)
                folium.GeoJson(
                    data=boundary_gdf[["geometry"]].to_json(),
                    name="Кордон міста",
                    show=True,
                    style_function=lambda x: {"color": "#d9534f", "weight": 3, "fill": False, "opacity": 0.9}
                ).add_to(m)
            except Exception:
                pass

            self.progress.emit(50)

            # POI за вибраним рівнем
            label_map = {
                "education": "Освіта",
                "health": "Медицина",
                "culture": "Культура",
                "greens_sport": "Зелена інфра / Спорт",
                "shopping_services": "Покупки / Сервіси",
                "transport": "Громадський транспорт",
            }

            level_cfg = LEVELS_QUERIES.get(self.level_name, {})
            for category, queries in level_cfg.items():
                try:
                    gdf = fetch_category(self.city_name, queries)  # уже точки, уже CRS=4326
                    if gdf.empty:
                        continue

                    fg = folium.FeatureGroup(name=f"POI — {label_map.get(category, category)} [{self.level_name}]",
                                             show=False)
                    cluster = MarkerCluster().add_to(fg)

                    for _, row in gdf.iterrows():
                        geom = row.geometry
                        if geom is None or geom.is_empty:
                            continue
                        name = str(row.get("name") or row.get("brand") or "Без назви")
                        poi_type = (
                            row.get("amenity")
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
                    print(f"[warn] Не вдалося завантажити POI '{category}': {e}")

            folium.LayerControl(collapsed=False).add_to(m)

            map_file = os.path.join("outputs", f"{self.safe_city_name}_map.html")
            m.save(map_file)
            self.progress.emit(100)
            self.finished.emit(map_file)
        except Exception as e:
            self.error.emit(f"Помилка рендеру карти: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("15-Minute City Map Viewer")

        screen_height = QApplication.primaryScreen().size().height()
        initial_height = int(screen_height * 0.9)
        initial_width = int(initial_height * (1831 / 2048) * 1.3)
        self.setGeometry(100, 100, initial_width, initial_height)

        self.cached_cities = load_cached_cities()
        self.selected_country = None

        container = QWidget()
        self.main_layout = QHBoxLayout(container)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # Ліва панель
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setAlignment(Qt.AlignTop)

        # Вибір країни
        self.left_layout.addWidget(QLabel("Країна:"))
        self.country_select = QComboBox()
        self.country_select.addItems(["Польща", "Україна"])
        self.country_select.currentIndexChanged.connect(self.country_selected)
        self.left_layout.addWidget(self.country_select)

        self.country_confirm = QLabel("Оберіть країну для пошуку міст")
        self.left_layout.addWidget(self.country_confirm)

        # Пошук міста
        self.left_layout.addWidget(QLabel("Пошук міста:"))
        self.city_input = QLineEdit()
        self.city_input.setPlaceholderText("Введіть частину назви міста")
        self.city_button = QToolButton(self.city_input)
        self.city_button.setText("↓")
        self.city_button.clicked.connect(self.search_city)
        self.city_input.setTextMargins(0, 0, 30, 0)
        self.city_input.resizeEvent = self._resize_city_button
        self.left_layout.addWidget(self.city_input)

        # Результати пошуку
        self.found_cities = QComboBox()
        self.left_layout.addWidget(self.found_cities)

        # Попередні міста
        self.left_layout.addWidget(QLabel("Попередні міста:"))
        self.prev_cities = QComboBox()
        self.prev_cities.addItems(self.cached_cities)
        self.left_layout.addWidget(self.prev_cities)

        # НОВЕ: вибір рівня доступності
        self.left_layout.addWidget(QLabel("Рівень доступності (POI):"))
        self.level_select = QComboBox()
        self.level_select.addItems(["medium (середній)", "minimum (база)", "maximum (макс)"])
        self.level_select.setCurrentIndex(0)  # medium за замовчуванням
        self.left_layout.addWidget(self.level_select)

        # Прогрес/статус
        self.progress_label = QLabel("")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.left_layout.addWidget(self.progress_label)
        self.left_layout.addWidget(self.progress_bar)

        # Кнопки
        self.btn_build = QPushButton("Побудувати карту")
        self.btn_clear_cache = QPushButton("Очистити кеш міст")

        for btn in (self.btn_build, self.btn_clear_cache):
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setMinimumHeight(40)

        self.left_layout.addWidget(self.btn_build)
        self.left_layout.addWidget(self.btn_clear_cache)

        self.btn_build.clicked.connect(self.build_map)
        self.btn_clear_cache.clicked.connect(self.clear_city_cache)

        # Ліва панель у скролі
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(self.left_panel)

        # Права панель (карта)
        self.web_view = QWebEngineView()

        # Пропорції 1:2
        self.main_layout.addWidget(left_scroll, 1)
        self.main_layout.addWidget(self.web_view, 2)

        self.setCentralWidget(container)

        # Тримачі потоків
        self._threads = []

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

    def search_city(self):
        if not self.selected_country:
            QMessageBox.information(self, "Увага", "Спочатку оберіть країну.")
            return
        query = self.city_input.text().strip()
        if not query:
            QMessageBox.information(self, "Увага", "Введіть частину назви міста.")
            return

        self.found_cities.clear()
        self.progress_label.setText("Пошук міст…")

        worker = CitySearchWorker(self.selected_country, query)
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

    def _on_search_error(self, message: str):
        self.progress_label.setText("")
        self.found_cities.clear()
        self.found_cities.addItem("Не знайдено")
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

        self.progress_label.setText("Завантаження даних з OSM…")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_build.setEnabled(False)
        self.btn_clear_cache.setEnabled(False)

        worker = GraphWorker(city)
        worker.progress.connect(self.progress_bar.setValue)
        worker.finished.connect(self.on_graph_ready)
        worker.error.connect(self._on_build_error)
        worker.finished.connect(lambda *_: self._cleanup_thread(worker))
        worker.error.connect(lambda *_: self._cleanup_thread(worker))
        self._threads.append(worker)
        worker.start()

    def on_graph_ready(self, gdf_edges, safe_city_name: str, city_name: str):
        self.progress_label.setText("Рендер карти…")
        self.progress_bar.setValue(0)

        level = self._current_level()
        mworker = MapRenderWorker(gdf_edges, safe_city_name, city_name, level)
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
        self.web_view.load(QUrl.fromLocalFile(os.path.abspath(map_file_path)))

    def _on_build_error(self, message: str):
        self.progress_label.setText("")
        self.progress_bar.setVisible(False)
        self.btn_build.setEnabled(True)
        self.btn_clear_cache.setEnabled(True)
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
        QMessageBox.information(self, "Кеш", "Список збережених міст очищено.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
