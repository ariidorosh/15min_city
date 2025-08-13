import os
import osmnx as ox

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _safe_name(city_name: str) -> str:
    return city_name.lower().replace(",", "").replace(" ", "_")


def get_city_graph(city_name: str, network_type: str = "walk"):
    """Повертає MultiDiGraph. Кешуємо у форматі OSMnx GraphML для сумісності.
    """
    safe = _safe_name(city_name)
    cache_file = os.path.join(CACHE_DIR, f"{safe}.graphml")

    if os.path.exists(cache_file):
        print(f"[INFO] Завантаження з кешу: {cache_file}")
        return ox.load_graphml(cache_file)

    print(f"[INFO] Завантаження мапи міста з OpenStreetMap: {city_name}")
    G = ox.graph_from_place(city_name, network_type=network_type, simplify=True)

    ox.save_graphml(G, cache_file)
    print(f"[INFO] Граф збережено: {cache_file}")
    return G