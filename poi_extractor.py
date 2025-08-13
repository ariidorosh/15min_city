import os
import argparse
from typing import Dict, List, Any
import osmnx as ox
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

ox.settings.use_cache = True
ox.settings.log_console = True


LEVELS_DOC: Dict[str, Dict[str, List[str]]] = {
    "minimum": {
        "education": [
            "Дитсадки/ясла; початкова школа поруч.",
            "OSM: amenity=childcare|kindergarten; amenity=school (school:level=primary).",
        ],
        "health": [
            "Первинна ланка: амбулаторія/клініка/сімейний лікар; аптека.",
            "OSM: amenity=clinic|doctors|pharmacy; healthcare=centre|clinic|doctor|pharmacy.",
        ],
        "culture": [
            "Бібліотека, громадський центр.",
            "OSM: amenity=library|community_centre.",
        ],
        "greens_sport": [
            "Парки/лісосмуги; дитячий майданчик; спортмайданчик; фітнес/спортцентр.",
            "OSM: leisure=park|playground|pitch|fitness_centre|sports_centre; natural=wood.",
        ],
        "shopping_services": [
            "Щоденні покупки + пошта.",
            "OSM: shop=convenience|greengrocer|bakery|butcher; amenity=post_office.",
        ],
        "transport": [
            "Будь-яка зупинка громадського транспорту.",
            "OSM: highway=bus_stop; railway=tram_stop; public_transport=platform.",
        ],
    },
    "medium": {
        "education": ["Як minimum."],
        "health": ["Як minimum."],
        "culture": ["Як minimum."],
        "greens_sport": ["Як minimum."],
        "shopping_services": [
            "Мінімум + супермаркет і місця харчування.",
            "OSM: shop=supermarket; amenity=restaurant|cafe|pub|bar.",
        ],
        "transport": ["Як minimum (пріоритет трамваю де є)."],
    },
    "maximum": {
        "education": [
            "Як minimum + середня школа.",
            "OSM: amenity=school (school:level=secondary).",
        ],
        "health": [
            "Як minimum + денний догляд для літніх.",
            "OSM: amenity=social_facility + social_facility=day_care + social_facility:for=senior.",
        ],
        "culture": [
            "Кіно, театр, музей/галерея, концертні/мистецькі простори.",
            "OSM: amenity=cinema|theatre|arts_centre|music_venue|concert_hall; tourism=museum|gallery.",
        ],
        "greens_sport": [
            "Як minimum + басейн.",
            "OSM: leisure=swimming_pool (+ sport=swimming).",
        ],
        "shopping_services": [
            "Як minimum/medium + ринок.",
            "OSM: amenity=marketplace.",
        ],
        "transport": [
            "Обов’язково трамвайна зупинка (де мережа існує).",
            "OSM: railway=tram_stop або public_transport=platform + tram=yes.",
        ],
    },
}


# Machine-readable запити для OSM
LEVELS_QUERIES: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "minimum": {
        "education": [
            {"tags": {"amenity": ["childcare", "kindergarten"]}},
            {"tags": {"amenity": ["school"]}, "post_filters": {"school:level": ["primary"]}},
        ],
        "health": [
            {"tags": {"amenity": ["pharmacy"]}},
            {"tags": {"amenity": ["clinic", "doctors"]}},
            {"tags": {"healthcare": ["centre", "clinic", "doctor", "pharmacy"]}},
        ],
        "culture": [
            {"tags": {"amenity": ["library", "community_centre"]}},
        ],
        "greens_sport": [
            {"tags": {"leisure": ["park", "playground", "pitch", "fitness_centre", "sports_centre"]}},
            {"tags": {"natural": ["wood"]}},
        ],
        "shopping_services": [
            {"tags": {"shop": ["convenience", "greengrocer", "bakery", "butcher"]}},
            {"tags": {"amenity": ["post_office"]}},
        ],
        "transport": [
            {"tags": {"highway": ["bus_stop"]}},
            {"tags": {"railway": ["tram_stop"]}},
            {"tags": {"public_transport": ["platform"]}},
        ],
    },
    "medium": {
        "education": [
            {"tags": {"amenity": ["childcare", "kindergarten"]}},
            {"tags": {"amenity": ["school"]}, "post_filters": {"school:level": ["primary"]}},
        ],
        "health": [
            {"tags": {"amenity": ["pharmacy"]}},
            {"tags": {"amenity": ["clinic", "doctors"]}},
            {"tags": {"healthcare": ["centre", "clinic", "doctor", "pharmacy"]}},
        ],
        "culture": [
            {"tags": {"amenity": ["library", "community_centre"]}},
        ],
        "greens_sport": [
            {"tags": {"leisure": ["park", "playground", "pitch", "fitness_centre", "sports_centre"]}},
            {"tags": {"natural": ["wood"]}},
        ],
        "shopping_services": [
            {"tags": {"shop": ["convenience", "greengrocer", "bakery", "butcher", "supermarket"]}},
            {"tags": {"amenity": ["post_office", "restaurant", "cafe", "pub", "bar"]}},
        ],
        "transport": [
            {"tags": {"highway": ["bus_stop"]}},
            {"tags": {"railway": ["tram_stop"]}},
            {"tags": {"public_transport": ["platform"]}},
        ],
    },
    "maximum": {
        "education": [
            {"tags": {"amenity": ["childcare", "kindergarten"]}},
            {"tags": {"amenity": ["school"]}, "post_filters": {"school:level": ["primary", "secondary"]}},
        ],
        "health": [
            {"tags": {"amenity": ["pharmacy"]}},
            {"tags": {"amenity": ["clinic", "doctors"]}},
            {"tags": {"healthcare": ["centre", "clinic", "doctor", "pharmacy"]}},
            {"tags": {"amenity": ["social_facility"]},
             "post_filters": {"social_facility": ["day_care"], "social_facility:for": ["senior"]}},
        ],
        "culture": [
            {"tags": {"amenity": ["library", "community_centre", "cinema", "theatre",
                                   "arts_centre", "music_venue", "concert_hall"]}},
            {"tags": {"tourism": ["museum", "gallery"]}},
        ],
        "greens_sport": [
            {"tags": {"leisure": ["park", "playground", "pitch", "fitness_centre",
                                  "sports_centre", "swimming_pool"]}},
            {"tags": {"natural": ["wood"]}},
            {"tags": {"sport": ["swimming"]}},
        ],
        "shopping_services": [
            {"tags": {"shop": ["convenience", "greengrocer", "bakery", "butcher", "supermarket"]}},
            {"tags": {"amenity": ["post_office", "restaurant", "cafe", "pub", "bar", "marketplace"]}},
        ],
        "transport": [
            {"tags": {"railway": ["tram_stop"]}},
            {"tags": {"public_transport": ["platform"]}},
        ],
    },
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def representative_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(
        lambda geom: geom.representative_point() if not isinstance(geom, Point) else geom
    )
    return gdf


def apply_post_filters(gdf: gpd.GeoDataFrame, post_filters: Dict[str, List[str]]) -> gpd.GeoDataFrame:
    if gdf.empty or not post_filters:
        return gdf
    mask = None
    for col, allowed in post_filters.items():
        s = gdf[col] if col in gdf.columns else pd.Series([None] * len(gdf))
        part = s.isin(allowed)
        mask = part if mask is None else (mask & part)
    return gdf[mask] if mask is not None else gdf


def fetch_category(place: str, queries: List[Dict[str, Any]]) -> gpd.GeoDataFrame:
    frames = []
    for q in queries:
        tags = q.get("tags", {})
        post = q.get("post_filters", {})
        try:
            g = ox.geometries_from_place(place, tags)
        except Exception as e:
            print(f"[warn] query failed {tags}: {e}")
            continue
        if g.empty:
            continue
        g = apply_post_filters(g, post)
        if g.empty:
            continue
        frames.append(g)

    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    g_all = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))
    try:
        g_all = g_all.to_crs(4326)
    except Exception:
        g_all.set_crs(4326, inplace=True)

    g_all = representative_points(g_all)
    subset_cols = [c for c in ["osmid", "name"] if c in g_all.columns]
    g_all = g_all.drop_duplicates(subset=subset_cols + ["geometry"]) if subset_cols else g_all.drop_duplicates("geometry")
    return g_all


def save_category(gdf: gpd.GeoDataFrame, out_path: str) -> None:
    if gdf.empty:
        print(f"[warn] empty -> skip save: {out_path}")
        return
    gdf.reset_index(drop=True).to_file(out_path, driver="GeoJSON")
    print(f"[ok] saved {out_path} ({len(gdf)} features)")


def export_level(place: str, level_name: str) -> None:
    print(f"\n=== LEVEL: {level_name} ===")
    level_cfg = LEVELS_QUERIES[level_name]
    base_dir = os.path.join("data", "raw", level_name)
    ensure_dir(base_dir)

    try:
        boundary = ox.geocode_to_gdf(place)
        boundary.to_file(os.path.join(base_dir, "boundary.geojson"), driver="GeoJSON")
        print(f"[ok] boundary saved for {place}")
    except Exception as e:
        print(f"[warn] boundary failed: {e}")

    for category, queries in level_cfg.items():
        gdf = fetch_category(place, queries)
        out = os.path.join(base_dir, f"poi_{category}.geojson")
        save_category(gdf, out)

    doc_lines = ["# Description of level: " + level_name]
    for cat, lines in LEVELS_DOC[level_name].items():
        doc_lines.append(f"\n## {cat}")
        for line in lines:
            doc_lines.append(f"- {line}")
    with open(os.path.join(base_dir, "_README.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(doc_lines))
    print(f"[ok] doc saved for level {level_name}")


def main(place: str, level: str):
    if level == "all":
        for lv in ["minimum", "medium", "maximum"]:
            export_level(place, lv)
    else:
        if level not in LEVELS_QUERIES:
            raise ValueError(f"Unknown level: {level}. Use one of minimum|medium|maximum|all.")
        export_level(place, level)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--place", default="Lviv, Ukraine", help="назва населеного пункту")
    ap.add_argument("--level", default="all", help="minimum|medium|maximum|all")
    args = ap.parse_args()
    main(args.place, args.level)
