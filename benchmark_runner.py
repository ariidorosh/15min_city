# benchmark_runner.py
from __future__ import annotations

import os
import time
import csv
from typing import Any, Dict, List, Optional, Tuple

import api

LatLon = Tuple[float, float]


# =========================
# Налаштування
# =========================
CITY = "Lviv, Ukraine"  # як у UI

# Швидкість ходьби для MVP: 4.8 км/год ~= 80 м/хв
WALK_SPEED_M_PER_MIN = 80.0

# Куди писати результати
try:
    from paths import DIR_OUTPUTS
except Exception:
    DIR_OUTPUTS = "outputs"


# =========================
# Сценарії
# =========================
SCENARIOS: Dict[str, Any] = {
    "routes": [
        {
            "name": "A_to_B",
            "start": (49.861047, 24.011570),
            "end": (49.837304, 24.033655),
        },
        {
            "name": "A_to_D",
            "start": (49.861047, 24.011570),
            "end": (49.810151, 24.045424),
        },
        {
            "name": "C_to_B",
            "start": (49.835301, 24.014359),
            "end": (49.837304, 24.033655),
        },
    ],
    "multistop": [
        {
            "name": "A_to_C_to_B",
            "points": [
                (49.861047, 24.011570),  # A
                (49.835301, 24.014359),  # C
                (49.837304, 24.033655),  # B
            ],
        },
        {
            "name": "A_to_C_to_D",
            "points": [
                (49.861047, 24.011570),  # A
                (49.835301, 24.014359),  # C
                (49.810151, 24.045424),  # D
            ],
        },
    ],
    "isochrones": [
        {
            "name": "Iso_A",
            "center": (49.861047, 24.011570),  # A
            "minutes": [5, 10, 15],
        }
    ],
}


# =========================
# Утиліти
# =========================
def _ensure_outputs_dir() -> None:
    os.makedirs(DIR_OUTPUTS, exist_ok=True)


def _ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000.0


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _minutes_to_cutoff_m(minutes: float) -> float:
    return float(minutes) * float(WALK_SPEED_M_PER_MIN)


# =========================
# Bench: routes
# =========================
def bench_routes(G) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    ok = 0
    total = 0
    max_abs_delta_len = 0.0
    sum_dij_ms = 0.0
    sum_ast_ms = 0.0
    cnt_pair = 0

    for sc in SCENARIOS["routes"]:
        total += 1
        name: str = sc["name"]
        start: LatLon = tuple(sc["start"])
        end: LatLon = tuple(sc["end"])

        # Dijkstra
        dij_err: Optional[str] = None
        dij_len: Optional[float] = None
        dij_nodes: Optional[int] = None
        dij_ms: Optional[float] = None

        t0 = time.perf_counter()
        try:
            r_dij = api.build_route(
                G, start=start, end=end, algorithm="dijkstra", weight="length", use_undirected=True
            )
            t1 = time.perf_counter()
            dij_ms = _ms(t0, t1)
            dij_len = float(r_dij.length_m)
            dij_nodes = int(len(r_dij.nodes))
        except Exception as e:
            t1 = time.perf_counter()
            dij_ms = _ms(t0, t1)
            dij_err = str(e)

        # A*
        ast_err: Optional[str] = None
        ast_len: Optional[float] = None
        ast_nodes: Optional[int] = None
        ast_ms: Optional[float] = None

        t2 = time.perf_counter()
        try:
            r_ast = api.build_route(
                G, start=start, end=end, algorithm="astar", weight="length", use_undirected=True
            )
            t3 = time.perf_counter()
            ast_ms = _ms(t2, t3)
            ast_len = float(r_ast.length_m)
            ast_nodes = int(len(r_ast.nodes))
        except Exception as e:
            t3 = time.perf_counter()
            ast_ms = _ms(t2, t3)
            ast_err = str(e)

        # Порівняння
        delta_len = None
        delta_pct = None
        if dij_len is not None and ast_len is not None and dij_len > 0:
            delta_len = ast_len - dij_len
            delta_pct = (delta_len / dij_len) * 100.0
            max_abs_delta_len = max(max_abs_delta_len, abs(delta_len))

        if dij_err is None and ast_err is None:
            ok += 1
            sum_dij_ms += float(dij_ms or 0.0)
            sum_ast_ms += float(ast_ms or 0.0)
            cnt_pair += 1

        rows.append(
            {
                "case": name,
                "start_lat": start[0],
                "start_lon": start[1],
                "end_lat": end[0],
                "end_lon": end[1],
                "dijkstra_ok": dij_err is None,
                "dijkstra_len_m": dij_len,
                "dijkstra_nodes": dij_nodes,
                "dijkstra_ms": dij_ms,
                "dijkstra_err": dij_err,
                "astar_ok": ast_err is None,
                "astar_len_m": ast_len,
                "astar_nodes": ast_nodes,
                "astar_ms": ast_ms,
                "astar_err": ast_err,
                "delta_len_m": delta_len,
                "delta_len_pct": delta_pct,
            }
        )

    summary = {
        "routes_total": total,
        "routes_ok_both": ok,
        "routes_ok_rate": (ok / total) if total else 0.0,
        "routes_avg_dijkstra_ms": (sum_dij_ms / cnt_pair) if cnt_pair else None,
        "routes_avg_astar_ms": (sum_ast_ms / cnt_pair) if cnt_pair else None,
        "routes_max_abs_delta_len_m": max_abs_delta_len,
    }
    return rows, summary


# =========================
# Bench: multistop
# =========================
def bench_multistop(G) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    total = 0
    ok = 0

    for sc in SCENARIOS["multistop"]:
        total += 1
        name: str = sc["name"]
        points: List[LatLon] = [tuple(p) for p in sc["points"]]

        # dijkstra
        dij_err: Optional[str] = None
        dij_ms: Optional[float] = None
        dij_len: Optional[float] = None

        t0 = time.perf_counter()
        try:
            r_dij = api.build_route_multi(
                G, points=points, algorithm="dijkstra", weight="length", use_undirected=True
            )
            t1 = time.perf_counter()
            dij_ms = _ms(t0, t1)
            # ФІКС: у твоєму MultiPathResult поле називається length_m
            dij_len = float(r_dij.length_m)
        except Exception as e:
            t1 = time.perf_counter()
            dij_ms = _ms(t0, t1)
            dij_err = str(e)

        # astar
        ast_err: Optional[str] = None
        ast_ms: Optional[float] = None
        ast_len: Optional[float] = None

        t2 = time.perf_counter()
        try:
            r_ast = api.build_route_multi(
                G, points=points, algorithm="astar", weight="length", use_undirected=True
            )
            t3 = time.perf_counter()
            ast_ms = _ms(t2, t3)
            # ФІКС: у твоєму MultiPathResult поле називається length_m
            ast_len = float(r_ast.length_m)
        except Exception as e:
            t3 = time.perf_counter()
            ast_ms = _ms(t2, t3)
            ast_err = str(e)

        delta_len = None
        delta_pct = None
        if dij_len is not None and ast_len is not None and dij_len > 0:
            delta_len = ast_len - dij_len
            delta_pct = (delta_len / dij_len) * 100.0

        if dij_err is None and ast_err is None:
            ok += 1

        rows.append(
            {
                "case": name,
                "points_count": len(points),
                "dijkstra_ok": dij_err is None,
                "dijkstra_len_m": dij_len,
                "dijkstra_ms": dij_ms,
                "dijkstra_err": dij_err,
                "astar_ok": ast_err is None,
                "astar_len_m": ast_len,
                "astar_ms": ast_ms,
                "astar_err": ast_err,
                "delta_len_m": delta_len,
                "delta_len_pct": delta_pct,
            }
        )

    summary = {
        "multistop_total": total,
        "multistop_ok_both": ok,
        "multistop_ok_rate": (ok / total) if total else 0.0,
    }
    return rows, summary


# =========================
# Bench: isochrones
# =========================
def bench_isochrones(G) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    total = 0
    ok = 0
    sum_ms = 0.0
    cnt = 0

    for sc in SCENARIOS["isochrones"]:
        name: str = sc["name"]
        center: LatLon = tuple(sc["center"])
        minutes_list: List[float] = list(sc["minutes"])

        for minutes in minutes_list:
            total += 1
            cutoff_m = _minutes_to_cutoff_m(minutes)

            err: Optional[str] = None
            runtime_ms: Optional[float] = None
            nodes_reached: Optional[int] = None
            area: Optional[float] = None

            t0 = time.perf_counter()
            try:
                iso = api.build_isochrone(G, center=center, cutoff=cutoff_m, weight="length")
                t1 = time.perf_counter()
                runtime_ms = _ms(t0, t1)

                nodes_reached = int(len(iso.nodes))
                if iso.polygon is not None:
                    area = float(iso.polygon.area)  # для MVP ок (площа в координатах CRS графа)

                ok += 1
                sum_ms += float(runtime_ms or 0.0)
                cnt += 1
            except Exception as e:
                t1 = time.perf_counter()
                runtime_ms = _ms(t0, t1)
                err = str(e)

            rows.append(
                {
                    "case": name,
                    "center_lat": center[0],
                    "center_lon": center[1],
                    "minutes": minutes,
                    "cutoff_m": cutoff_m,
                    "ok": err is None,
                    "runtime_ms": runtime_ms,
                    "nodes_reached": nodes_reached,
                    "polygon_area": area,
                    "err": err,
                }
            )

    summary = {
        "iso_total": total,
        "iso_ok": ok,
        "iso_ok_rate": (ok / total) if total else 0.0,
        "iso_avg_ms": (sum_ms / cnt) if cnt else None,
    }
    return rows, summary


# =========================
# Main
# =========================
def main() -> None:
    _ensure_outputs_dir()

    if not SCENARIOS["routes"] and not SCENARIOS["multistop"] and not SCENARIOS["isochrones"]:
        print("Нема сценаріїв. Заповни SCENARIOS у benchmark_runner.py (координати).")
        return

    print(f"[bench] City = {CITY}")
    print("[bench] Loading city graph...")
    G, ginfo = api.load_city_graph(CITY)
    print(f"[bench] Graph loaded. source={ginfo.get('source')} cache_action={ginfo.get('cache_action')}")

    # Прогрів (щоб перший запуск не псував середні)
    if SCENARIOS["routes"]:
        sc0 = SCENARIOS["routes"][0]
        try:
            api.build_route(
                G,
                start=tuple(sc0["start"]),
                end=tuple(sc0["end"]),
                algorithm="dijkstra",
                weight="length",
                use_undirected=True,
            )
        except Exception:
            pass

    routes_rows, routes_sum = bench_routes(G)
    multi_rows, multi_sum = bench_multistop(G)
    iso_rows, iso_sum = bench_isochrones(G)

    routes_csv = os.path.join(DIR_OUTPUTS, "bench_routes.csv")
    multi_csv = os.path.join(DIR_OUTPUTS, "bench_multistop.csv")
    iso_csv = os.path.join(DIR_OUTPUTS, "bench_isochrones.csv")

    _write_csv(routes_csv, routes_rows)
    _write_csv(multi_csv, multi_rows)
    _write_csv(iso_csv, iso_rows)

    print("\n=== SUMMARY ===")
    for k, v in {**routes_sum, **multi_sum, **iso_sum}.items():
        print(f"{k}: {v}")

    print("\n=== FILES ===")
    if routes_rows:
        print(routes_csv)
    if multi_rows:
        print(multi_csv)
    if iso_rows:
        print(iso_csv)


if __name__ == "__main__":
    main()
