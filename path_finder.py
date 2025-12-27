# path_finder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal, Iterable
import math

import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, MultiPoint, Polygon, Point

from logger_config import logger


LatLon = Tuple[float, float]  # (lat, lon)
Algorithm = Literal["dijkstra", "astar"]
SnapMode = Literal["edge", "node"]


# ============================================================
# Нормальні помилки з контекстом
# ============================================================
@dataclass(frozen=True)
class ErrorContext:
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.data)


class PathfinderError(Exception):
    default_code = "PATHFINDER_ERROR"

    def __init__(
        self,
        message: str = "Помилка в модулі path_finder",
        *,
        code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.code: str = code or self.default_code
        self.message: str = str(message)
        self.context: ErrorContext = ErrorContext(context or {})
        self.cause: Optional[BaseException] = cause

        super().__init__(self.message)

        if cause is not None:
            self.__cause__ = cause

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "type": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "context": self.context.to_dict(),
        }
        if self.cause is not None:
            out["cause"] = {
                "type": type(self.cause).__name__,
                "message": str(self.cause),
            }
        return out

    def __str__(self) -> str:
        ctx = self.context.to_dict()
        if not ctx:
            return f"[{self.code}] {self.message}"
        compact = ", ".join(f"{k}={ctx[k]!r}" for k in sorted(ctx.keys()))
        return f"[{self.code}] {self.message} | {compact}"


class InvalidCoordinateError(PathfinderError):
    default_code = "INVALID_COORDINATE"


class PathNotFoundError(PathfinderError):
    default_code = "PATH_NOT_FOUND"


# ============================================================
# Результати
# ============================================================
@dataclass(frozen=True)
class PathResult:
    start_node: int
    end_node: int
    nodes: List[int]
    coords: List[LatLon]          # [(lat, lon), ...]
    length_m: float               # довжина в метрах
    algorithm: Algorithm
    weight: str


@dataclass(frozen=True)
class IsochroneResult:
    center_node: int
    cutoff: float
    weight: str
    nodes: List[int]
    costs: Dict[int, float]
    polygon: Optional[Polygon]


@dataclass(frozen=True)
class SnapInfo:
    """
    input_latlon        — що вибрав користувач (будинок/клік)
    snapped_latlon      — точка НА дорозі (проєкція на найближче ребро)
    chosen_node         — (інформативно) "найближчий" вузол, але маршрут може вибрати інший кінець ребра
    edge_u/edge_v/key   — ребро, на яке спроєктувались
    edge_pos_01         — позиція snapped уздовж ребра (0..1) в напрямку u->v (після орієнтації геометрії)
    edge_length_m       — довжина ребра в метрах (OSMnx 'length' або fallback)
    edge_cost_to_u_m    — скільки метрів "по ребру" від snapped до u
    edge_cost_to_v_m    — скільки метрів "по ребру" від snapped до v
    """
    input_latlon: LatLon
    snapped_latlon: LatLon
    chosen_node: int
    chosen_node_latlon: LatLon
    mode: SnapMode
    dist_to_snapped_m: float
    dist_to_node_m: float
    edge_u: Optional[int] = None
    edge_v: Optional[int] = None
    edge_key: Optional[int] = None
    edge_pos_01: Optional[float] = None
    edge_length_m: Optional[float] = None
    edge_cost_to_u_m: Optional[float] = None
    edge_cost_to_v_m: Optional[float] = None


# ============================================================
# Утиліти
# ============================================================
def _validate_latlon(p: LatLon, name: str = "point") -> None:
    if not isinstance(p, (tuple, list)) or len(p) != 2:
        raise InvalidCoordinateError(
            f"{name}: очікую (lat, lon), отримав: {p!r}",
            context={"param": name, "value": p},
        )

    lat, lon = p
    try:
        lat = float(lat)
        lon = float(lon)
    except Exception as e:
        raise InvalidCoordinateError(
            f"{name}: координати мають бути числами, отримав: {p!r}",
            context={"param": name, "value": p},
            cause=e,
        )

    if math.isnan(lat) or math.isnan(lon):
        raise InvalidCoordinateError(
            f"{name}: координати не можуть бути NaN: {p!r}",
            context={"param": name, "value": p},
        )

    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        raise InvalidCoordinateError(
            f"{name}: координати поза діапазоном: {p!r}",
            context={"param": name, "value": p},
        )


def _nearest_nodes_func():
    if hasattr(ox, "distance") and hasattr(ox.distance, "nearest_nodes"):
        return ox.distance.nearest_nodes
    if hasattr(ox, "nearest_nodes"):
        return ox.nearest_nodes
    raise PathfinderError(
        "Не знайдено nearest_nodes у встановленому OSMnx",
        context={"osmnx_has_distance": hasattr(ox, "distance"), "osmnx_version": getattr(ox, "__version__", None)},
    )


def _nearest_edges_func():
    if hasattr(ox, "distance") and hasattr(ox.distance, "nearest_edges"):
        return ox.distance.nearest_edges
    if hasattr(ox, "nearest_edges"):
        return ox.nearest_edges
    if hasattr(ox, "distance") and hasattr(ox.distance, "get_nearest_edge"):
        return ox.distance.get_nearest_edge
    if hasattr(ox, "get_nearest_edge"):
        return ox.get_nearest_edge
    return None


def _node_latlon(G, node_id: int) -> LatLon:
    data = G.nodes[int(node_id)]
    return (float(data["y"]), float(data["x"]))  # (lat, lon)


def _haversine_m(a: LatLon, b: LatLon) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    s = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(min(1.0, math.sqrt(s)))
    return 6371000.0 * c


def nearest_node(G, latlon: LatLon) -> int:
    _validate_latlon(latlon, "latlon")
    lat, lon = float(latlon[0]), float(latlon[1])
    nn = _nearest_nodes_func()
    try:
        return int(nn(G, lon, lat))  # X=lon, Y=lat
    except Exception as e:
        raise PathfinderError(
            "Не вдалося знайти найближчий вузол у графі",
            context={"latlon": latlon},
            cause=e,
        )


def _pick_edge_tuple(raw) -> Tuple[int, int, Optional[int]]:
    if isinstance(raw, (tuple, list)):
        if len(raw) == 3:
            return int(raw[0]), int(raw[1]), int(raw[2])
        if len(raw) == 2:
            return int(raw[0]), int(raw[1]), None
    raise PathfinderError("Неочікуваний формат nearest_edges()", context={"raw": raw})


def _edge_attrs_exact(G, u: int, v: int, key: Optional[int]) -> Dict[str, Any]:
    data = G.get_edge_data(u, v)
    if not data:
        return {}
    if isinstance(data, dict) and key is None:
        k0 = sorted(list(data.keys()))[0]
        return data.get(k0) or {}
    if isinstance(data, dict) and key is not None:
        return data.get(key) or {}
    return {}


def _best_edge_attrs(G, u: int, v: int, weight: str) -> Dict[str, Any]:
    """
    Між u->v може бути кілька ребер (MultiDiGraph).
    Беремо ребро з мінімальним weight (як логіка shortest_path).
    """
    data = G.get_edge_data(u, v) or {}
    best_attrs: Dict[str, Any] = {}
    best_w = float("inf")

    for _k, attrs in data.items():
        w = attrs.get(weight)
        try:
            w = float(w)
        except Exception:
            w = float("inf")

        if w < best_w:
            best_w = w
            best_attrs = attrs or {}

    if not best_attrs and isinstance(data, dict) and data:
        try:
            k0 = sorted(list(data.keys()))[0]
            best_attrs = data.get(k0) or {}
        except Exception:
            best_attrs = {}

    return best_attrs


def _orient_line_lonlat_to_uv(
    line_lonlat: List[Tuple[float, float]],
    u_ll: LatLon,
    v_ll: LatLon
) -> List[Tuple[float, float]]:
    """
    У shapely coords: (lon, lat).
    Розвертаємо так, щоб шлях ішов від u до v.
    """
    if not line_lonlat or len(line_lonlat) < 2:
        return line_lonlat

    lon0, lat0 = line_lonlat[0]
    lon1, lat1 = line_lonlat[-1]

    a_ll = (float(lat0), float(lon0))  # start
    b_ll = (float(lat1), float(lon1))  # end

    score_keep = _haversine_m(a_ll, u_ll) + _haversine_m(b_ll, v_ll)
    score_flip = _haversine_m(a_ll, v_ll) + _haversine_m(b_ll, u_ll)

    if score_flip < score_keep:
        return list(reversed(line_lonlat))
    return line_lonlat


def _geometry_from_edge(
    G,
    u: int,
    v: int,
    *,
    key: Optional[int] = None,
    prefer_exact_key: bool = True,
    weight: str = "length"
) -> Tuple[LineString, float]:
    """
    Повертає (geom_lonlat, edge_length_m).

    geom_lonlat орієнтований u->v.
    edge_length_m беремо з attrs["length"] або fallback на haversine(u,v).
    """
    u_ll = _node_latlon(G, u)
    v_ll = _node_latlon(G, v)

    if prefer_exact_key:
        attrs = _edge_attrs_exact(G, u, v, key)
        if not attrs:
            attrs = _best_edge_attrs(G, u, v, weight=weight)
    else:
        attrs = _best_edge_attrs(G, u, v, weight=weight)

    edge_len_m = attrs.get("length")
    try:
        edge_len_m = float(edge_len_m)
    except Exception:
        edge_len_m = float(_haversine_m(u_ll, v_ll))

    geom = attrs.get("geometry")
    if geom is None:
        geom = LineString([(u_ll[1], u_ll[0]), (v_ll[1], v_ll[0])])
    else:
        try:
            if hasattr(geom, "geom_type") and geom.geom_type == "MultiLineString":
                parts = list(getattr(geom, "geoms", []) or [])
                if parts:
                    parts.sort(key=lambda g: getattr(g, "length", 0.0), reverse=True)
                    geom = parts[0]
                else:
                    geom = LineString([(u_ll[1], u_ll[0]), (v_ll[1], v_ll[0])])
        except Exception:
            geom = LineString([(u_ll[1], u_ll[0]), (v_ll[1], v_ll[0])])

    try:
        line = list(getattr(geom, "coords", []))
        line = _orient_line_lonlat_to_uv(line, u_ll, v_ll)
        geom = LineString(line)
    except Exception:
        geom = LineString([(u_ll[1], u_ll[0]), (v_ll[1], v_ll[0])])

    if edge_len_m <= 0:
        edge_len_m = float(_haversine_m(u_ll, v_ll))

    return geom, float(edge_len_m)


def _interpolate_along_linestring(
    geom: LineString,
    start_d: float,
    end_d: float,
    *,
    edge_length_m: float,
    step_m: float = 5.0,
    max_points: int = 80,
) -> List[LatLon]:
    """
    Інтерполяція по LineString між start_d і end_d (в одиницях geom.length),
    але крок задаємо в метрах, через пропорцію edge_length_m <-> geom.length.
    """
    try:
        total_d = float(abs(end_d - start_d))
        geom_len = float(getattr(geom, "length", 0.0))
    except Exception:
        geom_len = 0.0
        total_d = 0.0

    if geom_len <= 0 or edge_length_m <= 0:
        p0 = geom.interpolate(start_d)
        p1 = geom.interpolate(end_d)
        return [(float(p0.y), float(p0.x)), (float(p1.y), float(p1.x))]

    # крок у "геом-одиницях"
    step_d = (float(step_m) / float(edge_length_m)) * geom_len
    if step_d <= 0:
        step_d = geom_len / 20.0

    n = int(total_d / step_d) + 1
    n = max(1, min(max_points - 1, n))

    out: List[LatLon] = []
    for i in range(n + 1):
        t = i / float(n)
        d = start_d + (end_d - start_d) * t
        p = geom.interpolate(d)
        out.append((float(p.y), float(p.x)))

    return out


def _extend_dedup(out: List[LatLon], seg: List[LatLon]) -> None:
    for p in seg:
        if not out:
            out.append(p)
        else:
            if abs(out[-1][0] - p[0]) > 1e-10 or abs(out[-1][1] - p[1]) > 1e-10:
                out.append(p)


def snap_to_graph(G, latlon: LatLon, *, mode: SnapMode = "edge") -> SnapInfo:
    _validate_latlon(latlon, "latlon")
    in_ll: LatLon = (float(latlon[0]), float(latlon[1]))

    # --- mode=node (fallback) ---
    if mode == "node":
        n = nearest_node(G, in_ll)
        n_ll = _node_latlon(G, n)
        d = _haversine_m(in_ll, n_ll)
        return SnapInfo(
            input_latlon=in_ll,
            snapped_latlon=n_ll,
            chosen_node=int(n),
            chosen_node_latlon=n_ll,
            mode="node",
            dist_to_snapped_m=float(d),
            dist_to_node_m=float(d),
        )

    # --- mode=edge ---
    ne = _nearest_edges_func()
    if ne is None:
        return snap_to_graph(G, in_ll, mode="node")

    lat, lon = in_ll[0], in_ll[1]
    try:
        raw_edge = ne(G, lon, lat)  # X=lon, Y=lat
    except Exception as e:
        logger.warning("snap_to_graph: nearest_edges failed -> fallback to node (%s)", e)
        return snap_to_graph(G, in_ll, mode="node")

    u, v, key = _pick_edge_tuple(raw_edge)

    # геометрія ребра + довжина (в метрах)
    geom, edge_len_m = _geometry_from_edge(G, u, v, key=key, prefer_exact_key=True, weight="length")

    # snapped point (проєкція на геометрію)
    p = Point(lon, lat)
    try:
        proj_d = float(geom.project(p))
        proj_pt = geom.interpolate(proj_d)
        snapped_ll: LatLon = (float(proj_pt.y), float(proj_pt.x))
        geom_len = float(getattr(geom, "length", 0.0))
        pos01 = (proj_d / geom_len) if (geom_len > 0) else 0.0
        pos01 = max(0.0, min(1.0, float(pos01)))
    except Exception:
        return snap_to_graph(G, in_ll, mode="node")

    # скільки "по ребру" до u і до v
    cost_to_u = float(pos01) * float(edge_len_m)
    cost_to_v = (1.0 - float(pos01)) * float(edge_len_m)

    # для інфи залишимо chosen_node як ближчий "по ребру"
    chosen = u if cost_to_u <= cost_to_v else v
    chosen_ll = _node_latlon(G, chosen)

    dist_to_snapped = _haversine_m(in_ll, snapped_ll)
    dist_to_node = _haversine_m(in_ll, chosen_ll)

    return SnapInfo(
        input_latlon=in_ll,
        snapped_latlon=snapped_ll,
        chosen_node=int(chosen),
        chosen_node_latlon=chosen_ll,
        mode="edge",
        dist_to_snapped_m=float(dist_to_snapped),
        dist_to_node_m=float(dist_to_node),
        edge_u=int(u),
        edge_v=int(v),
        edge_key=int(key) if key is not None else None,
        edge_pos_01=float(pos01),
        edge_length_m=float(edge_len_m),
        edge_cost_to_u_m=float(cost_to_u),
        edge_cost_to_v_m=float(cost_to_v),
    )


def nodes_to_coords(G, nodes: Iterable[int]) -> List[LatLon]:
    coords: List[LatLon] = []
    for n in nodes:
        data = G.nodes[int(n)]
        coords.append((float(data["y"]), float(data["x"])))
    return coords


# ============================================================
# Координати маршруту по геометрії ребер (щоб не було "прямих")
# ============================================================
def route_coords_from_edges(G, path_nodes: List[int], weight: str = "length") -> List[LatLon]:
    """
    Формує coords маршруту по geometry ребер (з поворотами),
    а не як "вузол-вузол" прямими.
    """
    if not path_nodes or len(path_nodes) < 2:
        return nodes_to_coords(G, path_nodes)

    out: List[LatLon] = []

    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        u = int(u)
        v = int(v)
        u_ll = _node_latlon(G, u)
        v_ll = _node_latlon(G, v)

        attrs = _best_edge_attrs(G, u, v, weight=weight)
        geom = attrs.get("geometry")

        seg: List[LatLon]

        if geom is not None:
            try:
                if hasattr(geom, "geom_type") and geom.geom_type == "MultiLineString":
                    parts = list(getattr(geom, "geoms", []) or [])
                    if parts:
                        parts.sort(key=lambda g: getattr(g, "length", 0.0), reverse=True)
                        geom_use = parts[0]
                    else:
                        geom_use = None
                else:
                    geom_use = geom

                if geom_use is not None and hasattr(geom_use, "coords"):
                    line = list(geom_use.coords)  # [(lon, lat), ...]
                    line = _orient_line_lonlat_to_uv(line, u_ll, v_ll)
                    seg = [(float(lat), float(lon)) for (lon, lat) in line]
                else:
                    seg = [u_ll, v_ll]
            except Exception:
                seg = [u_ll, v_ll]
        else:
            seg = [u_ll, v_ll]

        _extend_dedup(out, seg)

    return out


def _path_weight_mdg(G, path_nodes: List[int], weight: str) -> float:
    total = 0.0
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        data = G.get_edge_data(u, v)
        if not data:
            raise PathfinderError(
                "Нема ребра між вузлами при підрахунку довжини",
                context={"u": u, "v": v, "weight": weight},
            )

        best = None
        for _k, attrs in data.items():
            w = attrs.get(weight)
            if w is None:
                continue
            try:
                w = float(w)
            except Exception:
                continue
            best = w if best is None else min(best, w)

        if best is None:
            raise PathfinderError(
                "Ребро не має потрібної ваги",
                context={"u": u, "v": v, "weight": weight},
            )

        total += best

    return float(total)


def path_linestring(coords: List[LatLon]) -> LineString:
    if len(coords) < 2:
        return LineString()
    return LineString([(lon, lat) for lat, lon in coords])


def _candidate_costs_for_snap(G, snap: SnapInfo) -> Dict[int, float]:
    """
    Для edge-snap повертаємо два кандидати: u та v з ціною добігання по ребру.
    Для node-snap — тільки chosen_node з ціною 0.
    """
    if snap.mode == "edge" and snap.edge_u is not None and snap.edge_v is not None:
        u = int(snap.edge_u)
        v = int(snap.edge_v)

        cu = snap.edge_cost_to_u_m
        cv = snap.edge_cost_to_v_m

        if cu is None:
            cu = _haversine_m(snap.snapped_latlon, _node_latlon(G, u))
        if cv is None:
            cv = _haversine_m(snap.snapped_latlon, _node_latlon(G, v))

        return {u: float(cu), v: float(cv)}

    return {int(snap.chosen_node): 0.0}


def _edge_segment_from_snapped_to_node(G, snap: SnapInfo, node_id: int, *, weight: str) -> List[LatLon]:
    """
    Повертає coords по геометрії ребра від snapped до вузла (u або v).
    """
    if snap.mode != "edge" or snap.edge_u is None or snap.edge_v is None:
        return []

    u = int(snap.edge_u)
    v = int(snap.edge_v)
    node_id = int(node_id)

    if node_id not in (u, v):
        return []

    geom, edge_len_m = _geometry_from_edge(
        G, u, v, key=snap.edge_key, prefer_exact_key=True, weight=weight
    )

    p = Point(float(snap.snapped_latlon[1]), float(snap.snapped_latlon[0]))
    try:
        d_snapped = float(geom.project(p))
    except Exception:
        return []

    d_node = 0.0 if node_id == u else float(getattr(geom, "length", 0.0))

    seg = _interpolate_along_linestring(
        geom,
        d_snapped,
        d_node,
        edge_length_m=edge_len_m,
        step_m=5.0,
        max_points=80,
    )
    return seg


def _edge_segment_from_node_to_snapped(G, snap: SnapInfo, node_id: int, *, weight: str) -> List[LatLon]:
    """
    Повертає coords по геометрії ребра від вузла (u або v) до snapped.
    """
    if snap.mode != "edge" or snap.edge_u is None or snap.edge_v is None:
        return []

    u = int(snap.edge_u)
    v = int(snap.edge_v)
    node_id = int(node_id)

    if node_id not in (u, v):
        return []

    geom, edge_len_m = _geometry_from_edge(
        G, u, v, key=snap.edge_key, prefer_exact_key=True, weight=weight
    )

    p = Point(float(snap.snapped_latlon[1]), float(snap.snapped_latlon[0]))
    try:
        d_snapped = float(geom.project(p))
    except Exception:
        return []

    d_node = 0.0 if node_id == u else float(getattr(geom, "length", 0.0))

    seg = _interpolate_along_linestring(
        geom,
        d_node,
        d_snapped,
        edge_length_m=edge_len_m,
        step_m=5.0,
        max_points=80,
    )
    return seg


# ============================================================
# Маршрут: Dijkstra / A*
# ============================================================
def find_shortest_path(
    G,
    start_latlon: LatLon,
    end_latlon: LatLon,
    weight: str = "length",
    algorithm: Algorithm = "dijkstra",
    use_undirected: bool = False,
    snap_mode: SnapMode = "edge",
) -> PathResult:
    _validate_latlon(start_latlon, "start_latlon")
    _validate_latlon(end_latlon, "end_latlon")

    Gwork = G.to_undirected() if use_undirected else G

    snap_s = snap_to_graph(Gwork, start_latlon, mode=snap_mode)
    snap_e = snap_to_graph(Gwork, end_latlon, mode=snap_mode)

    start_candidates = _candidate_costs_for_snap(Gwork, snap_s)
    end_candidates = _candidate_costs_for_snap(Gwork, snap_e)

    logger.info(
        "SNAP START: input=%s -> snapped=%s (%.1fm), edge=%s-%s key=%s, costs_to_uv=%s",
        snap_s.input_latlon,
        snap_s.snapped_latlon,
        snap_s.dist_to_snapped_m,
        snap_s.edge_u,
        snap_s.edge_v,
        snap_s.edge_key,
        {k: round(v, 2) for k, v in start_candidates.items()},
    )
    logger.info(
        "SNAP END:   input=%s -> snapped=%s (%.1fm), edge=%s-%s key=%s, costs_to_uv=%s",
        snap_e.input_latlon,
        snap_e.snapped_latlon,
        snap_e.dist_to_snapped_m,
        snap_e.edge_u,
        snap_e.edge_v,
        snap_e.edge_key,
        {k: round(v, 2) for k, v in end_candidates.items()},
    )

    # --------------------------------------------------------
    # Вибір найкращих кінців ребер (u/v) за сумарною довжиною:
    # start_extra + shortest_path_length + end_extra
    # --------------------------------------------------------
    best_pair: Optional[Tuple[int, int]] = None
    best_total: float = float("inf")
    best_core_len: float = float("inf")

    for s_node, s_extra in start_candidates.items():
        for e_node, e_extra in end_candidates.items():
            try:
                core_len = float(nx.shortest_path_length(Gwork, int(s_node), int(e_node), weight=weight))
            except nx.NetworkXNoPath:
                continue
            except nx.NodeNotFound:
                continue
            except Exception:
                continue

            total = float(s_extra) + float(core_len) + float(e_extra)
            if total < best_total:
                best_total = total
                best_pair = (int(s_node), int(e_node))
                best_core_len = float(core_len)

    if best_pair is None:
        raise PathNotFoundError(
            "Маршрут не знайдено між вибраними точками",
            context={
                "start_latlon": start_latlon,
                "end_latlon": end_latlon,
                "start_candidates": list(start_candidates.keys()),
                "end_candidates": list(end_candidates.keys()),
                "algorithm": algorithm,
                "weight": weight,
                "use_undirected": use_undirected,
                "snap_mode": snap_mode,
            },
        )

    start_node, end_node = best_pair
    start_extra = float(start_candidates.get(start_node, 0.0))
    end_extra = float(end_candidates.get(end_node, 0.0))

    # --------------------------------------------------------
    # Основний маршрут по графу (від start_node до end_node)
    # --------------------------------------------------------
    try:
        if algorithm == "dijkstra":
            path_nodes = nx.shortest_path(Gwork, start_node, end_node, weight=weight)
            core_len = float(nx.shortest_path_length(Gwork, start_node, end_node, weight=weight))

        elif algorithm == "astar":
            def heuristic(u: int, v: int) -> float:
                return _haversine_m(_node_latlon(Gwork, u), _node_latlon(Gwork, v))

            path_nodes = nx.astar_path(Gwork, start_node, end_node, heuristic=heuristic, weight=weight)
            core_len = _path_weight_mdg(Gwork, path_nodes, weight=weight)

        else:
            raise PathfinderError("Невідомий алгоритм", context={"algorithm": algorithm, "weight": weight})

    except nx.NetworkXNoPath as e:
        raise PathNotFoundError(
            "Маршрут не знайдено між вибраними точками",
            context={
                "start_latlon": start_latlon,
                "end_latlon": end_latlon,
                "start_node": start_node,
                "end_node": end_node,
                "algorithm": algorithm,
                "weight": weight,
                "use_undirected": use_undirected,
                "snap_mode": snap_mode,
            },
            cause=e,
        )
    except nx.NodeNotFound as e:
        raise PathfinderError(
            "Вузол не знайдено в графі",
            context={"start_node": start_node, "end_node": end_node},
            cause=e,
        )
    except PathfinderError:
        raise
    except Exception as e:
        raise PathfinderError(
            "Помилка побудови маршруту",
            context={
                "start_latlon": start_latlon,
                "end_latlon": end_latlon,
                "start_node": start_node,
                "end_node": end_node,
                "algorithm": algorithm,
                "weight": weight,
                "use_undirected": use_undirected,
                "snap_mode": snap_mode,
            },
            cause=e,
        )

    # --------------------------------------------------------
    # Координати:
    #  - стартуємо в snapped START (на дорозі)
    #  - по ребру до вибраного вузла (з поворотами)
    #  - далі core маршрут по ребрах (з поворотами)
    #  - потім по ребру до snapped END (на дорозі)
    # --------------------------------------------------------
    core_coords = route_coords_from_edges(Gwork, [int(n) for n in path_nodes], weight=weight)

    coords: List[LatLon] = []

    if snap_s.mode == "edge":
        prefix = _edge_segment_from_snapped_to_node(Gwork, snap_s, start_node, weight=weight)
        _extend_dedup(coords, prefix)
    else:
        # node-mode: стартуємо з вузла
        _extend_dedup(coords, [_node_latlon(Gwork, start_node)])

    _extend_dedup(coords, core_coords)

    if snap_e.mode == "edge":
        suffix = _edge_segment_from_node_to_snapped(Gwork, snap_e, end_node, weight=weight)
        _extend_dedup(coords, suffix)
    else:
        # node-mode: кінець у вузлі
        _extend_dedup(coords, [_node_latlon(Gwork, end_node)])

    length = float(core_len) + float(start_extra) + float(end_extra)

    logger.info(
        "Route built: start_node=%s end_node=%s core_nodes=%d coords=%d core_len=%.1f extra=(%.1f+%.1f) total=%.1f",
        start_node,
        end_node,
        len(path_nodes),
        len(coords),
        float(core_len),
        float(start_extra),
        float(end_extra),
        float(length),
    )

    return PathResult(
        start_node=int(start_node),
        end_node=int(end_node),
        nodes=[int(n) for n in path_nodes],
        coords=coords,
        length_m=float(length),
        algorithm=algorithm,
        weight=weight,
    )


# ============================================================
# Ізохрона
# ============================================================
def compute_isochrone(
    G,
    center_latlon: LatLon,
    cutoff: float,
    weight: str = "length",
    use_undirected: bool = False,
    build_polygon: bool = True,
    polygon_buffer_m: float = 0.0,
) -> IsochroneResult:
    _validate_latlon(center_latlon, "center_latlon")
    if cutoff <= 0:
        raise PathfinderError("cutoff має бути > 0", code="INVALID_CUTOFF", context={"cutoff": cutoff, "weight": weight})

    Gwork = G.to_undirected() if use_undirected else G
    center_node = nearest_node(Gwork, center_latlon)

    logger.info("Isochrone: center=%s (node=%s), cutoff=%s, weight=%s", center_latlon, center_node, cutoff, weight)

    try:
        costs = nx.single_source_dijkstra_path_length(Gwork, center_node, cutoff=cutoff, weight=weight)
    except Exception as e:
        raise PathfinderError(
            "Помилка обчислення ізохрони",
            context={"center_node": center_node, "cutoff": cutoff, "weight": weight},
            cause=e,
        )

    nodes = sorted(costs.keys())

    poly: Optional[Polygon] = None
    if build_polygon and nodes:
        pts = [(float(Gwork.nodes[n]["x"]), float(Gwork.nodes[n]["y"])) for n in nodes]  # (lon, lat)
        hull = MultiPoint(pts).convex_hull

        if isinstance(hull, Polygon):
            poly = hull

            if polygon_buffer_m and polygon_buffer_m > 0:
                lat0 = float(center_latlon[0])
                meters_per_deg_lat = 111_320.0
                meters_per_deg_lon = 111_320.0 * max(0.1, math.cos(math.radians(lat0)))
                deg = polygon_buffer_m / max(1.0, (meters_per_deg_lat + meters_per_deg_lon) / 2.0)
                poly = poly.buffer(deg)

    logger.info("Isochrone: nodes=%d", len(nodes))

    return IsochroneResult(
        center_node=int(center_node),
        cutoff=float(cutoff),
        weight=weight,
        nodes=[int(n) for n in nodes],
        costs={int(k): float(v) for k, v in costs.items()},
        polygon=poly,
    )


# ============================================================
# MVP: маршрут через категорію POI (наприклад "park")
# ============================================================

@dataclass(frozen=True)
class ViaStopSelection:
    category: str
    label: str
    poi_latlon: LatLon
    dist_start_to_poi_m: float
    dist_poi_to_end_m: float
    total_m: float


def _poi_candidates_for_category(gdf_all_poi, category: str):
    """
    Повертає GeoDataFrame підмножину POI по категорії.
    MVP: підтримує category='park'
    Працює з форматом (poi_key/poi_value) або з колонкою 'leisure'.
    """
    if gdf_all_poi is None or getattr(gdf_all_poi, "empty", True):
        return None

    cat = (category or "").strip().lower()
    if not cat:
        return None

    if cat == "park":
        # Основний варіант у твоєму пайплайні: poi_key/poi_value
        if ("poi_key" in gdf_all_poi.columns) and ("poi_value" in gdf_all_poi.columns):
            m = (gdf_all_poi["poi_key"] == "leisure") & (gdf_all_poi["poi_value"].astype(str) == "park")
            out = gdf_all_poi[m]
            if not getattr(out, "empty", True):
                return out

        # fallback: якщо є колонка leisure
        if "leisure" in gdf_all_poi.columns:
            out = gdf_all_poi[gdf_all_poi["leisure"].astype(str) == "park"]
            if not getattr(out, "empty", True):
                return out

    return None


def _geom_to_latlon_safe(geom) -> Optional[LatLon]:
    if geom is None:
        return None
    try:
        if geom.is_empty:
            return None
    except Exception:
        pass

    try:
        if isinstance(geom, Point):
            return (float(geom.y), float(geom.x))
        p = geom.representative_point()
        return (float(p.y), float(p.x))
    except Exception:
        return None


def _concat_coords_dedup(a: List[LatLon], b: List[LatLon]) -> List[LatLon]:
    out: List[LatLon] = []
    _extend_dedup(out, a or [])
    _extend_dedup(out, b or [])
    return out


def _concat_nodes_dedup(a: List[int], b: List[int]) -> List[int]:
    if not a:
        return list(b)
    if not b:
        return list(a)
    if int(a[-1]) == int(b[0]):
        return list(a) + list(b[1:])
    return list(a) + list(b)


def find_shortest_path_via_poi_category(
    G,
    gdf_all_poi,
    start_latlon: LatLon,
    end_latlon: LatLon,
    *,
    via_category: str,
    weight: str = "length",
    algorithm: Algorithm = "dijkstra",
    use_undirected: bool = False,
    snap_mode: SnapMode = "edge",
    # швидкість/якість:
    prefilter_max: int = 400,          # скільки парків беремо "поблизу" для кандидатів
    max_candidates_to_check: int = 25, # скільки реально пробуємо маршрутом
) -> Tuple[PathResult, ViaStopSelection]:
    """
    MVP: "пройти через парк" (або іншу категорію) і зробити сумарний шлях найкоротшим.

    Стратегія (проста і надійна):
      1) беремо всі POI категорії (park)
      2) сортуємо по відстані до середини між start/end
      3) беремо top-N кандидатів і для кожного рахуємо:
         len(start->poi) + len(poi->end) реальним роутером find_shortest_path
      4) вибираємо найменший.

    Повертає (PathResult, ViaStopSelection).
    """
    _validate_latlon(start_latlon, "start_latlon")
    _validate_latlon(end_latlon, "end_latlon")

    cat = (via_category or "").strip().lower()
    if not cat:
        raise PathfinderError("via_category порожня", code="INVALID_VIA_CATEGORY")

    gdf = _poi_candidates_for_category(gdf_all_poi, cat)
    if gdf is None or getattr(gdf, "empty", True):
        raise PathfinderError(
            "Не знайдено POI для категорії",
            code="NO_POI_FOR_CATEGORY",
            context={"category": cat},
        )

    # середина між start/end (для “розумного” prefilter)
    mid = ((float(start_latlon[0]) + float(end_latlon[0])) / 2.0, (float(start_latlon[1]) + float(end_latlon[1])) / 2.0)

    scored = []
    for _, row in gdf.iterrows():
        ll = _geom_to_latlon_safe(row.get("geometry"))
        if ll is None:
            continue

        d_mid = _haversine_m(ll, mid)
        label = str(row.get("name") or row.get("brand") or row.get("poi_value") or "POI")
        scored.append((float(d_mid), ll, label))

    if not scored:
        raise PathfinderError(
            "POI категорії є, але не вдалося дістати геометрію/точки",
            code="POI_GEOM_EMPTY",
            context={"category": cat},
        )

    scored.sort(key=lambda x: x[0])

    prefilter_max = max(10, int(prefilter_max))
    max_candidates_to_check = max(3, int(max_candidates_to_check))
    cand_pool = scored[:prefilter_max]
    cand_pool = cand_pool[:max_candidates_to_check]

    best_total = float("inf")
    best_r1: Optional[PathResult] = None
    best_r2: Optional[PathResult] = None
    best_sel: Optional[ViaStopSelection] = None

    for _dmid, poi_ll, label in cand_pool:
        try:
            r1 = find_shortest_path(
                G,
                start_latlon,
                poi_ll,
                weight=weight,
                algorithm=algorithm,
                use_undirected=use_undirected,
                snap_mode=snap_mode,
            )
            r2 = find_shortest_path(
                G,
                poi_ll,
                end_latlon,
                weight=weight,
                algorithm=algorithm,
                use_undirected=use_undirected,
                snap_mode=snap_mode,
            )
        except PathNotFoundError:
            continue
        except Exception:
            continue

        total = float(r1.length_m) + float(r2.length_m)
        if total < best_total:
            best_total = total
            best_r1 = r1
            best_r2 = r2
            best_sel = ViaStopSelection(
                category=cat,
                label=label,
                poi_latlon=poi_ll,
                dist_start_to_poi_m=float(r1.length_m),
                dist_poi_to_end_m=float(r2.length_m),
                total_m=float(total),
            )

    if best_r1 is None or best_r2 is None or best_sel is None:
        raise PathNotFoundError(
            "Не вдалося побудувати маршрут через цю категорію (кандидати не дали валідний шлях)",
            context={"category": cat, "checked": len(cand_pool)},
        )

    full_coords = _concat_coords_dedup(best_r1.coords, best_r2.coords)
    full_nodes = _concat_nodes_dedup(best_r1.nodes, best_r2.nodes)

    result = PathResult(
        start_node=int(best_r1.start_node),
        end_node=int(best_r2.end_node),
        nodes=[int(n) for n in full_nodes],
        coords=full_coords,
        length_m=float(best_sel.total_m),
        algorithm=algorithm,
        weight=weight,
    )

    return result, best_sel
