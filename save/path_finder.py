# path_finder.py
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal
import math

import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, MultiPoint, Polygon

from logger_config import logger
from utils import haversine_m


LatLon = Tuple[float, float]  # (lat, lon)
Algorithm = Literal["dijkstra", "astar"]


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
        message: str = "Помилка в модулі pathfinder",
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
    length_m: float               # якщо weight='length' -> метри, інакше "вартість" у одиницях weight
    algorithm: Algorithm
    weight: str


@dataclass(frozen=True)
class IsochroneResult:
    center_node: int
    cutoff: float                 # у одиницях weight (метри для 'length')
    weight: str
    nodes: List[int]
    costs: Dict[int, float]       # node -> cost
    polygon: Optional[Polygon]


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


@lru_cache(maxsize=1)
def _nearest_nodes_func():
    """
    Сумісність з різними версіями OSMnx:
    - новіші: ox.distance.nearest_nodes
    - старіші: ox.nearest_nodes
    """
    if hasattr(ox, "distance") and hasattr(ox.distance, "nearest_nodes"):
        return ox.distance.nearest_nodes
    if hasattr(ox, "nearest_nodes"):
        return ox.nearest_nodes
    raise PathfinderError(
        "Не знайдено nearest_nodes у встановленому OSMnx",
        context={"osmnx_has_distance": hasattr(ox, "distance"), "osmnx_version": getattr(ox, "__version__", None)},
    )


def _node_latlon(G, node: int) -> LatLon:
    data = G.nodes[node]
    # OSMnx convention: x=lon, y=lat
    return (float(data["y"]), float(data["x"]))


def nearest_node(G, latlon: LatLon) -> int:
    _validate_latlon(latlon, "latlon")
    lat, lon = float(latlon[0]), float(latlon[1])

    nn = _nearest_nodes_func()
    try:
        # У OSMnx порядок: X=lon, Y=lat
        return int(nn(G, lon, lat))
    except Exception as e:
        raise PathfinderError(
            "Не вдалося знайти найближчий вузол у графі",
            context={"latlon": latlon},
            cause=e,
        )


def nodes_to_coords(G, nodes: Iterable[int]) -> List[LatLon]:
    return [_node_latlon(G, int(n)) for n in nodes]


def _path_weight_mdg(G, path_nodes: List[int], weight: str) -> float:
    """
    Підрахунок ваги шляху для MultiDiGraph:
    для кожної пари (u,v) беремо мінімальне значення weight серед паралельних ребер.
    """
    total = 0.0
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        data = G.get_edge_data(u, v)
        if not data:
            raise PathfinderError(
                "Нема ребра між вузлами при підрахунку ваги",
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
    """LineString в (lon, lat) — так зручніше для геометрій."""
    if len(coords) < 2:
        return LineString()
    return LineString([(lon, lat) for lat, lon in coords])


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
) -> PathResult:
    _validate_latlon(start_latlon, "start_latlon")
    _validate_latlon(end_latlon, "end_latlon")

    Gwork = G.to_undirected() if use_undirected else G

    start_node = nearest_node(Gwork, start_latlon)
    end_node = nearest_node(Gwork, end_latlon)

    logger.info(
        "Pathfinder: start=%s -> end=%s (nodes %s -> %s), alg=%s, weight=%s",
        start_latlon, end_latlon, start_node, end_node, algorithm, weight,
    )

    try:
        if algorithm == "dijkstra":
            path_nodes = nx.shortest_path(Gwork, start_node, end_node, weight=weight)
            length = float(nx.shortest_path_length(Gwork, start_node, end_node, weight=weight))

        elif algorithm == "astar":
            def heuristic(u: int, v: int) -> float:
                return haversine_m(_node_latlon(Gwork, u), _node_latlon(Gwork, v))

            path_nodes = nx.astar_path(Gwork, start_node, end_node, heuristic=heuristic, weight=weight)
            length = _path_weight_mdg(Gwork, path_nodes, weight=weight)

        else:
            raise PathfinderError(
                "Невідомий алгоритм",
                context={"algorithm": algorithm, "weight": weight},
            )

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
            },
            cause=e,
        )

    coords = nodes_to_coords(Gwork, path_nodes)

    logger.info("Pathfinder: route nodes=%d, cost=%.1f (weight=%s)", len(path_nodes), length, weight)

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
# Ізохрона (зона досяжності)
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
        raise PathfinderError(
            "cutoff має бути > 0",
            code="INVALID_CUTOFF",
            context={"cutoff": cutoff, "weight": weight},
        )

    Gwork = G.to_undirected() if use_undirected else G
    center_node = nearest_node(Gwork, center_latlon)

    logger.info(
        "Isochrone: center=%s (node %s), cutoff=%s, weight=%s",
        center_latlon, center_node, cutoff, weight,
    )

    try:
        costs = nx.single_source_dijkstra_path_length(Gwork, center_node, cutoff=cutoff, weight=weight)
    except Exception as e:
        raise PathfinderError(
            "Помилка обчислення ізохрони",
            context={"center_node": center_node, "cutoff": cutoff, "weight": weight},
            cause=e,
        )

    nodes = sorted(int(n) for n in costs.keys())

    poly: Optional[Polygon] = None
    if build_polygon and nodes:
        pts = [(float(Gwork.nodes[n]["x"]), float(Gwork.nodes[n]["y"])) for n in nodes]  # (lon, lat)
        hull = MultiPoint(pts).convex_hull

        if isinstance(hull, Polygon):
            poly = hull

            if polygon_buffer_m and polygon_buffer_m > 0 and poly is not None:
                # Грубий буфер в градусах (достатньо для MVP-візуалізації на мапі).
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
        nodes=nodes,
        costs={int(k): float(v) for k, v in costs.items()},
        polygon=poly,
    )
