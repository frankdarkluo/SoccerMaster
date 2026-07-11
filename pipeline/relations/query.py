from __future__ import annotations

import math
import statistics


PLAYER_QUANTITIES = {"x", "y", "speed", "rel_x", "rel_y", "dist_ball",
                     "dist_nearest_opp", "depth_vs_line"}
TEAM_QUANTITIES = {"n_within_15m_of_ball", "opp_line_x"}
AGGS = {"min": min, "max": max, "mean": statistics.fmean,
        "last": lambda values: values[-1]}
QUANTITIES = PLAYER_QUANTITIES | TEAM_QUANTITIES | {"ball_speed"}
PREDICATES = {
    ">": lambda value, threshold: value > threshold,
    ">=": lambda value, threshold: value >= threshold,
    "<": lambda value, threshold: value < threshold,
    "<=": lambda value, threshold: value <= threshold,
    "==": lambda value, threshold: value == threshold,
}


def _valid_predicate(predicate: object) -> bool:
    return (isinstance(predicate, dict) and predicate.get("op") in PREDICATES
            and type(predicate.get("threshold")) in (int, float)
            and math.isfinite(predicate["threshold"]))


def predicate_passes(result: dict, predicate: dict) -> bool:
    op = predicate.get("op") if isinstance(predicate, dict) else None
    threshold = predicate.get("threshold") if isinstance(predicate, dict) else None
    return (
        op in PREDICATES
        and type(threshold) in (int, float)
        and math.isfinite(threshold)
        and result.get("value") is not None
        and PREDICATES[op](result["value"], threshold)
    )


def _unsupported(reason: str) -> dict:
    return {"value": None, "n_samples": 0, "note": f"UNSUPPORTED: {reason}"}


def _query_times(query: dict) -> tuple[float, float] | None:
    t0, t1 = query.get("t0"), query.get("t1")
    if (type(t0) not in (int, float) or type(t1) not in (int, float)
            or not all(math.isfinite(value) for value in (t0, t1)) or t0 > t1):
        return None
    return float(t0), float(t1)


def resolve_query(relations: dict, query: dict) -> dict:
    """Resolve one supported self-ask query against snapshots in its time range."""
    if not isinstance(query, dict):
        return _unsupported("query must be an object")
    times = _query_times(query)
    if times is None:
        return _unsupported("t0 and t1 must be ordered finite JSON numbers")
    t0, t1 = times
    quantity = query.get("quantity")
    if not isinstance(quantity, str) or quantity not in QUANTITIES:
        return _unsupported(f"unknown quantity {quantity!r}")
    agg_name = query.get("agg")
    if not isinstance(agg_name, str) or agg_name not in AGGS:
        return _unsupported(f"unknown aggregation {agg_name!r}")
    if "jersey" in query and not isinstance(query["jersey"], str):
        return _unsupported("jersey must be a string")
    team = query.get("team")
    if team is not None and (not isinstance(team, str) or team not in {"left", "right"}):
        return _unsupported("team must be left or right")
    if quantity in TEAM_QUANTITIES and team is None:
        return _unsupported("team quantity requires team")
    if quantity in PLAYER_QUANTITIES and not isinstance(query.get("jersey"), str):
        return _unsupported("player quantity requires jersey")
    agg = AGGS[agg_name]
    values = []
    for snapshot in relations.get("snapshots", []):
        if not t0 - 1e-6 <= snapshot["t"] <= t1 + 1e-6:
            continue
        if quantity == "ball_speed":
            values.append(snapshot["ball"]["speed"])
        elif quantity in TEAM_QUANTITIES:
            value = snapshot["teams"].get(query["team"], {}).get(quantity)
            if value is not None:
                values.append(value)
        elif quantity in PLAYER_QUANTITIES:
            for player in snapshot["players"]:
                if (str(player.get("jersey")) == str(query.get("jersey"))
                        and (not query.get("team")
                             or player["team"] == query["team"])
                        and quantity in player):
                    values.append(player[quantity])
    if not values:
        return {"value": None, "n_samples": 0,
                "note": "NO DATA in range — claim unsupported"}
    return {"value": round(agg(values), 2), "n_samples": len(values)}
