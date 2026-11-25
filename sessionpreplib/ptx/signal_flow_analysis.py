"""Balance validation, path tracing, and semantic-tier DAG layout for Pro Tools Signal Flow."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import math
from typing import Any

from .parser import PtxSession

_PARALLEL_INF_DB = -140.0
_BALANCE_TOL_DB = 0.15

COLUMN_TITLES: tuple[str, ...] = (
    "AUDIO TRACKS",
    "ROUTING FOLDERS",
    "SUBGROUP AUXES",
    "STEM AUXES",
    "MIX BUS CHAIN",
    "MASTER",
)
COLUMN_WIDTHS: tuple[int, ...] = (156, 148, 184, 184, 198, 160)
COLUMN_GAP_X = 34
ROW_GAP_Y = 16
CANVAS_PAD_X = 16
CANVAS_PAD_Y = 38


@dataclass(frozen=True)
class PreFaderSendCaveat:
    """Active pre-fader send whose level shifts by its source bus's Slot A Trim."""

    source: str
    destination: str
    slot: str
    shift_db: float


@dataclass
class BalanceValidationReport:
    """Formal mathematical verification of rough-mix balance and solver convergence."""

    status: str = "verified"  # verified | converged_shifted | override_drift | caveat | unconverged | blocked
    converged: bool = True
    passes: int = 1
    leaf_fader_shift_db: float = 0.0
    global_path_delta_db: float | None = None
    max_balance_spread_db: float = 0.0
    active_leaf_count: int = 0
    staged_bus_count: int = 0
    unsimulated_insert_busses: list[str] = field(default_factory=list)
    track_path_deltas: dict[str, float] = field(default_factory=dict)
    bus_net_deltas: dict[str, float] = field(default_factory=dict)
    overridden_busses: dict[str, float] = field(default_factory=dict)
    clamped_busses: dict[str, float] = field(default_factory=dict)
    blocked_busses: dict[str, str] = field(default_factory=dict)
    pre_fader_send_caveats: list[PreFaderSendCaveat] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphLayoutNode:
    """Positioned node card in the semantic-tier Signal Flow DAG view."""

    name: str
    column: int
    role: str  # leaf | folder | bus | terminal | exempt | master | blocked
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class GraphLayoutEdge:
    """Directed routing edge between two positioned DAG nodes."""

    source: str
    target: str
    bus_name: str
    kind: str = "main"
    is_pre_fader: bool = False
    is_intra_column: bool = False


@dataclass(frozen=True)
class SignalFlowGraphLayout:
    """Complete pure-Python layout geometry for rendering the Signal Flow DAG."""

    columns: list[tuple[int, str, float, float]]
    nodes: dict[str, GraphLayoutNode]
    edges: list[GraphLayoutEdge]
    canvas_width: float
    canvas_height: float


@dataclass(frozen=True)
class PathTraceResult:
    """Upstream/downstream highlight set and algebraic path equation for a selected node."""

    selected_node: str
    representative_leaf: str | None
    highlighted_nodes: frozenset[str]
    highlighted_edges: frozenset[tuple[str, str]]
    path_nodes: tuple[str, ...]
    total_delta_db: float | None
    equation_text: str


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_nodes_and_edges(graph: Any) -> tuple[dict[str, Any], list[Any], list[str]]:
    nodes = _get(graph, "nodes") or {}
    edges = list(_get(graph, "edges") or ())
    order = list(_get(graph, "topological_order") or nodes.keys())
    return dict(nodes), edges, order


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or isinstance(val, bool):
        return default
    try:
        fv = float(val)
    except (TypeError, ValueError):
        return default
    return fv if math.isfinite(fv) else default


def _fmt_signed(val: float, *, explicit_zero: bool = False) -> str:
    v = 0.0 if abs(val) < 0.05 else round(float(val), 1)
    if v == 0.0:
        return "+0.0" if explicit_zero else "0.0"
    return f"+{v:.1f}" if v > 0 else f"{v:.1f}"


def _active_template_trim_db(node: Any) -> float:
    return _safe_float(_get(node, "template_trim_db", 0.0)) if _get(node, "slot_a_trim_active", False) else 0.0


def _is_active_node(node: Any) -> bool:
    return not _get(node, "is_muted") and not _get(node, "is_inactive") and not _get(node, "is_parallel_parked")


def _scan_pre_fader_send_caveats(
    ptx: PtxSession,
    nodes: Mapping[str, Any],
) -> list[PreFaderSendCaveat]:
    """Find active pre-fader sends whose level shifts relative to post-fader main (even when include_sends=False)."""
    caveats: list[PreFaderSendCaveat] = []
    for trk in ptx.tracks:
        node = nodes.get(trk.name)
        if node is None or not _is_active_node(node):
            continue
        ttype = str(_get(node, "track_type", ""))
        if ttype in ("Audio", "Instrument"):
            if _get(node, "pre_fader_peak_db") is None:
                continue
            shift_db = round(-_safe_float(_get(node, "recommended_fader_db", 0.0)), 1)
        elif _get(node, "staging_strategy") in ("trim_and_fader", "trim_only"):
            shift_db = round(_safe_float(_get(node, "recommended_trim_db", 0.0)) - _active_template_trim_db(node), 1)
        else:
            continue
        if abs(shift_db) < 0.05:
            continue
        for snd in trk.sends:
            if not snd.is_pre_fader or snd.is_muted:
                continue
            if snd.volume_db is not None and snd.volume_db <= _PARALLEL_INF_DB:
                continue
            dest = snd.destination.name if snd.destination else ""
            if dest:
                caveats.append(PreFaderSendCaveat(trk.name, dest, snd.slot, shift_db))
    return caveats


def _main_outgoing_map(edges: list[Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for edge in edges:
        if _get(edge, "kind", "main") == "main":
            out.setdefault(str(_get(edge, "source")), []).append(str(_get(edge, "target")))
    return out


def _leaf_total_delta(node: Any) -> float:
    return (
        _safe_float(_get(node, "baked_gain_db", 0.0))
        + _safe_float(_get(node, "clip_gain_db", 0.0))
        + _safe_float(_get(node, "recommended_fader_db", 0.0))
    )


def _trace_leaf_path_delta(
    leaf_name: str,
    nodes: Mapping[str, Any],
    main_out: Mapping[str, list[str]],
) -> float | None:
    leaf = nodes[leaf_name]
    if not _is_active_node(leaf) or _get(leaf, "post_fader_peak_db") is None:
        return None
    total = _leaf_total_delta(leaf)
    visited: set[str] = {leaf_name}
    cur = leaf_name
    while True:
        raw_nxts = [n for n in main_out.get(cur, ()) if n in nodes and n not in visited]
        if not raw_nxts:
            break
        active_nxts = [n for n in raw_nxts if _is_active_node(nodes[n])]
        if not active_nxts:
            return None
        nxt = active_nxts[0]
        visited.add(nxt)
        total += _safe_float(_get(nodes[nxt], "net_gain_delta_db", 0.0))
        cur = nxt
    return round(total, 2)


def _format_validation_summary(rep: BalanceValidationReport, *, attenuate_only: bool) -> str:
    if rep.status == "blocked":
        first, err = next(iter(rep.blocked_busses.items()))
        short_err = err.split(":", 1)[-1].strip() if ":" in err else err
        extra = f" (+{len(rep.blocked_busses) - 1} more)" if len(rep.blocked_busses) > 1 else ""
        return f"Slot A blocked on {first}{extra}: {short_err}"
    if rep.status == "override_drift":
        names = ", ".join(f"{k} {_fmt_signed(v)} dB" for k, v in list(rep.overridden_busses.items())[:2])
        return f"Balance shifted by override ({names} \u00b7 spread {rep.max_balance_spread_db:.1f} dB)"
    if rep.status == "unconverged":
        if rep.clamped_busses:
            first, exc = next(iter(rep.clamped_busses.items()))
            return f"Could not converge {first} within fader limits (+{exc:.1f} dB above target \u00b7 balance preserved \u00b1{rep.max_balance_spread_db:.1f} dB)"
        return f"Balance spread {rep.max_balance_spread_db:.1f} dB across {rep.active_leaf_count} tracks"
    if rep.status == "caveat":
        n = len(rep.pre_fader_send_caveats)
        return f"Main balance verified (\u00b1{rep.max_balance_spread_db:.1f} dB) \u00b7 {n} pre-fader send{'s' if n != 1 else ''} shifted by gain staging"
    if rep.status == "converged_shifted":
        return (
            f"Balance converged via {_fmt_signed(rep.leaf_fader_shift_db)} dB global leaf fader shift "
            f"(Pass {rep.passes} \u00b7 \u00b1{rep.max_balance_spread_db:.1f} dB)"
        )
    if not attenuate_only:
        return f"Unity-fader staging active ({rep.staged_bus_count} busses staged)"
    delta_txt = f"Global path \u0394: {_fmt_signed(rep.global_path_delta_db)} dB" if rep.global_path_delta_db is not None else "No active leaves"
    return (
        f"Balance invariant verified (\u00b1{rep.max_balance_spread_db:.1f} dB across {rep.active_leaf_count} tracks) "
        f"\u00b7 {delta_txt} \u00b7 {rep.staged_bus_count} staged busses"
    )


def _collect_validation_metrics(
    nodes: Mapping[str, Any],
    main_out: Mapping[str, list[str]],
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, str],
    list[str],
    int,
]:
    track_deltas: dict[str, float] = {}
    bus_deltas: dict[str, float] = {}
    overridden: dict[str, float] = {}
    clamped: dict[str, float] = {}
    blocked: dict[str, str] = {}
    unsim_ins: list[str] = []
    staged_count = 0
    for name, node in nodes.items():
        ttype = str(_get(node, "track_type", ""))
        if _get(node, "slot_a_blocked_error"):
            blocked[name] = str(_get(node, "slot_a_blocked_error"))
        if ttype in ("Audio", "Instrument"):
            delta = _trace_leaf_path_delta(name, nodes, main_out)
            if delta is not None:
                track_deltas[name] = delta
            continue
        if _get(node, "staged"):
            staged_count += 1
            bus_deltas[name] = round(_safe_float(_get(node, "net_gain_delta_db", 0.0)), 2)
        if _get(node, "is_overridden"):
            overridden[name] = round(
                _safe_float(_get(node, "recommended_fader_db", 0.0)) - _safe_float(_get(node, "auto_fader_db", 0.0)), 1,
            )
        exc = _safe_float(_get(node, "trim_clamped_excess_db", 0.0))
        if exc > _BALANCE_TOL_DB and not _get(node, "is_overridden"):
            clamped[name] = round(exc, 2)
        trim_plug = _get(node, "slot_a_trim_plugin")
        non_trim_ins = [p for p in (_get(node, "active_inserts") or ()) if p != trim_plug]
        if non_trim_ins and _get(node, "pre_fader_peak_db") is not None and _is_active_node(node):
            unsim_ins.append(name)
    return track_deltas, bus_deltas, overridden, clamped, blocked, unsim_ins, staged_count


def validate_signal_flow_balance(
    ptx: PtxSession,
    nodes: Mapping[str, Any],
    edges: list[Any],
    *,
    attenuate_only: bool,
    passes: int = 1,
    leaf_fader_shift_db: float = 0.0,
) -> BalanceValidationReport:
    """Compute formal balance-preservation and convergence diagnostics for a SignalFlowGraph."""
    track_deltas, bus_deltas, overridden, clamped, blocked, unsim_ins, staged_count = _collect_validation_metrics(
        nodes, _main_outgoing_map(edges),
    )
    vals = [v for v in track_deltas.values() if math.isfinite(v)]
    spread = round(max(vals) - min(vals), 2) if len(vals) >= 2 else 0.0
    global_delta = round(sorted(vals)[len(vals) // 2], 2) if vals else None
    caveats = _scan_pre_fader_send_caveats(ptx, nodes)

    if blocked:
        status, converged = "blocked", False
    elif clamped:
        status, converged = "unconverged", False
    elif overridden:
        status, converged = "override_drift", True
    elif attenuate_only and spread > _BALANCE_TOL_DB:
        status, converged = "unconverged", False
    elif caveats:
        status, converged = "caveat", True
    elif abs(leaf_fader_shift_db) >= 0.05:
        status, converged = "converged_shifted", True
    else:
        status, converged = "verified", True

    rep = BalanceValidationReport(
        status=status,
        converged=converged,
        passes=passes,
        leaf_fader_shift_db=round(leaf_fader_shift_db, 1),
        global_path_delta_db=global_delta,
        max_balance_spread_db=spread,
        active_leaf_count=len(vals),
        staged_bus_count=staged_count,
        unsimulated_insert_busses=unsim_ins,
        track_path_deltas=track_deltas,
        bus_net_deltas=bus_deltas,
        overridden_busses=overridden,
        clamped_busses=clamped,
        blocked_busses=blocked,
        pre_fader_send_caveats=caveats,
    )
    rep.summary = _format_validation_summary(rep, attenuate_only=attenuate_only)
    return rep


def _classify_graph_column(name: str, node: Any, main_in: Mapping[str, list[str]], nodes: Mapping[str, Any]) -> tuple[int, str]:
    ttype = str(_get(node, "track_type", ""))
    feeders = main_in.get(name, ())
    has_aux_feeder = any(_get(nodes.get(src), "track_type") == "Aux" for src in feeders)
    if _get(node, "slot_a_blocked_error"):
        if _get(node, "is_terminal") or _get(node, "is_post_terminal"):
            col = 4
        elif ttype == "RoutingFolder":
            col = 1
        else:
            col = 3 if has_aux_feeder else 2
        return col, "blocked"
    if ttype in ("Audio", "Instrument"):
        return 0, "leaf"
    if ttype == "RoutingFolder":
        return 1, ("bus" if _get(node, "staged") else "folder")
    if ttype == "Master":
        return 5, "master"
    if _get(node, "is_terminal"):
        return 4, "terminal"
    if _get(node, "is_post_terminal"):
        return 4, "exempt"
    col = 3 if has_aux_feeder else 2
    return col, ("bus" if _get(node, "staged") else "folder")


def _card_height_for_role(role: str) -> float:
    return {
        "leaf": 58.0,
        "folder": 72.0,
        "bus": 108.0,
        "terminal": 114.0,
        "blocked": 114.0,
        "exempt": 72.0,
        "master": 88.0,
    }.get(role, 88.0)


def _is_node_visible_in_graph(node: Any, *, active_only: bool) -> bool:
    if not active_only:
        return True
    if _get(node, "slot_a_blocked_error"):
        return True
    if _get(node, "is_inactive") or _get(node, "is_parallel_parked"):
        return False
    return _get(node, "pre_fader_peak_db") is not None or _get(node, "pre_insert_peak_db") is not None


def _place_column_nodes(
    col: int,
    col_x_pos: float,
    entries: list[tuple[str, str]],
    vis_edges: list[Any],
    node_col: Mapping[str, int],
    main_in: Mapping[str, list[str]],
    order_rank: Mapping[str, int],
    placed: dict[str, GraphLayoutNode],
) -> float:
    intra_parents: dict[str, list[str]] = {}
    for e in vis_edges:
        s, d = str(_get(e, "source")), str(_get(e, "target"))
        if node_col.get(s) == col and node_col.get(d) == col and s != d:
            intra_parents.setdefault(d, []).append(s)

    def _intra_depth(nm: str, seen: frozenset[str] = frozenset()) -> int:
        ps = [p for p in intra_parents.get(nm, ()) if p not in seen]
        return 0 if not ps else 1 + max(_intra_depth(p, seen | {nm}) for p in ps)

    depths = {nm: _intra_depth(nm) for nm, _ in entries}

    def _sort_key(item: tuple[str, str]) -> tuple[int, float, int]:
        nm = item[0]
        ups = [placed[s].y + placed[s].height * 0.5 for s in main_in.get(nm, ()) if s in placed and node_col.get(s) != col]
        bary = sum(ups) / len(ups) if ups else float(order_rank.get(nm, 0) * 100)
        return (depths[nm], bary, order_rank.get(nm, 0))

    entries.sort(key=_sort_key)
    cur_y = float(CANVAS_PAD_Y)
    w = float(COLUMN_WIDTHS[col])
    for nm, role in entries:
        h = _card_height_for_role(role)
        ups = [placed[s].y + placed[s].height * 0.5 for s in main_in.get(nm, ()) if s in placed and node_col.get(s) != col]
        if ups and depths[nm] == 0:
            cur_y = max(cur_y, (sum(ups) / len(ups)) - h * 0.5)
        placed[nm] = GraphLayoutNode(nm, col, role, col_x_pos, round(cur_y, 1), w, h)
        cur_y += h + ROW_GAP_Y
    return cur_y


def compute_signal_flow_graph_layout(
    graph: Any,
    *,
    active_only: bool = True,
) -> SignalFlowGraphLayout:
    """Compute semantic-tier column coordinates with barycenter vertical ordering."""
    nodes, edges, order = _extract_nodes_and_edges(graph)
    order_rank = {name: idx for idx, name in enumerate(order)}
    visible = {name for name, n in nodes.items() if _is_node_visible_in_graph(n, active_only=active_only)}
    main_in: dict[str, list[str]] = {n: [] for n in visible}
    vis_edges: list[Any] = []
    for e in edges:
        src, dst = str(_get(e, "source")), str(_get(e, "target"))
        if src in visible and dst in visible:
            vis_edges.append(e)
            if _get(e, "kind", "main") == "main":
                main_in[dst].append(src)

    by_col: dict[int, list[tuple[str, str]]] = {}
    node_col: dict[str, int] = {}
    for name in order:
        if name in visible:
            col, role = _classify_graph_column(name, nodes[name], main_in, nodes)
            by_col.setdefault(col, []).append((name, role))
            node_col[name] = col

    active_cols = sorted(by_col)
    columns_meta: list[tuple[int, str, float, float]] = []
    placed: dict[str, GraphLayoutNode] = {}
    cur_x = float(CANVAS_PAD_X)
    max_y = float(CANVAS_PAD_Y + 120)
    for col in active_cols:
        w = float(COLUMN_WIDTHS[col])
        columns_meta.append((col, COLUMN_TITLES[col], cur_x, w))
        max_y = max(max_y, _place_column_nodes(col, cur_x, by_col[col], vis_edges, node_col, main_in, order_rank, placed))
        cur_x += w + COLUMN_GAP_X

    layout_edges = [
        GraphLayoutEdge(
            source=str(_get(e, "source")),
            target=str(_get(e, "target")),
            bus_name=str(_get(e, "bus_name", "")),
            kind=str(_get(e, "kind", "main")),
            is_pre_fader=bool(_get(e, "is_pre_fader", False)),
            is_intra_column=(node_col.get(str(_get(e, "source"))) == node_col.get(str(_get(e, "target")))),
        )
        for e in vis_edges
        if str(_get(e, "source")) in placed and str(_get(e, "target")) in placed
    ]
    return SignalFlowGraphLayout(
        columns=columns_meta,
        nodes=placed,
        edges=layout_edges,
        canvas_width=max(320.0, cur_x - COLUMN_GAP_X + CANVAS_PAD_X),
        canvas_height=max(240.0, max_y + 12.0),
    )


def _format_node_equation_step(name: str, node: Any) -> str:
    if _get(node, "slot_a_blocked_error"):
        return f"{name} (BLOCKED)"
    ttype = str(_get(node, "track_type", ""))
    if ttype in ("Audio", "Instrument"):
        baked = _safe_float(_get(node, "baked_gain_db", 0.0))
        cg = _safe_float(_get(node, "clip_gain_db", 0.0))
        fdr = _safe_float(_get(node, "recommended_fader_db", 0.0))
        tot = baked + cg + fdr
        if abs(baked) >= 0.05 and abs(cg) < 0.05:
            return f"{name} ({_fmt_signed(baked)} baked {_fmt_signed(fdr, explicit_zero=True)} Fdr = {_fmt_signed(tot)})"
        return f"{name} ({_fmt_signed(cg)} CG {_fmt_signed(fdr, explicit_zero=True)} Fdr = {_fmt_signed(tot)})"
    if ttype == "Master":
        return name
    strat = str(_get(node, "staging_strategy", "none"))
    tr_del = _safe_float(_get(node, "recommended_trim_db", 0.0)) - _active_template_trim_db(node)
    f_del = _safe_float(_get(node, "recommended_fader_db", 0.0)) - _safe_float(_get(node, "template_fader_db", 0.0))
    if strat == "trim_and_fader":
        return f"{name} ({_fmt_signed(tr_del)}{_fmt_signed(f_del, explicit_zero=True)})"
    if strat == "trim_only":
        return f"{name} ({_fmt_signed(tr_del)}{_fmt_signed(f_del, explicit_zero=True)})" if abs(f_del) >= 0.05 else f"{name} ({_fmt_signed(tr_del)})"
    if strat == "fader_only":
        return f"{name} ({_fmt_signed(f_del)})"
    return f"{name} (0)"


def _bfs_highlight_sets(
    selected_node: str,
    main_in: Mapping[str, list[str]],
    main_out: Mapping[str, list[str]],
) -> tuple[frozenset[str], frozenset[tuple[str, str]]]:
    hi_nodes: set[str] = {selected_node}
    hi_edges: set[tuple[str, str]] = set()
    for adj, is_up in ((main_in, True), (main_out, False)):
        q: deque[str] = deque([selected_node])
        while q:
            cur = q.popleft()
            for nb in adj.get(cur, ()):
                hi_edges.add((nb, cur) if is_up else (cur, nb))
                if nb not in hi_nodes:
                    hi_nodes.add(nb)
                    q.append(nb)
    return frozenset(hi_nodes), frozenset(hi_edges)


def trace_signal_flow_path(graph: Any, selected_node: str) -> PathTraceResult:
    """Trace upstream feeders + downstream main chain for *selected_node* and format its equation."""
    nodes, edges, _order = _extract_nodes_and_edges(graph)
    if selected_node not in nodes:
        return PathTraceResult(selected_node, None, frozenset(), frozenset(), (), None, "")
    main_in: dict[str, list[str]] = {}
    main_out: dict[str, list[str]] = {}
    for e in edges:
        if _get(e, "kind", "main") != "main":
            continue
        s, d = str(_get(e, "source")), str(_get(e, "target"))
        if s in nodes and d in nodes:
            main_out.setdefault(s, []).append(d)
            main_in.setdefault(d, []).append(s)

    hi_nodes, hi_edges = _bfs_highlight_sets(selected_node, main_in, main_out)

    up_chain: list[str] = [selected_node]
    seen_up: set[str] = {selected_node}
    cur = selected_node
    while True:
        srcs = [
            s for s in main_in.get(cur, ())
            if s not in seen_up and _get(nodes[s], "pre_fader_peak_db") is not None and _is_active_node(nodes[s])
        ]
        if not srcs:
            break
        cur = srcs[0]
        seen_up.add(cur)
        up_chain.append(cur)
    up_chain.reverse()

    dn_chain: list[str] = []
    seen_dn: set[str] = set(up_chain)
    cur = selected_node
    while True:
        raw_dsts = [d for d in main_out.get(cur, ()) if d not in seen_dn]
        if not raw_dsts:
            break
        active_dsts = [d for d in raw_dsts if _is_active_node(nodes[d])]
        cur = active_dsts[0] if active_dsts else raw_dsts[0]
        seen_dn.add(cur)
        dn_chain.append(cur)

    full_chain = tuple(up_chain + dn_chain)
    rep_leaf = full_chain[0] if _get(nodes[full_chain[0]], "track_type") in ("Audio", "Instrument") else None
    total_delta: float | None = None
    if rep_leaf is not None:
        total = _leaf_total_delta(nodes[rep_leaf])
        for nm in full_chain[1:]:
            total += _safe_float(_get(nodes[nm], "net_gain_delta_db", 0.0))
        total_delta = round(total, 2)

    eq = " \u2192 ".join(_format_node_equation_step(nm, nodes[nm]) for nm in full_chain)
    return PathTraceResult(
        selected_node=selected_node,
        representative_leaf=rep_leaf,
        highlighted_nodes=hi_nodes,
        highlighted_edges=hi_edges,
        path_nodes=full_chain,
        total_delta_db=total_delta,
        equation_text=eq,
    )
