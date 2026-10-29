"""
Runway Slot Scheduler: BST vs AVL + Visual Trees + Gantt + Tarmac Routing
-------------------------------------------------------------------------
Install:  pip install streamlit plotly
Run:      streamlit run runway.py

Tabs
  1. Scheduler          - book slots (stored in an AVL tree, mirrored in a BST),
                          Gantt timeline, Graphviz drawings of both trees, data table
  2. Stress Test        - 1,000-slot BST vs AVL insertion benchmark (time.perf_counter)
  3. Tarmac Routing     - weighted airport graph (adjacency list) + Dijkstra's algorithm
                          from a runway exit to the nearest available gate
"""

import heapq
import math
import random
import time  # used for benchmarking (time.perf_counter)
from datetime import time as dtime
from typing import Dict, Iterable, List, Optional, Set, Tuple

import streamlit as st

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:  # the rest of the app still works without Plotly
    HAS_PLOTLY = False


# ----------------------------------------------------------------------------
# Data structures: flight slots (BST / AVL)
# ----------------------------------------------------------------------------
class FlightNode:
    """A single flight's runway time block, stored as a tree node."""

    def __init__(self, flight_id: str, start_time: float, end_time: float):
        self.flight_id = flight_id
        self.start_time = start_time  # e.g. 14.0 for 2:00 PM
        self.end_time = end_time      # e.g. 14.5 for 2:30 PM
        self.left: Optional["FlightNode"] = None
        self.right: Optional["FlightNode"] = None
        self.height = 1               # maintained by the AVL tree only

    def overlaps(self, start: float, end: float) -> bool:
        """Half-open overlap test: back-to-back slots (end == start) are allowed."""
        return start < self.end_time and end > self.start_time


class SlotTree:
    """Shared behaviour for both tree types."""

    # Palette for "height factor" colouring (index = node height - 1, capped).
    HEIGHT_COLORS = ["#e8f5e9", "#c8e6c9", "#bbdefb", "#90caf9", "#ce93d8", "#f48fb1", "#ef9a9a"]
    ROOT_COLOR = "#ffb300"

    def __init__(self):
        self.root: Optional[FlightNode] = None
        self.size = 0
        self.comparisons = 0  # number of nodes visited during insertions

    # --- traversal / queries -------------------------------------------------
    def in_order(self) -> List[FlightNode]:
        """All nodes sorted by start_time (iterative in-order traversal)."""
        result, stack, current = [], [], self.root
        while stack or current:
            while current:
                stack.append(current)
                current = current.left
            current = stack.pop()
            result.append(current)
            current = current.right
        return result

    def contains_id(self, flight_id: str) -> bool:
        return any(n.flight_id.lower() == flight_id.lower() for n in self.in_order())

    def true_heights(self) -> Dict[int, int]:
        """
        Real height of every node (leaf = 1), keyed by id(node). Computed iteratively
        so it is correct for the plain BST (which doesn't track heights) and safe for
        degenerate, very deep trees.
        """
        heights: Dict[int, int] = {}
        if self.root is None:
            return heights
        order, stack = [], [self.root]
        while stack:  # pre-order; reversed gives children before parents
            node = stack.pop()
            order.append(node)
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        for node in reversed(order):
            lh = heights[id(node.left)] if node.left else 0
            rh = heights[id(node.right)] if node.right else 0
            heights[id(node)] = 1 + max(lh, rh)
        return heights

    def tree_height(self) -> int:
        return self.true_heights().get(id(self.root), 0) if self.root else 0

    # --- Graphviz rendering --------------------------------------------------
    def to_dot(self, max_nodes: int = 60) -> str:
        """
        Build a Graphviz DOT description of the tree for st.graphviz_chart().
        Each node shows flight_id, [start-end], its height h and balance factor bf.
        The root is amber; every other node is shaded by its height (taller = darker).
        Returns "" when the tree is empty or has more than max_nodes nodes.
        """
        if self.root is None or self.size > max_nodes:
            return ""

        heights = self.true_heights()
        lines = [
            "digraph Tree {",
            '  graph [ordering=out, nodesep=0.25, ranksep=0.55, bgcolor="transparent"];',
            '  node  [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, '
            'color="#455a64", fontcolor="#102027", penwidth=1.2];',
            '  edge  [color="#78909c", arrowsize=0.6];',
        ]
        counter = 0

        def new_id() -> str:
            nonlocal counter
            counter += 1
            return f"n{counter}"

        ids: Dict[int, str] = {}
        stack = [self.root]
        order = []
        while stack:
            node = stack.pop()
            ids[id(node)] = new_id()
            order.append(node)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)

        for node in order:
            h = heights[id(node)]
            lh = heights[id(node.left)] if node.left else 0
            rh = heights[id(node.right)] if node.right else 0
            bf = lh - rh
            is_root = node is self.root
            fill = self.ROOT_COLOR if is_root else self.HEIGHT_COLORS[min(h - 1, len(self.HEIGHT_COLORS) - 1)]
            # A balance factor outside [-1, 1] means the AVL property is violated (BST only).
            border = "#c62828" if abs(bf) > 1 else "#455a64"
            pen = 2.5 if (is_root or abs(bf) > 1) else 1.2
            label = (
                f"{_dot_escape(node.flight_id)}"
                f"\\n[{short_time(node.start_time)}-{short_time(node.end_time)}]"
                f"\\nh={h}  bf={bf:+d}"
            )
            lines.append(
                f'  {ids[id(node)]} [label="{label}", fillcolor="{fill}", color="{border}", penwidth={pen}];'
            )

        for node in order:
            for child, side in ((node.left, "L"), (node.right, "R")):
                if child:
                    lines.append(f"  {ids[id(node)]} -> {ids[id(child)]};")
                elif node.left or node.right:
                    # Invisible placeholder keeps a lone child on the correct side.
                    ghost = new_id()
                    lines.append(f'  {ghost} [style=invis, label="", width=0.1, height=0.1];')
                    lines.append(f"  {ids[id(node)]} -> {ghost} [style=invis];")

        lines.append("}")
        return "\n".join(lines)


class BinarySearchTree(SlotTree):
    """Standard (unbalanced) BST keyed on start_time, with conflict detection."""

    def insert(self, flight_id: str, start_time: float, end_time: float) -> Tuple[bool, str]:
        """
        Insert a flight if its block doesn't overlap an existing one.
        Because stored slots never overlap, any conflict must involve the new slot's
        in-order predecessor or successor, and both lie on the search path, so
        checking each node along the path is a complete conflict check.
        """
        if end_time <= start_time:
            return False, "End time must be after start time."

        new_node = FlightNode(flight_id, start_time, end_time)
        if self.root is None:
            self.root = new_node
            self.size += 1
            return True, "Slot booked."

        current = self.root
        while True:
            self.comparisons += 1
            if current.overlaps(start_time, end_time):
                return False, (
                    f"Conflict with {current.flight_id} "
                    f"({format_time(current.start_time)} – {format_time(current.end_time)})."
                )
            if start_time < current.start_time:
                if current.left is None:
                    current.left = new_node
                    break
                current = current.left
            else:
                if current.right is None:
                    current.right = new_node
                    break
                current = current.right

        self.size += 1
        return True, "Slot booked."


class AVLTree(SlotTree):
    """Self-balancing AVL tree with height tracking and conflict detection."""

    def __init__(self):
        super().__init__()
        self.rotations = {"LL": 0, "RR": 0, "LR": 0, "RL": 0}
        self._conflict: Optional[FlightNode] = None

    # --- height / balance helpers -------------------------------------------
    @staticmethod
    def _height(node: Optional[FlightNode]) -> int:
        return node.height if node else 0

    def _update_height(self, node: FlightNode) -> None:
        node.height = 1 + max(self._height(node.left), self._height(node.right))

    def _balance_factor(self, node: Optional[FlightNode]) -> int:
        return self._height(node.left) - self._height(node.right) if node else 0

    # --- rotations ----------------------------------------------------------
    def _rotate_right(self, y: FlightNode) -> FlightNode:
        x = y.left
        y.left, x.right = x.right, y
        self._update_height(y)
        self._update_height(x)
        return x

    def _rotate_left(self, x: FlightNode) -> FlightNode:
        y = x.right
        x.right, y.left = y.left, x
        self._update_height(x)
        self._update_height(y)
        return y

    # --- insertion ----------------------------------------------------------
    def insert(self, flight_id: str, start_time: float, end_time: float) -> Tuple[bool, str]:
        if end_time <= start_time:
            return False, "End time must be after start time."

        self._conflict = None
        new_node = FlightNode(flight_id, start_time, end_time)
        new_root = self._insert(self.root, new_node)

        if self._conflict is not None:
            c = self._conflict
            return False, (
                f"Conflict with {c.flight_id} "
                f"({format_time(c.start_time)} – {format_time(c.end_time)})."
            )
        self.root = new_root
        self.size += 1
        return True, "Slot booked."

    def _insert(self, node: Optional[FlightNode], new: FlightNode) -> FlightNode:
        if node is None:
            return new

        self.comparisons += 1
        if node.overlaps(new.start_time, new.end_time):
            self._conflict = node
            return node  # reject: leave the tree untouched

        if new.start_time < node.start_time:
            node.left = self._insert(node.left, new)
        else:
            node.right = self._insert(node.right, new)

        if self._conflict is not None:  # conflict found deeper; nothing changed
            return node

        self._update_height(node)
        balance = self._balance_factor(node)

        # Left-heavy
        if balance > 1:
            if self._balance_factor(node.left) >= 0:      # Left-Left -> Right rotation
                self.rotations["LL"] += 1
                return self._rotate_right(node)
            self.rotations["LR"] += 1                      # Left-Right -> Left then Right
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)

        # Right-heavy
        if balance < -1:
            if self._balance_factor(node.right) <= 0:     # Right-Right -> Left rotation
                self.rotations["RR"] += 1
                return self._rotate_left(node)
            self.rotations["RL"] += 1                      # Right-Left -> Right then Left
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        return node


# ----------------------------------------------------------------------------
# Data structures: airport graph (CO5) + Dijkstra
# ----------------------------------------------------------------------------
class AirportGraph:
    """
    Taxiways, intersections, runway exits and gates as an undirected weighted graph,
    stored as an adjacency list: {node: [(neighbour, distance_in_metres), ...]}.
    """

    def __init__(self):
        self.adj: Dict[str, List[Tuple[str, float]]] = {}
        self.kind: Dict[str, str] = {}  # node -> "exit" | "taxiway" | "intersection" | "gate"

    def add_node(self, name: str, kind: str) -> None:
        self.adj.setdefault(name, [])
        self.kind[name] = kind

    def add_edge(self, a: str, b: str, distance: float) -> None:
        if a not in self.adj or b not in self.adj:
            raise KeyError("Add both nodes before connecting them.")
        self.adj[a].append((b, distance))
        self.adj[b].append((a, distance))

    def edges(self) -> List[Tuple[str, str, float]]:
        """Each undirected edge once."""
        seen, out = set(), []
        for a, nbrs in self.adj.items():
            for b, w in nbrs:
                key = tuple(sorted((a, b)))
                if key not in seen:
                    seen.add(key)
                    out.append((a, b, w))
        return out

    def nodes_of_kind(self, kind: str) -> List[str]:
        return [n for n, k in self.kind.items() if k == kind]

    def dijkstra(
        self, source: str, blocked: Iterable[str] = ()
    ) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
        """
        Single-source shortest paths with a min-heap, O((V + E) log V).
        Nodes in `blocked` (closed taxiways / occupied gates) can't be entered.
        Returns (distance, previous) maps; unreachable nodes have distance inf.
        """
        blocked_set: Set[str] = set(blocked)
        dist = {n: math.inf for n in self.adj}
        prev: Dict[str, Optional[str]] = {n: None for n in self.adj}
        dist[source] = 0.0
        heap = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue  # stale heap entry
            for v, w in self.adj[u]:
                if v in blocked_set:
                    continue
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        return dist, prev

    @staticmethod
    def build_path(prev: Dict[str, Optional[str]], target: str) -> List[str]:
        path, node = [], target
        while node is not None:
            path.append(node)
            node = prev[node]
        return path[::-1]

    def shortest_path(
        self, source: str, target: str, blocked: Iterable[str] = ()
    ) -> Tuple[float, List[str]]:
        """Shortest (distance, path) from source to target; (inf, []) if unreachable."""
        dist, prev = self.dijkstra(source, blocked)
        if math.isinf(dist[target]):
            return math.inf, []
        return dist[target], self.build_path(prev, target)

    def to_dot(
        self,
        route: Optional[List[str]] = None,
        occupied: Iterable[str] = (),
        closed: Iterable[str] = (),
    ) -> str:
        """DOT drawing of the network; the computed route is highlighted in red."""
        route = route or []
        route_edges = {tuple(sorted(p)) for p in zip(route, route[1:])}
        route_nodes, occupied_set, closed_set = set(route), set(occupied), set(closed)

        lines = [
            "graph Airport {",
            '  graph [rankdir=LR, nodesep=0.45, ranksep=0.8, bgcolor="transparent"];',
            '  node  [fontname="Helvetica", fontsize=11, style=filled, penwidth=1.3, color="#455a64", fontcolor="#102027"];',
            '  edge  [fontname="Helvetica", fontsize=10, color="#90a4ae", fontcolor="#37474f"];',
        ]
        for name, kind in self.kind.items():
            if name in closed_set:
                shape, fill = "box", "#cfd8dc"
            elif kind == "exit":
                shape, fill = "invhouse", "#fff176"
            elif kind == "gate":
                shape, fill = "box", ("#ef9a9a" if name in occupied_set else "#a5d6a7")
            elif kind == "taxiway":
                shape, fill = "ellipse", "#b3e5fc"
            else:
                shape, fill = "circle", "#e1bee7"
            pen = ', penwidth=3, color="#d50000"' if name in route_nodes else ""
            extra = ", style=\"filled,dashed\"" if name in closed_set else ""
            lines.append(f'  "{name}" [shape={shape}, fillcolor="{fill}"{pen}{extra}];')
        for a, b, w in self.edges():
            if tuple(sorted((a, b))) in route_edges:
                lines.append(f'  "{a}" -- "{b}" [label="{w:g} m", color="#d50000", penwidth=3.5, fontcolor="#d50000"];')
            else:
                lines.append(f'  "{a}" -- "{b}" [label="{w:g} m"];')
        lines.append("}")
        return "\n".join(lines)


def build_default_airport() -> AirportGraph:
    """Sample airport layout. Edge weights are taxi distances in metres."""
    g = AirportGraph()
    for n in ("Runway_Exit", "Runway_Exit_East"):
        g.add_node(n, "exit")
    for n in ("Taxiway_A", "Taxiway_B", "Taxiway_C"):
        g.add_node(n, "taxiway")
    for n in ("Intersection_1", "Intersection_2", "Intersection_3"):
        g.add_node(n, "intersection")
    for n in ("Gate_A1", "Gate_A2", "Gate_B1", "Gate_B2", "Gate_C1"):
        g.add_node(n, "gate")

    for a, b, w in [
        ("Runway_Exit", "Taxiway_A", 300),
        ("Runway_Exit_East", "Taxiway_C", 350),
        ("Taxiway_A", "Intersection_1", 250),
        ("Taxiway_A", "Taxiway_B", 400),
        ("Taxiway_B", "Intersection_2", 200),
        ("Taxiway_C", "Intersection_2", 300),
        ("Taxiway_C", "Intersection_3", 250),
        ("Intersection_1", "Intersection_2", 300),
        ("Intersection_2", "Intersection_3", 260),
        ("Intersection_1", "Gate_A1", 150),
        ("Intersection_1", "Gate_A2", 200),
        ("Intersection_2", "Gate_B1", 180),
        ("Intersection_2", "Gate_B2", 220),
        ("Intersection_3", "Gate_C1", 140),
    ]:
        g.add_edge(a, b, w)
    return g


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def format_time(decimal_hours: float) -> str:
    """14.5 -> '02:30 PM' (hours beyond 24 wrap, which only matters for benchmark data)."""
    total_minutes = round(decimal_hours * 60)
    h, m = divmod(total_minutes, 60)
    suffix = "AM" if h % 24 < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12:02d}:{m:02d} {suffix}"


def short_time(decimal_hours: float) -> str:
    """14.5 -> '14:30' (compact 24-hour form for diagram labels)."""
    total_minutes = round(decimal_hours * 60)
    h, m = divmod(total_minutes, 60)
    return f"{h:02d}:{m:02d}"


def to_decimal(t: dtime) -> float:
    """datetime.time -> decimal hours (14:30 -> 14.5)"""
    return t.hour + t.minute / 60


def _dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def build_gantt(flights: List[FlightNode]):
    """Horizontal Gantt/timeline: one bar per flight across the 24-hour day."""
    fig = go.Figure()
    ids = [f.flight_id for f in flights]
    fig.add_trace(
        go.Bar(
            y=ids,
            x=[f.end_time - f.start_time for f in flights],
            base=[f.start_time for f in flights],
            orientation="h",
            marker=dict(color="#1e88e5", line=dict(color="#0d47a1", width=1)),
            text=[f"{short_time(f.start_time)}-{short_time(f.end_time)}" for f in flights],
            textposition="inside",
            insidetextanchor="middle",
            customdata=[
                [format_time(f.start_time), format_time(f.end_time), round((f.end_time - f.start_time) * 60)]
                for f in flights
            ],
            hovertemplate="<b>%{y}</b><br>%{customdata[0]} – %{customdata[1]}"
                          "<br>%{customdata[2]} min<extra></extra>",
        )
    )
    lo = max(0, math.floor(min(f.start_time for f in flights)) - 1)
    hi = min(24, math.ceil(max(f.end_time for f in flights)) + 1)
    ticks = list(range(lo, hi + 1))
    fig.update_layout(
        height=max(240, 46 * len(flights) + 110),
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        bargap=0.3,
        xaxis=dict(
            range=[lo, hi], tickvals=ticks, ticktext=[f"{t:02d}:00" for t in ticks],
            title="Time of day", gridcolor="rgba(128,128,128,0.25)",
        ),
        yaxis=dict(autorange="reversed", categoryorder="array", categoryarray=ids, title=""),
    )
    return fig


def generate_slots(n: int, shuffle: bool, seed: int) -> List[Tuple[str, float, float]]:
    """
    Generate n valid, mutually non-overlapping random slots (random gaps and durations)
    so every insertion succeeds and the work measured is pure tree insertion.
    shuffle=True  -> insertion in random order (typical case)
    shuffle=False -> insertion in chronological order (worst case for a plain BST)
    """
    rng = random.Random(seed)
    slots, cursor = [], 0.0
    for i in range(n):
        start = cursor + rng.uniform(0.0, 0.05)
        end = start + rng.uniform(0.05, 0.25)  # 3 to 15 minute blocks
        slots.append((f"FL{i + 1:04d}", start, end))
        cursor = end
    if shuffle:
        rng.shuffle(slots)
    return slots


def run_benchmark(n: int, shuffle: bool, seed: int) -> dict:
    slots = generate_slots(n, shuffle, seed)
    results = {}
    for name, tree in (("BST", BinarySearchTree()), ("AVL", AVLTree())):
        start = time.perf_counter()
        for fid, s, e in slots:
            tree.insert(fid, s, e)
        elapsed = time.perf_counter() - start
        results[name] = {
            "time_ms": elapsed * 1000,
            "height": tree.tree_height(),
            "comparisons": tree.comparisons,
            "inserted": tree.size,
            "rotations": sum(tree.rotations.values()) if isinstance(tree, AVLTree) else 0,
        }
    return results


# Sample flights, deliberately in chronological order so the BST degenerates
# into a chain while the AVL tree rotates to stay balanced.
SAMPLE_FLIGHTS = [
    ("AI-101", 6.0, 6.25), ("6E-202", 6.5, 6.75), ("UK-303", 7.0, 7.5),
    ("SG-404", 8.0, 8.25), ("AI-505", 9.0, 9.5), ("6E-606", 10.0, 10.25),
    ("UK-707", 11.0, 11.5), ("SG-808", 12.0, 12.25), ("AI-909", 13.0, 13.5),
    ("6E-110", 14.0, 14.5),
]


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Runway Slot Scheduler", page_icon="🛫", layout="wide")

if "avl" not in st.session_state:
    st.session_state.avl = AVLTree()
    st.session_state.bst = BinarySearchTree()
if "airport" not in st.session_state:
    st.session_state.airport = build_default_airport()

avl: AVLTree = st.session_state.avl
bst: BinarySearchTree = st.session_state.bst
airport: AirportGraph = st.session_state.airport

st.title("🛫 Runway Slot Scheduler")
st.caption(
    "Flights are stored in an AVL tree (mirrored in a plain BST for comparison). "
    "Overlapping slots are rejected; back-to-back slots are allowed."
)

tab_schedule, tab_bench, tab_route = st.tabs(
    ["📅 Scheduler", "⚡ Stress Test / Benchmark", "🛬 Tarmac Routing (CO5 Graph)"]
)

# ---------------------------- Scheduler tab ---------------------------------
with tab_schedule:
    form_col, info_col = st.columns([1, 1.4])

    with form_col:
        st.subheader("Book a slot")
        with st.form("booking_form"):
            flight_id = st.text_input("Flight ID", placeholder="e.g. AI-202")
            col1, col2 = st.columns(2)
            start = col1.time_input("Start time", value=dtime(14, 0), step=300)
            end = col2.time_input("End time", value=dtime(14, 30), step=300)
            submitted = st.form_submit_button("Book slot", type="primary", use_container_width=True)

        if submitted:
            fid = flight_id.strip()
            if not fid:
                st.error("Please enter a Flight ID.")
            elif avl.contains_id(fid):
                st.error(f"Flight ID '{fid}' is already scheduled.")
            else:
                ok, msg = avl.insert(fid, to_decimal(start), to_decimal(end))
                if ok:
                    bst.insert(fid, to_decimal(start), to_decimal(end))  # keep the BST in sync
                    st.success(
                        f"{fid} booked: {format_time(to_decimal(start))} – {format_time(to_decimal(end))}"
                    )
                else:
                    st.error(f"Booking rejected. {msg}")

        b1, b2 = st.columns(2)
        if b1.button("Load sample flights", use_container_width=True,
                     help="Adds 10 flights in chronological order (skips any that conflict)."):
            for fid, s, e in SAMPLE_FLIGHTS:
                if not avl.contains_id(fid):
                    ok, _ = avl.insert(fid, s, e)
                    if ok:
                        bst.insert(fid, s, e)
            st.rerun()
        if b2.button("Clear all bookings", use_container_width=True, disabled=avl.size == 0):
            st.session_state.avl = AVLTree()
            st.session_state.bst = BinarySearchTree()
            st.rerun()

    with info_col:
        st.subheader(f"Scheduled flights ({avl.size})")
        m1, m2, m3 = st.columns(3)
        m1.metric("Flights", avl.size)
        m2.metric("AVL height", avl.tree_height())
        m3.metric("BST height", bst.tree_height())

        flights = avl.in_order()
        if not flights:
            st.info("No flights scheduled yet. Book one, or load the sample flights.")
        elif HAS_PLOTLY:
            st.plotly_chart(build_gantt(flights), use_container_width=True)
        else:
            st.warning("Install Plotly (`pip install plotly`) to see the timeline chart.")

    if flights:
        with st.expander("📋 Schedule table", expanded=False):
            st.dataframe(
                [
                    {
                        "Flight ID": n.flight_id,
                        "Start": format_time(n.start_time),
                        "End": format_time(n.end_time),
                        "Duration (min)": round((n.end_time - n.start_time) * 60),
                    }
                    for n in flights
                ],
                use_container_width=True,
                hide_index=True,
            )

    st.divider()
    st.subheader("🌳 Tree structure")
    st.caption(
        "Amber = root. Other nodes are shaded by height (taller subtree = darker). "
        "h = node height, bf = balance factor (left height − right height). "
        "A red border means |bf| > 1, i.e. the node is out of balance."
    )
    if avl.size == 0:
        st.info("Book some flights to see the trees.")
    else:
        t_avl, t_bst = st.columns(2)
        for col, title, tree in ((t_avl, "AVL Tree", avl), (t_bst, "Binary Search Tree", bst)):
            with col:
                st.markdown(f"**{title}** · height {tree.tree_height()}")
                dot = tree.to_dot()
                if dot:
                    st.graphviz_chart(dot, use_container_width=True)
                else:
                    st.info("Too many nodes to draw legibly (limit 60).")

# ---------------------------- Benchmark tab ---------------------------------
with tab_bench:
    st.subheader("BST vs AVL: insertion benchmark")
    st.write(
        "Generates random, non-overlapping flight slots and inserts the same sequence "
        "into a fresh BST and a fresh AVL tree, timing each with `time.perf_counter()`."
    )

    c1, c2, c3 = st.columns(3)
    n_slots = c1.number_input("Number of slots", min_value=100, max_value=5000, value=1000, step=100)
    order = c2.selectbox(
        "Insertion order",
        ["Random order", "Chronological order (worst case for BST)"],
    )
    seed = c3.number_input("Random seed", min_value=0, value=42, step=1)

    if st.button("Run benchmark", type="primary", use_container_width=True):
        with st.spinner("Inserting slots..."):
            st.session_state.bench = run_benchmark(
                int(n_slots), order.startswith("Random"), int(seed)
            )
            st.session_state.bench_meta = (int(n_slots), order)

    if "bench" in st.session_state:
        res = st.session_state.bench
        n_done, order_done = st.session_state.bench_meta
        b, a = res["BST"], res["AVL"]

        st.markdown(f"**Results: {n_done:,} slots, {order_done.lower()}**")

        faster, slower = ("AVL", "BST") if a["time_ms"] < b["time_ms"] else ("BST", "AVL")
        ratio = res[slower]["time_ms"] / max(res[faster]["time_ms"], 1e-9)
        st.success(f"🏆 {faster} Tree was faster: {ratio:.2f}x quicker than the {slower} Tree.")

        col_b, col_a = st.columns(2)
        with col_b:
            st.markdown("##### Binary Search Tree")
            st.metric("Total time", f"{b['time_ms']:.2f} ms")
            st.metric("Tree height", b["height"])
            st.metric("Node visits", f"{b['comparisons']:,}")
        with col_a:
            st.markdown("##### AVL Tree")
            st.metric(
                "Total time",
                f"{a['time_ms']:.2f} ms",
                delta=f"{a['time_ms'] - b['time_ms']:+.2f} ms vs BST",
                delta_color="inverse",
            )
            st.metric("Tree height", a["height"])
            st.metric("Node visits", f"{a['comparisons']:,}")

        st.caption(
            f"AVL performed {a['rotations']:,} rotations to stay balanced. "
            f"Inserted: BST {b['inserted']:,}, AVL {a['inserted']:,}."
        )

        st.bar_chart({"Time (ms)": {"BST": b["time_ms"], "AVL": a["time_ms"]}})

        st.info(
            "Why results vary: on random input a BST is already roughly balanced, so it can match or "
            "beat the AVL tree because AVL pays rotation overhead. On chronological input the BST "
            "degenerates into a linked list (height ≈ n, O(n²) total work) while the AVL tree stays at "
            "height ≈ log₂(n). Run both modes to see the trade-off."
        )

# ---------------------------- Tarmac routing tab ----------------------------
with tab_route:
    st.subheader("Tarmac routing: runway exit → gate (Dijkstra)")
    st.write(
        "The airfield is a weighted graph stored as an adjacency list (edge weight = taxi distance "
        "in metres). Dijkstra's algorithm finds the shortest route from the runway exit to a free gate."
    )

    exits = airport.nodes_of_kind("exit")
    gates = airport.nodes_of_kind("gate")
    closable = airport.nodes_of_kind("taxiway") + airport.nodes_of_kind("intersection")

    ctl1, ctl2, ctl3 = st.columns(3)
    source = ctl1.selectbox("Runway exit (start)", exits)
    occupied = ctl2.multiselect("Occupied gates", gates, default=["Gate_A1"])
    closed = ctl3.multiselect("Closed taxiways / intersections", closable, default=[])

    ctl4, ctl5 = st.columns(2)
    available = [g for g in gates if g not in occupied]
    target_choice = ctl4.selectbox("Destination", ["Nearest available gate (auto)"] + available)
    speed_kts = ctl5.slider("Taxi speed (knots)", min_value=5, max_value=30, value=15)
    speed_mps = speed_kts * 0.514444  # knots -> metres per second

    # One Dijkstra run gives the shortest distance to every node.
    blocked = set(closed) | set(occupied)
    dist, prev = airport.dijkstra(source, blocked=closed)  # gates are only *destinations*

    reachable = [(g, dist[g]) for g in available if not math.isinf(dist[g])]
    route: List[str] = []
    chosen: Optional[str] = None
    if target_choice.startswith("Nearest"):
        if reachable:
            chosen = min(reachable, key=lambda x: x[1])[0]
    else:
        chosen = target_choice if not math.isinf(dist[target_choice]) else None

    if chosen is not None:
        route = airport.build_path(prev, chosen)
        total_m = dist[chosen]
        minutes = total_m / speed_mps / 60
        st.success(f"✅ Optimal route to **{chosen}**:  " + "  →  ".join(route))
        r1, r2, r3 = st.columns(3)
        r1.metric("Total distance", f"{total_m:,.0f} m")
        r2.metric("Estimated taxi time", f"{minutes:.1f} min")
        r3.metric("Segments", len(route) - 1)
    elif not available:
        st.error("All gates are occupied. No gate can be assigned.")
    else:
        st.error("No route exists: the closed taxiways cut this exit off from the selected gate(s).")

    left, right = st.columns([1.6, 1])
    with left:
        st.markdown("**Airport network** (red = computed route, green gate = free, red gate = occupied, "
                    "dashed grey = closed)")
        st.graphviz_chart(airport.to_dot(route=route, occupied=occupied, closed=closed),
                          use_container_width=True)
    with right:
        st.markdown(f"**Shortest distance from `{source}`**")
        rows = []
        for g in gates:
            if g in occupied:
                status, d, t = "Occupied", None, None
            elif math.isinf(dist[g]):
                status, d, t = "Unreachable", None, None
            else:
                status, d, t = "Free", round(dist[g]), round(dist[g] / speed_mps / 60, 1)
            rows.append({"Gate": g, "Status": status, "Distance (m)": d, "Time (min)": t})
        st.dataframe(rows, use_container_width=True, hide_index=True)

        with st.expander("Adjacency list"):
            st.code(
                "\n".join(
                    f"{node:<17}-> " + ", ".join(f"{n} ({w:g})" for n, w in nbrs)
                    for node, nbrs in airport.adj.items()
                ),
                language="text",
            )