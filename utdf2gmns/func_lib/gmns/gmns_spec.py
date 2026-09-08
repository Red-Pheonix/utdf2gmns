'''
##############################################################
# Rewrite utdf2gmns output as spec-compliant GMNS.
##############################################################

utdf2gmns emits its own dialect: `length_m` / `free_speed_mps` instead of
`length` / `free_speed`, lane ranges as `ib_lane_indices` lists rather than
`start_ib_lane` / `end_ib_lane`, signal timing as a nested `signal.json` rather
than the GMNS signal tables, and no `config.csv` at all.

Anything that reads GMNS through `gmnspy` — including any `gmns2y` converter —
rejects or misreads that. This module maps it onto the real spec.

    from utdf2gmns import to_spec_gmns
    to_spec_gmns("utdf_to_gmns", "utdf_to_gmns")   # in place is fine

Reference: https://github.com/zephyr-data-specs/GMNS
'''
import csv
import json
import os

LANE_WIDTH_M = 3.6576   # 12 ft, the usual US default

# movement.type in GMNS is a closed enum
MVMT_TYPES = {"left", "right", "uturn", "thru", "merge", "diverge"}


def _read(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write(path: str, rows: list, fieldnames: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _lane_range(spec, total_lanes: int) -> tuple:
    """utdf2gmns lane indices -> GMNS lane numbers.

    The two number lanes from opposite sides:

        utdf2gmns   0 .. N-1, 0 = rightmost (curb)      — SUMO's convention
        GMNS        1 .. N,   1 = leftmost (centreline)

    so a lane is `total - index`, and a range flips end for end. Getting this
    backwards still produces a network — just one where every turn is fed from
    the wrong lane — so it is done in one place and nowhere else.
    """
    if spec is None or not total_lanes:
        return None, None
    parts = [p.strip() for p in str(spec).split(",") if p.strip() != ""]
    if not parts:
        return None, None
    try:
        idx = [int(float(p)) for p in parts]
    except ValueError:
        return None, None
    start = total_lanes - max(idx)   # innermost index -> lowest GMNS number
    end = total_lanes - min(idx)
    return max(start, 1), max(end, 1)


def _phase_num(key) -> int:
    """'D1' -> 1. Returns None for non-phase keys such as 'brp_info'."""
    digits = "".join(c for c in str(key) if c.isdigit())
    return int(digits) if digits else None


def _brp(value) -> tuple:
    """'212' -> (barrier=2, ring=1, position=2).

    Confirmed against the sibling `brp_info` map, which is keyed
    {barrier: {ring: [phases]}}.
    """
    text = str(value or "").strip()
    if len(text) != 3 or not text.isdigit():
        return None, None, None
    return int(text[0]), int(text[1]), int(text[2])


def _num(value, default=None):
    try:
        text = str(value).strip()
        return float(text) if text not in ("", "None", "nan") else default
    except (TypeError, ValueError):
        return default


def to_spec_gmns(src_dir: str, dst_dir: str = "", *, crs: int = 4326,
                 dataset_name: str = "", timeplans: list = None,
                 verbose: bool = True) -> str:
    """Convert a utdf2gmns output folder into spec-compliant GMNS.

    Args:
        src_dir (str): folder holding utdf2gmns output (node.csv, link.csv,
            lane.csv, movement.csv, signal.json, utdf_timeplans.csv).
        dst_dir (str): where to write the spec files. Defaults to src_dir,
            which rewrites in place — every input is read before anything is
            written, so that is safe.
        crs (int): EPSG code to declare in config.csv. utdf2gmns writes
            lon/lat, so 4326 unless the caller reprojected.
        dataset_name (str): name for config.csv, defaults to the folder name.
        timeplans (list): UTDF Timeplans rows as dicts, supplying cycle length
            and offset. Read from utdf_timeplans.csv when not given, which is
            only present if the folder was written with incl_utdf=True.

    Returns:
        str: dst_dir.
    """
    dst_dir = dst_dir or src_dir
    os.makedirs(dst_dir, exist_ok=True)
    dataset_name = dataset_name or os.path.basename(os.path.abspath(dst_dir))

    # every read happens before every write, so src_dir == dst_dir is safe
    nodes = _read(os.path.join(src_dir, "node.csv"))
    links = _read(os.path.join(src_dir, "link.csv"))
    lanes = _read(os.path.join(src_dir, "lane.csv"))
    movements = _read(os.path.join(src_dir, "movement.csv"))
    if timeplans is None:
        timeplans = _read(os.path.join(src_dir, "utdf_timeplans.csv"))

    signal_path = os.path.join(src_dir, "signal.json")
    if os.path.exists(signal_path):
        with open(signal_path, encoding="utf-8") as f:
            signals = json.load(f)
    else:
        signals = {}

    say = print if verbose else (lambda *a, **k: None)

    # ---------------------------------------------------------------- node
    signal_nodes = {str(k) for k in signals}
    out_nodes = []
    for n in nodes:
        nid = str(n.get("node_id", "")).strip()
        if not nid:
            continue
        out_nodes.append({
            "node_id": nid,
            "name": (n.get("DESCRIPTION") or "").strip(),
            "x_coord": n.get("x_coord"),
            "y_coord": n.get("y_coord"),
            "z_coord": "",
            "node_type": (n.get("TYPE_DESC") or n.get("node_type") or "").strip(),
            "ctrl_type": "signal" if nid in signal_nodes else "",
        })
    _write(os.path.join(dst_dir, "node.csv"), out_nodes,
           ["node_id", "name", "x_coord", "y_coord", "z_coord", "node_type", "ctrl_type"])

    # ---------------------------------------------------------------- link
    # utdf2gmns already emits SI, so values pass through unchanged and
    # config.csv below declares metres and m/s.
    lanes_per_link = {}
    out_links = []
    for lk in links:
        lid = str(lk.get("link_id", "")).strip()
        if not lid:
            continue
        n_lanes = int(_num(lk.get("lanes"), 0) or 0)
        lanes_per_link[lid] = n_lanes
        out_links.append({
            "link_id": lid,
            "name": "",
            "from_node_id": lk.get("from_node_id"),
            "to_node_id": lk.get("to_node_id"),
            "directed": "true",
            "geometry": lk.get("geometry", ""),
            "dir_flag": 1,
            "length": _num(lk.get("length_m")),
            "free_speed": _num(lk.get("free_speed_mps")),
            "lanes": n_lanes,
            "allowed_uses": "all",
            "row_width": round(max(n_lanes, 1) * LANE_WIDTH_M, 4),
        })
    _write(os.path.join(dst_dir, "link.csv"), out_links,
           ["link_id", "name", "from_node_id", "to_node_id", "directed", "geometry",
            "dir_flag", "length", "free_speed", "lanes", "allowed_uses", "row_width"])

    # ---------------------------------------------------------------- lane
    # `allowed_uses` matters: a converter drops any lane it cannot confirm is
    # open to vehicles, so an empty value silently deletes the network.
    out_lanes = []
    for ln in lanes:
        lane_id = str(ln.get("lane_id", "")).strip()
        if not lane_id:
            continue
        link_id = str(ln.get("link_id", "")).strip()
        total = lanes_per_link.get(link_id, 0)
        idx = int(_num(ln.get("lane_num"), 0) or 0)
        out_lanes.append({
            "lane_id": lane_id,
            "link_id": link_id,
            # same flip as the movements: GMNS counts 1..N from the centreline
            "lane_num": max(total - idx, 1) if total else idx + 1,
            "allowed_uses": "all",
            "width": LANE_WIDTH_M,
        })
    _write(os.path.join(dst_dir, "lane.csv"), out_lanes,
           ["lane_id", "link_id", "lane_num", "allowed_uses", "width"])

    # ------------------------------------------------------------ movement
    out_mvmts = []
    name_to_mvmt = {}          # (node_id, 'SBL') -> mvmt_id, for the signal tables
    bad_types = set()
    for mv in movements:
        mid = str(mv.get("mvmt_id", "")).strip()
        if not mid:
            continue
        node_id = str(mv.get("node_id", "")).strip()
        mtype = (mv.get("type") or "").strip().lower()
        if mtype not in MVMT_TYPES:
            bad_types.add(mtype)
            continue
        sib, eib = _lane_range(mv.get("ib_lane_indices"),
                               lanes_per_link.get(str(mv.get("ib_link_id", "")).strip()))
        sob, eob = _lane_range(mv.get("ob_lane_indices"),
                               lanes_per_link.get(str(mv.get("ob_link_id", "")).strip()))
        mname = (mv.get("movement_name") or "").strip()
        if mname:
            name_to_mvmt[(node_id, mname.upper())] = mid
        out_mvmts.append({
            "mvmt_id": mid,
            "node_id": node_id,
            "name": mname,
            "ib_link_id": mv.get("ib_link_id"),
            "start_ib_lane": sib,
            "end_ib_lane": eib,
            "ob_link_id": mv.get("ob_link_id"),
            "start_ob_lane": sob,
            "end_ob_lane": eob,
            "type": mtype,
            "ctrl_type": "signal" if node_id in signal_nodes else "",
        })
    _write(os.path.join(dst_dir, "movement.csv"), out_mvmts,
           ["mvmt_id", "node_id", "name", "ib_link_id", "start_ib_lane", "end_ib_lane",
            "ob_link_id", "start_ob_lane", "end_ob_lane", "type", "ctrl_type"])
    if bad_types:
        say(f"  ! dropped movements with non-spec type: {sorted(bad_types)}")

    # -------------------------------------------------------------- signals
    cycle_by_node = {}
    offset_by_node = {}
    for row in timeplans:
        rec = (row.get("RECORDNAME") or "").strip()
        nid = str(row.get("INTID", "")).strip()
        if rec == "Cycle Length":
            cycle_by_node[nid] = _num(row.get("DATA"))
        elif rec == "Offset":
            offset_by_node[nid] = _num(row.get("DATA"))

    controllers, plans, phases, phase_mvmts = [], [], [], []
    unmatched = 0
    for nid, phase_map in signals.items():
        nid = str(nid)
        controllers.append({"controller_id": nid, "node_id": nid})
        plan_id = f"{nid}_1"
        plans.append({
            "timing_plan_id": plan_id,
            "controller_id": nid,
            "time_day": "",
            "cycle_length": cycle_by_node.get(nid, ""),
            "offset": offset_by_node.get(nid, ""),
        })

        for key, ph in phase_map.items():
            num = _phase_num(key)
            if num is None or not isinstance(ph, dict):
                continue          # skips 'brp_info'
            barrier, ring, position = _brp(ph.get("BRP"))
            yellow = _num(ph.get("Yellow"), 0) or 0
            allred = _num(ph.get("AllRed"), 0) or 0
            phases.append({
                "timing_phase_id": f"{nid}_{num}",
                "timing_plan_id": plan_id,
                "signal_phase_num": num,
                "ring": ring,
                "barrier": barrier,
                "position": position,
                "min_green": _num(ph.get("MinGreen")),
                "max_green": _num(ph.get("MaxGreen")),
                "extension": _num(ph.get("VehExt")),
                "clearance": round(yellow + allred, 3),
                "walk_time": _num(ph.get("Walk")),
                "ped_clearance": _num(ph.get("DontWalk")),
            })

            for field, protection in (("protected", "protected"),
                                      ("permitted", "permitted")):
                for mname in (ph.get(field) or []):
                    mid = name_to_mvmt.get((nid, str(mname).strip().upper()))
                    if mid is None:
                        unmatched += 1
                        continue
                    phase_mvmts.append({
                        "signal_phase_mvmt_id": f"{nid}_{num}_{mname}",
                        # timing_phase_id is the join readers actually use to
                        # get from a phase to its movements
                        "timing_phase_id": f"{nid}_{num}",
                        "controller_id": nid,
                        "signal_phase_num": num,
                        "mvmt_id": mid,
                        "protection": protection,
                    })

    _write(os.path.join(dst_dir, "signal_controller.csv"), controllers,
           ["controller_id", "node_id"])
    _write(os.path.join(dst_dir, "signal_timing_plan.csv"), plans,
           ["timing_plan_id", "controller_id", "time_day", "cycle_length", "offset"])
    _write(os.path.join(dst_dir, "signal_timing_phase.csv"), phases,
           ["timing_phase_id", "timing_plan_id", "signal_phase_num", "ring", "barrier",
            "position", "min_green", "max_green", "extension", "clearance",
            "walk_time", "ped_clearance"])
    _write(os.path.join(dst_dir, "signal_phase_mvmt.csv"), phase_mvmts,
           ["signal_phase_mvmt_id", "timing_phase_id", "controller_id",
            "signal_phase_num", "mvmt_id", "protection"])
    if unmatched:
        say(f"  ! {unmatched} phase movement(s) had no matching movement row")

    # --------------------------------------------------------------- config
    # utdf2gmns writes lon/lat and SI throughout, so say so explicitly rather
    # than letting a reader assume US units.
    _write(os.path.join(dst_dir, "config.csv"), [{
        "dataset_name": dataset_name,
        "short_length": "meter",
        "long_length": "meter",
        "speed": "mps",
        "crs": crs,
        "geometry_field_format": "wkt",
        "version_number": "0.96",
        "id_type": "string",
    }], ["dataset_name", "short_length", "long_length", "speed", "crs",
         "geometry_field_format", "version_number", "id_type"])

    say(f"  :Rewrote GMNS to the spec in \n    {dst_dir}")
    say(f"    {len(out_nodes)} nodes, {len(out_links)} links, {len(out_lanes)} lanes, "
        f"{len(out_mvmts)} movements")
    say(f"    {len(controllers)} signals, {len(phases)} phases, "
        f"{len(phase_mvmts)} phase-movements")
    return dst_dir
