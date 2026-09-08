'''
##############################################################
# Georeference a UTDF network by consensus rather than by one anchor.
##############################################################

A UTDF file contains no latitude or longitude. Synchro stores nodes on a private
grid measured from an arbitrary origin, so something has to tie that grid to the
earth.

The single-anchor approach geocodes intersection NAMES and accepts the FIRST one
whose forward ("A & B") and reversed ("B & A") lookups agree within a threshold,
then places every node by offsetting from that one point. That check tests
whether the geocoder is self-consistent, not whether it is right: a confidently
wrong geocode returns the same wrong point twice, agrees perfectly, and is
accepted. On the Bullhead SR 95 corridor "SR 95 & Fairway Vlg Blvd" resolves
44 km away with a forward/reverse gap of 0.000 km. If it wins the race, the
whole network moves 44 km, and nothing downstream complains.

This module geocodes a spread of intersections instead and fits one similarity
transform (scale + rotation + translation) by RANSAC consensus, which:

  1. outvotes a bad geocode instead of letting it decide,
  2. measures scale instead of assuming Synchro units are exactly feet,
  3. measures rotation instead of assuming Synchro +Y is true north.

Measured against 8 OSM-matched intersections on that corridor: 13.8 m mean /
26.6 m max error before, 3.3 m / 4.5 m after, with the 44 km outlier rejected
automatically and its own intersection still landing within 3.2 m.
'''
import math

from utdf2gmns.func_lib.utdf.geocoding_intersection import geocoder_geocoding_from_address

FT_TO_M = 0.3048        # metres per international foot
INLIER_TOL_M = 40.0     # a geocoded centroid further off than this is an outlier
MIN_INLIERS = 3
MAX_GEOCODE = 25        # geocoding hundreds of names is slow and rate-limited


def spread(points: list, k: int) -> list:
    """Farthest-point sampling: indices of k well-separated points.

    Control points bunched together fit rotation badly, and geocoding every
    intersection in a large network is slow. Spreading them fixes both.
    """
    if len(points) <= k:
        return list(range(len(points)))
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    chosen = [max(range(len(points)), key=lambda i: math.dist(points[i], (cx, cy)))]
    while len(chosen) < k:
        chosen.append(max((i for i in range(len(points)) if i not in chosen),
                          key=lambda i: min(math.dist(points[i], points[c]) for c in chosen)))
    return chosen


def similarity(src: list, dst: list):
    """Best scale + rotation + translation taking point set src onto dst.

    Umeyama's closed-form solution.
    """
    n = len(src)
    if n == 0:
        return None
    sx, sy = sum(p[0] for p in src) / n, sum(p[1] for p in src) / n
    dx, dy = sum(p[0] for p in dst) / n, sum(p[1] for p in dst) / n
    src_c = [(p[0] - sx, p[1] - sy) for p in src]
    dst_c = [(p[0] - dx, p[1] - dy) for p in dst]
    s_xx = sum(src_c[k][0] * dst_c[k][0] + src_c[k][1] * dst_c[k][1] for k in range(n))
    s_xy = sum(src_c[k][0] * dst_c[k][1] - src_c[k][1] * dst_c[k][0] for k in range(n))
    norm = sum(a * a + b * b for a, b in src_c)
    if norm == 0:
        return None
    return (math.hypot(s_xx, s_xy) / norm, math.atan2(s_xy, s_xx), (sx, sy), (dx, dy))


def apply_similarity(params: tuple, point: tuple) -> tuple:
    """Push one point through a transform returned by similarity()."""
    scale, theta, (sx, sy), (dx, dy) = params
    a, b = point[0] - sx, point[1] - sy
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return (scale * (cos_t * a - sin_t * b) + dx,
            scale * (sin_t * a + cos_t * b) + dy)


def consensus_fit(src: list, dst: list, *, inlier_tol_m: float = INLIER_TOL_M,
                  min_inliers: int = MIN_INLIERS):
    """RANSAC over point pairs: the transform most intersections agree with wins.

    Returns (params, inlier_indices), or (None, indices) when too few points
    agree to trust any transform.
    """
    best_inliers, _best_params = [], None
    for i in range(len(src)):
        for j in range(i + 1, len(src)):
            params = similarity([src[i], src[j]], [dst[i], dst[j]])
            if not params:
                continue
            inliers = [k for k in range(len(src))
                       if math.dist(apply_similarity(params, src[k]), dst[k]) <= inlier_tol_m]
            if len(inliers) > len(best_inliers):
                best_inliers, _best_params = inliers, params

    if len(best_inliers) < min_inliers:
        return None, best_inliers

    inliers = best_inliers
    for _ in range(2):                   # refit on the inliers, then reselect
        params = similarity([src[k] for k in inliers], [dst[k] for k in inliers])
        inliers = [k for k in range(len(src))
                   if math.dist(apply_similarity(params, src[k]), dst[k]) <= inlier_tol_m]
    return similarity([src[k] for k in inliers], [dst[k] for k in inliers]), inliers


def _metres_per_degree(lat_deg: float) -> tuple:
    """Local metres per degree of latitude and longitude."""
    lat = math.radians(lat_deg)
    m_lat = 111132.92 - 559.82 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat)
    m_lon = 111412.84 * math.cos(lat) - 93.5 * math.cos(3 * lat)
    return m_lat, m_lon


def geocode_intersection_names(names: list, region_name: str, *,
                               cache: dict = None, verbose: bool = True) -> dict:
    """Geocode intersection names, reusing and updating an optional cache.

    A failed lookup returns [0, 0]; that is stored as None so it can never be
    mistaken for a real point at Null Island.
    """
    cache = {} if cache is None else cache
    todo = [n for n in names if n not in cache]
    for i, name in enumerate(todo, 1):
        if verbose:
            print(f"    geocoding {i}/{len(todo)}: {name}")
        try:
            result = geocoder_geocoding_from_address(f"{name},{region_name}")
        except Exception as err:            # a dead lookup must not kill the fit
            if verbose:
                print(f"      failed: {err}")
            result = [0, 0]
        is_null = not result or (abs(result[0]) < 1e-9 and abs(result[1]) < 1e-9)
        cache[name] = None if is_null else list(result)
    return cache


def fit_network_coordinates(df_utdf_intersection, df_nodes, region_name: str, *,
                            net_unit: str = "feet, mph",
                            max_geocode: int = MAX_GEOCODE,
                            inlier_tol_m: float = INLIER_TOL_M,
                            min_inliers: int = MIN_INLIERS,
                            cache: dict = None,
                            verbose: bool = True) -> dict:
    """Georeference a UTDF network by consensus over many geocoded intersections.

    Args:
        df_utdf_intersection: output of generate_intersection_from_Links, with
            synchro_INTID and intersection_name columns.
        df_nodes: the UTDF Nodes table, with INTID, X and Y columns.
        region_name (str): city/state appended to each name before geocoding.
        net_unit (str): the network's unit string, e.g. "feet, mph".
        max_geocode (int): cap on how many names to geocode.
        cache (dict): name -> [lon, lat], read and extended in place.

    Returns:
        dict: {"transform", "lon0", "lat0", "m_lon", "m_lat", "seed",
               "scale", "rotation_deg", "n_inliers", "n_points", "rejected"},
        or {} when too few intersections could be geocoded to fit anything.
    """
    unit_scale = FT_TO_M if "feet" in str(net_unit).lower() else 1.0

    names = {str(row["synchro_INTID"]).strip(): row["intersection_name"]
             for _, row in df_utdf_intersection.iterrows()}
    xy = {str(row["INTID"]).strip(): (float(row["X"]), float(row["Y"]))
          for _, row in df_nodes.iterrows()}

    pool = [(int_id, name) for int_id, name in names.items() if int_id in xy]
    if not pool:
        if verbose:
            print("  ! no intersection names could be paired with node coordinates")
        return {}

    keep = spread([xy[int_id] for int_id, _ in pool], max_geocode)
    picked = [pool[i] for i in keep]
    if verbose:
        print(f"  :using {len(picked)} of {len(pool)} named intersections as control points")

    cache = geocode_intersection_names(sorted({name for _, name in picked}),
                                       region_name, cache=cache, verbose=verbose)

    candidates = [(int_id, xy[int_id], cache[name])
                  for int_id, name in picked if cache.get(name)]
    if len(candidates) < min_inliers:
        if verbose:
            print(f"  ! only {len(candidates)} usable geocodes, need {min_inliers}; "
                  "check region_name")
        return {}

    lat0 = sum(c[2][1] for c in candidates) / len(candidates)
    lon0 = sum(c[2][0] for c in candidates) / len(candidates)
    m_lat, m_lon = _metres_per_degree(lat0)

    synchro = [(c[1][0] * unit_scale, c[1][1] * unit_scale) for c in candidates]
    enu = [((c[2][0] - lon0) * m_lon, (c[2][1] - lat0) * m_lat) for c in candidates]

    params, inliers = consensus_fit(synchro, enu, inlier_tol_m=inlier_tol_m,
                                    min_inliers=min_inliers)
    if params is None:
        if verbose:
            print("  ! no consistent set of geocoded intersections; check region_name")
        return {}

    scale, theta, _, _ = params
    rejected = []
    for k in range(len(candidates)):
        if k not in inliers:
            off_m = math.dist(apply_similarity(params, synchro[k]), enu[k])
            rejected.append((candidates[k][0], names[candidates[k][0]], off_m))

    if verbose:
        print(f"  :consensus fit on {len(inliers)}/{len(candidates)} intersections")
        print(f"    scale    {scale:.5f}  ({(scale - 1) * 100:+.3f} % vs assuming exact feet)")
        print(f"    rotation {math.degrees(theta):+.4f} deg vs assuming true north")
        for int_id, name, off_m in rejected:
            off = f"{off_m / 1000:.1f} km" if off_m >= 1000 else f"{off_m:.0f} m"
            print(f"    rejected INT {int_id:>5}  {name!r}  {off} off")
        if not rejected:
            print("    no outliers")

    seed = candidates[inliers[0]]
    return {
        "transform": params,
        "lon0": lon0,
        "lat0": lat0,
        "m_lon": m_lon,
        "m_lat": m_lat,
        "unit_scale": unit_scale,
        # a known-good anchor, so the existing single-anchor path can still run
        "seed": {"INTID": str(seed[0]),
                 "x_coord": float(seed[2][0]),
                 "y_coord": float(seed[2][1])},
        "scale": scale,
        "rotation_deg": math.degrees(theta),
        "n_inliers": len(inliers),
        "n_points": len(candidates),
        "rejected": rejected,
    }


def apply_network_fit(fit: dict, network_nodes: dict) -> dict:
    """Overwrite every node's lon/lat using a fit from fit_network_coordinates."""
    if not fit:
        return network_nodes
    params = fit["transform"]
    lon0, lat0 = fit["lon0"], fit["lat0"]
    m_lon, m_lat = fit["m_lon"], fit["m_lat"]
    unit_scale = fit["unit_scale"]

    for node in network_nodes.values():
        east, north = apply_similarity(
            params, (float(node["X"]) * unit_scale, float(node["Y"]) * unit_scale))
        node["x_coord"] = east / m_lon + lon0
        node["y_coord"] = north / m_lat + lat0
    return network_nodes
