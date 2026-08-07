"""
Automatic cross-camera ReID matcher (MTMC-style).

Replaces manual clicking with an appearance-driven pipeline inspired by the AI
City Challenge multi-target multi-camera (MTMC) solutions:

  1. Aggregate a robust appearance embedding per single-camera track
     (sample several crops across the track, extract IBN-ReID features, average).
  2. Gate candidate matches by spatio-temporal compatibility on the synced
     timeline (tracks must co-exist within a tolerance window).
  3. Re-rank distances with k-reciprocal encoding (Zhong et al.), the same
     technique used by the winning AI City entries.
  4. Cluster tracks into global IDs with a union-find that forbids two tracks
     from the same camera in one identity.

Output is written to the project's matches.json using the existing schema, so
the current visualization / export tooling works unchanged.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

import cv2

from ..core.project import Project
from ..core import tracks_io
from .extractor import ReIDExtractor


# --------------------------------------------------------------------------- #
# Track feature aggregation
# --------------------------------------------------------------------------- #
def _sample_indices(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    step = n / k
    return [int(i * step) for i in range(k)]


def _collect_track_crops(
    project: Project,
    extractor: ReIDExtractor,
    samples_per_track: int,
    min_box_size: int,
    min_track_len: int,
) -> dict:
    """
    Build one averaged embedding per track for every camera.

    Returns a dict keyed by (cam_name, track_id) -> {
        "feature": (D,) float32, "frames": (lo, hi), "frame_set": set[int]
    }
    """
    result: dict[tuple[str, int], dict] = {}

    for cam in project.cameras:
        tracks = tracks_io.load_tracks(project.tracks_path(cam))
        cap = cv2.VideoCapture(str(project.video_path(cam)))
        if not cap.isOpened():
            logger.warning(f"  Could not open video for '{cam.name}'")
            continue

        # Group the (frame -> crop) reads per track, but read each needed frame
        # once by iterating frames in ascending order.
        needed: dict[int, list[tuple[int, list[float]]]] = {}
        track_meta: dict[int, dict] = {}
        for tid, track in tracks.items():
            frames = track["frames"]
            boxes = track["boxes"]
            if len(frames) < min_track_len:
                continue
            idxs = _sample_indices(len(frames), samples_per_track)
            for j in idxs:
                needed.setdefault(frames[j], []).append((tid, boxes[j]))
            track_meta[tid] = {
                "lo": frames[0],
                "hi": frames[-1],
                "frame_set": set(frames),
                "crops": [],
            }

        for frame_idx in sorted(needed.keys()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            for tid, box in needed[frame_idx]:
                x1, y1, x2, y2 = box
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
                    continue
                track_meta[tid]["crops"].append(frame[y1:y2, x1:x2])
        cap.release()

        # Extract features per track and average.
        for tid, meta in track_meta.items():
            crops = meta["crops"]
            if not crops:
                continue
            feats = extractor.extract(crops)
            if feats.shape[0] == 0:
                continue
            mean_feat = feats.mean(axis=0)
            norm = np.linalg.norm(mean_feat) + 1e-12
            result[(cam.name, int(tid))] = {
                "feature": (mean_feat / norm).astype(np.float32),
                "lo": meta["lo"],
                "hi": meta["hi"],
                "frame_set": meta["frame_set"],
            }
        logger.info(f"  {cam.name}: embedded {sum(1 for k in result if k[0] == cam.name)} tracks")

    return result


# --------------------------------------------------------------------------- #
# k-reciprocal re-ranking (Zhong et al., CVPR 2017)
# --------------------------------------------------------------------------- #
def re_ranking(features: np.ndarray, k1: int = 20, k2: int = 6, lambda_value: float = 0.3) -> np.ndarray:
    """All-vs-all k-reciprocal re-ranked distance matrix for L2-normalized feats."""
    n = features.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    # Euclidean^2 on normalized vectors == 2 - 2*cos_sim.
    dist = np.clip(2.0 - 2.0 * (features @ features.T), 0.0, None)
    original_dist = dist / (np.max(dist, axis=0) + 1e-12)
    V = np.zeros_like(original_dist, dtype=np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    k1 = min(k1, n - 1) if n > 1 else 1
    for i in range(n):
        forward = initial_rank[i, : k1 + 1]
        backward = initial_rank[forward, : k1 + 1]
        fi = np.where(backward == i)[1]
        k_recip = forward[fi]
        k_recip_exp = k_recip
        half = int(round(k1 / 2.0)) + 1
        for j in k_recip:
            cand = initial_rank[j, :half]
            cand_back = initial_rank[cand, :half]
            fj = np.where(cand_back == j)[1]
            cand_recip = cand[fj]
            if len(np.intersect1d(cand_recip, k_recip)) > (2.0 / 3.0) * len(cand_recip):
                k_recip_exp = np.append(k_recip_exp, cand_recip)
        k_recip_exp = np.unique(k_recip_exp)
        weight = np.exp(-original_dist[i, k_recip_exp])
        V[i, k_recip_exp] = (weight / np.sum(weight)).astype(np.float32)

    if k2 > 1:
        V_qe = np.zeros_like(V)
        for i in range(n):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe

    inv_index = [np.where(V[:, i] != 0)[0] for i in range(n)]
    jaccard = np.zeros_like(original_dist)
    for i in range(n):
        temp_min = np.zeros((n,), dtype=np.float32)
        ind_nonzero = np.where(V[i, :] != 0)[0]
        ind_images = [inv_index[ind] for ind in ind_nonzero]
        for j in range(len(ind_nonzero)):
            temp_min[ind_images[j]] += np.minimum(V[i, ind_nonzero[j]], V[ind_images[j], ind_nonzero[j]])
        jaccard[i] = 1.0 - temp_min / (2.0 - temp_min + 1e-12)

    final_dist = jaccard * (1.0 - lambda_value) + original_dist * lambda_value
    np.fill_diagonal(final_dist, 0.0)
    return final_dist.astype(np.float32)


# --------------------------------------------------------------------------- #
# Constrained clustering into global IDs
# --------------------------------------------------------------------------- #
class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.cameras: list[set] = [set() for _ in range(n)]

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self.cameras[ra] & self.cameras[rb]:
            return False  # same camera in both -> would duplicate a camera
        self.parent[rb] = ra
        self.cameras[ra] |= self.cameras[rb]
        return True


def _temporally_compatible(a: dict, b: dict, gap_tol: int) -> bool:
    lo = max(a["lo"], b["lo"])
    hi = min(a["hi"], b["hi"])
    if hi >= lo:
        return True  # real overlap
    gap = lo - hi
    return gap <= gap_tol


def _covisible_frames(a: dict, b: dict) -> int:
    """Number of frames where both tracks are simultaneously visible."""
    fa, fb = a["frame_set"], b["frame_set"]
    # iterate over the smaller set for speed
    if len(fa) > len(fb):
        fa, fb = fb, fa
    return sum(1 for f in fa if f in fb)


def _representative_frame(members: list[dict]) -> int:
    """Frame where the most members are simultaneously visible."""
    from collections import Counter

    counter: Counter = Counter()
    for m in members:
        counter.update(m["frame_set"])
    if not counter:
        return int(members[0]["lo"])
    return int(counter.most_common(1)[0][0])


# --------------------------------------------------------------------------- #
# Spatial gate (shared ground frame)
# --------------------------------------------------------------------------- #
def _attach_world_trajectories(project, data: dict, ground_frame, stride: int = 2) -> None:
    """
    Add ``data[(cam, tid)]["world"] = {frame: (x, y)}`` — the track's foot points
    mapped into the shared ground frame — for every track that has calibration.
    """
    from ..core import tracks_io as _tio

    def _foot(box):
        x1, y1, x2, y2 = box
        return [(x1 + x2) * 0.5, float(y2)]

    for cam in project.cameras:
        if not ground_frame.has(cam.name):
            continue
        tracks = _tio.load_tracks(project.tracks_path(cam))
        for (ck, tid), entry in data.items():
            if ck != cam.name or tid not in tracks:
                continue
            tr = tracks[tid]
            frames, boxes = tr["frames"], tr["boxes"]
            idx = range(0, len(frames), stride)
            feet = [_foot(boxes[i]) for i in idx]
            fr = [frames[i] for i in idx]
            if not feet:
                continue
            w = ground_frame.project_feet(cam.name, feet)
            entry["world"] = {int(fr[k]): (float(w[k, 0]), float(w[k, 1]))
                              for k in range(len(fr)) if np.isfinite(w[k]).all()}


def _world_distance(a: dict, b: dict, min_shared: int = 2) -> float | None:
    """
    Median shared-frame distance between two tracks' foot points, or None if they
    lack enough co-visible frames with world positions.
    """
    wa = a.get("world")
    wb = b.get("world")
    if not wa or not wb:
        return None
    shared = set(wa) & set(wb)
    if len(shared) < min_shared:
        return None
    ds = []
    for f in shared:
        ax, ay = wa[f]
        bx, by = wb[f]
        ds.append(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)
    ds.sort()
    return float(ds[len(ds) // 2])


def auto_match(
    project: Project,
    samples_per_track: int = 6,
    min_box_size: int = 24,
    min_track_len: int = 5,
    dist_threshold: float = 0.45,
    min_covis: int = 3,
    mutual_only: bool = True,
    use_reranking: bool = False,
    weights_path: str | None = None,
    ground_frame_path: str | None = None,
    max_world_dist: float = 0.75,
    world_min_shared: int = 2,
    require_world: bool = True,
) -> list[dict]:
    """
    Compute cross-camera matches automatically and save them to matches.json.

    Strategy (AI City MTMC style, tuned for annotation pre-labeling):
      * appearance embedding per track (vehicle-ReID features),
      * candidates gated by frame-level co-visibility (same vehicle crossing a
        fixed intersection is seen simultaneously in overlapping cameras),
      * if a shared ground frame is available, also gate by real-world distance
        between the tracks' foot points (the AIC21-MTMC spatial constraint),
      * mutual nearest-neighbor + distance threshold -> high precision,
      * union-find clustering into global IDs (no camera used twice per ID).

    Returns the list of match dicts written.
    """
    project.load()
    if len(project.cameras) < 2:
        raise ValueError("Automatic matching needs at least 2 cameras.")

    extractor = ReIDExtractor(weights_path=weights_path)
    logger.info("  Extracting per-track appearance features ...")
    data = _collect_track_crops(
        project, extractor, samples_per_track, min_box_size, min_track_len
    )
    keys = list(data.keys())
    if len(keys) < 2:
        raise ValueError("Not enough tracks with valid crops to match.")

    # Optional spatial gate: load a shared ground frame if present.
    ground_frame = None
    if ground_frame_path is None:
        default_gf = project.matches_path.parent / "ground_frame.json"
        if default_gf.exists():
            ground_frame_path = str(default_gf)
    if ground_frame_path:
        from ..core.ground_align import load_ground_frame
        ground_frame = load_ground_frame(ground_frame_path)
        _attach_world_trajectories(project, data, ground_frame)
        logger.info(
            f"  Spatial gate ON (shared ground frame, "
            f"max_world_dist={max_world_dist}, min_shared={world_min_shared})"
        )

    features = np.stack([data[k]["feature"] for k in keys], axis=0)
    logger.info(
        f"  Matching {len(keys)} tracks "
        f"({'re-ranked' if use_reranking else 'cosine'} dist, "
        f"covis>={min_covis}, mutual_nn={mutual_only}, thr={dist_threshold}) ..."
    )
    if use_reranking:
        dist = re_ranking(features)
    else:
        dist = np.clip(1.0 - features @ features.T, 0.0, None)

    n = len(keys)
    cams = [k[0] for k in keys]

    # For each track, find its best temporally-compatible cross-camera partner
    # per other camera. best_partner[i][cam] = (dist, j) or absent.
    best_partner: list[dict[str, tuple[float, int]]] = [dict() for _ in range(n)]
    for i in range(n):
        di = data[keys[i]]
        for j in range(n):
            if cams[j] == cams[i]:
                continue
            if dist[i, j] > dist_threshold:
                continue
            if _covisible_frames(di, data[keys[j]]) < min_covis:
                continue
            if ground_frame is not None:
                wd = _world_distance(di, data[keys[j]], world_min_shared)
                if wd is None:
                    if require_world:
                        continue  # no spatial confirmation -> reject (high precision)
                elif wd > max_world_dist:
                    continue  # too far apart on the ground to be the same vehicle
            cur = best_partner[i].get(cams[j])
            if cur is None or dist[i, j] < cur[0]:
                best_partner[i][cams[j]] = (float(dist[i, j]), j)

    # Keep edges that are mutual nearest neighbors (i<->j) for high precision.
    edges: list[tuple[float, int, int]] = []
    for i in range(n):
        for cam_b, (d_ij, j) in best_partner[i].items():
            back = best_partner[j].get(cams[i])
            if not mutual_only or (back is not None and back[1] == i):
                edges.append((d_ij, min(i, j), max(i, j)))
    # deduplicate
    edges = sorted(set(edges), key=lambda e: e[0])

    uf = _UnionFind(n)
    for idx in range(n):
        uf.cameras[idx].add(cams[idx])
    for _d, i, j in edges:
        uf.union(i, j)

    # Gather clusters with >= 2 cameras.
    clusters: dict[int, list[int]] = {}
    for idx in range(n):
        clusters.setdefault(uf.find(idx), []).append(idx)

    matches: list[dict] = []
    for members in clusters.values():
        member_cams = {cams[m] for m in members}
        if len(member_cams) < 2:
            continue
        member_data = [data[keys[m]] for m in members]
        rep_frame = _representative_frame(member_data)
        track_map: dict[str, int | None] = {name: None for name in project.camera_names}
        for m in members:
            cam_name, tid = keys[m]
            track_map[cam_name] = int(tid)
        matches.append({"frame": rep_frame, "tracks": track_map})

    matches.sort(key=lambda m: m["frame"])
    tracks_io.save_matches(project.matches_path, matches)
    n_full = sum(
        1 for m in matches
        if sum(v is not None for v in m["tracks"].values()) == len(project.cameras)
    )
    logger.info(
        f"  Automatic matching produced {len(matches)} global IDs "
        f"({n_full} span all {len(project.cameras)} cameras)"
    )
    return matches
