import os
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_PATH = os.path.join(BASE_DIR, "data/source/source.ply")
TARGET_PATH = os.path.join(BASE_DIR, "data/target/target.ply")
DEFORMED_PATH = os.path.join(BASE_DIR, "data/save_deformed/deformed.ply")
CORR_DIR = os.path.join(BASE_DIR, "data/correspondence/")
CORR_NAME = "correspondence.npy"


def load_ply_points(path):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        raise ValueError(f"No points loaded from {path}")
    return pts


def adaptive_threshold(target_pts, multiplier=3.0):
    """Multiplier x median nearest-neighbour distance of the target cloud."""
    tree = cKDTree(target_pts)
    d, _ = tree.query(target_pts, k=2)
    return multiplier * float(np.median(d[:, 1]))


def mutual_nn_with_threshold(deformed_pts, target_pts, dist_threshold):
    """
    deformed_pts: (N,3)   target_pts: (M,3)
    Returns indices (def_idx, tgt_idx) and distances for pairs that
    (a) are mutual nearest neighbours, and
    (b) have deformed->target distance < dist_threshold.
    """
    tree_tgt = cKDTree(target_pts)
    d_d2t, idx_d2t = tree_tgt.query(deformed_pts, k=1)

    tree_def = cKDTree(deformed_pts)
    _, idx_t2d = tree_def.query(target_pts, k=1)

    def_idx = np.arange(len(deformed_pts))
    mutual = idx_t2d[idx_d2t] == def_idx
    within = d_d2t < dist_threshold
    keep = mutual & within
    return def_idx[keep], idx_d2t[keep], d_d2t[keep]


def parse_args():
    p = argparse.ArgumentParser(
        description="Find source<->target correspondences via mutual nearest "
                    "neighbour between deformed source and target point clouds."
    )
    p.add_argument("--source", default=SOURCE_PATH)
    p.add_argument("--target", default=TARGET_PATH)
    p.add_argument("--deformed", default=DEFORMED_PATH)
    p.add_argument("--output", default=os.path.join(CORR_DIR, CORR_NAME),
                   help="Path to save the (M,2) int correspondence array as .npy")
    p.add_argument("--threshold", type=float, default=None,
                   help="Absolute distance threshold (target frame). "
                        "If omitted, uses adaptive_mult * median NN distance of target.")
    p.add_argument("--adaptive-mult", type=float, default=3.0,
                   help="Multiplier for the adaptive threshold (default 3.0).")
    return p.parse_args()


def main():
    args = parse_args()

    src_pts = load_ply_points(args.source)
    tgt_pts = load_ply_points(args.target)
    def_pts = load_ply_points(args.deformed)

    if len(src_pts) != len(def_pts):
        raise ValueError(
            f"source ({len(src_pts)}) and deformed ({len(def_pts)}) point counts "
            f"differ; src[i] <-> deformed[i] alignment is required."
        )

    threshold = args.threshold
    if threshold is None:
        threshold = adaptive_threshold(tgt_pts, multiplier=args.adaptive_mult)

    src_idx, tgt_idx, dists = mutual_nn_with_threshold(def_pts, tgt_pts, threshold)
    pairs = np.stack([src_idx, tgt_idx], axis=1).astype(np.int64)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.output, pairs)

    pct = 100.0 * len(pairs) / max(len(def_pts), 1)
    mean_d = float(dists.mean()) if len(dists) else float("nan")
    print(f"threshold={threshold:.6f}  "
          f"pairs={len(pairs)}/{len(def_pts)} ({pct:.1f}%)  "
          f"mean dist={mean_d:.6f}  -> {args.output}")


if __name__ == "__main__":
    main()
