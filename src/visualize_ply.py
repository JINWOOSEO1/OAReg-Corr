import os
import argparse
import numpy as np
import open3d as o3d


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_PATH = os.path.join(REPO_ROOT, "data/source/source.ply")
TARGET_PATH = os.path.join(REPO_ROOT, "data/target/target.ply")
DEFORMED_PATH = os.path.join(REPO_ROOT, "data/save_deformed/deformed.ply")
CORR_PATH = os.path.join(REPO_ROOT, "data/correspondence/correspondence.npy")

COLOR_SOURCE = [1.0, 0.4, 0.4]    # red
COLOR_TARGET = [0.4, 0.7, 1.0]    # blue
COLOR_DEFORMED = [0.4, 1.0, 0.4]  # green
COLOR_LINE = [0.15, 0.15, 0.15]   # dark grey

SAMPLE_RATIO = 0.05  # fraction of correspondence pairs to draw


def load_pcd(path, color, keep_colors=False):
    pcd = o3d.io.read_point_cloud(path)
    if not pcd.has_points():
        raise ValueError(f"No points loaded from {path}")
    if not keep_colors or not pcd.has_colors():
        pcd.paint_uniform_color(color)
    return pcd


def make_line_set(pts_a, idx_a, pts_b, idx_b, color):
    """LineSet connecting pts_a[idx_a[k]] <-> pts_b[idx_b[k]]."""
    n = len(idx_a)
    pts = np.vstack([pts_a[idx_a], pts_b[idx_b]])
    lines = np.stack([np.arange(n), np.arange(n) + n], axis=1)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (n, 1)))
    return ls


def show(geoms, title):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries(
        geoms + [frame], window_name=title, point_show_normal=False,
    )


def view_overlay(args):
    geoms = []
    keep = not args.paint
    if not args.no_source and os.path.exists(args.source):
        print(f"  source   (red)   : {args.source}")
        geoms.append(load_pcd(args.source, COLOR_SOURCE, keep_colors=keep))
    if not args.no_target and os.path.exists(args.target):
        print(f"  target   (blue)  : {args.target}")
        geoms.append(load_pcd(args.target, COLOR_TARGET, keep_colors=keep))
    if not args.no_deformed and os.path.exists(args.deformed):
        print(f"  deformed (green) : {args.deformed}")
        geoms.append(load_pcd(args.deformed, COLOR_DEFORMED, keep_colors=keep))

    if not geoms:
        print("Nothing to display.")
        return
    show(geoms, "PLY viewer: red=source, blue=target, green=deformed")


def load_correspondence_pairs(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"correspondence file not found: {path}")
    pairs = np.load(path)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"expected (M,2) int array in {path}, got {pairs.shape}")
    src_idx = pairs[:, 0].astype(np.int64)
    tgt_idx = pairs[:, 1].astype(np.int64)

    n_total = len(pairs)
    n_target = max(1, int(round(n_total * SAMPLE_RATIO)))
    sel = np.linspace(0, n_total - 1, n_target).astype(np.int64)
    src_idx, tgt_idx = src_idx[sel], tgt_idx[sel]
    print(f"  correspondence   : {path}  pairs={n_total} shown={len(src_idx)}")
    return src_idx, tgt_idx


def view_correspondence(args, use_deformed):
    src_idx, tgt_idx = load_correspondence_pairs(args.correspondence)

    tgt_pcd = load_pcd(args.target, COLOR_TARGET)
    tgt_pts = np.asarray(tgt_pcd.points)

    if use_deformed:
        if not os.path.exists(args.deformed):
            raise FileNotFoundError(f"deformed ply not found: {args.deformed}")
        dfm_pcd = load_pcd(args.deformed, COLOR_DEFORMED)
        dfm_pts = np.asarray(dfm_pcd.points)
        # deformed[i] aligns with source[i], so use src_idx into deformed.
        ls = make_line_set(dfm_pts, src_idx, tgt_pts, tgt_idx, COLOR_LINE)
        print(f"  deformed (green) : {args.deformed}")
        show([dfm_pcd, tgt_pcd, ls], "deformed<->target correspondences")
    else:
        src_pcd = load_pcd(args.source, COLOR_SOURCE)
        src_pts = np.asarray(src_pcd.points)
        offset = (tgt_pts.mean(axis=0) + np.array([0.3, 0.0, 0.0])) - src_pts.mean(axis=0)
        src_pcd.translate(offset)
        src_pts = np.asarray(src_pcd.points)
        ls = make_line_set(src_pts, src_idx, tgt_pts, tgt_idx, COLOR_LINE)
        print(f"  source   (red)   : {args.source}  (shifted by {offset})")
        print(f"  target   (blue)  : {args.target}")
        show([src_pcd, tgt_pcd, ls], "source<->target correspondences")


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize source.ply / target.ply (and optional deformed.ply, "
                    "correspondences) from the OAReg pipeline."
    )
    p.add_argument("--mode",
                   choices=["overlay", "correspondence", "deformed-correspondence"],
                   default="overlay",
                   help="overlay: draw source/target/deformed together; "
                        "correspondence: lines between source and target; "
                        "deformed-correspondence: lines between deformed and target.")
    p.add_argument("--source", default=SOURCE_PATH)
    p.add_argument("--target", default=TARGET_PATH)
    p.add_argument("--deformed", default=DEFORMED_PATH)
    p.add_argument("--correspondence", default=CORR_PATH)
    p.add_argument("--no-source", action="store_true")
    p.add_argument("--no-target", action="store_true")
    p.add_argument("--no-deformed", action="store_true")
    p.add_argument("--paint", action="store_true",
                   help="Ignore embedded RGB; paint each cloud a uniform color.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "overlay":
        view_overlay(args)
    elif args.mode == "correspondence":
        view_correspondence(args, use_deformed=False)
    elif args.mode == "deformed-correspondence":
        view_correspondence(args, use_deformed=True)


if __name__ == "__main__":
    main()
