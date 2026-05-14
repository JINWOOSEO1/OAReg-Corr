#!/usr/bin/env python
"""
Interactively segment the object in **one reference view per side** with SAM2,
then propagate those clicks to the remaining views by unprojecting them to 3D
via the reference view's depth map and re-projecting through each other view's
intrinsics/pose. SAM2 then runs on every view (interactive on the reference,
auto on the rest) and the fused colored point clouds (world frame) are saved
to data/source/source.ply and data/target/target.ply, with XYZ in meters and
RGB stored per-point.

Run with the `instant` conda env (the only one that has `sam2` installed):

    /home/seojinwoo/miniforge3/envs/instant/bin/python \
        scripts/build_segmented_pointclouds.py

The first view (cam1) is the reference by default; override with --ref-cam.
Click 3-5 prompts on the reference view; the others are auto-prompted.

Click controls (reference view only):
    Left click   add foreground point
    Right click  add background point
    p            preview SAM2 mask with current points
    u            undo last point
    r            reset all points
    Enter        confirm and finalize mask
    q            cancel and exit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import open3d as o3d

from dataclasses import dataclass
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@dataclass
class CameraView:
    rgb: np.ndarray              # (H, W, 3) uint8
    depth: np.ndarray            # (H, W) float32, meters; <=0 invalid
    K: np.ndarray                # (3, 3) float64
    extrinsic: np.ndarray        # (4, 4) float64 — convention: x_cam = R @ x_world + t
    H: int
    W: int


@dataclass
class MultiViewInput:
    views: List[CameraView]      # views[0] is the keypoint-picking camera (cam1)
    kpts: np.ndarray             # (N, 2) pixel coords in views[0]


# Kept for callers that still want a single-view container.
@dataclass
class RGBDInput:
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    kpts: np.ndarray
    H: int
    W: int


class DataProcessor:

    @staticmethod
    def _read_rgb(path) -> np.ndarray:
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

    @staticmethod
    def _read_depth(path) -> np.ndarray:
        ext = str(path).lower().rsplit(".", 1)[-1]
        if ext == "npy":
            depth = np.load(path).astype(np.float32)
        elif ext == "npz":
            with np.load(path) as f:
                key = "depth" if "depth" in f.files else f.files[0]
                depth = f[key].astype(np.float32)
        else:
            depth = np.array(Image.open(path), dtype=np.float32)
            if depth.max() > 100.0:
                depth = depth / 1000.0
        if depth.ndim == 3:
            depth = depth[..., 0]
        return depth

    @staticmethod
    def load_multiview(rgb_paths: Sequence,
                       depth_paths: Sequence,
                       Ks: Sequence,
                       extrinsics: Sequence,
                       kpts: np.ndarray) -> MultiViewInput:
        if not (len(rgb_paths) == len(depth_paths) == len(Ks) == len(extrinsics)):
            raise ValueError("rgb_paths/depth_paths/Ks/extrinsics must all have the same length")
        views: List[CameraView] = []
        for rgb_p, dp, K, ext in zip(rgb_paths, depth_paths, Ks, extrinsics):
            rgb = DataProcessor._read_rgb(rgb_p)
            depth = DataProcessor._read_depth(dp)
            H, W = rgb.shape[:2]
            views.append(CameraView(
                rgb=rgb, depth=depth,
                K=np.asarray(K, dtype=np.float64).reshape(3, 3),
                extrinsic=np.asarray(ext, dtype=np.float64).reshape(4, 4),
                H=H, W=W,
            ))
        return MultiViewInput(views=views,
                              kpts=np.asarray(kpts, dtype=np.float32).reshape(-1, 2))

    @staticmethod
    def load(rgb_path,
             depth_path,
             K: np.ndarray,
             kpts: np.ndarray) -> RGBDInput:
        rgb = DataProcessor._read_rgb(rgb_path)
        H, W = rgb.shape[:2]
        depth = (DataProcessor._read_depth(depth_path) if depth_path is not None
                 else np.zeros((H, W), dtype=np.float32))
        return RGBDInput(
            rgb=rgb,
            depth=depth,
            K=np.asarray(K, dtype=np.float32).reshape(3, 3),
            kpts=np.asarray(kpts, dtype=np.float32).reshape(-1, 2),
            H=H, W=W,
        )

    @staticmethod
    def unproject_depth_to_pointmap(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Per-pixel 3D point in the *camera* frame; invalid depth becomes NaN."""
        H, W = depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        u = np.arange(W, dtype=np.float32)
        v = np.arange(H, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        z = depth.astype(np.float32)
        invalid = z <= 0
        x = (uu - cx) * z / fx
        y = (vv - cy) * z / fy
        pm = np.stack([x, y, z], axis=-1)
        pm[invalid] = np.nan
        return pm

    @staticmethod
    def _bilinear_sample(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]
        x = np.clip(xy[:, 0], 0, W - 1)
        y = np.clip(xy[:, 1], 0, H - 1)
        x0 = np.floor(x).astype(np.int64); x1 = np.clip(x0 + 1, 0, W - 1)
        y0 = np.floor(y).astype(np.int64); y1 = np.clip(y0 + 1, 0, H - 1)
        wx = (x - x0).astype(np.float32); wy = (y - y0).astype(np.float32)
        Ia = img[y0, x0]; Ib = img[y0, x1]; Ic = img[y1, x0]; Id = img[y1, x1]
        out = (Ia.T * (1 - wx) * (1 - wy) + Ib.T * wx * (1 - wy)
               + Ic.T * (1 - wx) * wy + Id.T * wx * wy).T
        return out

    @staticmethod
    def kpts_to_cam(kpts: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
        """2D pixel keypoints + depth -> 3D in *camera* frame."""
        z = DataProcessor._bilinear_sample(depth[..., None], kpts).reshape(-1)
        valid = z > 0
        if not valid.all():
            for i in np.where(~valid)[0]:
                u, v = int(round(kpts[i, 0])), int(round(kpts[i, 1]))
                ys, xs = np.where(depth > 0)
                if len(xs) == 0:
                    z[i] = 1.0
                    continue
                d2 = (xs - u) ** 2 + (ys - v) ** 2
                j = np.argmin(d2)
                z[i] = depth[ys[j], xs[j]]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x = (kpts[:, 0] - cx) * z / fx
        y = (kpts[:, 1] - cy) * z / fy
        return np.stack([x, y, z], axis=-1).astype(np.float64)

    @staticmethod
    def cam_to_world(P_cam: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
        """x_cam = R @ x_world + t  =>  x_world = R^T @ (x_cam - t)."""
        R = extrinsic[:3, :3]; t = extrinsic[:3, 3]
        return ((P_cam - t[None, :]) @ R).astype(np.float64)

    @staticmethod
    def world_to_cam(P_world: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
        R = extrinsic[:3, :3]; t = extrinsic[:3, 3]
        return (P_world @ R.T + t[None, :]).astype(np.float64)

    @staticmethod
    def kpts_to_world(kpts: np.ndarray, view: CameraView) -> np.ndarray:
        """Pixel keypoints in `view` -> 3D world-frame points."""
        P_cam = DataProcessor.kpts_to_cam(kpts, view.depth, view.K)
        return DataProcessor.cam_to_world(P_cam, view.extrinsic)

    @staticmethod
    def world_to_pixel(P_world: np.ndarray,
                       view: CameraView):
        """World -> pixel coords + per-point depth in `view`'s camera frame."""
        P_cam = DataProcessor.world_to_cam(P_world, view.extrinsic)
        z = P_cam[:, 2]
        z_safe = np.where(z > 1e-6, z, 1e-6)
        K = view.K
        u = K[0, 0] * P_cam[:, 0] / z_safe + K[0, 2]
        v = K[1, 1] * P_cam[:, 1] / z_safe + K[1, 2]
        return np.stack([u, v], axis=-1), z

    @staticmethod
    def kpts_3d(kpts: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
        return DataProcessor.kpts_to_cam(kpts, depth, K)

    # Backward-compat alias used by older code paths.
    kpts_to_3d = kpts_3d


DEFAULT_SAM2_REPO = Path("/home/seojinwoo/workspace/codes/instant_policy/sam2_repo")
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
DEFAULT_SAM2_CKPT = DEFAULT_SAM2_REPO / "checkpoints/sam2.1_hiera_base_plus.pt"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_camera_params(path: Path, n_cams: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    arr = np.load(path)
    Ks, extrinsics = [], []
    for i in range(1, n_cams + 1):
        Ks.append(np.asarray(arr[f"cam{i}_intrinsic"], dtype=np.float64))
        extrinsics.append(np.asarray(arr[f"cam{i}_extrinsic"], dtype=np.float64))
    return Ks, extrinsics


class SAM2PromptUI:
    """cv2 window that collects pos/neg clicks and previews SAM2 masks live."""

    def __init__(self, predictor):
        self.predictor = predictor

    def run(self, window: str, rgb_uint8: np.ndarray) -> np.ndarray | None:
        # Embed the image once; subsequent predict() calls are cheap.
        self.predictor.set_image(rgb_uint8)
        bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
        H, W = bgr.shape[:2]
        points: List[Tuple[float, float]] = []
        labels: List[int] = []          # 1 = foreground, 0 = background
        cur_mask: np.ndarray | None = None
        state = {"dirty": True}

        def predict_mask() -> np.ndarray | None:
            pos = [labels[i] == 1 for i in range(len(labels))]
            if not any(pos):
                return None
            pc = np.asarray(points, dtype=np.float32)
            pl = np.asarray(labels, dtype=np.int32)
            masks, scores, _ = self.predictor.predict(
                point_coords=pc,
                point_labels=pl,
                multimask_output=False,
            )
            return masks[0].astype(bool)

        def redraw():
            canvas = bgr.copy()
            if cur_mask is not None:
                tint = np.zeros_like(canvas)
                tint[cur_mask] = (0, 255, 0)  # green
                canvas = cv2.addWeighted(canvas, 1.0, tint, 0.4, 0.0)
                # Draw mask outline for crispness.
                m_u8 = cur_mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(canvas, contours, -1, (0, 255, 255), 1)
            for (x, y), lab in zip(points, labels):
                color = (0, 255, 0) if lab == 1 else (0, 0, 255)
                cv2.circle(canvas, (int(x), int(y)), 5, color, -1)
                cv2.circle(canvas, (int(x), int(y)), 6, (255, 255, 255), 1)
            n_pos = sum(1 for v in labels if v == 1)
            n_neg = sum(1 for v in labels if v == 0)
            msg1 = f"+{n_pos}  -{n_neg}   L=fg  R=bg  [p]=preview  [u]=undo  [r]=reset  [Enter]=confirm  [q]=cancel"
            cv2.putText(canvas, msg1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 2)
            cv2.imshow(window, canvas)

        def on_mouse(event, x, y, flags, _):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((float(x), float(y)))
                labels.append(1)
                state["dirty"] = True
            elif event == cv2.EVENT_RBUTTONDOWN:
                points.append((float(x), float(y)))
                labels.append(0)
                state["dirty"] = True

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, W, H)
        cv2.moveWindow(window, 100, 100)
        try:
            cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass
        cv2.imshow(window, bgr)
        cv2.waitKey(50)
        cv2.setMouseCallback(window, on_mouse)
        redraw()

        while True:
            if state["dirty"]:
                # Auto-refresh mask whenever clicks change.
                cur_mask = predict_mask()
                state["dirty"] = False
                redraw()
            k = cv2.waitKey(20) & 0xFF
            if k == ord('q'):
                cv2.destroyWindow(window)
                raise SystemExit(f"[build_segmented_pointclouds] cancelled in window {window!r}")
            if k == ord('u') and points:
                points.pop(); labels.pop()
                state["dirty"] = True
            elif k == ord('r'):
                points.clear(); labels.clear()
                cur_mask = None
                redraw()
            elif k == ord('p'):
                state["dirty"] = True
            elif k in (13, 10):
                # Enter with no clicks => treat the view as having no visible
                # object (e.g. fully occluded) and return None so the caller
                # can skip it.
                break
        cv2.destroyWindow(window)
        return cur_mask, list(points), list(labels)

    def predict_with_prompts(self,
                             rgb_uint8: np.ndarray,
                             points: np.ndarray,
                             labels: np.ndarray) -> np.ndarray | None:
        """Run SAM2 once with the given prompts and no UI.

        Used for views whose prompts were propagated from another view's clicks
        via depth+pose, so the user does not need to click again.
        """
        pc = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        pl = np.asarray(labels, dtype=np.int32).reshape(-1)
        if pc.size == 0 or not np.any(pl == 1):
            return None
        self.predictor.set_image(rgb_uint8)
        masks, _, _ = self.predictor.predict(
            point_coords=pc,
            point_labels=pl,
            multimask_output=False,
        )
        return masks[0].astype(bool)


_OCV_TO_BLENDER_CAM = np.array([1.0, -1.0, -1.0], dtype=np.float64)


def unproject_pixels_to_world(
    pixels: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    extrinsic: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lift 2D pixels from a view into the world frame using its depth map.

    Mirrors view_pointcloud's convention: per-pixel depth unprojects into the
    OpenCV camera frame, which is flipped Y/Z into the Blender frame before
    applying the (cam->world) extrinsic stored in camera_params.npz.

    Returns
    -------
    P_world : (N, 3) float64 points in world coordinates.
    valid   : (N,) bool — False where the depth lookup was missing or <= 0.
    """
    H, W = depth.shape[:2]
    u = pixels[:, 0]
    v = pixels[:, 1]
    iu = np.clip(np.round(u).astype(int), 0, W - 1)
    iv = np.clip(np.round(v).astype(int), 0, H - 1)
    z = depth[iv, iu].astype(np.float64)
    valid = np.isfinite(z) & (z > 0)
    z_safe = np.where(valid, z, 1.0)
    fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
    x = (u - cx) * z_safe / fx
    y = (v - cy) * z_safe / fy
    P_cam_ocv = np.stack([x, y, z_safe], axis=1)
    P_cam_blender = P_cam_ocv * _OCV_TO_BLENDER_CAM
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    P_world = P_cam_blender @ R.T + t[None, :]
    return P_world, valid


def project_points_3d_to_view(
    P_world: np.ndarray,
    K: np.ndarray,
    extrinsic: np.ndarray,
    H: int,
    W: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project world-frame 3D points into a camera view's pixel coordinates.

    Inverts the cam->world extrinsic (Blender convention) and the OpenCV<->
    Blender axis flip applied in view_pointcloud, then runs the pinhole model.

    Returns
    -------
    pixels  : (N, 2) float64 (u, v); may fall outside the image rectangle.
    in_view : (N,) bool — True iff the point is in front of the camera AND
              its (u, v) lies inside [0, W) x [0, H).
    """
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    P_cam_blender = (P_world - t[None, :]) @ R
    P_cam_ocv = P_cam_blender * _OCV_TO_BLENDER_CAM
    z = P_cam_ocv[:, 2]
    in_front = z > 1e-6
    z_safe = np.where(in_front, z, 1.0)
    u = K[0, 0] * (P_cam_ocv[:, 0] / z_safe) + K[0, 2]
    v = K[1, 1] * (P_cam_ocv[:, 1] / z_safe) + K[1, 2]
    in_view = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pixels = np.stack([u, v], axis=1)
    return pixels, in_view


def view_pointcloud(rgb: np.ndarray,
                    depth: np.ndarray,
                    K: np.ndarray,
                    extrinsic: np.ndarray,
                    mask: np.ndarray) -> np.ndarray:
    # camera_params.npz stores T_cam->world in Blender/OpenGL convention
    # (+Y up, camera looks toward -Z). unproject_depth_to_pointmap returns
    # points in OpenCV camera frame (+Y down, +Z forward), so flip Y/Z to
    # Blender cam frame before applying the pose. Verified empirically:
    # this brings the per-view world clouds into <5 cm median NN agreement,
    # versus ~37 cm under DataProcessor.cam_to_world's OpenCV world->cam
    # interpretation.
    pointmap = DataProcessor.unproject_depth_to_pointmap(depth, K)
    valid = mask & np.isfinite(pointmap[..., 2]) & (depth > 0)
    if not valid.any():
        return np.zeros((0, 6), dtype=np.float32)
    P_cam_ocv = pointmap[valid].astype(np.float64)
    P_cam_blender = P_cam_ocv * _OCV_TO_BLENDER_CAM
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    P_world = P_cam_blender @ R.T + t[None, :]
    colors = rgb[valid].astype(np.float32) / 255.0
    return np.concatenate([P_world.astype(np.float32), colors], axis=1)


def build_predictor(sam2_repo: Path, sam2_config: str, sam2_ckpt: Path, device: str):
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(sam2_config, str(sam2_ckpt), device=device)
    return SAM2ImagePredictor(model)


def save_mask_png(out_dir: Path, side: str, cam_idx: int, mask: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{side}_cam{cam_idx}_mask.png"),
                (mask.astype(np.uint8) * 255))


def save_ply(path: Path, arr: np.ndarray) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:, :3].astype(np.float64))
    if arr.shape[1] >= 6:
        rgb = np.clip(arr[:, 3:6].astype(np.float64), 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(str(path), pcd)


# Number of pixels to shave off the SAM2 mask boundary before unprojection.
# Boundary pixels often straddle object/background and produce 3D outliers when
# back-projected, so we trim the mask 2 px inward (3x3 kernel, 2 iterations).
MASK_ERODE_PX = 2


def erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=px)
    return eroded.astype(bool)


def process_side(side: str,
                 side_cfg: dict,
                 Ks: List[np.ndarray],
                 extrinsics: List[np.ndarray],
                 ui: SAM2PromptUI,
                 save_masks_dir: Path | None,
                 ref_cam: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rgb_paths = side_cfg["rgb"]
    depth_paths = side_cfg["depth"]
    n_cams = len(rgb_paths)
    if not (len(depth_paths) == len(Ks) == len(extrinsics) == n_cams):
        raise ValueError(
            f"[{side}] cam count mismatch: rgb={len(rgb_paths)} depth={len(depth_paths)} "
            f"K={len(Ks)} extrinsic={len(extrinsics)}"
        )
    if not (0 <= ref_cam < n_cams):
        raise ValueError(f"[{side}] --ref-cam={ref_cam} out of range for n_cams={n_cams}")

    Ks_arr = [np.asarray(K, dtype=np.float64).reshape(3, 3) for K in Ks]
    ext_arr = [np.asarray(E, dtype=np.float64).reshape(4, 4) for E in extrinsics]

    rgbs: List[np.ndarray] = []
    depths: List[np.ndarray] = []
    for i in range(n_cams):
        rgb = DataProcessor._read_rgb(rgb_paths[i])
        depth = DataProcessor._read_depth(depth_paths[i])
        if depth.shape[:2] != rgb.shape[:2]:
            raise ValueError(
                f"[{side} cam{i+1}] rgb/depth shape mismatch: "
                f"{rgb.shape[:2]} vs {depth.shape[:2]}"
            )
        rgbs.append(rgb)
        depths.append(depth)

    print(f"[{side}] click 3-5 prompts on the reference view cam{ref_cam+1}; "
          f"the other {n_cams - 1} view(s) will be SAM2'd automatically")
    ref_title = f"{side.upper()} cam{ref_cam+1} [REF]  ({Path(rgb_paths[ref_cam]).name})"
    print(f"[{side}] cam{ref_cam+1} (REF): prompting in window {ref_title!r}")
    ref_mask, ref_pts, ref_labs = ui.run(ref_title, rgbs[ref_cam])
    if ref_mask is None or not ref_pts:
        raise RuntimeError(
            f"[{side}] reference view cam{ref_cam+1} received no clicks — "
            "cannot propagate to the other views"
        )

    pts2d = np.asarray(ref_pts, dtype=np.float64)
    labs = np.asarray(ref_labs, dtype=np.int32)
    P_world, depth_ok = unproject_pixels_to_world(
        pts2d, depths[ref_cam], Ks_arr[ref_cam], ext_arr[ref_cam]
    )
    n_ok = int(depth_ok.sum())
    print(
        f"[{side}] cam{ref_cam+1} (REF): {n_ok}/{len(pts2d)} clicks have valid "
        f"depth for propagation"
    )
    P_world_ok = P_world[depth_ok]
    labs_ok = labs[depth_ok]
    if not np.any(labs_ok == 1):
        raise RuntimeError(
            f"[{side}] no foreground click on cam{ref_cam+1} has valid depth — "
            "cannot propagate to other views"
        )

    fg_clicks_world = P_world_ok[labs_ok == 1].astype(np.float64)

    clouds: List[np.ndarray] = []
    for i in range(n_cams):
        if i == ref_cam:
            mask = ref_mask
            print(f"[{side}] cam{i+1} (REF): using interactive mask")
        else:
            H, W = rgbs[i].shape[:2]
            pixels, in_frame = project_points_3d_to_view(
                P_world_ok, Ks_arr[i], ext_arr[i], H, W
            )
            pts_view = pixels[in_frame]
            labs_view = labs_ok[in_frame]
            n_fg = int((labs_view == 1).sum())
            n_bg = int((labs_view == 0).sum())
            print(
                f"[{side}] cam{i+1}: projected prompts = {n_fg} fg / {n_bg} bg "
                f"(of {len(P_world_ok)} 3D clicks)"
            )
            if n_fg == 0:
                print(f"[{side}] cam{i+1}: no foreground prompt in frame — skipping view")
                continue
            mask = ui.predict_with_prompts(rgbs[i], pts_view, labs_view)
            if mask is None:
                print(f"[{side}] cam{i+1}: SAM2 returned no mask — skipping")
                continue

        n_raw = int(mask.sum())
        mask = erode_mask(mask, MASK_ERODE_PX)
        n_mask = int(mask.sum())
        print(f"[{side}] cam{i+1}: mask covers {n_mask} pixels "
              f"(eroded {MASK_ERODE_PX}px from {n_raw})")
        if n_mask == 0:
            print(f"[{side}] cam{i+1}: mask empty after erosion — skipping view")
            continue
        if save_masks_dir is not None:
            save_mask_png(save_masks_dir, side, i + 1, mask)

        cloud = view_pointcloud(rgbs[i], depths[i], Ks_arr[i], ext_arr[i], mask)
        print(f"[{side}] cam{i+1}: -> {cloud.shape[0]} valid points")
        clouds.append(cloud)

    fused = np.concatenate(clouds, axis=0).astype(np.float32) if clouds else np.zeros((0, 6), np.float32)
    if fused.shape[0] == 0:
        raise RuntimeError(f"[{side}] no valid points across all views — refusing to save empty cloud")
    return fused, fg_clicks_world


def warmup_cv2_qt() -> None:
    # cv2's Qt backend grabs the GUI thread on its first window call. If PyTorch
    # / CUDA initialize before that first call, Qt deadlocks on namedWindow
    # (visible as a flood of "QObject::moveToThread ..." warnings followed by
    # an indefinite hang). Creating a throwaway window here forces Qt to wake
    # up before SAM2 / torch initializes.
    cv2.namedWindow("_qt_warmup", cv2.WINDOW_NORMAL)
    cv2.imshow("_qt_warmup", np.zeros((1, 1, 3), dtype=np.uint8))
    cv2.waitKey(1)
    cv2.destroyWindow("_qt_warmup")
    cv2.waitKey(1)


def main():
    warmup_cv2_qt()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("data/images/config.json"))
    ap.add_argument("--camera-params", type=Path, default=None,
                    help="Override camera_params .npz path (defaults to value in config.json or data/images/camera_params.npz)")
    ap.add_argument("--sam2-repo", type=Path, default=DEFAULT_SAM2_REPO)
    ap.add_argument("--sam2-config", type=str, default=DEFAULT_SAM2_CONFIG)
    ap.add_argument("--sam2-ckpt", type=Path, default=DEFAULT_SAM2_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ref-cam", type=int, default=0,
                    help="0-indexed view used for interactive clicks; the other "
                         "views are auto-prompted by re-projecting those clicks "
                         "via the reference view's depth and pose.")
    ap.add_argument("--out-source", type=Path, default=Path("data/source/source.ply"))
    ap.add_argument("--out-target", type=Path, default=Path("data/target/target.ply"))
    ap.add_argument("--out-corr", type=Path, default=Path("data/images/user_correspondence.npy"),
                    help="Where to save the (N, 2, 3) world-frame correspondence "
                         "pairs derived from the user's foreground SAM2 clicks "
                         "(paired source/target by click order).")
    ap.add_argument("--save-masks", action="store_true",
                    help="Also dump per-view PNG masks to data/images/masks/ for debugging")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cam_params_path = (args.camera_params
                       or Path(cfg.get("camera_params", "data/images/camera_params.npz")))
    n_cams = len(cfg["source"]["rgb"])
    if len(cfg["target"]["rgb"]) != n_cams:
        raise ValueError(
            f"source/target view counts differ: {n_cams} vs {len(cfg['target']['rgb'])}"
        )
    Ks, extrinsics = load_camera_params(cam_params_path, n_cams)

    print(f"[build_segmented_pointclouds] loading SAM2 from {args.sam2_ckpt}")
    predictor = build_predictor(args.sam2_repo, args.sam2_config, args.sam2_ckpt, args.device)
    ui = SAM2PromptUI(predictor)

    masks_dir = Path("data/images/masks") if args.save_masks else None

    src, src_clicks = process_side("source", cfg["source"], Ks, extrinsics, ui, masks_dir, args.ref_cam)
    tgt, tgt_clicks = process_side("target", cfg["target"], Ks, extrinsics, ui, masks_dir, args.ref_cam)

    args.out_source.parent.mkdir(parents=True, exist_ok=True)
    args.out_target.parent.mkdir(parents=True, exist_ok=True)
    save_ply(args.out_source, src)
    save_ply(args.out_target, tgt)
    print(f"[done] wrote {args.out_source}  points={src.shape[0]}")
    print(f"[done] wrote {args.out_target}  points={tgt.shape[0]}")

    if src_clicks.shape[0] != tgt_clicks.shape[0]:
        raise RuntimeError(
            f"correspondence count mismatch: source has {src_clicks.shape[0]} "
            f"foreground click(s), target has {tgt_clicks.shape[0]} — click the "
            "same number of foreground points on each reference view so they can "
            "be paired by click order"
        )
    corr_3d = np.stack([src_clicks, tgt_clicks], axis=1).astype(np.float32)
    args.out_corr.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_corr, corr_3d)
    print(f"[done] wrote {args.out_corr}  pairs={corr_3d.shape[0]}")


if __name__ == "__main__":
    main()
