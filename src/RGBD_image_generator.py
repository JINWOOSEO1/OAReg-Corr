import mujoco
import numpy as np
import cv2
from pathlib import Path
import argparse

def main(args=None):
    if args.is_target:
        model = mujoco.MjModel.from_xml_path("data/images/target.xml")
    else:
        model = mujoco.MjModel.from_xml_path("data/images/source.xml")
    data = mujoco.MjData(model)

    W, H = 640, 480
    renderer = mujoco.Renderer(model, width=W, height=H)
    cam_list = ['cam1', 'cam2', 'cam3']

    mujoco.mj_forward(model, data)
    cam_params = {}
    for cam_name in cam_list:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)

        fovy = model.cam_fovy[cam_id]
        fy = 0.5 * H / np.tan(np.deg2rad(fovy) / 2)
        fx = fy
        cx, cy = W / 2.0, H / 2.0
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)

        T_wc = np.eye(4, dtype=np.float64)
        T_wc[:3, :3] = data.cam_xmat[cam_id].reshape(3, 3)
        T_wc[:3, 3]  = data.cam_xpos[cam_id]

        cam_params[f"{cam_name}_intrinsic"] = K
        cam_params[f"{cam_name}_extrinsic"] = T_wc

    np.savez("data/images/camera_params.npz", **cam_params)

    mujoco.mj_step(model, data)

    for cam_name in cam_list:
        renderer.update_scene(data, camera=cam_name)
        rgb_image = renderer.render()

        renderer.enable_depth_rendering()
        depth_image = renderer.render()
        renderer.disable_depth_rendering()

        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        if args.is_target:
            cv2.imwrite(f"data/images/target_{cam_name}.jpg", rgb_image)
            np.save(f"data/images/target_{cam_name}.npy", depth_image)
        else:
            cv2.imwrite(f"data/images/source_{cam_name}.jpg", rgb_image)
            np.save(f"data/images/source_{cam_name}.npy", depth_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RGB-D images from MuJoCo simulation.")
    parser.add_argument("--is-target", action="store_true",
                        help="Whether to generate target images (default: source images).")
    args = parser.parse_args()
    main(args)