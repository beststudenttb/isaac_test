"""Collect Isaac camera images for ball x/dist labels.

Run from the project directory:

    ./IsaacLab/isaaclab.sh -p collect_cv_dataset.py --num-envs 64 --samples 10000
"""

import argparse
import csv
import math
import shutil
import time
from pathlib import Path

from isaaclab.app import AppLauncher

ROBOT_USD = "./ball_robot.usd"
OUT_DIR = "./data_isaac"
SPLITS = ("train", "test", "val")
SPLIT_PROBS = (0.7, 0.2, 0.1)
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
FOCAL_LENGTH = 24.0
FOV_X_DEG = 80.0
ROOM_SIZE = 16.0
WALL_HEIGHT = 2.0
WALL_THICKNESS = 0.1
ROBOT_Z = 0.5
TARGET_RADIUS = 0.23
DIST_MIN = 2.0
DIST_MAX = 7.0
ANGLE_DEG = 45.0
DIST_BIN_EDGES = (1.0, 1.3, 1.7, 2.2, 3.0, 4.2, 5.8, 8.0)

parser = argparse.ArgumentParser(description="Collect CV dataset from Isaac tiled cameras.")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--samples", type=int, default=10000)
parser.add_argument("--spacing", type=float, default=16.0)
parser.add_argument("--out-dir", type=Path, default=Path(OUT_DIR))
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--visible-only", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.rendering_mode = "performance"
args_cli.headless = True
args_cli.livestream = 0

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from torchvision.io import write_png

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationContext
from pxr import Gf, UsdGeom


def horizontal_aperture() -> float:
    return 2.0 * FOCAL_LENGTH * math.tan(math.radians(FOV_X_DEG) * 0.5)


def make_dirs(out_dir: Path):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    dirs = {}
    for split in SPLITS:
        img_dir = out_dir / split / "img"
        label_dir = out_dir / split / "label"
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        dirs[split] = (img_dir, label_dir)
    return dirs


def add_room(env_path: str):
    floor = sim_utils.CuboidCfg(
        size=(ROOM_SIZE, ROOM_SIZE, 0.02),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.82, 0.78)),
    )
    floor.func(f"{env_path}/floor", floor, translation=(0.0, 0.0, -0.01))

    wall_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.65, 0.65, 0.65))
    wall_x = sim_utils.CuboidCfg(size=(WALL_THICKNESS, ROOM_SIZE, WALL_HEIGHT), visual_material=wall_mat)
    wall_y = sim_utils.CuboidCfg(size=(ROOM_SIZE, WALL_THICKNESS, WALL_HEIGHT), visual_material=wall_mat)
    half = ROOM_SIZE * 0.5
    wall_x.func(f"{env_path}/wall_x_pos", wall_x, translation=(half, 0.0, WALL_HEIGHT * 0.5))
    wall_x.func(f"{env_path}/wall_x_neg", wall_x, translation=(-half, 0.0, WALL_HEIGHT * 0.5))
    wall_y.func(f"{env_path}/wall_y_pos", wall_y, translation=(0.0, half, WALL_HEIGHT * 0.5))
    wall_y.func(f"{env_path}/wall_y_neg", wall_y, translation=(0.0, -half, WALL_HEIGHT * 0.5))


def add_env(env_path: str, x: float, y: float):
    sim_utils.create_prim(env_path, "Xform", translation=(x, y, 0.0))
    add_room(env_path)
    sim_utils.create_prim(f"{env_path}/Robot", "Xform", usd_path=ROBOT_USD, translation=(0.0, 0.0, ROBOT_Z))

    target = sim_utils.SphereCfg(
        radius=TARGET_RADIUS,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    )
    target.func(f"{env_path}/target", target, translation=(4.0, 0.0, TARGET_RADIUS))

    stage = sim_utils.get_current_stage()
    target_prim = stage.GetPrimAtPath(f"{env_path}/target")
    target_xform = UsdGeom.Xformable(target_prim)
    for op in target_xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    return target_xform.AddTranslateOp()


def design_scene():
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light.func("/World/light", light)
    sim_utils.create_prim("/World/envs", "Xform")

    num_envs = int(args_cli.num_envs)
    cols = int(math.ceil(math.sqrt(num_envs)))
    spacing = float(args_cli.spacing)
    target_ops = []
    env_origins = []
    for i in range(num_envs):
        row = i // cols
        col = i % cols
        x = (col - (cols - 1) * 0.5) * spacing
        y = (row - (math.ceil(num_envs / cols) - 1) * 0.5) * spacing
        env_origins.append((x, y, 0.0))
        target_ops.append(add_env(f"/World/envs/env_{i}", x, y))

    camera_cfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/head/eye/front_camera",
        update_period=0.0,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=FOCAL_LENGTH,
            focus_distance=400.0,
            horizontal_aperture=horizontal_aperture(),
            clipping_range=(0.1, 100.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )
    return TiledCamera(camera_cfg), target_ops, env_origins


def sample_target(rng: np.random.Generator):
    dist = float(rng.uniform(DIST_MIN, DIST_MAX))
    angle = float(rng.uniform(-math.radians(ANGLE_DEG), math.radians(ANGLE_DEG)))
    forward = dist * math.cos(angle)
    lateral = dist * math.sin(angle)
    return forward, lateral, dist, angle


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def label_from_camera(cam_pos: np.ndarray, cam_quat: np.ndarray, intrinsic: np.ndarray, target_pos: np.ndarray, dist: float):
    rot = quat_to_matrix(cam_quat)
    cam_vec = rot.T @ (target_pos - cam_pos)
    cam_forward = float(cam_vec[0])
    cam_left = float(cam_vec[1])
    fx = float(intrinsic[0, 0])
    cx = float(intrinsic[0, 2])
    px_x = cx - fx * cam_left / cam_forward
    x_norm = (px_x - cx) / cx
    visible = int(cam_forward > 0.0 and 0.0 <= px_x < IMAGE_WIDTH)
    dist_bin = int(sum(dist > edge for edge in DIST_BIN_EDGES))
    if not visible:
        return -IMAGE_WIDTH * 0.5, -1.0, 0.0, 0, 0
    return px_x, x_norm, dist, dist_bin, 1


def save_sample(img_dir: Path, label_dir: Path, index: int, image: torch.Tensor, label: tuple):
    name = f"{index:08d}"
    img_path = img_dir / f"{name}.png"
    label_path = label_dir / f"{name}.txt"
    if img_path.exists() or label_path.exists():
        raise FileExistsError(f"sample already exists: {name}")
    write_png(image.permute(2, 0, 1).cpu(), str(img_path))
    _px_x, x_norm, dist, dist_bin, visible = label
    label_path.write_text(
        f"{x_norm:.6f} {dist:.6f} {dist_bin:d} 0.000000 0.000000 0.000000 {visible:d}\n",
        encoding="utf-8",
    )


def choose_split(rng: np.random.Generator) -> str:
    value = float(rng.random())
    if value < SPLIT_PROBS[0]:
        return "train"
    if value < SPLIT_PROBS[0] + SPLIT_PROBS[1]:
        return "test"
    return "val"


def main():
    dirs = make_dirs(Path(args_cli.out_dir))
    meta_files = {}
    meta_writers = {}
    for split in SPLITS:
        meta_path = Path(args_cli.out_dir) / split / "meta.csv"
        meta_files[split] = meta_path.open("w", newline="", encoding="utf-8")
        meta_writers[split] = csv.DictWriter(
            meta_files[split],
            fieldnames=["index", "env_id", "px_x", "x_norm", "dist", "angle_deg", "target_x", "target_y", "visible"],
        )
        meta_writers[split].writeheader()

    rng = np.random.default_rng(int(args_cli.seed))
    render_cfg = sim_utils.RenderCfg(rendering_mode="performance")
    sim_cfg = sim_utils.SimulationCfg(dt=0.04, device=args_cli.device, render=render_cfg)
    sim = SimulationContext(sim_cfg)
    camera, target_ops, env_origins = design_scene()
    sim.reset()
    camera.reset()
    for _ in range(5):
        sim.step()
        camera.update(sim.get_physics_dt())

    total = int(args_cli.samples)
    num_envs = int(args_cli.num_envs)
    split_index = {split: 0 for split in SPLITS}
    split_count = {split: 0 for split in SPLITS}
    saved = 0
    frames = 0
    last_time = time.perf_counter()
    print(
        f"[INFO] collect start samples={total} envs={num_envs} split_prob=train/test/val={SPLIT_PROBS} "
        f"res={IMAGE_WIDTH}x{IMAGE_HEIGHT} fov_x={FOV_X_DEG:.1f} angle=+/-{ANGLE_DEG:.1f} "
        f"dist={DIST_MIN:.1f}-{DIST_MAX:.1f}m out={args_cli.out_dir} clear_existing=1"
    )

    while simulation_app.is_running() and saved < total:
        batch = []
        for op in target_ops:
            forward, lateral, dist, angle = sample_target(rng)
            op.Set(Gf.Vec3d(forward, lateral, TARGET_RADIUS))
            batch.append((forward, lateral, dist, angle))

        sim.step()
        camera.update(sim.get_physics_dt())
        images = camera.data.output["rgb"]
        cam_pos = camera.data.pos_w.cpu().numpy()
        cam_quat = camera.data.quat_w_world.cpu().numpy()
        intrinsics = camera.data.intrinsic_matrices.cpu().numpy()
        count = min(num_envs, total - saved)
        for env_id in range(count):
            forward, lateral, dist, angle = batch[env_id]
            env_x, env_y, env_z = env_origins[env_id]
            target_pos = np.array([env_x + forward, env_y + lateral, env_z + TARGET_RADIUS], dtype=np.float64)
            label = label_from_camera(cam_pos[env_id], cam_quat[env_id], intrinsics[env_id], target_pos, dist)
            if bool(args_cli.visible_only) and not label[4]:
                continue
            split = choose_split(rng)
            img_dir, label_dir = dirs[split]
            sample_index = split_index[split]
            save_sample(img_dir, label_dir, sample_index, images[env_id], label)
            px_x, x_norm, label_dist, _dist_bin, visible = label
            meta_writers[split].writerow(
                {
                    "index": sample_index,
                    "env_id": env_id,
                    "px_x": f"{px_x:.6f}",
                    "x_norm": f"{x_norm:.6f}",
                    "dist": f"{dist:.6f}",
                    "angle_deg": f"{math.degrees(angle):.6f}",
                    "target_x": f"{forward:.6f}",
                    "target_y": f"{lateral:.6f}",
                    "visible": visible,
                }
            )
            split_index[split] += 1
            split_count[split] += 1
            saved += 1

        frames += 1
        now = time.perf_counter()
        if now - last_time >= 1.0:
            print(
                f"[INFO] saved={saved}/{total} train={split_count['train']} test={split_count['test']} "
                f"val={split_count['val']} render_fps={frames / (now - last_time):.1f}"
            )
            frames = 0
            last_time = now

    for meta_file in meta_files.values():
        meta_file.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
