"""Benchmark pretrained classifier forward speed on multi-env camera images.

Run from the project directory:

    ./IsaacLab/isaaclab.sh -p scripts/benchmark_classifier_multi.py --num-envs 64 --model both --duration 20
    ./IsaacLab/isaaclab.sh -p scripts/benchmark_classifier_multi.py --num-envs 64 --model mobile --duration 20
    ./IsaacLab/isaaclab.sh -p scripts/benchmark_classifier_multi.py --num-envs 64 --model ours --duration 20
"""

import argparse
import math
import os
import sys
import time
import traceback

from isaaclab.app import AppLauncher

ROBOT_USD = "./assets/robots/ball_robot.usd"
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
FOCAL_LENGTH = 24.0
FOV_X_DEG = 80.0
ROOM_SIZE = 16.0
WALL_HEIGHT = 2.0
WALL_THICKNESS = 0.1
ROBOT_Z = 0.5
TARGET_RADIUS = 0.23

parser = argparse.ArgumentParser(description="Benchmark classifier forward speed with Isaac tiled cameras.")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--spacing", type=float, default=16.0)
parser.add_argument("--duration", type=float, default=20.0)
parser.add_argument(
    "--model",
    choices=["resnet18", "mobile", "mobile_large", "mobilenet_v3_small", "mobilenet_v3_large", "ours", "both"],
    default="both",
)
parser.add_argument("--weights", choices=["default", "none"], default="default")
parser.add_argument("--weights-path", type=str, default="")
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--graceful-close", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.rendering_mode = "quality"
args_cli.headless = True
args_cli.livestream = 0

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from torchvision import models

sys.path.insert(0, "./src")
from cv_extractor.simple_cnn import BallVisionNet

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationContext


def horizontal_aperture() -> float:
    return 2.0 * FOCAL_LENGTH * math.tan(math.radians(FOV_X_DEG) * 0.5)


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


def design_scene():
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light.func("/World/light", light)
    sim_utils.create_prim("/World/envs", "Xform")

    num_envs = int(args_cli.num_envs)
    cols = int(math.ceil(math.sqrt(num_envs)))
    rows = int(math.ceil(num_envs / cols))
    spacing = float(args_cli.spacing)
    for i in range(num_envs):
        row = i // cols
        col = i % cols
        x = (col - (cols - 1) * 0.5) * spacing
        y = (row - (rows - 1) * 0.5) * spacing
        add_env(f"/World/envs/env_{i}", x, y)

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
    return TiledCamera(camera_cfg)


def load_state_dict(model: torch.nn.Module, path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)


def model_key(name: str) -> str:
    if name == "mobilenet_v3_small":
        return "mobile"
    if name == "mobilenet_v3_large":
        return "mobile_large"
    return name


def make_model(name: str, device: torch.device):
    name = model_key(name)
    weights = None
    kind = "classifier"
    if name == "resnet18":
        if args_cli.weights == "default":
            weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    elif name == "mobile":
        if args_cli.weights == "default":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
    elif name == "mobile_large":
        if args_cli.weights == "default":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)
    elif name == "ours":
        model = BallVisionNet(pretrained=False)
        kind = "ours"
    else:
        raise ValueError(f"unknown model: {name}")

    if args_cli.weights_path and kind == "classifier":
        load_state_dict(model, args_cli.weights_path, device)
    model.eval().to(device)
    return model, kind


def preprocess(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = images.to(device)
    if x.shape[-1] == 4:
        x = x[..., :3]
    x = x.permute(0, 3, 1, 2).contiguous().float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def sync_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    device = torch.device(args_cli.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model_names = ["resnet18", "mobile", "ours"] if args_cli.model == "both" else [model_key(args_cli.model)]
    loaded_models = []
    for name in model_names:
        model, kind = make_model(name, device)
        loaded_models.append((name, model, kind))

    render_cfg = sim_utils.RenderCfg(rendering_mode="quality")
    sim_cfg = sim_utils.SimulationCfg(dt=0.04, device=args_cli.device, render=render_cfg)
    sim = SimulationContext(sim_cfg)
    camera = design_scene()
    sim.reset()
    camera.reset()

    for _ in range(5):
        sim.step()
        camera.update(sim.get_physics_dt())

    print(
        f"[INFO] benchmark ready envs={int(args_cli.num_envs)} res={IMAGE_WIDTH}x{IMAGE_HEIGHT} "
        f"models={','.join(model_names)} weights={args_cli.weights} device={device} rendering=quality"
    )

    step_count = 0
    total_steps = 0
    image_count = 0
    forward_time = {name: 0.0 for name in model_names}
    last_print = time.perf_counter()
    start = last_print

    while simulation_app.is_running():
        sim.step()
        camera.update(sim.get_physics_dt())
        images = camera.data.output["rgb"]
        classifier_batch = None

        for name, model, kind in loaded_models:
            with torch.inference_mode():
                if kind == "ours":
                    model_input = images
                else:
                    if classifier_batch is None:
                        classifier_batch = preprocess(images, device)
                    model_input = classifier_batch
                sync_if_needed(device)
                start_forward = time.perf_counter()
                model(model_input)
                sync_if_needed(device)
                forward_time[name] += time.perf_counter() - start_forward

        now = time.perf_counter()
        total_steps += 1
        if total_steps <= int(args_cli.warmup):
            if total_steps == int(args_cli.warmup):
                step_count = 0
                image_count = 0
                forward_time = {name: 0.0 for name in model_names}
                last_print = now
                print(f"[INFO] warmup done steps={int(args_cli.warmup)}", flush=True)
            continue

        step_count += 1
        image_count += int(images.shape[0])
        if step_count >= int(args_cli.warmup) and now - last_print >= 1.0:
            wall_dt = now - last_print
            total_img_fps = image_count / wall_dt
            loop_fps = step_count / wall_dt
            parts = [f"sim_fps={loop_fps:.1f}", f"img_fps={total_img_fps:.0f}"]
            for name in model_names:
                model_fps = image_count / forward_time[name] if forward_time[name] > 0.0 else 0.0
                parts.append(f"{name}_fps={model_fps:.0f}")
            print(" ".join(parts), flush=True)
            step_count = 0
            image_count = 0
            forward_time = {name: 0.0 for name in model_names}
            last_print = now

        if float(args_cli.duration) > 0.0 and now - start >= float(args_cli.duration):
            break


def close_and_exit(exit_code: int):
    sys.stdout.flush()
    sys.stderr.flush()
    if bool(args_cli.graceful_close):
        simulation_app.close()
    os._exit(exit_code)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        exit_code = 130
        print("[INFO] benchmark interrupted", flush=True)
    except Exception:
        exit_code = 1
        traceback.print_exc()
    finally:
        close_and_exit(exit_code)
