"""Preview many simple robot environments in one Isaac Lab process.

Run from the project directory:

    ./IsaacLab/isaaclab.sh -p scripts/robot_multi_preview.py --num-envs 64
"""

import argparse
import math
import time

from isaaclab.app import AppLauncher

TARGET_X = 4.0
ROBOT_SPEED = 1.0
RESET_DISTANCE = 0.3
SIM_DT = 0.01
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

parser = argparse.ArgumentParser(description="Preview many simple ball robot environments.")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--spacing", type=float, default=16.0)
parser.add_argument("--no-cameras", action="store_true")
parser.add_argument("--camera-count", type=int, default=None)
parser.add_argument("--camera-every", type=int, default=1)
parser.add_argument("--duration", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = not bool(args_cli.no_cameras) and (
    args_cli.camera_count is None or int(args_cli.camera_count) > 0
)
args_cli.rendering_mode = "quality"
args_cli.headless = True
args_cli.livestream = 0

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationContext
import omni.usd
from pxr import Gf, UsdGeom


def spawn_robot(env_path: str, x: float, y: float):
    sim_utils.create_prim(env_path, "Xform", translation=(x, y, 0.0))
    sim_utils.create_prim(f"{env_path}/Robot", "Xform", translation=(0.0, 0.0, 0.0))

    body = sim_utils.SphereCfg(
        radius=0.35,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.95)),
    )
    body.func(f"{env_path}/Robot/body", body, translation=(0.0, 0.0, 0.35))

    head = sim_utils.SphereCfg(
        radius=0.16,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.75, 0.15)),
    )
    head.func(f"{env_path}/Robot/head", head, translation=(0.0, 0.0, 0.82))

    camera_body = sim_utils.CuboidCfg(
        size=(0.18, 0.08, 0.08),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02, 0.02, 0.02)),
    )
    camera_body.func(f"{env_path}/Robot/camera_body", camera_body, translation=(0.18, 0.0, 0.85))

    lens = sim_utils.SphereCfg(
        radius=0.035,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.08)),
    )
    lens.func(f"{env_path}/Robot/lens", lens, translation=(0.28, 0.0, 0.85))

    target = sim_utils.SphereCfg(
        radius=0.18,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    )
    target.func(f"{env_path}/target", target, translation=(TARGET_X, 0.0, 0.18))

    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath(f"{env_path}/Robot")
    robot_xform = UsdGeom.Xformable(robot_prim)
    for op in robot_xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    return robot_xform.AddTranslateOp()


def design_scene():
    num_envs = int(args_cli.num_envs)
    cols = int(math.ceil(math.sqrt(num_envs)))
    rows = int(math.ceil(num_envs / cols))
    spacing = float(args_cli.spacing)
    ground_size = max(cols, rows) * spacing

    ground = sim_utils.GroundPlaneCfg(size=(ground_size, ground_size))
    ground.func("/World/ground", ground)

    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light.func("/World/light", light)

    sim_utils.create_prim("/World/envs", "Xform")
    robot_ops = []

    for i in range(num_envs):
        row = i // cols
        col = i % cols
        x = (col - (cols - 1) * 0.5) * spacing
        y = (row - (rows - 1) * 0.5) * spacing
        robot_ops.append(spawn_robot(f"/World/envs/env_{i}", x, y))

    if bool(args_cli.no_cameras):
        return None, robot_ops

    camera_count = int(args_cli.num_envs) if args_cli.camera_count is None else int(args_cli.camera_count)
    camera_count = max(0, min(camera_count, int(args_cli.num_envs)))
    if camera_count <= 0:
        return None, robot_ops
    if camera_count == int(args_cli.num_envs):
        camera_path = "/World/envs/env_.*/Robot/front_camera"
    elif camera_count == 1:
        camera_path = "/World/envs/env_0/Robot/front_camera"
    else:
        env_ids = "|".join(str(i) for i in range(camera_count))
        camera_path = f"/World/envs/env_({env_ids})/Robot/front_camera"

    cameras_cfg = TiledCameraCfg(
        prim_path=camera_path,
        update_period=0.0,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.30, 0.0, 0.85),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )
    return TiledCamera(cameras_cfg), robot_ops


def compute_state():
    image_width = float(IMAGE_WIDTH)
    fov_x = math.radians(70.0)
    fx = image_width / (2.0 * math.tan(0.5 * fov_x))

    target_x = TARGET_X
    target_y = 0.0
    camera_x = 0.30
    camera_y = 0.0

    forward = target_x - camera_x
    lateral = target_y - camera_y
    px_x = image_width * 0.5 + fx * lateral / forward
    x_norm = (px_x - image_width * 0.5) / (image_width * 0.5)
    d = forward
    return x_norm, d


def main():
    render_cfg = sim_utils.RenderCfg(rendering_mode="quality")
    sim_cfg = sim_utils.SimulationCfg(dt=SIM_DT, device=args_cli.device, render=render_cfg)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[8.0, -8.0, 6.0], target=[0.0, 0.0, 0.5])

    scene = design_scene()
    camera, robot_ops = scene
    robot_x = [0.0 for _ in robot_ops]
    sim.reset()
    if camera is not None:
        camera.reset()

    if bool(args_cli.no_cameras):
        camera_count = 0
    elif args_cli.camera_count is None:
        camera_count = int(args_cli.num_envs)
    else:
        camera_count = max(0, min(int(args_cli.camera_count), int(args_cli.num_envs)))
    camera_every = max(1, int(args_cli.camera_every))
    cols = int(math.ceil(math.sqrt(int(args_cli.num_envs))))
    rows = int(math.ceil(int(args_cli.num_envs) / cols))
    ground_size = max(cols, rows) * float(args_cli.spacing)
    print(
        f"[INFO] Multi preview ready. num_envs={int(args_cli.num_envs)}, "
        f"grid={cols}x{rows}, spacing={float(args_cli.spacing):.1f}, "
        f"ground={ground_size:.1f}m, target={TARGET_X:.1f}m, speed={ROBOT_SPEED:.1f}m/s, "
        f"reset_dist={RESET_DISTANCE:.1f}m, cameras={camera_count}, camera_every={camera_every}, "
        f"sensor=tiled_camera, rendering_mode=quality, viewport=off"
    )

    frames = 0
    image_count = 0
    last_time = time.perf_counter()
    start_time = last_time
    last_reset_time = last_time
    last_reset_sim_time = 0.0
    reset_count = 0
    reset_dt = 0.0
    reset_sim_dt = 0.0
    sim_time = 0.0
    x_norm, d = compute_state()

    while simulation_app.is_running():
        dt = sim.get_physics_dt()
        sim_time += dt
        step_resets = 0
        for i, op in enumerate(robot_ops):
            robot_x[i] += ROBOT_SPEED * dt
            if TARGET_X - robot_x[i] < RESET_DISTANCE:
                robot_x[i] = 0.0
                reset_count += 1
                step_resets += 1
            op.Set(Gf.Vec3d(robot_x[i], 0.0, 0.0))
        if step_resets > 0:
            reset_now = time.perf_counter()
            reset_dt = reset_now - last_reset_time
            reset_sim_dt = sim_time - last_reset_sim_time
            last_reset_time = reset_now
            last_reset_sim_time = sim_time

        sim.step()
        if camera is not None and frames % camera_every == 0:
            camera.update(dt)
            images = camera.data.output["rgb"]
            image_count += int(images.shape[0])
        else:
            _obs = (x_norm, d)
        frames += 1
        now = time.perf_counter()
        if now - last_time >= 1.0:
            fps = frames / (now - last_time)
            rtf = fps * sim.get_physics_dt()
            img_fps = image_count / (now - last_time)
            print(f"sim_fps={fps:.1f} img_fps={img_fps:.0f} rtf={rtf:.2f}")
            reset_count = 0
            frames = 0
            image_count = 0
            last_time = now
        if float(args_cli.duration) > 0.0 and now - start_time >= float(args_cli.duration):
            break


if __name__ == "__main__":
    main()
    simulation_app.close()
