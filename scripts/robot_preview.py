"""Preview a simple code-generated robot in Isaac Lab.

Run from the project directory:

    ./IsaacLab/isaaclab.sh -p scripts/robot_preview.py
"""

import argparse
import math
import time

from isaaclab.app import AppLauncher

TARGET_X = 4.0
ROBOT_SPEED = 1.0
RESET_DISTANCE = 0.3
SIM_DT = 0.04
ROOM_SIZE = 16.0
WALL_HEIGHT = 2.0
WALL_THICKNESS = 0.1
ROBOT_Z = 0.5
FLOAT_AMP = 0.01
FLOAT_PERIOD = 0.5
HEAD_W = math.pi / 4.0
ROBOT_USD = "./assets/robots/ball_robot.usd"

parser = argparse.ArgumentParser(description="Preview the simple ball robot.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext
from pxr import Gf, UsdGeom


def design_scene():
    floor = sim_utils.CuboidCfg(
        size=(ROOM_SIZE, ROOM_SIZE, 0.02),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.82, 0.78)),
    )
    floor.func("/World/floor", floor, translation=(0.0, 0.0, -0.01))

    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light.func("/World/light", light)

    wall_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.65, 0.65, 0.65))
    wall_x = sim_utils.CuboidCfg(size=(WALL_THICKNESS, ROOM_SIZE, WALL_HEIGHT), visual_material=wall_mat)
    wall_y = sim_utils.CuboidCfg(size=(ROOM_SIZE, WALL_THICKNESS, WALL_HEIGHT), visual_material=wall_mat)
    half = ROOM_SIZE * 0.5
    wall_x.func("/World/wall_x_pos", wall_x, translation=(half, 0.0, WALL_HEIGHT * 0.5))
    wall_x.func("/World/wall_x_neg", wall_x, translation=(-half, 0.0, WALL_HEIGHT * 0.5))
    wall_y.func("/World/wall_y_pos", wall_y, translation=(0.0, half, WALL_HEIGHT * 0.5))
    wall_y.func("/World/wall_y_neg", wall_y, translation=(0.0, -half, WALL_HEIGHT * 0.5))

    sim_utils.create_prim("/World/Robot", "Xform", usd_path=ROBOT_USD, translation=(0.0, 0.0, ROBOT_Z))

    target = sim_utils.SphereCfg(
        radius=0.15,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    )
    target.func("/World/target", target, translation=(TARGET_X, 0.0, 0.15))

    camera_cfg = CameraCfg(
        prim_path="/World/Robot/head/eye/front_camera",
        update_period=0.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=40.2768,
            clipping_range=(0.1, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )
    camera = Camera(camera_cfg)

    stage = sim_utils.get_current_stage()
    robot_prim = stage.GetPrimAtPath("/World/Robot")
    robot_xform = UsdGeom.Xformable(robot_prim)
    robot_op = None
    for op in robot_xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            robot_op = op
            break
    if robot_op is None:
        robot_op = robot_xform.AddTranslateOp()

    head_prim = stage.GetPrimAtPath("/World/Robot/head")
    head_xform = UsdGeom.Xformable(head_prim)
    head_rot_op = None
    for op in head_xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
            head_rot_op = op
    if head_rot_op is None:
        head_rot_op = head_xform.AddRotateZOp()
    return camera, robot_op, head_rot_op


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=SIM_DT, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.0, -4.0, 2.6], target=[0.0, 0.0, 0.5])

    camera, robot_op, head_rot_op = design_scene()
    sim.reset()
    camera.reset()
    if not bool(args_cli.headless):
        from omni.kit.viewport.utility import create_viewport_window, get_viewport_from_window_name

        viewport = get_viewport_from_window_name("Viewport")
        viewport.set_active_camera("/OmniverseKit_Persp")
        camera_window = create_viewport_window("Robot Camera", width=640, height=480)
        camera_window.viewport_api.set_active_camera("/World/Robot/head/eye/front_camera")
        print(
            "[INFO] Robot preview ready. Main viewport is normal, Robot Camera viewport uses front_camera. "
            f"target={TARGET_X:.1f}m speed={ROBOT_SPEED:.1f}m/s reset_dist={RESET_DISTANCE:.1f}m."
        )
    else:
        print(
            "[INFO] Robot preview ready in headless mode. "
            f"target={TARGET_X:.1f}m speed={ROBOT_SPEED:.1f}m/s reset_dist={RESET_DISTANCE:.1f}m."
        )

    frames = 0
    last_time = time.perf_counter()
    last_reset_time = last_time
    last_reset_sim_time = 0.0
    reset_dt = 0.0
    reset_sim_dt = 0.0
    sim_time = 0.0
    robot_x = 0.0

    while simulation_app.is_running():
        dt = sim.get_physics_dt()
        sim_time += dt
        robot_x += ROBOT_SPEED * dt
        if TARGET_X - robot_x < RESET_DISTANCE:
            robot_x = 0.0
            now_reset = time.perf_counter()
            reset_dt = now_reset - last_reset_time
            reset_sim_dt = sim_time - last_reset_sim_time
            last_reset_time = now_reset
            last_reset_sim_time = sim_time
            print(f"[INFO] Reset robot. reset_wall={reset_dt:.2f}s reset_sim={reset_sim_dt:.2f}s")
        robot_z = ROBOT_Z + FLOAT_AMP * math.sin(2.0 * math.pi * sim_time / FLOAT_PERIOD)
        robot_op.Set(Gf.Vec3d(robot_x, 0.0, robot_z))
        head_angle = HEAD_W * sim_time
        head_rot_op.Set(math.degrees(head_angle))

        sim.step()
        camera.update(dt)
        frames += 1
        now = time.perf_counter()
        if now - last_time >= 1.0:
            fps = frames / (now - last_time)
            rtf = fps * sim.get_physics_dt()
            print(
                f"[INFO] FPS: {fps:.1f} RTF: {rtf:.2f}x sim_t={sim_time:.2f}s "
                f"x={robot_x:.2f} z={robot_z:.2f} last_reset_wall={reset_dt:.2f}s "
                f"last_reset_sim={reset_sim_dt:.2f}s"
            )
            frames = 0
            last_time = now


if __name__ == "__main__":
    main()
    simulation_app.close()
