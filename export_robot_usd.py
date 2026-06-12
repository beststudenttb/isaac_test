"""Export the simple ball robot as a standalone USD asset.

Run from the project directory:

    ./IsaacLab/isaaclab.sh -p export_robot_usd.py
"""

import argparse
import math
import os

from isaaclab.app import AppLauncher

ROBOT_USD = "./ball_robot.usd"
EYE_PITCH_DEG = 15.0

parser = argparse.ArgumentParser(description="Export the ball robot USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.livestream = 0
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.sim import schemas
from isaaclab.sim.utils import export_prim_to_file


def build_robot():
    sim_utils.create_prim("/World/Robot", "Xform")
    sim_utils.create_prim("/World/Robot/body", "Xform", translation=(0.0, 0.0, 0.0))
    sim_utils.create_prim("/World/Robot/head", "Xform", translation=(0.0, 0.0, 0.30))
    eye_pitch = math.radians(EYE_PITCH_DEG)
    eye_rot = (math.cos(eye_pitch * 0.5), 0.0, math.sin(eye_pitch * 0.5), 0.0)
    sim_utils.create_prim("/World/Robot/head/eye", "Xform", translation=(0.13, 0.0, 0.0), orientation=eye_rot)
    collision = sim_utils.CollisionPropertiesCfg(collision_enabled=True)

    body = sim_utils.SphereCfg(
        radius=0.23,
        collision_props=collision,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.95)),
    )
    body.func("/World/Robot/body/shape", body, translation=(0.0, 0.0, 0.0))

    head = sim_utils.SphereCfg(
        radius=0.15,
        collision_props=collision,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.75, 0.15)),
    )
    head.func("/World/Robot/head/shape", head, translation=(0.0, 0.0, 0.0))

    eye = sim_utils.CuboidCfg(
        size=(0.08, 0.28, 0.06),
        collision_props=collision,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02, 0.02, 0.02)),
    )
    eye.func("/World/Robot/head/eye/shape", eye, translation=(0.0, 0.0, 0.0))

    schemas.define_rigid_body_properties(
        "/World/Robot",
        sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
    )
    schemas.define_mass_properties("/World/Robot", sim_utils.MassPropertiesCfg(mass=1.0))


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.04, device=args_cli.device))
    build_robot()
    sim.reset()

    out_path = os.path.abspath(ROBOT_USD)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    export_prim_to_file(out_path, "/World/Robot", "/Robot")
    print(f"[INFO] Exported robot USD: {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
