"""Remove physics APIs from the robot USD.

Run from the project root:

    ./IsaacLab/isaaclab.sh -p scripts/strip_robot_usd_physics.py
"""

from __future__ import annotations

from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import PhysxSchema, Usd, UsdPhysics


USD_PATH = Path("assets/robots/ball_robot.usd")


def main():
    stage = Usd.Stage.Open(str(USD_PATH))
    if stage is None:
        raise RuntimeError(f"cannot open {USD_PATH}")

    api_types = (
        UsdPhysics.RigidBodyAPI,
        UsdPhysics.MassAPI,
        UsdPhysics.CollisionAPI,
        UsdPhysics.MeshCollisionAPI,
        PhysxSchema.PhysxRigidBodyAPI,
        PhysxSchema.PhysxCollisionAPI,
    )

    removed = 0
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        for api_type in api_types:
            if prim.HasAPI(api_type):
                prim.RemoveAPI(api_type)
                removed += 1

    stage.GetRootLayer().Save()
    Path("/tmp/strip_robot_usd_physics_result.txt").write_text(
        f"removed {removed} physics APIs from {USD_PATH}\n", encoding="utf-8"
    )
    print(f"[INFO] removed {removed} physics APIs from {USD_PATH}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
