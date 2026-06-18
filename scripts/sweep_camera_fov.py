"""Randomly flash target positions under render settings and save one video.

Run from the project root:

    python scripts/sweep_camera_fov.py

The default mode is a controller: it launches Isaac once per render config,
saves camera frames, then combines them into one labeled video. The internal
``--single`` mode is used by the controller and should normally not be called
directly.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
RENDER_MODES = ("performance", "balanced", "quality")
AA_MODES = ("Off", "FXAA", "TAA", "DLAA", "DLSS_PERF", "DLSS_BALANCED", "DLSS_QUALITY", "DLSS_AUTO")
DLSS_MODE_MAP = {
    "DLSS_PERF": 0,
    "DLSS_BALANCED": 1,
    "DLSS_QUALITY": 2,
    "DLSS_AUTO": 3,
}


def is_single_mode() -> bool:
    return "--single" in sys.argv


def build_parser(single: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Randomly flash a target ball across camera FOV.")
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=Path("camera_sweep_all.mp4"))
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--frame-dir", type=Path, default=None)
    parser.add_argument("--distance", type=float, default=4.0)
    parser.add_argument("--start-deg", type=float, default=-30.0)
    parser.add_argument("--end-deg", type=float, default=30.0)
    parser.add_argument("--step-deg", type=float, default=0.2)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--height-min", type=float, default=0.0)
    parser.add_argument("--height-max", type=float, default=0.3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--rendering-modes", nargs="+", choices=RENDER_MODES, default=["balanced"])
    parser.add_argument(
        "--aa-modes",
        nargs="+",
        choices=AA_MODES,
        default=["Off", "DLSS_PERF", "DLSS_BALANCED", "DLSS_QUALITY", "DLSS_AUTO"],
    )
    parser.add_argument("--render-mode", choices=RENDER_MODES, default="quality")
    parser.add_argument("--aa", choices=["Off", "FXAA", "DLSS", "TAA", "DLAA"], default=None)
    parser.add_argument("--dlss-mode", type=int, choices=[0, 1, 2, 3], default=None)
    parser.add_argument("--spp", type=int, default=None)
    parser.add_argument("--denoiser", action="store_true")
    parser.add_argument("--no-shadows", action="store_true")
    parser.add_argument("--no-reflections", action="store_true")
    parser.add_argument("--no-gi", action="store_true")
    parser.add_argument("--no-ao", action="store_true")
    parser.add_argument("--no-translucency", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    if single:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    return parser


def frame_count(args) -> int:
    count = int(args.frames)
    if count <= 0:
        raise ValueError("--frames must be positive")
    return count


def aa_to_args(name: str) -> tuple[str, int | None]:
    if name in DLSS_MODE_MAP:
        return "DLSS", DLSS_MODE_MAP[name]
    return name, None


def config_name(render_mode: str, aa_name: str) -> str:
    return f"{render_mode}_{aa_name}".replace("/", "_")


def labeled_frame(path: Path, label: str):
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    pad = 6
    bbox = draw.textbbox((0, 0), label)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x0 = image.width - width - pad * 2
    y0 = 0
    draw.rectangle((x0, y0, image.width, height + pad * 2), fill=(0, 0, 0))
    draw.text((x0 + pad, y0 + pad), label, fill=(255, 255, 255))
    return np.asarray(image)


def ensure_empty_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"frame dir is not empty: {path}")


def run_controller(args, unknown: list[str]):
    run_dir = args.run_dir
    if run_dir is None:
        stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        run_dir = Path("camera_sweep_runs") / stamp
    run_dir.mkdir(parents=True, exist_ok=False)

    configs = [(render, aa) for render in args.rendering_modes for aa in args.aa_modes]
    print(f"[INFO] configs={len(configs)} frames_per_config={frame_count(args)} out={args.out}")

    for render, aa_name in configs:
        aa, dlss_mode = aa_to_args(aa_name)
        frame_dir = run_dir / config_name(render, aa_name)
        ensure_empty_dir(frame_dir)
        cmd = [
            str(PROJECT_ROOT / "IsaacLab" / "isaaclab.sh"),
            "-p",
            str(SCRIPT_PATH),
            "--single",
            "--frame-dir",
            str(frame_dir),
            "--distance",
            str(args.distance),
            "--start-deg",
            str(args.start_deg),
            "--end-deg",
            str(args.end_deg),
            "--step-deg",
            str(args.step_deg),
            "--frames",
            str(args.frames),
            "--height-min",
            str(args.height_min),
            "--height-max",
            str(args.height_max),
            "--fps",
            str(args.fps),
            "--render-mode",
            render,
            "--aa",
            aa,
            "--seed",
            str(args.seed),
        ]
        if dlss_mode is not None:
            cmd.extend(["--dlss-mode", str(dlss_mode)])
        if args.spp is not None:
            cmd.extend(["--spp", str(args.spp)])
        for flag in ("denoiser", "no_shadows", "no_reflections", "no_gi", "no_ao", "no_translucency"):
            if bool(getattr(args, flag)):
                cmd.append("--" + flag.replace("_", "-"))
        cmd.extend(unknown)
        print(f"[INFO] run {render} {aa_name}")
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=int(args.fps))
    try:
        for render, aa_name in configs:
            frame_dir = run_dir / config_name(render, aa_name)
            label = f"{render} / {aa_name}"
            for frame_path in sorted(frame_dir.glob("*.png")):
                writer.append_data(labeled_frame(frame_path, label))
    finally:
        writer.close()
    print(f"[INFO] saved video: {args.out}")
    print(f"[INFO] frames kept in: {run_dir}")


def rgb_to_numpy(image):
    import torch

    image = image.detach()
    if image.shape[-1] > 3:
        image = image[..., :3]
    if image.dtype != torch.uint8:
        image = torch.clamp(image, 0.0, 255.0)
        if image.max() <= 1.0:
            image = image * 255.0
        image = image.to(torch.uint8)
    return image.cpu().numpy()


def run_single(args):
    if args.frame_dir is None:
        raise ValueError("--single requires --frame-dir")

    import torch
    from isaaclab.app import AppLauncher

    args.enable_cameras = True
    args.headless = True
    args.livestream = 0
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import isaaclab.sim as sim_utils

        sys.path.insert(0, str(PROJECT_ROOT))
        from src.env import BallEnv, BallEnvCfg

        cfg = BallEnvCfg()
        cfg.seed = int(args.seed)
        cfg.episode_length_s = 3600.0
        cfg.scene.num_envs = 1
        cfg.scene.env_spacing = 16.0
        cfg.sim.device = args.device
        cfg.sim.render = sim_utils.RenderCfg(
            rendering_mode=str(args.render_mode),
            antialiasing_mode=args.aa,
            enable_dlssg=False,
            dlss_mode=args.dlss_mode,
            samples_per_pixel=args.spp,
            enable_dl_denoiser=True if bool(args.denoiser) else None,
            enable_shadows=False if bool(args.no_shadows) else None,
            enable_reflections=False if bool(args.no_reflections) else None,
            enable_global_illumination=False if bool(args.no_gi) else None,
            enable_ambient_occlusion=False if bool(args.no_ao) else None,
            enable_translucency=False if bool(args.no_translucency) else None,
        )
        cfg.use_camera = True

        env = BallEnv(cfg)
        env.reset()
        actions = torch.zeros((1, int(env.cfg.action_space)), device=env.device)
        frame_dir = args.frame_dir
        frame_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(int(args.seed))
        count = frame_count(args)
        start_time = time.perf_counter()
        try:
            for index in range(count):
                angle_deg = float(rng.uniform(float(args.start_deg), float(args.end_deg)))
                height = float(rng.uniform(float(args.height_min), float(args.height_max)))
                angle = math.radians(float(angle_deg))
                dist = float(args.distance)
                env.target_xy[0, 0] = dist * math.cos(angle)
                env.target_xy[0, 1] = dist * math.sin(angle)
                env.write_target_pose(torch.tensor([0], device=env.device, dtype=torch.long))
                target_pos = env.target_view.get_world_poses()[0][0]
                target_pos[2] += height
                env.target_view.set_world_poses(
                    positions=target_pos.unsqueeze(0),
                    indices=[0],
                )
                env.step(actions)
                frame = rgb_to_numpy(env.camera.data.output["rgb"][0])
                imageio.imwrite(frame_dir / f"{index:04d}_a{angle_deg:+06.2f}_h{height:.3f}.png", frame)
                if index % 50 == 0:
                    label = env.project_target()
                    print(
                        f"frame={index:04d}/{count} angle={angle_deg:+.2f} height={height:.3f} "
                        f"px={float(label['px_x'][0].item()):.2f} dist={float(label['dist'][0].item()):.2f}"
                    )
        finally:
            env.close()
        elapsed = time.perf_counter() - start_time
        print(
            f"[INFO] saved {count} frames to {frame_dir} "
            f"render={args.render_mode} aa={args.aa} dlss={args.dlss_mode} time={elapsed:.2f}s"
        )
    finally:
        simulation_app.close()


def main():
    single = is_single_mode()
    parser = build_parser(single)
    if single:
        args = parser.parse_args()
        run_single(args)
    else:
        args, unknown = parser.parse_known_args()
        run_controller(args, unknown)


if __name__ == "__main__":
    main()
