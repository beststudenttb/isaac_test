"""Train MDP latent state model from teacher image transitions.

Run from the project root:

    python train/mdp_state.py --data-dir ./data_mdp_teacher_50k
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import mdp_state_cfg as cfg
from src.mdp_state import MDPStateNet


IMAGE_WIDTH = 224.0
FAIL_FAR = 8.0


class MDPDataset(Dataset):
    def __init__(self, data_dir: Path, seq_len: int):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "img"
        self.seq_len = int(seq_len)
        rows = self._read_rows(self.data_dir / "transitions.csv")
        groups = defaultdict(list)
        for row in rows:
            groups[(row["env_id"], row["episode"])].append(row)
        self.trajs = []
        for traj in groups.values():
            traj.sort(key=lambda row: row["t"])
            if len(traj) >= self.seq_len:
                self.trajs.append(traj)
        if not self.trajs:
            raise RuntimeError(f"no trajectory with seq_len={self.seq_len}: {self.data_dir}")
        self.starts = []
        for traj_id, traj in enumerate(self.trajs):
            for start in range(0, len(traj) - self.seq_len + 1):
                self.starts.append((traj_id, start))

    def _read_rows(self, path: Path) -> list[dict]:
        rows = []
        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                rows.append(
                    {
                        "env_id": int(row["env_id"]),
                        "episode": int(row["episode"]),
                        "t": int(row["t"]),
                        "img": row["img"],
                        "next_img": row["next_img"],
                        "a_x": float(row["a_x"]),
                        "a_y": float(row["a_y"]),
                        "a_w": float(row["a_w"]),
                        "reward": float(row["reward"]),
                        "done": float(row["done"]),
                        "px_x": float(row["px_x"]),
                        "dist": float(row["dist"]),
                    }
                )
        return rows

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        traj_id, start = self.starts[index]
        rows = self.trajs[traj_id][start : start + self.seq_len]
        images = []
        actions = []
        rewards = []
        dones = []
        probes = []
        for row in rows:
            images.append(read_image(str(self.img_dir / row["img"])))
            actions.append([row["a_x"], row["a_y"], row["a_w"]])
            rewards.append([row["reward"]])
            dones.append([row["done"]])
            probes.append([self.norm_x(row["px_x"], row["dist"]), self.norm_d(row["dist"])])
        images.append(read_image(str(self.img_dir / rows[-1]["next_img"])))
        return {
            "image": torch.stack(images, dim=0),
            "action": torch.tensor(actions, dtype=torch.float32),
            "reward": torch.tensor(rewards, dtype=torch.float32),
            "done": torch.tensor(dones, dtype=torch.float32),
            "probe": torch.tensor(probes, dtype=torch.float32),
        }

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        items = [self[random.randrange(len(self))] for _ in range(int(batch_size))]
        return {key: torch.stack([item[key] for item in items], dim=0) for key in items[0]}

    def norm_x(self, px_x: float, dist: float) -> float:
        if dist <= 0.0:
            return -1.0
        return max(-1.0, min(1.0, px_x / (IMAGE_WIDTH * 0.5) - 1.0))

    def norm_d(self, dist: float) -> float:
        if dist <= 0.0:
            return 0.0
        return max(0.0, min(1.0, dist / FAIL_FAR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MDP latent state model.")
    parser.add_argument("--data-dir", type=Path, default=Path(cfg.TRAIN_CFG["dataset_dir"]))
    parser.add_argument("--out-dir", type=Path, default=Path(cfg.TRAIN_CFG["output_dir"]))
    parser.add_argument("--updates", type=int, default=int(cfg.TRAIN_CFG["updates"]))
    parser.add_argument("--batch-size", type=int, default=int(cfg.TRAIN_CFG["batch_size"]))
    parser.add_argument("--seq-len", type=int, default=int(cfg.TRAIN_CFG["seq_len"]))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def returns(reward: torch.Tensor, done: torch.Tensor, gamma: float) -> torch.Tensor:
    out = torch.zeros_like(reward)
    g = torch.zeros_like(reward[:, 0])
    for t in reversed(range(reward.shape[1])):
        g = reward[:, t] + float(gamma) * g * (1.0 - done[:, t])
        out[:, t] = g
    return out


def var_loss(z: torch.Tensor) -> torch.Tensor:
    std = torch.sqrt(z.float().var(dim=0) + 1e-4)
    return F.relu(1.0 - std).mean()


def train_step(model: MDPStateNet, batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    image = batch["image"].to(device)
    action = batch["action"].to(device)
    reward = batch["reward"].to(device)
    done = batch["done"].to(device)
    probe = batch["probe"].to(device)
    ret = returns(reward, done, float(cfg.TRAIN_CFG["gamma"]))

    batch_size, seq_len = action.shape[:2]
    state = model.initial(batch_size, device)
    prev_action = torch.zeros(batch_size, int(cfg.MODEL_CFG["action_dim"]), device=device)
    obs = model.observe(image[:, 0], prev_action, state)

    dyn_losses = []
    reward_losses = []
    done_losses = []
    value_losses = []
    probe_losses = []
    probe_x_errs = []
    probe_d_errs = []
    z_list = []

    for t in range(seq_len):
        pred = model.imagine(obs, action[:, t])
        next_post = model.observe(image[:, t + 1], action[:, t], obs)

        dyn_losses.append(F.smooth_l1_loss(pred["next_z_pred"], next_post["z"].detach()))
        reward_losses.append(F.smooth_l1_loss(pred["reward_pred"], reward[:, t]))
        done_losses.append(F.binary_cross_entropy_with_logits(pred["done_logit"], done[:, t]))
        value_losses.append(F.smooth_l1_loss(obs["value"], ret[:, t]))
        probe_losses.append(F.smooth_l1_loss(obs["probe"], probe[:, t]))
        probe_x_errs.append(((obs["probe"][:, 0:1] - probe[:, t, 0:1]).abs() * (IMAGE_WIDTH * 0.5)).mean())
        probe_d_errs.append(((obs["probe"][:, 1:2] - probe[:, t, 1:2]).abs() * FAIL_FAR).mean())
        z_list.append(obs["z"])

        keep = 1.0 - done[:, t]
        obs = {
            "belief": next_post["belief"] * keep,
            "z": next_post["z"] * keep,
            "state_feature": next_post["state_feature"] * keep,
            "value": next_post["value"] * keep,
            "probe": next_post["probe"] * keep,
        }

    z_all = torch.cat(z_list, dim=0)
    dyn = torch.stack(dyn_losses).mean()
    rew = torch.stack(reward_losses).mean()
    done_loss = torch.stack(done_losses).mean()
    val = torch.stack(value_losses).mean()
    probe_loss = torch.stack(probe_losses).mean()
    var = var_loss(z_all)
    loss = (
        float(cfg.TRAIN_CFG["dyn_w"]) * dyn
        + float(cfg.TRAIN_CFG["reward_w"]) * rew
        + float(cfg.TRAIN_CFG["done_w"]) * done_loss
        + float(cfg.TRAIN_CFG["value_w"]) * val
        + float(cfg.TRAIN_CFG["probe_w"]) * probe_loss
        + float(cfg.TRAIN_CFG["var_w"]) * var
    )
    return {
        "loss": loss,
        "dyn": dyn.detach(),
        "reward": rew.detach(),
        "done": done_loss.detach(),
        "value": val.detach(),
        "probe": probe_loss.detach(),
        "probe_x_err": torch.stack(probe_x_errs).mean().detach(),
        "probe_d_err": torch.stack(probe_d_errs).mean().detach(),
        "var": var.detach(),
        "z_std": z_all.detach().float().std(dim=0).mean(),
    }


def save(path: Path, model: MDPStateNet, update: int, info: dict[str, float]) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "model_cfg": cfg.MODEL_CFG,
            "train_cfg": cfg.TRAIN_CFG,
            "update": int(update),
            "info": info,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    random.seed(int(cfg.TRAIN_CFG["seed"]))
    torch.manual_seed(int(cfg.TRAIN_CFG["seed"]))
    device = pick_device(args.device)

    if args.out_dir.exists():
        raise FileExistsError(f"output dir already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    dataset = MDPDataset(args.data_dir, int(args.seq_len))
    model = MDPStateNet(**cfg.MODEL_CFG).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.TRAIN_CFG["learning_rate"]))

    log_path = args.out_dir / "log.csv"
    fields = ["update", "loss", "dyn", "reward", "done", "value", "probe", "probe_x_err", "probe_d_err", "var", "z_std"]
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        print(
            f"[INFO] train mdp data={args.data_dir} out={args.out_dir} seq={args.seq_len} "
            f"batch={args.batch_size} updates={args.updates} device={device}"
        )
        last = {}
        for update in range(1, int(args.updates) + 1):
            batch = dataset.sample(int(args.batch_size))
            info_t = train_step(model, batch, device)
            loss = info_t["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()

            info = {key: float(value.detach().cpu()) for key, value in info_t.items()}
            row = {"update": update, **info}
            writer.writerow(row)
            file.flush()
            last = info
            print(
                f"update={update} loss={info['loss']:.4f} dyn={info['dyn']:.4f} "
                f"rew={info['reward']:.4f} done={info['done']:.4f} probe={info['probe']:.4f} "
                f"probe_x={info['probe_x_err']:.2f}px probe_d={info['probe_d_err']:.2f}m "
                f"var={info['var']:.4f} z_std={info['z_std']:.4f}"
            )
            if update % int(cfg.TRAIN_CFG["save_every"]) == 0:
                save(args.out_dir / f"model_{update}.pt", model, update, info)

    save(args.out_dir / "last.pt", model, int(args.updates), last)
    print(f"[INFO] saved {args.out_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
