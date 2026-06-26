"""Train MDP latent state model from teacher image transitions.

Run from the project root:

    python train/mdp_state.py --data-dir ./data_mdp_teacher_50k
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mdp_state_cfg as cfg
from src.cv_extractor.mdp_state import MDPStateNet


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
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=str(cfg.DEVICE))
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


def transition_contrast(pred: torch.Tensor, target: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    logits = pred @ target.t() / float(tau)
    labels = torch.arange(pred.shape[0], device=pred.device)
    loss = F.cross_entropy(logits, labels)
    retrieval1 = (logits.argmax(dim=1) == labels).float().mean()
    return loss, retrieval1


def effective_rank(z: torch.Tensor) -> torch.Tensor:
    z = z.float() - z.float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(z)
    prob = singular / singular.sum().clamp_min(1e-8)
    entropy = -(prob * prob.clamp_min(1e-8).log()).sum()
    return entropy.exp()


def vicreg_var(z: torch.Tensor, gamma: float) -> torch.Tensor:
    std = torch.sqrt(z.float().var(dim=0) + 1e-4)
    return F.relu(float(gamma) - std).mean()


def vicreg_cov(z: torch.Tensor) -> torch.Tensor:
    z = z.float()
    z = z - z.mean(dim=0, keepdim=True)
    n = max(z.shape[0] - 1, 1)
    cov = (z.t() @ z) / n
    off_diag_sq = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag_sq / z.shape[1]


def train_step(model: MDPStateNet, batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    image = batch["image"].to(device)
    action = batch["action"].to(device)
    reward = batch["reward"].to(device)
    done = batch["done"].to(device)
    probe = batch["probe"].to(device)
    ret = returns(reward, done, float(cfg.GAMMA))

    batch_size, seq_len = action.shape[:2]
    state = model.initial(batch_size, device)
    prev_action = torch.zeros(batch_size, int(cfg.ACTION_DIM), device=device)
    obs = model.observe(image[:, 0], prev_action, state)

    dyn_losses = []
    idm_losses = []
    reward_losses = []
    done_losses = []
    value_losses = []
    probe_losses = []
    probe_x_errs = []
    probe_d_errs = []
    z_list = []
    pred_list = []
    target_list = []

    for t in range(seq_len):
        pred = model.imagine(obs, action[:, t])
        next_post = model.observe(image[:, t + 1], action[:, t], obs)

        dyn_losses.append(F.smooth_l1_loss(pred["next_z_pred_unit"], next_post["z_unit"].detach()))
        idm_losses.append(F.smooth_l1_loss(model.inverse(obs["z_img"], next_post["z_img"]), action[:, t]))
        reward_losses.append(F.smooth_l1_loss(pred["reward_pred"], reward[:, t]))
        done_losses.append(F.binary_cross_entropy_with_logits(pred["done_logit"], done[:, t]))
        value_losses.append(F.smooth_l1_loss(obs["value"], ret[:, t]))
        probe_pred = obs["probe"]
        probe_losses.append(F.smooth_l1_loss(probe_pred, probe[:, t]))
        probe_x_errs.append(((probe_pred[:, 0:1] - probe[:, t, 0:1]).abs() * (IMAGE_WIDTH * 0.5)).mean())
        probe_d_errs.append(((probe_pred[:, 1:2] - probe[:, t, 1:2]).abs() * FAIL_FAR).mean())
        z_list.append(obs["z"])
        pred_list.append(pred["next_z_pred_unit"])
        target_list.append(next_post["z_unit"].detach())

        keep = 1.0 - done[:, t]
        obs = {
            "belief": next_post["belief"] * keep,
            "z": next_post["z"] * keep,
            "z_img": next_post["z_img"] * keep,
            "state_feature": next_post["state_feature"] * keep,
            "value": next_post["value"] * keep,
            "probe": next_post["probe"] * keep,
        }

    z_all = torch.cat(z_list, dim=0)
    pred_all = torch.cat(pred_list, dim=0)
    target_all = torch.cat(target_list, dim=0)
    dyn = torch.stack(dyn_losses).mean()
    contrast, retrieval1 = transition_contrast(pred_all, target_all, float(cfg.CONTRAST_TAU))
    idm = torch.stack(idm_losses).mean()
    rew = torch.stack(reward_losses).mean()
    done_loss = torch.stack(done_losses).mean()
    val = torch.stack(value_losses).mean()
    probe_loss = torch.stack(probe_losses).mean()
    eff_rank = effective_rank(z_all.detach())
    var = vicreg_var(z_all, float(cfg.VAR_GAMMA))
    cov = vicreg_cov(z_all)
    loss = (
        float(cfg.DYN_W) * dyn
        + float(cfg.CONTRAST_W) * contrast
        + float(cfg.IDM_W) * idm
        + float(cfg.VAR_W) * var
        + float(cfg.COV_W) * cov
        + float(cfg.REWARD_W) * rew
        + float(cfg.DONE_W) * done_loss
        + float(cfg.VALUE_W) * val
        + float(cfg.PROBE_W) * probe_loss
    )
    return {
        "loss": loss,
        "dyn": dyn.detach(),
        "contrast": contrast.detach(),
        "idm": idm.detach(),
        "reward": rew.detach(),
        "done": done_loss.detach(),
        "value": val.detach(),
        "probe": probe_loss.detach(),
        "probe_x_err": torch.stack(probe_x_errs).mean().detach(),
        "probe_d_err": torch.stack(probe_d_errs).mean().detach(),
        "var": var.detach(),
        "cov": cov.detach(),
        "retrieval1": retrieval1.detach(),
        "eff_rank": eff_rank.detach(),
        "z_std": z_all.detach().float().std(dim=0).mean(),
    }


def model_cfg() -> dict:
    return {
        "input_shape": cfg.INPUT_SHAPE,
        "z_dim": int(cfg.Z_DIM),
        "action_dim": int(cfg.ACTION_DIM),
        "belief_dim": int(cfg.BELIEF_DIM),
        "feature_dim": int(cfg.FEATURE_DIM),
        "hidden_dim": int(cfg.HIDDEN_DIM),
        "pretrained": bool(cfg.PRETRAINED),
        "freeze_backbone": bool(cfg.FREEZE_BACKBONE),
        "train_layer3": bool(cfg.TRAIN_LAYER3),
    }


def train_cfg() -> dict:
    return {
        "dataset_dir": str(cfg.DATASET_DIR),
        "output_dir": str(cfg.OUT_DIR),
        "batch_size": int(cfg.BATCH_SIZE),
        "seq_len": int(cfg.SEQ_LEN),
        "updates": int(cfg.UPDATES),
        "learning_rate": float(cfg.LR),
        "gamma": float(cfg.GAMMA),
        "dyn_w": float(cfg.DYN_W),
        "contrast_w": float(cfg.CONTRAST_W),
        "contrast_tau": float(cfg.CONTRAST_TAU),
        "idm_w": float(cfg.IDM_W),
        "var_w": float(cfg.VAR_W),
        "cov_w": float(cfg.COV_W),
        "var_gamma": float(cfg.VAR_GAMMA),
        "reward_w": float(cfg.REWARD_W),
        "value_w": float(cfg.VALUE_W),
        "done_w": float(cfg.DONE_W),
        "probe_w": float(cfg.PROBE_W),
        "save_every": int(cfg.SAVE_EVERY),
        "seed": int(cfg.SEED),
    }


def save(path: Path, model: MDPStateNet, update: int, info: dict[str, float], run_cfg: dict) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "model_cfg": model_cfg(),
            "train_cfg": run_cfg,
            "update": int(update),
            "info": info,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    random.seed(int(cfg.SEED))
    torch.manual_seed(int(cfg.SEED))
    device = pick_device(args.device)
    run_cfg = train_cfg()

    out_dir = Path(cfg.OUT_DIR)
    data_dir = Path(cfg.DATASET_DIR)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    dataset = MDPDataset(data_dir, int(cfg.SEQ_LEN))
    model = MDPStateNet(**model_cfg()).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.LR))

    log_path = out_dir / "log.csv"
    fields = [
        "update",
        "loss",
        "dyn",
        "contrast",
        "idm",
        "reward",
        "done",
        "value",
        "probe",
        "probe_x_err",
        "probe_d_err",
        "var",
        "cov",
        "retrieval1",
        "eff_rank",
        "z_std",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        print(
            f"[INFO] train mdp data={data_dir} out={out_dir} seq={cfg.SEQ_LEN} "
            f"batch={cfg.BATCH_SIZE} updates={cfg.UPDATES} device={device}"
        )
        last = {}
        for update in range(1, int(cfg.UPDATES) + 1):
            batch = dataset.sample(int(cfg.BATCH_SIZE))
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
                f"con={info['contrast']:.4f} idm={info['idm']:.4f} "
                f"rew={info['reward']:.4f} done={info['done']:.4f} probe={info['probe']:.4f} "
                f"probe_x={info['probe_x_err']:.2f}px probe_d={info['probe_d_err']:.2f}m "
                f"var={info['var']:.4f} cov={info['cov']:.4f} "
                f"retrieval1={info['retrieval1']:.3f} eff_rank={info['eff_rank']:.2f} z_std={info['z_std']:.4f}"
            )
            if update % int(cfg.SAVE_EVERY) == 0:
                save(out_dir / f"model_{update}.pt", model, update, info, run_cfg)

    save(out_dir / "last.pt", model, int(cfg.UPDATES), last, run_cfg)
    print(f"[INFO] saved {out_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
