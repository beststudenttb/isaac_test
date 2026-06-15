"""Train or evaluate the Isaac image -> x/dist model.

Run from the project root:

    python train/train_cv.py --train --model resnet --updates 1000 --batch-size 64
    python train/train_cv.py --val --model old-mobile --size 1000
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.io import read_image

sys.path.insert(0, "./src")

from cv_extractor.simple_cnn import BallVisionNet, MobileBallNet, OldBallVisionNet, OldMobileBallNet

DATA_DIR = Path("./data_isaac")
LR = 1e-4
EVAL_EVERY = 20
MODEL_CLASSES = {
    "resnet": BallVisionNet,
    "mobile": MobileBallNet,
    "old": OldBallVisionNet,
    "old-mobile": OldMobileBallNet,
}
MODEL_DIRS = {
    "resnet": Path("./models/cv_resnet"),
    "mobile": Path("./models/cv_mobile"),
    "old": Path("./models/cv_old"),
    "old-mobile": Path("./models/cv_old_mobile"),
}


class CVDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, augment: bool = False):
        self.image_paths = sorted((data_dir / split / "img").glob("*.png"))
        self.label_dir = data_dir / split / "label"
        self.augment = bool(augment)
        if not self.image_paths:
            raise RuntimeError(f"no images found: {data_dir / split / 'img'}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        label = np.loadtxt(self.label_dir / f"{image_path.stem}.txt")
        image = read_image(str(image_path)).permute(1, 2, 0).contiguous()
        if self.augment:
            image = self.augment_image(image)
        return {
            "image": image,
            "x": torch.tensor(label[0], dtype=torch.float32),
            "dist": torch.tensor(label[1], dtype=torch.float32),
        }

    def augment_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float()
        image = image * float(np.random.uniform(0.9, 1.1))
        image = (image - 127.5) * float(np.random.uniform(0.9, 1.1)) + 127.5
        if float(np.random.random()) < 0.3:
            image = image + torch.randn_like(image) * 3.0
        return image.clamp(0.0, 255.0).to(torch.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate image -> x/dist models.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--val", action="store_true")
    parser.add_argument("--model", choices=tuple(MODEL_CLASSES), default="resnet")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--size", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = MODEL_DIRS[args.model]
    return args


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def loss_fn(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], device: torch.device):
    x_label = batch["x"].float().view(-1, 1).to(device)
    dist_label = batch["dist"].float().view(-1, 1).to(device)
    x_loss = F.smooth_l1_loss(output["x_pred"], x_label)
    dist_loss = F.smooth_l1_loss(output["distance_pred"], dist_label)
    return x_loss + dist_loss, x_loss, dist_loss


def make_eval_loader(dataset: Dataset, batch_size: int, size: int) -> DataLoader:
    if int(size) > 0 and int(size) < len(dataset):
        indices = torch.randperm(len(dataset))[: int(size)].tolist()
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)


def evaluate(model: torch.nn.Module, dataset: Dataset, batch_size: int, size: int, device: torch.device):
    model.eval()
    loader = make_eval_loader(dataset, batch_size, size)
    loss_sum = 0.0
    x_sum = 0.0
    dist_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            output = model(batch["image"])
            loss, x_loss, dist_loss = loss_fn(output, batch, device)
            n = int(batch["x"].shape[0])
            loss_sum += float(loss.detach()) * n
            x_sum += float(x_loss.detach()) * n
            dist_sum += float(dist_loss.detach()) * n
            count += n
    model.train()
    return loss_sum / count, x_sum / count, dist_sum / count, count


def print_predictions(model: torch.nn.Module, dataset: Dataset, batch_size: int, size: int, device: torch.device):
    model.eval()
    loader = make_eval_loader(dataset, batch_size, size)
    x_err_sum = 0.0
    dist_err_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            output = model(batch["image"])
            x_gt = batch["x"].float().view(-1, 1).to(device)
            dist_gt = batch["dist"].float().view(-1, 1).to(device)
            x_pred = output["x_pred"]
            dist_pred = output["distance_pred"]
            x_scale = float(batch["image"].shape[2]) * 0.5
            x_gt = x_gt * x_scale
            x_pred = x_pred * x_scale
            x_err = (x_pred - x_gt).abs()
            dist_err = (dist_pred - dist_gt).abs()

            x_gt_list = x_gt.detach().cpu().flatten().tolist()
            x_pred_list = x_pred.detach().cpu().flatten().tolist()
            x_err_list = x_err.detach().cpu().flatten().tolist()
            dist_gt_list = dist_gt.detach().cpu().flatten().tolist()
            dist_pred_list = dist_pred.detach().cpu().flatten().tolist()
            dist_err_list = dist_err.detach().cpu().flatten().tolist()
            for i in range(len(x_gt_list)):
                print(
                    f"i={count + i} "
                    f"gt_x={x_gt_list[i]:.2f} pred_x={x_pred_list[i]:.2f} err_x={x_err_list[i]:.2f} "
                    f"gt_dist={dist_gt_list[i]:.2f} pred_dist={dist_pred_list[i]:.2f} err_dist={dist_err_list[i]:.2f}"
                )

            n = int(batch["x"].shape[0])
            x_err_sum += float(x_err.sum().detach())
            dist_err_sum += float(dist_err.sum().detach())
            count += n
    model.train()
    print(f"mean_err x={x_err_sum / count:.2f} dist={dist_err_sum / count:.2f} n={count}")


def make_model(name: str, pretrained: bool) -> torch.nn.Module:
    return MODEL_CLASSES[name](pretrained=pretrained)


def save(path: Path, model: torch.nn.Module, model_name: str, update: int, test_loss: float):
    torch.save(
        {
            "model": model_name,
            "model_class": model.__class__.__name__,
            "model_state_dict": model.state_dict(),
            "update": int(update),
            "test_loss": float(test_loss),
        },
        path,
    )


def train(args: argparse.Namespace, device: torch.device):
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_set = CVDataset(args.data_dir, "train", augment=True)
    test_set = CVDataset(args.data_dir, "test", augment=False)
    train_loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True, num_workers=0)

    model = make_model(args.model, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    log_file = (args.out_dir / "log.csv").open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(log_file, fieldnames=["update", "train_loss", "train_x", "train_dist", "test_loss", "test_x", "test_dist", "test_count"])
    writer.writeheader()

    best_test = float("inf")
    update = 0
    train_loss_sum = 0.0
    train_x_sum = 0.0
    train_dist_sum = 0.0
    train_count = 0
    print(
        f"train start model={args.model} device={device} train={len(train_set)} test={len(test_set)} "
        f"batch={args.batch_size} updates={args.updates} out={args.out_dir}"
    )

    try:
        while update < int(args.updates):
            for batch in train_loader:
                if update >= int(args.updates):
                    break
                update += 1
                output = model(batch["image"])
                loss, x_loss, dist_loss = loss_fn(output, batch, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                n = int(batch["x"].shape[0])
                train_loss_sum += float(loss.detach()) * n
                train_x_sum += float(x_loss.detach()) * n
                train_dist_sum += float(dist_loss.detach()) * n
                train_count += n

                if update % EVAL_EVERY == 0 or update >= int(args.updates):
                    train_loss = train_loss_sum / train_count
                    train_x = train_x_sum / train_count
                    train_dist = train_dist_sum / train_count
                    test_loss, test_x, test_dist, test_count = evaluate(model, test_set, int(args.batch_size), int(args.size), device)
                    print(
                        f"update={update} train={train_loss:.5f} x={train_x:.5f} dist={train_dist:.5f} "
                        f"test={test_loss:.5f} x={test_x:.5f} dist={test_dist:.5f} n={test_count}"
                    )
                    writer.writerow(
                        {
                            "update": update,
                            "train_loss": train_loss,
                            "train_x": train_x,
                            "train_dist": train_dist,
                            "test_loss": test_loss,
                            "test_x": test_x,
                            "test_dist": test_dist,
                            "test_count": test_count,
                        }
                    )
                    log_file.flush()
                    save(args.out_dir / "last.pt", model, args.model, update, test_loss)
                    if test_loss < best_test:
                        best_test = test_loss
                        save(args.out_dir / "best.pt", model, args.model, update, test_loss)
                    train_loss_sum = 0.0
                    train_x_sum = 0.0
                    train_dist_sum = 0.0
                    train_count = 0
    finally:
        log_file.close()


def val(args: argparse.Namespace, device: torch.device):
    val_set = CVDataset(args.data_dir, "val", augment=False)
    model = make_model(args.model, pretrained=False).to(device)
    ckpt = torch.load(args.out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print_predictions(model, val_set, int(args.batch_size), int(args.size), device)


def main():
    args = parse_args()
    device = pick_device(args.device)
    if args.train:
        np.random.seed(0)
        torch.manual_seed(0)
        train(args, device)
    if args.val:
        val(args, device)


if __name__ == "__main__":
    main()
