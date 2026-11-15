import argparse
import copy
import json
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader, random_split

from models import build_model


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders(cfg, device):
    data_cfg = cfg["data"]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    train_full = torchvision.datasets.FashionMNIST(
        root=data_cfg["root"], train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.FashionMNIST(
        root=data_cfg["root"], train=False, download=True, transform=transform
    )
    val_size = int(data_cfg.get("val_size", 5000))
    train_size = len(train_full) - val_size
    g = torch.Generator().manual_seed(cfg["seed"])
    train_set, val_set = random_split(train_full, [train_size, val_size], generator=g)

    pin = device.type == "cuda"
    nw = int(data_cfg.get("num_workers", 0))
    bs = int(data_cfg.get("batch_size", 128))

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=pin,
            generator=torch.Generator().manual_seed(cfg["seed"]) if shuffle else None,
        )

    return make_loader(train_set, True), make_loader(val_set, False), make_loader(test_set, False)


def build_optimizer(model, cfg):
    t = cfg["training"]
    opt = t.get("optimizer", "adam").lower()
    lr = float(t["lr"])
    wd = float(t.get("weight_decay", 0.0))
    if opt == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if opt in ("sgd", "sgd_momentum"):
        return optim.SGD(model.parameters(), lr=lr, momentum=float(t.get("momentum", 0.9)), weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {opt}")


def build_scheduler(optimizer, cfg):
    t = cfg["training"]
    name = t.get("scheduler", "none").lower()
    if name == "none":
        return None
    if name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=int(t.get("step_size", 3)), gamma=float(t.get("gamma", 0.1)))
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(t.get("t_max", t["epochs"])))
    if name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=float(t.get("plateau_factor", 0.5)), patience=int(t.get("plateau_patience", 2))
        )
    raise ValueError(f"Unknown scheduler: {name}")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch, history, best_val_acc, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "history": history,
        "best_val_acc": best_val_acc,
        "config": cfg,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt


def train(cfg, device=None):
    device = device or get_device()
    set_seed(cfg["seed"])

    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(cfg, device)
    model = build_model(cfg["model"]["name"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    start_epoch = 1
    history = []
    best_val_acc = 0.0

    resume_from = cfg["training"].get("resume_from", "")
    if resume_from:
        ckpt = load_checkpoint(Path(resume_from), model, optimizer, scheduler, device)
        start_epoch = int(ckpt["epoch"]) + 1
        history = ckpt.get("history", [])
        best_val_acc = float(ckpt.get("best_val_acc", 0.0))
        print(f"Resumed from {resume_from} at epoch {start_epoch}")

    accum_steps = int(cfg["training"].get("grad_accum_steps", 1))
    epochs = int(cfg["training"]["epochs"])

    mlflow.set_experiment(cfg.get("experiment_name", "lab6"))
    with mlflow.start_run(run_name=cfg.get("run_name")):
        # Log all params for reproducibility
        mlflow.log_params({
            "seed": cfg["seed"],
            "model": cfg["model"]["name"],
            "batch_size": cfg["data"]["batch_size"],
            "num_workers": cfg["data"]["num_workers"],
            "epochs": epochs,
            "lr": cfg["training"]["lr"],
            "optimizer": cfg["training"].get("optimizer", "adam"),
            "scheduler": cfg["training"].get("scheduler", "none"),
            "grad_accum_steps": accum_steps,
        })
        mlflow.log_artifact(__file__)  # log config

        for epoch in range(start_epoch, epochs + 1):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            optimizer.zero_grad(set_to_none=True)

            for step, (images, labels) in enumerate(train_loader, start=1):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels) / accum_steps
                loss.backward()

                if step % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                running_loss += loss.item() * accum_steps * labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

            if len(train_loader) % accum_steps != 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss = running_loss / total
            train_acc = 100.0 * correct / total
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            # scheduler step
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
            }
            history.append(row)
            mlflow.log_metrics(
                {"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "lr": current_lr},
                step=epoch,
            )

            print(f"Epoch {epoch}/{epochs} | train loss {train_loss:.4f} acc {train_acc:.2f}% | "
                  f"val loss {val_loss:.4f} acc {val_acc:.2f}% | lr {current_lr:.6f}")

            # checkpoints
            save_checkpoint(out_dir / "checkpoint_last.pth", model, optimizer, scheduler, epoch, history, best_val_acc, cfg)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(out_dir / "checkpoint_best.pth", model, optimizer, scheduler, epoch, history, best_val_acc, cfg)
            if cfg["output"].get("save_every_epoch", True):
                save_checkpoint(out_dir / f"checkpoint_epoch_{epoch}.pth", model, optimizer, scheduler, epoch, history, best_val_acc, cfg)

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Test: loss {test_loss:.4f}, acc {test_acc:.2f}%")
        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})

        # save artifacts
        history_path = out_dir / "history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        mlflow.log_artifact(str(history_path))
        mlflow.log_artifact(str(out_dir / "checkpoint_best.pth"))

    return {"history": history, "test_loss": test_loss, "test_acc": test_acc, "best_val_acc": best_val_acc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    base = Path(args.config).resolve().parent
    for k in ("root",):
        cfg["data"][k] = str((base / cfg["data"][k]).resolve())
    cfg["output"]["results_dir"] = str((base / cfg["output"]["results_dir"]).resolve())
    if cfg["training"].get("resume_from"):
        cfg["training"]["resume_from"] = str((base / cfg["training"]["resume_from"]).resolve())
    train(cfg)
