import argparse
import copy
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from torch.profiler import ProfilerActivity, profile

from train import build_optimizer, build_scheduler, evaluate, get_dataloaders, get_device, set_seed, train
from models import build_model


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_paths(cfg, base: Path):
    cfg = copy.deepcopy(cfg)
    cfg["data"]["root"] = str((base / cfg["data"]["root"]).resolve())
    cfg["output"]["results_dir"] = str((base / cfg["output"]["results_dir"]).resolve())
    return cfg


def profile_one_epoch(cfg, device):
    """Task 2"""
    set_seed(cfg["seed"])
    train_loader, _, _ = get_dataloaders(cfg, device)
    model = build_model(cfg["model"]["name"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    model.train()
    # limited number of steps for speed 
    steps_to_profile = min(50, len(train_loader))

    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for i, (images, labels) in enumerate(train_loader):
            if i >= steps_to_profile:
                break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            prof.step()

    table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=30)
    print(table)

    totals = {"data_loading": 0.0, "forward": 0.0, "backward": 0.0, "optimizer": 0.0, "other": 0.0}
    total_cpu = 0.0
    for evt in prof.key_averages():
        t = evt.cpu_time_total
        total_cpu += t
        name = (evt.key or "").lower()
        if "dataloader" in name or "enumerate" in name:
            totals["data_loading"] += t
        elif "backward" in name or "autograd" in name or "accumulategrad" in name:
            totals["backward"] += t
        elif "optimizer" in name or "adam" in name or "sgd" in name or "step" in name and "aten::" not in name:
            totals["optimizer"] += t
        elif "aten::" in name or "conv" in name or "linear" in name or "relu" in name or "loss" in name or "forward" in name:
            totals["forward"] += t
        else:
            totals["other"] += t

    # heuristic fallback
    if total_cpu == 0:
        total_cpu = 1.0
    shares = {k: 100.0 * v / total_cpu for k, v in totals.items()}

    print("\nApproximate time share (by CPU time in profiled ops):")
    for k, v in shares.items():
        print(f"  {k:15s}: {v:6.2f}%")
    print("NOTE: data loading may appear under DataLoader/enumerate; for precise breakdown also compare wall-clock "
          "of loader iteration vs compute. See printed profiler table above.")

    out = Path(cfg["output"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "profile_table.txt").write_text(table, encoding="utf-8")
    with open(out / "profile_shares.json", "w", encoding="utf-8") as f:
        json.dump({"cpu_time_totals_us": totals, "shares_percent": shares}, f, indent=2)
    return shares


def study_workers(cfg, device):
    """Task 3"""
    values = [0, 1, 2, 4, 8]
    results = []
    for nw in values:
        c = copy.deepcopy(cfg)
        c["data"]["num_workers"] = nw
        c["training"]["epochs"] = min(1, c["training"]["epochs"])  
        c["run_name"] = f"workers_{nw}"
        set_seed(c["seed"])
        train_loader, _, _ = get_dataloaders(c, device)
        model = build_model(c["model"]["name"]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(model, c)

        # measure one epoch 
        model.train()
        n_samples = 0
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            n_samples += labels.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        throughput = n_samples / elapsed if elapsed > 0 else 0
        results.append({"num_workers": nw, "epoch_time_s": elapsed, "throughput_samples_s": throughput})
        print(f"workers={nw}: {elapsed:.2f}s, {throughput:.1f} samples/s")

    out = Path(cfg["output"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "workers_study.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    xs = [r["num_workers"] for r in results]
    ys = [r["throughput_samples_s"] for r in results]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("num_workers")
    plt.ylabel("Throughput (samples/s)")
    plt.title("DataLoader num_workers vs Throughput")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out / "workers_bottleneck.png", dpi=200)
    plt.close()
    print(f"Saved plot to {out / 'workers_bottleneck.png'}")
    return results


def study_batch_size(cfg, device):
    """Task 4"""
    batch_sizes = [32, 64, 128, 256, 512]
    results = []
    for bs in batch_sizes:
        c = copy.deepcopy(cfg)
        c["data"]["batch_size"] = bs
        c["training"]["epochs"] = 5
        c["run_name"] = f"batch_{bs}"
        c["output"] = copy.deepcopy(c["output"])
        c["output"]["results_dir"] = str(Path(cfg["output"]["results_dir"]) / f"batch_{bs}")
        set_seed(c["seed"])
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            train_loader, val_loader, test_loader = get_dataloaders(c, device)
            model = build_model(c["model"]["name"]).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = build_optimizer(model, c)

            # throughput 
            model.train()
            n_samples = 0
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                out = model(images)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                n_samples += labels.size(0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            throughput = n_samples / (time.perf_counter() - t0)

            # 5-epoch training 
            metrics = train(c, device=device)
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else None
            results.append({
                "batch_size": bs,
                "throughput_samples_s": throughput,
                "peak_memory_mb": peak_mem,
                "final_test_accuracy": metrics["test_acc"],
            })
            print(f"bs={bs}: thr={throughput:.1f}, mem={peak_mem}, acc={metrics['test_acc']:.2f}%")
        except RuntimeError as e:
            print(f"bs={bs} failed (likely OOM): {e}")
            results.append({"batch_size": bs, "error": str(e)})

    out = Path(cfg["output"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "batch_study.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    # csv
    keys = ["batch_size", "throughput_samples_s", "peak_memory_mb", "final_test_accuracy", "error"]
    with open(out / "batch_study.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in keys})
    return results


def study_schedules(cfg, device):
    """Task 5"""
    configs = []
    # LR schedules with Adam
    for sched in ["step", "cosine", "plateau"]:
        c = copy.deepcopy(cfg)
        c["training"]["epochs"] = 10
        c["training"]["optimizer"] = "adam"
        c["training"]["scheduler"] = sched
        c["run_name"] = f"adam_{sched}"
        c["output"]["results_dir"] = str(Path(cfg["output"]["results_dir"]) / f"sched_adam_{sched}")
        configs.append((f"Adam+{sched}", c))
    # Optimizers
    for opt in ["sgd_momentum", "adam"]:
        c = copy.deepcopy(cfg)
        c["training"]["epochs"] = 10
        c["training"]["optimizer"] = opt
        c["training"]["scheduler"] = "cosine"
        c["run_name"] = f"opt_{opt}"
        c["output"]["results_dir"] = str(Path(cfg["output"]["results_dir"]) / f"opt_{opt}_cosine")
        configs.append((f"{opt}+cosine", c))

    results = []
    for label, c in configs:
        print(f"\n=== Running {label} ===")
        metrics = train(c, device=device)
        results.append({"setting": label, "test_acc": metrics["test_acc"], "test_loss": metrics["test_loss"], "best_val_acc": metrics["best_val_acc"]})

    out = Path(cfg["output"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "schedules_optimizers.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    return results


def study_accumulation(cfg, device):
    """Task 6"""
    out = Path(cfg["output"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)

    base = copy.deepcopy(cfg)
    base["training"]["epochs"] = 5

    c_large = copy.deepcopy(base)
    c_large["data"]["batch_size"] = 256
    c_large["training"]["grad_accum_steps"] = 1
    c_large["run_name"] = "batch256"
    c_large["output"]["results_dir"] = str(out / "accum_batch256")
    print("\n=== Effective batch 256 (actual 256) ===")
    m_large = train(c_large, device=device)

    c_acc = copy.deepcopy(base)
    c_acc["data"]["batch_size"] = 64
    c_acc["training"]["grad_accum_steps"] = 4
    c_acc["run_name"] = "batch64_accum4"
    c_acc["output"]["results_dir"] = str(out / "accum_batch64x4")
    print("\n=== Effective batch 256 (64 x 4 accumulation) ===")
    m_acc = train(c_acc, device=device)

    summary = {
        "batch256": {"test_acc": m_large["test_acc"], "history": m_large["history"]},
        "batch64_accum4": {"test_acc": m_acc["test_acc"], "history": m_acc["history"]},
    }
    with open(out / "grad_accum_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # plot curves
    for key, metric in [("val_acc", "Validation Accuracy"), ("train_loss", "Train Loss")]:
        plt.figure(figsize=(6, 4))
        for label, hist in [("batch 256", m_large["history"]), ("batch 64 x accum 4", m_acc["history"])]:
            plt.plot([h["epoch"] for h in hist], [h[key] for h in hist], marker="o", label=label)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"Gradient accumulation comparison: {metric}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(out / f"grad_accum_{key}.png", dpi=200)
        plt.close()
    print(f"Saved comparison artifacts in {out}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["profile", "workers", "batch", "schedules", "accum"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    base = Path(args.config).resolve().parent
    cfg = ensure_paths(load_config(args.config), base)
    device = get_device()
    print("Device:", device)

    if args.task == "profile":
        profile_one_epoch(cfg, device)
    elif args.task == "workers":
        study_workers(cfg, device)
    elif args.task == "batch":
        study_batch_size(cfg, device)
    elif args.task == "schedules":
        study_schedules(cfg, device)
    elif args.task == "accum":
        study_accumulation(cfg, device)
