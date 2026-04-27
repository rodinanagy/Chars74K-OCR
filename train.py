import argparse
import py7zr
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split

from model import (
    Chars74K, Chars74kDataset, TransformedSubset,
    CLASSES, NUM_CLASSES, train_tf, val_tf
)


def get_device():
    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return torch.device("cuda")
    print("No GPU - using CPU")
    return torch.device("cpu")


def extract_dataset(extract_to: Path):
    fnt_check = next(extract_to.rglob("Fnt"), None) if extract_to.exists() else None
    if fnt_check and any(fnt_check.glob("Sample*")):
        print(f"Dataset already extracted at {fnt_check} - skipping.")
        return fnt_check

    archives = sorted(Path(".").glob("EnglishFnt*.7z"))
    assert archives, "Dataset file (EnglishFnt*.7z) not found."
    extract_to.mkdir(exist_ok=True)
    for archive in archives:
        print(f"Extracting {archive.name} ...", end=" ")
        with py7zr.SevenZipFile(archive, mode="r") as z:
            z.extractall(path=extract_to)
        print("Done")

    fnt_root = next(extract_to.rglob("Fnt"), None)
    if fnt_root is None:
        fnt_root = next(extract_to.rglob("Sample*"), None).parent
    return fnt_root


def build_loaders(fnt_root, batch_size, val_split):
    full_dataset = Chars74kDataset(fnt_root, transform=None)
    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val
    train_subset, val_subset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_ds = TransformedSubset(train_subset, train_tf)
    val_ds = TransformedSubset(val_subset, val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Total samples : {len(full_dataset)}")
    print(f"Train         : {len(train_ds)}")
    print(f"Val           : {len(val_ds)}")
    return train_loader, val_loader


def train(args):
    device = get_device()

    fnt_root = extract_dataset(Path(args.data_dir))
    samples = sorted(fnt_root.glob("Sample*"))
    print(f"\nFnt root      : {fnt_root}")
    print(f"Classes found : {len(samples)}  ({samples[0].name} … {samples[-1].name})")

    train_loader, val_loader = build_loaders(fnt_root, args.batch_size, args.val_split)

    model = Chars74K(num_classes=NUM_CLASSES).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    model_out = Path(args.model_out)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = train_correct = train_total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            train_correct += (out.argmax(1) == labels).sum().item()
            train_total += imgs.size(0)
        scheduler.step()

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                val_correct += (model(imgs).argmax(1) == labels).sum().item()
                val_total += imgs.size(0)

        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={train_loss/train_total:.4f}  train={train_acc:.1f}%  val={val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_out)

    print(f"\nBest val accuracy : {best_val_acc:.2f}%")
    print(f"Model saved to    : {model_out}")
    return model, val_loader, device


def evaluate(model, val_loader, device):
    model.eval()
    correct_per_class = np.zeros(NUM_CLASSES)
    total_per_class = np.zeros(NUM_CLASSES)

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            for pred, label in zip(preds, labels):
                total_per_class[label] += 1
                correct_per_class[label] += (pred == label).item()

    overall = correct_per_class.sum() / total_per_class.sum() * 100
    print(f"Overall accuracy: {overall:.2f}%\n")
    print(f"{'Char':<6} {'Acc':>6}")
    print("-" * 14)
    for i, (c, tot) in enumerate(zip(CLASSES, total_per_class)):
        if tot > 0:
            acc = correct_per_class[i] / tot * 100
            if acc < 60:
                flag = " <<---"
            elif acc < 80:
                flag = " <-"
            else:
                flag = ""
            print(f"  {c!r:<4} {acc:6.1f}%{flag}")


def main():
    parser = argparse.ArgumentParser(description="Train Chars74K on Chars74K Fnt dataset")
    parser.add_argument("--data-dir", default="chars74k", help="Directory to extract/find the dataset (default: chars74k)")
    parser.add_argument("--model-out", default="chars74k.pth", help="Where to save the best model (default: chars74k.pth)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--eval-only", action="store_true", help="Skip training, just run evaluation on the val set")
    args = parser.parse_args()

    if args.eval_only:
        device = get_device()
        fnt_root = extract_dataset(Path(args.data_dir))
        _, val_loader = build_loaders(fnt_root, args.batch_size, args.val_split)
        model = Chars74K(num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(args.model_out, map_location=device))
        evaluate(model, val_loader, device)
    else:
        model, val_loader, device = train(args)
        print("\n--- Per-class accuracy on val set ---")
        evaluate(model, val_loader, device)


if __name__ == "__main__":
    main()
