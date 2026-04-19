import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.data.dataset import MusicDataset, load_music_sentiment_data
from src.evaluation.metrics import summarize_predictions
from src.models.mlp import MusicSentimentMLP
from src.utils.config_loader import Config


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, label_map):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            running_loss += loss.item() * features.size(0)
            predictions = logits.argmax(dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    metrics = summarize_predictions(all_labels, all_predictions, label_map)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics


def save_checkpoint(path, model, scaler, label_map, feature_names, config):
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "label_map": label_map,
            "feature_names": feature_names,
            "config": config,
        },
        path,
    )


def main():
    config = Config("configs/config.yaml").config

    requested_device = config["training"]["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA was requested but is not available. Falling back to CPU.")
    else:
        device = torch.device(requested_device)

    features, labels, label_map, feature_names = load_music_sentiment_data(
        config["data"]["csv_path"],
        drop_columns=config["features"].get("drop_columns", []),
    )

    x_train, x_val, y_train, y_val = train_test_split(
        features,
        labels,
        test_size=config["training"]["val_size"],
        random_state=config["training"]["random_state"],
        stratify=labels,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    train_dataset = MusicDataset(x_train, y_train)
    val_dataset = MusicDataset(x_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    model = MusicSentimentMLP(
        input_dim=len(feature_names),
        num_classes=len(label_map),
        hidden_dims=config["model"]["hidden_dims"],
        dropout=config["model"]["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])
    checkpoint_path = config["training"]["checkpoint_path"]

    print(f"Training with {len(feature_names)} features and {len(label_map)} classes")
    print(f"Label mapping: {label_map}")

    best_f1 = -1.0
    best_metrics = None

    for epoch in range(config["training"]["epochs"]):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device, label_map)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_metrics = val_metrics
            save_checkpoint(
                checkpoint_path,
                model,
                scaler,
                label_map,
                feature_names,
                config,
            )

        print(
            f"Epoch {epoch + 1}/{config['training']['epochs']} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

    print(f"\nBest checkpoint saved to: {checkpoint_path}")
    print(f"Best validation accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Best validation macro F1: {best_metrics['macro_f1']:.4f}")
    print("Confusion matrix:")
    print(best_metrics["confusion_matrix"])
    print("\nClassification report:")
    print(best_metrics["classification_report"])


if __name__ == "__main__":
    main()
