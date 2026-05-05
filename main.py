import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.data.dataset import MusicDataset, load_music_sentiment_data
from src.evaluation.metrics import summarize_predictions
from src.models.cnn import MusicSentimentCNN
from src.models.mlp import MusicSentimentMLP
from src.models.multimodal import MultimodalMusicSentimentModel
from src.utils.config_loader import Config


def move_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(value, device) for value in batch)
    return batch


def get_batch_size(features):
    if torch.is_tensor(features):
        return features.size(0)
    if isinstance(features, dict):
        first_value = next(iter(features.values()))
        return get_batch_size(first_value)
    if isinstance(features, (list, tuple)):
        return get_batch_size(features[0])
    raise ValueError("Unsupported batch structure for inferring batch size.")


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for features, labels in loader:
        features = move_to_device(features, device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * get_batch_size(features)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, label_map):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features = move_to_device(features, device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            running_loss += loss.item() * get_batch_size(features)
            predictions = logits.argmax(dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    metrics = summarize_predictions(all_labels, all_predictions, label_map)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics


def standardize_audio_features(train_features, val_features):
    mean = float(train_features.mean())
    std = float(train_features.std())
    if std == 0:
        std = 1.0

    train_features = (train_features - mean) / std
    val_features = (val_features - mean) / std
    preprocessing = {"type": "audio_standardization", "mean": mean, "std": std}
    return train_features, val_features, preprocessing


def standardize_multimodal_features(train_features, val_features):
    metadata_scaler = StandardScaler()
    train_metadata = metadata_scaler.fit_transform(train_features["metadata"])
    val_metadata = metadata_scaler.transform(val_features["metadata"])

    train_audio, val_audio, audio_preprocessing = standardize_audio_features(
        train_features["audio"],
        val_features["audio"],
    )

    preprocessing = {
        "type": "multimodal",
        "metadata": {
            "type": "standard_scaler",
            "mean": metadata_scaler.mean_,
            "scale": metadata_scaler.scale_,
        },
        "audio": audio_preprocessing,
    }
    processed_train = {"metadata": train_metadata, "audio": train_audio}
    processed_val = {"metadata": val_metadata, "audio": val_audio}
    return processed_train, processed_val, preprocessing


def save_checkpoint(path, model, preprocessing, label_map, feature_names, config):
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "label_map": label_map,
        "feature_names": feature_names,
        "config": config,
        "preprocessing": preprocessing,
    }

    if preprocessing["type"] == "standard_scaler":
        checkpoint["scaler_mean"] = preprocessing["mean"]
        checkpoint["scaler_scale"] = preprocessing["scale"]
    elif preprocessing["type"] == "multimodal":
        checkpoint["metadata_scaler_mean"] = preprocessing["metadata"]["mean"]
        checkpoint["metadata_scaler_scale"] = preprocessing["metadata"]["scale"]
        checkpoint["audio_mean"] = preprocessing["audio"]["mean"]
        checkpoint["audio_std"] = preprocessing["audio"]["std"]

    torch.save(
        checkpoint,
        path,
    )


def build_model(model_config, feature_names, num_classes):
    model_type = model_config.get("type", "mlp").lower()
    dropout = model_config.get("dropout", 0.2)

    if model_type == "mlp":
        hidden_dims = model_config.get("hidden_dims", [128, 64])
        model = MusicSentimentMLP(
            input_dim=len(feature_names),
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    elif model_type == "cnn":
        model = MusicSentimentCNN(
            input_shape=feature_names,
            num_classes=num_classes,
            channels=model_config.get("cnn_channels", [16, 32, 64]),
            classifier_hidden_dim=model_config.get("classifier_hidden_dim", 128),
            dropout=dropout,
        )
    elif model_type == "multimodal":
        model = MultimodalMusicSentimentModel(
            metadata_input_dim=len(feature_names["metadata"]),
            audio_input_shape=feature_names["audio"],
            num_classes=num_classes,
            metadata_hidden_dims=model_config.get("hidden_dims", [128, 64]),
            audio_channels=model_config.get("cnn_channels", [16, 32, 64]),
            embedding_dim=model_config.get("fusion_embedding_dim", 128),
            fusion_hidden_dim=model_config.get("fusion_hidden_dim", 128),
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unsupported model type '{model_type}'. Use 'mlp', 'cnn', or 'multimodal'."
        )

    return model_type, model


def slice_features(features, indices):
    if isinstance(features, dict):
        return {key: value[indices] for key, value in features.items()}
    return features[indices]


def describe_feature_names(feature_names):
    if isinstance(feature_names, dict):
        return (
            f"metadata={len(feature_names['metadata'])} features, "
            f"audio_shape={feature_names['audio']}"
        )
    return f"{len(feature_names)} features"


def compute_class_weights(labels, num_classes, device):
    class_counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    class_counts[class_counts == 0] = 1.0
    total_samples = float(class_counts.sum())
    weights = total_samples / (num_classes * class_counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_training(config):
    model_type = config["model"].get("type", "mlp").lower()

    requested_device = config["training"]["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA was requested but is not available. Falling back to CPU.")
    else:
        device = torch.device(requested_device)

    features, labels, label_map, feature_names = load_music_sentiment_data(
        config["data"]["csv_path"],
        drop_columns=config["features"].get("drop_columns", []),
        model_type=model_type,
        audio_config=config.get("audio", {}),
    )

    indices = np.arange(len(labels))
    train_indices, val_indices, y_train, y_val = train_test_split(
        indices,
        labels,
        test_size=config["training"]["val_size"],
        random_state=config["training"]["random_state"],
        stratify=labels,
    )
    x_train = slice_features(features, train_indices)
    x_val = slice_features(features, val_indices)

    if model_type == "mlp":
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        preprocessing = {
            "type": "standard_scaler",
            "mean": scaler.mean_,
            "scale": scaler.scale_,
        }
    elif model_type == "cnn":
        x_train, x_val, preprocessing = standardize_audio_features(x_train, x_val)
    else:
        x_train, x_val, preprocessing = standardize_multimodal_features(x_train, x_val)

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

    model_type, model = build_model(
        config["model"],
        feature_names=feature_names,
        num_classes=len(label_map),
    )
    model = model.to(device)

    class_weights = compute_class_weights(
        y_train,
        num_classes=len(label_map),
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])
    checkpoint_path = config["training"].get(
        "checkpoint_path",
        "artifacts/best_{model_type}.pt",
    ).format(model_type=model_type)

    print(
        f"Training {model_type.upper()} with "
        f"{describe_feature_names(feature_names)} and {len(label_map)} classes"
    )
    print(f"Label mapping: {label_map}")
    print(f"Class weights: {class_weights.cpu().tolist()}")
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
                preprocessing,
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

    return {
        "model_type": model_type,
        "checkpoint_path": checkpoint_path,
        "feature_description": describe_feature_names(feature_names),
        "class_weights": class_weights.cpu().tolist(),
        "best_metrics": best_metrics,
    }


def main():
    config = Config("configs/config.yaml").config
    return run_training(config)


if __name__ == "__main__":
    main()
