from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.audio.audio_features import AudioFeatureExtractor


MFCC_COLUMN_PATTERN = re.compile(r"^mfcc_(\d+)$")


def _map_emotions(df):
    if "emotion" in df.columns:
        return df.dropna(subset=["emotion"]).copy()

    if "sentiment" not in df.columns:
        raise ValueError("Dataset must contain either an 'emotion' or 'sentiment' column.")

    emotion_map = {
        "happy": "happy",
        "surprise": "happy",
        "sad": "sad",
        "neutral": "calm",
        "angry": "energetic",
        "fear": "energetic",
        "disgust": "energetic",
    }

    mapped = df.copy()
    mapped["emotion"] = mapped["sentiment"].map(emotion_map)
    return mapped.dropna(subset=["emotion"])


def _sorted_mfcc_columns(columns):
    pairs = []
    for column in columns:
        match = MFCC_COLUMN_PATTERN.match(column)
        if match:
            pairs.append((int(match.group(1)), column))
    return [column for _, column in sorted(pairs)]


def _resolve_audio_path(path_value, base_dir):
    raw_path = Path(str(path_value))
    if raw_path.is_absolute():
        return raw_path
    return Path(base_dir) / raw_path


def _build_label_data(df):
    label_names = sorted(df["emotion"].unique().tolist())
    label_map = {label: idx for idx, label in enumerate(label_names)}
    labels = df["emotion"].map(label_map).to_numpy(dtype=np.int64)
    return labels, label_map


def _load_metadata_features(df, drop_columns, audio_path_column):
    feature_frame = df.drop(columns=["sentiment", "emotion"], errors="ignore").copy()

    if drop_columns:
        existing_columns = [col for col in drop_columns if col in feature_frame.columns]
        feature_frame = feature_frame.drop(columns=existing_columns)

    mfcc_columns = _sorted_mfcc_columns(feature_frame.columns)
    drop_for_metadata = list(mfcc_columns)
    if audio_path_column and audio_path_column in feature_frame.columns:
        drop_for_metadata.append(audio_path_column)
    if drop_for_metadata:
        feature_frame = feature_frame.drop(columns=drop_for_metadata, errors="ignore")

    categorical_columns = feature_frame.select_dtypes(include=["object"]).columns.tolist()
    if categorical_columns:
        feature_frame = pd.get_dummies(
            feature_frame,
            columns=categorical_columns,
            dtype=float,
        )

    features = feature_frame.to_numpy(dtype=np.float32)
    return features, feature_frame.columns.tolist()


def _load_audio_features(df, feature_frame, audio_config):
    audio_path_column = audio_config.get("audio_path_column")
    base_dir = audio_config.get("base_dir", "data")
    feature_type = audio_config.get("feature_type", "mfcc")
    n_mfcc = audio_config.get("n_mfcc", 40)
    n_mels = audio_config.get("n_mels", 128)
    max_frames = audio_config.get("max_frames", 128)

    if audio_path_column and audio_path_column in feature_frame.columns:
        extractor = AudioFeatureExtractor(sample_rate=audio_config.get("sample_rate", 22050))
        audio_tensors = []

        for path_value in feature_frame[audio_path_column]:
            resolved_path = _resolve_audio_path(path_value, base_dir)
            if feature_type == "mel_spectrogram":
                feature_map = extractor.mel_spectrogram(
                    resolved_path,
                    n_mels=n_mels,
                    max_frames=max_frames,
                )
            elif feature_type == "mfcc":
                feature_map = extractor.mfcc(
                    resolved_path,
                    n_mfcc=n_mfcc,
                    max_frames=max_frames,
                )
            else:
                raise ValueError(
                    f"Unsupported audio feature type '{feature_type}'. "
                    "Use 'mfcc' or 'mel_spectrogram'."
                )

            audio_tensors.append(feature_map[np.newaxis, :, :])

        features = np.stack(audio_tensors).astype(np.float32)
        feature_shape = list(features.shape[1:])
        return features, feature_shape

    mfcc_columns = _sorted_mfcc_columns(feature_frame.columns)
    if not mfcc_columns:
        raise ValueError(
            "CNN training requires either an audio path column or MFCC columns in the CSV."
        )

    mfcc_values = feature_frame[mfcc_columns].to_numpy(dtype=np.float32)
    features = mfcc_values[:, np.newaxis, :, np.newaxis]
    feature_shape = [1, len(mfcc_columns), 1]
    return features, feature_shape


def _load_multimodal_features(df, drop_columns, audio_config):
    audio_path_column = None
    if audio_config:
        audio_path_column = audio_config.get("audio_path_column")

    metadata_features, metadata_feature_names = _load_metadata_features(
        df,
        drop_columns=drop_columns,
        audio_path_column=audio_path_column,
    )

    feature_frame = df.drop(columns=["sentiment", "emotion"], errors="ignore").copy()
    if drop_columns:
        existing_columns = [col for col in drop_columns if col in feature_frame.columns]
        feature_frame = feature_frame.drop(columns=existing_columns)

    audio_features, audio_feature_shape = _load_audio_features(
        df,
        feature_frame,
        audio_config or {},
    )
    features = {"metadata": metadata_features, "audio": audio_features}
    feature_names = {
        "metadata": metadata_feature_names,
        "audio": audio_feature_shape,
    }
    return features, feature_names


def load_music_sentiment_data(csv_path, drop_columns=None, model_type="mlp", audio_config=None):
    df = pd.read_csv(csv_path)
    df = _map_emotions(df)

    labels, label_map = _build_label_data(df)
    feature_frame = df.drop(columns=["sentiment", "emotion"], errors="ignore").copy()

    if drop_columns:
        existing_columns = [col for col in drop_columns if col in feature_frame.columns]
        feature_frame = feature_frame.drop(columns=existing_columns)

    audio_path_column = None
    if audio_config:
        audio_path_column = audio_config.get("audio_path_column")

    if model_type == "mlp":
        features, feature_names = _load_metadata_features(
            df,
            drop_columns=drop_columns,
            audio_path_column=audio_path_column,
        )
    elif model_type == "cnn":
        features, feature_names = _load_audio_features(
            df,
            feature_frame,
            audio_config or {},
        )
    elif model_type == "multimodal":
        features, feature_names = _load_multimodal_features(
            df,
            drop_columns=drop_columns,
            audio_config=audio_config or {},
        )
    else:
        raise ValueError(
            f"Unsupported model type '{model_type}'. Use 'mlp', 'cnn', or 'multimodal'."
        )

    print("\nEmotion Distribution:")
    print(df["emotion"].value_counts())
    print("\nLabel Mapping:")
    print(label_map)
    if isinstance(features, dict):
        print("\nFeature Shape:")
        print("  metadata:", features["metadata"].shape)
        print("  audio:", features["audio"].shape)
    else:
        print("\nFeature Shape:", features.shape)

    return features, labels, label_map, feature_names


class MusicDataset(Dataset):
    def __init__(self, features, labels):
        if isinstance(features, dict):
            self.features = {
                key: torch.tensor(value, dtype=torch.float32)
                for key, value in features.items()
            }
        else:
            self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if isinstance(self.features, dict):
            feature_item = {key: value[idx] for key, value in self.features.items()}
            return feature_item, self.labels[idx]
        return self.features[idx], self.labels[idx]
