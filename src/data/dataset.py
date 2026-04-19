import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_music_sentiment_data(csv_path, drop_columns=None):
    df = pd.read_csv(csv_path)

    label_names = sorted(df["sentiment"].unique().tolist())
    label_map = {label: idx for idx, label in enumerate(label_names)}
    labels = df["sentiment"].map(label_map).to_numpy(dtype=np.int64)

    feature_frame = df.drop(columns=["sentiment"]).copy()

    if drop_columns:
        existing_columns = [column for column in drop_columns if column in feature_frame.columns]
        feature_frame = feature_frame.drop(columns=existing_columns)

    categorical_columns = feature_frame.select_dtypes(include=["object"]).columns.tolist()
    if categorical_columns:
        feature_frame = pd.get_dummies(
            feature_frame,
            columns=categorical_columns,
            dtype=float,
        )

    features = feature_frame.to_numpy(dtype=np.float32)
    return features, labels, label_map, feature_frame.columns.tolist()


class MusicDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
