import torch
import torch.nn as nn


class MultimodalMusicSentimentModel(nn.Module):
    def __init__(
        self,
        metadata_input_dim,
        audio_input_shape,
        num_classes,
        metadata_hidden_dims=None,
        audio_channels=None,
        embedding_dim=128,
        fusion_hidden_dim=128,
        dropout=0.2,
    ):
        super().__init__()

        if metadata_hidden_dims is None:
            metadata_hidden_dims = [128, 64]
        if audio_channels is None:
            audio_channels = [16, 32, 64]

        metadata_layers = []
        current_dim = metadata_input_dim
        for hidden_dim in metadata_hidden_dims:
            metadata_layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim

        metadata_layers.extend(
            [
                nn.Linear(current_dim, embedding_dim),
                nn.ReLU(),
            ]
        )
        self.metadata_encoder = nn.Sequential(*metadata_layers)

        audio_layers = []
        in_channels = audio_input_shape[0]
        for out_channels in audio_channels:
            audio_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Dropout2d(dropout),
                ]
            )
            in_channels = out_channels

        self.audio_encoder = nn.Sequential(*audio_layers)
        self.audio_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.audio_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(audio_channels[-1], embedding_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(self, features):
        if not isinstance(features, dict):
            raise ValueError(
                "MultimodalMusicSentimentModel expects a dict with 'metadata' and 'audio'."
            )

        metadata = features["metadata"]
        audio = features["audio"]

        if metadata.ndim != 2:
            raise ValueError(
                "Metadata input must be a 2D tensor shaped [batch_size, num_features]."
            )
        if audio.ndim != 4:
            raise ValueError(
                "Audio input must be a 4D tensor shaped "
                "[batch_size, channels, freq_bins, time_steps]."
            )

        metadata_embedding = self.metadata_encoder(metadata)
        audio_embedding = self.audio_encoder(audio)
        audio_embedding = self.audio_pool(audio_embedding)
        audio_embedding = self.audio_projection(audio_embedding)

        fused = torch.cat([metadata_embedding, audio_embedding], dim=1)
        return self.classifier(fused)
