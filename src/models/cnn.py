import torch.nn as nn


class MusicSentimentCNN(nn.Module):
    def __init__(
        self,
        input_shape,
        num_classes,
        channels=None,
        classifier_hidden_dim=128,
        dropout=0.3,
    ):
        super().__init__()

        if channels is None:
            channels = [16, 32, 64]

        conv_layers = []
        in_channels = input_shape[0]

        for out_channels in channels:
            conv_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Dropout2d(dropout),
                ]
            )
            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, features):
        if features.ndim != 4:
            raise ValueError(
                "MusicSentimentCNN expects a 4D tensor shaped "
                "[batch_size, channels, freq_bins, time_steps]"
            )

        x = self.feature_extractor(features)
        x = self.pool(x)
        return self.classifier(x)
