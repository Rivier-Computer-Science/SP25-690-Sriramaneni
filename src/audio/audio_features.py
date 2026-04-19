from pathlib import Path

import librosa
import numpy as np


class AudioFeatureExtractor:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate

    def _load_audio(self, path):
        audio_path = Path(path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return audio, sr

    @staticmethod
    def _fix_time_axis(feature_map, max_frames):
        if feature_map.shape[1] > max_frames:
            return feature_map[:, :max_frames]

        if feature_map.shape[1] < max_frames:
            pad_width = max_frames - feature_map.shape[1]
            feature_map = np.pad(feature_map, ((0, 0), (0, pad_width)), mode="constant")

        return feature_map

    def mfcc(self, path, n_mfcc=40, max_frames=128):
        audio, sr = self._load_audio(path)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return self._fix_time_axis(mfcc.astype(np.float32), max_frames)

    def mel_spectrogram(self, path, n_mels=128, max_frames=128):
        audio, sr = self._load_audio(path)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return self._fix_time_axis(mel_db.astype(np.float32), max_frames)
