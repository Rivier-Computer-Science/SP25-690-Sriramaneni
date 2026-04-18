import librosa
import numpy as np

class AudioFeatureExtractor:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate

    def mfcc(self, path, n_mfcc=40):
        """
        Stable MFCC extraction for ML pipeline
        """
        audio, sr = librosa.load(path, sr=self.sample_rate)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

        # normalize into fixed vector
        return np.mean(mfcc.T, axis=0)

    def mel_spectrogram(self, path):
        """
        CNN/Transformer input representation
        """
        audio, sr = librosa.load(path, sr=self.sample_rate)

        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        return librosa.power_to_db(mel)