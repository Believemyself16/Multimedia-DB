import numpy as np
import librosa

def calculate_spectral_centroid(audio_path, sr=22050):
    """
    Tính trọng tâm phổ của tệp âm thanh.

    Parameters:
        audio_path (str): Đường dẫn đến tệp âm thanh.
        sr (int): Tần số lấy mẫu.

    Returns:
        float: Trọng tâm phổ.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return np.mean(centroid)
