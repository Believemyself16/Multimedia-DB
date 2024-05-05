import numpy as np
import librosa

def extract_chroma(audio_path, sr=22050):
    """
    Trích xuất đặc trưng Chroma từ tệp âm thanh.

    Parameters:
        audio_path (str): Đường dẫn đến tệp âm thanh.
        sr (int): Tần số lấy mẫu.

    Returns:
        np.ndarray: Ma trận chứa các hệ số Chroma.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return chroma.T
