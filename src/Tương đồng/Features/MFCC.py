import numpy as np
import librosa

def extract_mfcc(audio_path, sr=22050, n_mfcc=13):
    """
    Trích xuất đặc trưng MFCC từ tệp âm thanh.

    Parameters:
        audio_path (str): Đường dẫn đến tệp âm thanh.
        sr (int): Tần số lấy mẫu.
        n_mfcc (int): Số lượng hệ số MFCC.

    Returns:
        np.ndarray: Ma trận chứa các hệ số MFCC.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs.T
