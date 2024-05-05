# đo lường tính đồng nhất của dữ liệu âm thanh

import numpy as np
import librosa

def calculate_zcr(audio_path, frame_length=2048, hop_length=512):
    """
    Tính tỷ lệ vượt qua không của tệp âm thanh.

    Parameters:
        audio_path (str): Đường dẫn đến tệp âm thanh.
        frame_length (int): Độ dài khung (số mẫu).
        hop_length (int): Bước nhảy giữa các khung (số mẫu).

    Returns:
        float: Tỷ lệ vượt qua không.
    """
    y, sr = librosa.load(audio_path)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)
    return np.mean(zcr)
