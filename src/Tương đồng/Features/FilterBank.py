import librosa

def extract_filterbank(audio_path):
    # Đọc file âm thanh
    y, sr = librosa.load(audio_path)
    
    # Trích xuất đặc trưng Filter Bank
    filterbank_features = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    
    return filterbank_features
