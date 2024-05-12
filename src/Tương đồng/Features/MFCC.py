import librosa

def extract_mfcc(audio_path):
    # Đọc file âm thanh
    y, sr = librosa.load(audio_path)
    
    # Trích xuất đặc trưng MFCC
    mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    return mfcc_features
