import librosa

def extract_chroma(audio_path):
    # Đọc file âm thanh và tính chroma
    y, sr = librosa.load(audio_path)
    
    #trích xuất đặc trưng chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr) #y là mảng chứa dữ liệu âm thanh, sr là tần số lấy mẫu
    
    return chroma
