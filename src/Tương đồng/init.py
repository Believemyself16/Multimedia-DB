from Features.Chroma import extract_chroma
from Features.FilterBank import extract_filterbank
from Features.MFCC import extract_mfcc

import pandas as pd
import librosa
import os
os.chdir(r"C:\Users\Asus\OneDrive - m2xk\Desktop\Multimedia-DB\src\AudioFileCollection")

# Thư mục chứa các file âm thanh
audio_dir = "."

# Khởi tạo một list để chứa dữ liệu
data = []

# Lặp qua các file âm thanh trong thư mục
for filename in os.listdir(audio_dir):
    audio_path = os.path.join(audio_dir, filename)
    # Trích xuất các đặc trưng âm thanh
    y, sr = librosa.load(audio_path)
    # chroma_features = extract_chroma(audio_path)
    # filterbank_features = extract_filterbank(audio_path)
    mfcc_features = extract_mfcc(audio_path)
    mfcc_features = mfcc_features[:13, :]
    # Thêm thông tin vào list data
    data.append({
        'filename': filename,
        # 'chroma': chroma_features.tolist(),
        'mfcc': mfcc_features.T.tolist(),
        # 'filterbank': filterbank_features.tolist()
    })

# Tạo DataFrame từ list data
df = pd.DataFrame(data)

# Lưu DataFrame vào file CSV
df.to_csv('audio_features.csv', index=False)