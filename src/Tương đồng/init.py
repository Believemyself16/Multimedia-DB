from Features.MFCC import extract_mfcc
from Features.ZeroCrossingRate import calculate_zcr
from Features.SpectralCentroid import calculate_spectral_centroid
from Features.Chroma import extract_chroma
from scipy.spatial.distance import cosine
from scipy.io import wavfile
import os

# Load và trích xuất thuộc tính của tất cả các file âm thanh trong bộ dữ liệu
data_path = "C:\\Users\\Asus\\OneDrive - m2xk\\Desktop\\Multimedia-DB\\src\\AudioFileCollection"
database = {}

for filename in os.listdir(data_path):
    audio_path = os.path.join(data_path, filename)
    frequency_sampling, audio_signal = wavfile.read(audio_path)
    mfcc_features = extract_mfcc(audio_signal, frequency_sampling)
    zcr = calculate_zcr(audio_signal)
    spectral_centroid = calculate_spectral_centroid(audio_signal, frequency_sampling)
    chroma_features = extract_chroma(audio_signal, frequency_sampling)
    database[filename] = {"MFCC": mfcc_features, "ZCR": zcr, "SpectralCentroid": spectral_centroid, "Chroma": chroma_features}

# Trích xuất thuộc tính từ file âm thanh mới
new_audio_path = "path_to_your_new_audio_file"
frequency_sampling, new_audio_signal = wavfile.read(new_audio_path)
new_mfcc_features = extract_mfcc(new_audio_signal, frequency_sampling)
new_zcr = calculate_zcr(new_audio_signal)
new_spectral_centroid = calculate_spectral_centroid(new_audio_signal, frequency_sampling)
new_chroma_features = extract_chroma(new_audio_signal, frequency_sampling)

# Tìm kiếm file giống nhất
similarities = {}
for filename, features in database.items():
    mfcc_similarity = 1 - cosine(features["MFCC"].mean(axis=0), new_mfcc_features.mean(axis=0))
    zcr_similarity = 1 - abs(features["ZCR"] - new_zcr)
    spectral_centroid_similarity = 1 - abs(features["SpectralCentroid"] - new_spectral_centroid)
    chroma_similarity = 1 - cosine(features["Chroma"].mean(axis=0), new_chroma_features.mean(axis=0))
    similarity_score = (mfcc_similarity + zcr_similarity + spectral_centroid_similarity + chroma_similarity) / 4
    similarities[filename] = similarity_score

# Sắp xếp và hiển thị các file giống nhất
top_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]
for match in top_matches:
    print("File:", match[0], " - Similarity Score:", match[1])
