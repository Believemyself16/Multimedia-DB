# from flask import Flask, render_template, request
# import os
# import librosa
# import numpy as np
# import pandas as pd
# from scipy.spatial.distance import euclidean
# from pydub import AudioSegment

# app = Flask(__name__)

# # Đường dẫn tới thư mục chứa các tệp âm thanh của cơ sở dữ liệu
# database_audio_folder = r"C:\Users\Asus\OneDrive - m2xk\Desktop\Multimedia-DB\src\TrimmedAudioFile"

# # Đường dẫn tới tệp CSV chứa các đặc trưng của cơ sở dữ liệu
# features_csv = r"C:\Users\Asus\OneDrive - m2xk\Desktop\Multimedia-DB\src\TrimmedAudioFile\audio_features.csv"

# # Đọc các đặc trưng đã lưu từ CSV
# features_df = pd.read_csv(features_csv)

# # Hàm trích xuất đặc trưng MFCC từ tệp âm thanh
# def extract_features(file_path, n_mfcc=13):
#     y, sr = librosa.load(file_path, sr=None)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#     mfcc_mean = np.mean(mfcc, axis=1)
#     return mfcc_mean

# # Tính toán khoảng cách Euclidean giữa tệp đầu vào và từng tệp trong DataFrame
# def calculate_distances(input_features, features_df, database_folder):
#     distances = []
#     for index, row in features_df.iterrows():
#         filename = row['filename']
#         file_path = os.path.join(database_folder, filename)
#         feature_vector = row[1:].values.astype(float)  # Lấy đặc trưng MFCC từ DataFrame
#         distance = euclidean(input_features, feature_vector)
#         distances.append((filename, distance))
#     return distances

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         # Lấy file âm thanh từ request
#         audio_file = request.files['audio']
        
#         # Lưu file âm thanh vào thư mục tạm thời
#         temp_path = 'temp_audio.wav'
#         audio_file.save(temp_path)
        
#         # Trích xuất đặc trưng từ file âm thanh đầu vào
#         input_features = extract_features(temp_path)
        
#         # Tính toán khoảng cách và tìm 3 tệp âm thanh giống nhất
#         distances = calculate_distances(input_features, features_df, database_audio_folder)
#         distances.sort(key=lambda x: x[1])  # Sắp xếp theo khoảng cách tăng dần
        
#         # Lấy 3 tệp âm thanh giống nhất
#         top_3_similar_files = distances[:3]
        
#         # Chuyển đổi tệp âm thanh sang định dạng WAV để hiển thị trên giao diện
#         similar_audio_paths = []
#         for filename, _ in top_3_similar_files:
#             file_path = os.path.join(database_audio_folder, filename)
#             sound = AudioSegment.from_file(file_path)
#             temp_audio_path = f'temp_{filename}.wav'
#             sound.export(temp_audio_path, format='wav')
#             similar_audio_paths.append(file_path)
        
#         return render_template('index.html', similar_audio_paths=similar_audio_paths)
    
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for
import os
import librosa
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from pydub import AudioSegment

app = Flask(__name__)

# Đường dẫn tới thư mục chứa các tệp âm thanh của cơ sở dữ liệu
database_audio_folder = r"C:\Users\Asus\OneDrive - m2xk\Desktop\Multimedia-DB\src\TrimmedAudioFile"

# Đường dẫn tới tệp CSV chứa các đặc trưng của cơ sở dữ liệu
features_csv = r"C:\Users\Asus\OneDrive - m2xk\Desktop\Multimedia-DB\src\TrimmedAudioFile\audio_features.csv"

# Đọc các đặc trưng đã lưu từ CSV
features_df = pd.read_csv(features_csv)

# Thư mục lưu trữ các tệp âm thanh được tải lên từ giao diện
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Kiểm tra và tạo thư mục UPLOAD_FOLDER nếu nó không tồn tại
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Hàm trích xuất đặc trưng MFCC từ tệp âm thanh
def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

# Tính toán khoảng cách Euclidean giữa tệp đầu vào và từng tệp trong DataFrame
def calculate_distances(input_features, features_df, database_folder):
    distances = []
    for index, row in features_df.iterrows():
        filename = row['filename']
        file_path = os.path.join(database_folder, filename)
        feature_vector = row[1:].values.astype(float)  # Lấy đặc trưng MFCC từ DataFrame
        distance = euclidean(input_features, feature_vector)
        distances.append((filename, distance))
    return distances

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Kiểm tra xem có file âm thanh nào được tải lên không
        if 'audio' not in request.files:
            return redirect(request.url)
        
        audio_file = request.files['audio']
        
        # Kiểm tra xem tệp có tên không
        if audio_file.filename == '':
            return redirect(request.url)
        
        # Lưu trữ tệp âm thanh vào thư mục UPLOAD_FOLDER
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
        audio_file.save(audio_path)
        
        # Trích xuất đặc trưng từ file âm thanh đầu vào
        input_features = extract_features(audio_path)
        
        # Tính toán khoảng cách và tìm 3 tệp âm thanh giống nhất
        distances = calculate_distances(input_features, features_df, database_audio_folder)
        distances.sort(key=lambda x: x[1])  # Sắp xếp theo khoảng cách tăng dần
        
        # Lấy 3 tệp âm thanh giống nhất
        top_3_similar_files = distances[:3]
        
        # Chuyển đổi tệp âm thanh sang định dạng WAV để hiển thị trên giao diện
        similar_audio_paths = []
        for filename, _ in top_3_similar_files:
            file_path = os.path.join(database_audio_folder, filename)
            sound = AudioSegment.from_file(file_path)
            temp_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{filename}.wav')
            sound.export(temp_audio_path, format='wav')
            similar_audio_paths.append(temp_audio_path)
        
        return render_template('index.html', similar_audio_paths=similar_audio_paths)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
