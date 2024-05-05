# Đặc trưng: độ cao thấp của âm thanh
# Tính toán cao độ Pitch của file âm thanh

import numpy as np
from aubio import source, pitch


def funcPitch(path, pitch):
    win_s = 4096 # kích thước của sổ FFT
    hop_s = 512 # kích thước bước nhảy giữa các cửa sổ
    samplerate = 16000 # tần số lấy mẫu
    
    # tạo đối tượng âm thanh
    s = source(path, samplerate, hop_s)
    
    samplerate = s.samplerate 
    tolerance = 0.8 # ngưỡng chấp nhận được của kết quả

    pitch_o = pitch("yin", win_s, hop_s, samplerate) # tạo đối tượng pitch
    pitch_o.set_unit("midi") # đơn vị
    pitch_o.set_tolerance(tolerance)

    # tạo mảng lưu trữ giá trị Pitch và độ tin cậy
    pitches = []
    confidences = []
    
    # tính tổng số frame của file
    total_frames = 0
    while True:
        samples, read = s() # lấy mẫu từ đối tượng s
        
        # tính pitch và độ tin cậy, lưu vào mảng
        pitch = pitch_o(samples)[0]
        pitches += [pitch]
        confidence = pitch_o.get_confidence()
        confidences += [confidence]
        total_frames += read
        if read < hop_s: break

    # tạo mảng kết quả, tính số bước cho mỗi phân đoạn
    result = []
    step = int(len(pitches)/7)

    # chia các giá trị Pitch thành các phân đoạn, tính toán trung bình Pitch cho mỗi phân đoạn
    for i in range(0, 7, 1): # chia thành 7 đoạn
        max = step*(i+1)
        if(i == 6):
            max = len(pitches) - 1
        pitchLocal = []
        for j in range(step*i, max, 1):
            pitchLocal.append(pitches[j])
        result.append(np.array(pitchLocal).mean())  # tính trung bình 

    # trả về mảng kết quả
    return result