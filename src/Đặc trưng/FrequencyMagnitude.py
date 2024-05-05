#frequency magnitude xác định độ lớn của các thành phần tần số trong âm thanh

import os
import matplotlib.pyplot as plt
import librosa, librosa.display
import IPython.display as ipd
import numpy as np

# Mảng X_mag trả về một mảng bao gồm: 
# Giá trị của phần tử = giá trị magnitude, 
# vị trí của phần tử đó = hz tương ứng => Dùng 2 mảng để lưu lại 2 giá trị đó

def funcFrequencyMagnitude(path):
    file, sr = librosa.load(path, duration = 6) #load file âm thanh vào librosa
    
    # dùng biến đổi fourier để chuyển đổi tín hiệu từ miền thời gian sang miền tần số 
    X = np.fft.fft(file) # mảng X là mảng chứa dãy tần số và mật độ của nó
    X_mag = np.absolute(X) # giá trị tuyệt đối của phổ tần số (phần số thực) 

    # tạo mảng f (khoảng tần số từ 0 tới tần số sr)
    f = np.linspace(0, sr, len(X_mag))
    f_bins = int(len(X_mag))  

    #lấy 1 phần của mảng f và mảng X_mag
    f[:f_bins], X_mag[:f_bins]

    # mảng ghi tần số có mức độ xuất hiện lớn nhất
    freq = []
    freq = [0 for i in range(6)]

    #mảng ghi mức độ xuất hiện
    magnitude = []
    magnitude = [0 for i in range(6)]

    position = 0
    max = 0
    pos = 0

    #duyệt tìm tần số có mức độ xuất hiện lớn nhất
    for i in range(0, len(X_mag)):        
        if(X_mag[i] > max):
            max = X_mag[i]
            pos = i
        if(i> 0):
            if( (i % int(len(X_mag)/6)) == 0 and i != len(X_mag)-1):            
                freq[position] = pos
                magnitude[position] = max
                position += 1
                max = 0 
            if(i == len(X_mag)-1):            
                freq[position] = pos
                magnitude[position] = max
    
    #kết hợp tần số và độ lớn của tần số đó
    pairs = list(zip(freq, magnitude)) 
    return pairs
