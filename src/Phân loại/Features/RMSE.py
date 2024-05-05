#RMSE Root Mean Square Error để đo lường sự khác biệt giữa tín hiệu âm thanh gốc và tín hiệu âm thanh được xử lý
#xử lý tín hiệu và kiểm tra chất lượng âm thanh, cung cấp thông tin về mức độ năng lượng của tín hiệu âm thanh

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd

def funcRMSE(path):
    file1, sr = librosa.load(path, duration = 6) #file âm thanh, tần số lấy mẫu, độ dài 6s
    FRAME_SIZE = 1024 #kích thước frame
    HOP_LENGTH = 512 #khoảng cách giữa các frame
    
    rms = librosa.feature.rms(file1, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0] #RMS là mảng năng lượng trung bình mặc định mà librosa trả về
    arr = np.array_split(rms, 6) #chia mảng rms thành 6 phần

    result = [0 for i in range(6)] 
    window = 0

    #phần trăm khác nhau
    for i in range(len(arr)):
        sum = 0
        count = 0
        for j in range(len(arr[i])-1):  
            if(arr[i][j] != 0):
                #trung bình cộng hiệu 2 giá trị cạnh nhau / giá trị tiếp theo để tính xem năng lượng trung bình ở giá trị tiếp theo lệch bao nhiêu % so với giá trị hiện tại 
                sum += ((abs(arr[i][j] - arr[i][j+1]))/arr[i][j])*100 
            else:
                sum += 0
            count += 1
        avg = sum/count
        result[window] = avg
        window += 1
    return result
