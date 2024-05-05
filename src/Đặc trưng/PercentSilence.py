#Tính tỉ lệ các phần im lặng trong file (nhỏ hơn ngưỡng nghe được)
import librosa.display
import matplotlib.pyplot as plt

def Find_Max(arr, f, l): #mảng số nguyên bắt đầu ở chỉ mục f, kết thúc ở l
    max = arr[f] #max là phần tử đầu tiên trong mảng
    
    for i in range(f, l):
        if max < arr[i]:
            max = arr[i]
            
    return max

def Find_Min(arr, f, l):
    min = arr[f] #min là phần tử đầu tiên trong mảng
    
    for i in range(f, l):
        if min > arr[i]:
            min = arr[i]
            
    return min

def KhoangLang(arr, f, l):
    max = Find_Max(arr, f, l)
    min = Find_Min(arr, f, l)
    tb = (max + min) / 2 #trung bình là max và min của khoảng
    nguong = tb * 0.3 #tìm ngưỡng nghe được (30% của trung bình)
    dem = 0
    
    for i in range(f, l):
        if arr[i] > nguong:
            dem = dem + 1

    return 100 - (dem / (l - f)) * 100 #tỉ lệ của khoảng im lặng

def funcPercentSilence(path):
    y, sr = librosa.load(path)  #y là biên độ theo thời gian, sr là tần số lấy mẫu
    result = []
    for i in range(0, 6): #trong khoảng thời gian 6s
        result.append(KhoangLang(y, i * sr, (i + 1) * sr)) #tính tỉ lệ khoảng im lặng của mỗi phần, lưu vào mảng
    result.append(KhoangLang(y, 6, len(y)))
    return result
