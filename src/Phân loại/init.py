# chạy chương trình, dùng 4 đặc trưng trên, lấy file âm thanh và xử lí, lưu vào file csv

from Features.Pitch import funcPitch
from aubio import pitch
from Features.RMSE import funcRMSE
from Features.PercentSilence import funcPercentSilence
from Features.FrequencyMagnitude import funcFrequencyMagnitude

import os

listsubpath = []
for x in os.walk("E:/Learn/tool_instrument_voice_recognition/src/File âm thanh"):
    listsubpath.append(x[0].replace("\\", "/"))
listsubpath.pop(0)