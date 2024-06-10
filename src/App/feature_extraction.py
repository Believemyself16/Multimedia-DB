import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import dct

# Hàm tiền xử lý tín hiệu
def pre_emphasis(signal, pre_emphasis_coefficient=0.97):
    return np.append(signal[0], signal[1:] - pre_emphasis_coefficient * signal[:-1])

# Hàm chia tín hiệu thành các khung
def framing(signal, sample_rate, frame_size=0.025, frame_stride=0.01):
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames

# Hàm áp dụng cửa sổ Hamming
def hamming_window(frames):
    return frames * np.hamming(frames.shape[1])

# Hàm tính phổ FFT
def fft_spectrum(frames, NFFT=512):
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))
    return pow_frames

# Hàm tính các bộ lọc Mel
def mel_filterbanks(pow_frames, sample_rate, nfilt=40, NFFT=512):
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    return filter_banks

# Hàm tính toán MFCC
def mfcc(signal, sample_rate, pre_emphasis_coefficient=0.97, frame_size=0.025, frame_stride=0.01, NFFT=512, nfilt=40, num_ceps=13, cep_lifter=22):
    emphasized_signal = pre_emphasis(signal, pre_emphasis_coefficient)
    frames = framing(emphasized_signal, sample_rate, frame_size, frame_stride)
    frames = hamming_window(frames)
    pow_frames = fft_spectrum(frames, NFFT)
    filter_banks = mel_filterbanks(pow_frames, sample_rate, nfilt, NFFT)
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(num_ceps + 1)]
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift
    return mfcc

# Hàm trích xuất đặc trưng MFCC từ tệp âm thanh
def extract_features(file_path, n_mfcc=13):
    sample_rate, signal = wav.read(file_path)
    mfcc_features = mfcc(signal, sample_rate, num_ceps=n_mfcc)
    mfcc_mean = np.mean(mfcc_features, axis=0)
    return mfcc_mean
