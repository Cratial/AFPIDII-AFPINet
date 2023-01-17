import sys

import librosa
import pandas as pd
import os
import numpy as np
import torch
from tqdm import tqdm
import pathlib
import h5py
import torchaudio
import matplotlib.pyplot as plt

ori_data_dir = '../acoustic_footstep/AFPID-Raw'

train_df = pd.DataFrame(columns=["filename_wave", "filename_hcraft", "filename_spec", "person_label"])
test_df = pd.DataFrame(columns=["filename_wave", "filename_hcraft", "filename_spec", "person_label"])

# generated dataset path
afpid_dataset_dir = "../acoustic_footstep/AFPID_FE2/"

wave_cachedir = "waveform"
hcraft_cachedir = "handcraft"
spec_cachedir = "spectrogram"

# load the footstep events separation sampling points
f = h5py.File(os.path.join(ori_data_dir, 'fse_sep_tim_vec_cell.mat'), 'r')
fse_sep_tim_vec_cell = [f[element[0]][:] for element in f['fse_sep_tim_vec']]

# sub1, sub2, ...., sub13.
ori_sub_dir = sorted(os.listdir(ori_data_dir))


def fea_normalize(fea_ver):
    normalized_fea_ver = []
    for _, fea in enumerate(fea_ver):
        normalized_fea = (fea - np.mean(fea)) / np.std(fea)
        normalized_fea_ver.append(normalized_fea)
    return normalized_fea_ver


sample_num = 1

# for i in range(13):
i = 0
sub_dir = os.path.join(ori_data_dir, ori_sub_dir[i])
audio_dir = sorted(os.listdir(sub_dir))
print("========subject: {}========".format(i+1))

# iterate over each recorded audio file
# for j in range(len(audio_dir)):
j = 0
# select the corresponding FEs separation sampling points
fse_sep_tim_vec = np.squeeze(fse_sep_tim_vec_cell[i * 6 + j]).astype(int)
# load audio
audio_fil = os.path.join(sub_dir, audio_dir[j])
input_audio, sr = librosa.load(audio_fil, sr=16000)
print("session: {}".format(j + 1))

# generate acoustic footstep audio samples
# j starts from 5 to skip the silence beginning

# for k in range(4, len(fse_sep_tim_vec) - 2):
k = 25  # 25, 26,27, 28, 29
sample_audio = input_audio[fse_sep_tim_vec[k]:fse_sep_tim_vec[k + 2]]
# plt.plot(sample_audio)
# plt.show()

# padding the original audio wave to a fixed length for feature extraction branch 1, 3,
# but not for branch 2 (lstm, or gru). fixed window at 1.2s or 1.6s
fixed_sample_len = int(sr * 1.4)
padded = np.zeros(fixed_sample_len, dtype='float32')
wave = sample_audio[:fixed_sample_len]
padded[0:len(wave)] = wave  # ==> for saving

# a. original waveform for feature extraction branch 1.
wave_form = padded

# window length belongs to 20-40 ms, here we choose 40ms, with an overlapping length of 30ms.
win_len = int(0.04 * sr)
step_len = int(0.02 * sr)

# c. mel-spectrogram map for feature extraction branch 3.
# spec = librosa.feature.melspectrogram(padded, n_fft=2048, hop_length=256, n_mels=128, sr=48000, fmin=0,
#                                       fmax=24000)
spec_map = librosa.feature.melspectrogram(sample_audio, sr=sr, n_fft=2048, hop_length=step_len, n_mels=64,
                                          fmin=0, fmax=8000)
spec_map = np.log(spec_map)

# z-score normalization
std = spec_map.std()
mean = spec_map.mean()
spec_map = (spec_map - mean) / std

# padding the spectrogram
m = spec_map.shape[0]
fixed_pad_len = int(fixed_sample_len/step_len)
spec_map_padded = np.zeros((m, fixed_pad_len), dtype='float32')
tmp = spec_map[:, :fixed_pad_len]
spec_map_padded[:, 0:tmp.shape[1]] = tmp  # ==> for saving

# b. hand-crafted audio features for feature extraction branch 2.
# Compute root-mean-square (RMS)
fea_rms = librosa.feature.rms(sample_audio, frame_length=win_len, hop_length=step_len)

# zero-crossing rate
fea_zcr = librosa.feature.zero_crossing_rate(sample_audio, frame_length=win_len, hop_length=step_len)

# # spectrogram base
# stft_S = (np.abs(
#     librosa.stft(padded, n_fft=2048, hop_length=step_len, win_length=win_len, window="hann", center=True,
#                  pad_mode="reflect")))**1.0  # 1 for energy, 2 for power, etc.
# mel-spectrogram
mel_spec_map = librosa.feature.melspectrogram(sample_audio, sr=sr, n_fft=2048, hop_length=step_len,
                                              n_mels=128, fmin=0, fmax=8000)

# spectral centroid
fea_spectral_centroid = librosa.feature.spectral_centroid(sample_audio, sr=16000, S=mel_spec_map,
                                                          n_fft=2048, hop_length=step_len,
                                                          win_length=win_len)
# spectral_centroid = librosa.feature.spectral_centroid(padded, sr=16000, n_fft=2048,
#                                                       hop_length=step_len, win_length=win_len)

# spectral_flatness
fea_spectral_flatness = librosa.feature.spectral_flatness(sample_audio, n_fft=2048, hop_length=step_len,
                                                          win_length=win_len)

# Mel-frequency cepstral coefficients (MFCCs)
fea_mfcc = librosa.feature.mfcc(sample_audio, sr=sr, S=librosa.power_to_db(mel_spec_map), n_mfcc=30,
                                dct_type=2, norm="ortho")

# MFCCs-delta
fea_mfcc_delta = librosa.feature.delta(fea_mfcc)

# pitch
# fea_pitch = torchaudio.functional.detect_pitch_frequency(torch.tensor(padded), sr)

# normalize
fea_normalized = fea_normalize(
    [fea_rms, fea_zcr, fea_spectral_centroid, fea_spectral_flatness, fea_mfcc, fea_mfcc_delta])

# concat hand-crafted audio feature
handcraft_fea = np.vstack(fea_normalized)  # ==> for saving

# padding the hand-craft feature to the same dimension
m = handcraft_fea.shape[0]
hcraft_fea_padded = np.zeros((m, fixed_pad_len), dtype='float32')
tmp = handcraft_fea[:, :fixed_pad_len]
hcraft_fea_padded[:, 0:tmp.shape[1]] = tmp  # ==> for saving

# ==========>> visualization <===========
footstep_audio_sample = sample_audio
fas_t = np.arange(0,len(footstep_audio_sample))/sr
padded_wave = padded
padded_hcraft = hcraft_fea_padded
padded_spec = spec_map_padded

plt.figure(1,[9,3])
print("the {}th sample".format(k))
plt.plot(fas_t, footstep_audio_sample)
plt.xlabel('Time (S)')
plt.ylabel('Amplitude')
plt.show()

