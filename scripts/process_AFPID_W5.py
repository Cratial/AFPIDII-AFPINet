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

ori_data_dir = "../data//acoustic_footstep/AFPID-Raw"

train_df = pd.DataFrame(columns=["filename_wave", "filename_hcraft", "filename_spec", "person_label"])
test_df = pd.DataFrame(columns=["filename_wave", "filename_hcraft", "filename_spec", "person_label"])

# generated dataset path
afpid_dataset_dir = "../data/acoustic_footstep/AFPID_W5"

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
for i in range(13):
    sub_dir = os.path.join(ori_data_dir, ori_sub_dir[i])
    audio_dir = sorted(os.listdir(sub_dir))
    print("========subject: {}========".format(i + 1))
    # iterate over each recorded audio file
    for j in range(len(audio_dir)):
        # select the corresponding FEs separation sampling points
        fse_sep_tim_vec = np.squeeze(fse_sep_tim_vec_cell[i * 6 + j]).astype(int)
        # load audio
        audio_fil = os.path.join(sub_dir, audio_dir[j])
        input_audio, sr = librosa.load(audio_fil, sr=16000)
        print("session: {}".format(j + 1))

        # generate acoustic footstep audio samples
        # j starts from 5 to skip the silence beginning
        start_indx = fse_sep_tim_vec[4]
        ended_indx = fse_sep_tim_vec[len(fse_sep_tim_vec)-1]

        # for two consecutive footstep events
        # fixed_sample_len = int(sr * 1.4)
        # for a single footstep event
        # fixed_sample_len = int(sr * 0.7)
        fixed_sample_len = int(sr * 0.5)
        end_indx = start_indx + fixed_sample_len
        # while end_indx <= ended_indx:
        while end_indx+fixed_sample_len <= ended_indx:
            # for k in range(4, len(fse_sep_tim_vec) - 2):
            # for two consecutive footstep events
            # sample_audio = input_audio[fse_sep_tim_vec[k]:fse_sep_tim_vec[k + 2]]

            # for a single footstep event
            # sample_audio = input_audio[fse_sep_tim_vec[k]:fse_sep_tim_vec[k + 1]]
            # plt.plot(sample_audio)
            # plt.show()

            sample_audio = input_audio[start_indx:end_indx]
            start_indx = end_indx
            end_indx = start_indx + fixed_sample_len

            padded = sample_audio

            # a. original waveform for feature extraction branch 1.
            wave_form = padded

            # window length belongs to 20-40 ms, here we choose 40ms, with an overlapping length of 30ms.
            # fot two consecutive footstep events
            # win_len = int(0.04 * sr)
            # step_len = int(0.02 * sr)

            # for a single footstep event
            win_len = int(0.02 * sr)
            step_len = int(0.01 * sr)

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
            fixed_pad_len = int(fixed_sample_len / step_len)
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

            # saving waveform, handcraft feature, spectrogram
            sample_fname = audio_fil.split('/')[-1][1:5]
            session = int(sample_fname[-1])

            fname_wave = os.path.join(wave_cachedir, f"afpid_w5_s{sample_fname}_waveform_{sample_num}.npy")
            fname_hcraft = os.path.join(hcraft_cachedir, f"afpid_w5_s{sample_fname}_handcraft_{sample_num}.npy")
            fname_spec = os.path.join(spec_cachedir, f"afpid_w5_s{sample_fname}_spectrogram_{sample_num}.npy")
            sample_num += 1

            # saving
            # np.save(os.path.join(self.spectrogram_cachedir, item['filename_audio']).replace(".wav", ".npy"), spec)
            np.save(os.path.join(afpid_dataset_dir, fname_wave), wave_form)
            np.save(os.path.join(afpid_dataset_dir, fname_hcraft), hcraft_fea_padded)
            np.save(os.path.join(afpid_dataset_dir, fname_spec), spec_map_padded)

            if session > 4:
                test_df = test_df.append(
                    {"filename_wave": fname_wave, "filename_hcraft": fname_hcraft, "filename_spec": fname_spec,
                     "person_label": f"S{sample_fname[:2]}"}, ignore_index=True)
            else:
                train_df = train_df.append(
                    {"filename_wave": fname_wave, "filename_hcraft": fname_hcraft, "filename_spec": fname_spec,
                     "person_label": f"S{sample_fname[:2]}"}, ignore_index=True)

train_df.to_csv(os.path.join(afpid_dataset_dir, 'AFPID_W5_train.csv'))
test_df.to_csv(os.path.join(afpid_dataset_dir, 'AFPID_W5_test.csv'))

print("Finished to create the AFPID_W5 dataset with two consecutive footstep events to form on sample!")

