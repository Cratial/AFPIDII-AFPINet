from torchvision import datasets
import torchvision.transforms as tv_transforms
from base import BaseDataLoader
import torch.utils.data as data
import librosa
import os
import pandas as pd
import torch
import numpy as np
import pathlib
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, TimeMask
from utils.spec_timeshift_transform import TimeShift
from sklearn.model_selection import train_test_split


def load_dataframe(path):
    df = pd.read_csv(path)
    return df.to_dict('records')


def load_numpy(path):
    try:
        data = np.load(path)
    except Exception as e:
        print(e)
        return None
    return data


class VariousTrainingDataset(data.Dataset):
    """
        For the person identification with various training samples
    """

    def __init__(self, data_dir, wave_cache_dir, hand_cache_dir, spec_cache_dir, w_shift=False, h_shift=False, s_shift=False):
        # self.data_arr = load_dataframe(data_dir)
        dataset_name = data_dir.rsplit("/", 1)[1].rsplit('_')[-1].rsplit('.')[0]
        if dataset_name == 'train':
            ori_data_meta = pd.read_csv(data_dir)
            new_data_index = []
            for i in range(1, 14):
                sub_name = f'S{str(i).zfill(2)}'
                sub_label_index = ori_data_meta[ori_data_meta.person_label == sub_name].index.tolist()
                new_data_index.append(sub_label_index)
            # x_tr ,x_te, y_tr, y_te = train_test_split(ori_data_meta, person_label, test_size=1300, random_state=13)

            # randomly sampling
            np.random.seed(13)
            train_sample_per_sub = 50
            train_samp_index = np.random.permutation(np.arange(1055))[:train_sample_per_sub]
            # train_samp_index = np.arange(1200)[:train_sample_per_sub]
            new_data_meta = []
            for i in range(13):
                new_data_meta.append(ori_data_meta.loc[np.array(new_data_index[i])[train_samp_index]])

            self.data_arr = pd.concat(new_data_meta, axis=0).to_dict('records')
        else:
            self.data_arr = load_dataframe(data_dir)
        # --->>>>>>>>>>>-----------<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.data_dir = data_dir.rsplit("/", 1)[0]  # dataset directory
        self.w_shift = w_shift
        self.h_shift = h_shift
        self.s_shift = s_shift
        self.waveform_cachedir = wave_cache_dir
        self.handcraft_cachedir = hand_cache_dir
        self.spectrogram_cachedir = spec_cache_dir

        # waveform augmentation
        wave_transforms = []
        if w_shift:
            wave_transforms.append(Shift(min_fraction=-0.5, max_fraction=0.5, p=1.0))

        if len(wave_transforms) == 0:
            self.wave_transform = None
        else:
            self.wave_transform = Compose(wave_transforms)

        # handcraft augmentation
        # hcraft_transforms = []
        # if h_shift:
        #     hcraft_transforms.append(TimeShift())
        #
        # if len(hcraft_transforms) == 0:
        #     self.hcraft_transform = None
        # else:
        #     self.hcraft_transform = tv_transforms.Compose(hcraft_transforms)
        self.hcraft_transform = None

        # spectrogram augmentation
        spec_transforms = []
        if s_shift:
            spec_transforms.append(TimeShift())

        if len(spec_transforms) == 0:
            self.spec_transform = None
        else:
            self.spec_transform = tv_transforms.Compose(spec_transforms)

    def __len__(self):
        return len(self.data_arr)

    def __getitem__(self, idx):
        item = self.data_arr[idx]
        person_label = item['person_label']
        label_encoded = int(person_label[1:])-1
        dataset_dir = self.data_dir

        wave = load_numpy(os.path.join(dataset_dir, item['filename_wave']))
        hcraft = load_numpy(os.path.join(dataset_dir, item['filename_hcraft']))
        spec = load_numpy(os.path.join(dataset_dir, item['filename_spec']))

        # Add transforms
        if self.wave_transform is not None:
            wave = self.wave_transform(wave, sample_rate=16000)

        if self.hcraft_transform is not None:
            hcraft = self.hcraft_transform(hcraft)

        if self.spec_transform is not None:
            spec = self.spec_transform(spec)

        #         return spec, scene_encoded
        return {"wave": wave, "hcraft": hcraft, "spec": spec}, label_encoded


class MultiModalAugmentationDataset(data.Dataset):
    """
        For the baseline performance of AFPI-Net
    """

    def __init__(self, data_dir, wave_cache_dir, hand_cache_dir, spec_cache_dir, w_shift=False, h_shift=False, s_shift=False):
        self.data_arr = load_dataframe(data_dir)
        self.data_dir = data_dir.rsplit("/", 1)[0]  # dataset directory
        self.w_shift = w_shift
        self.h_shift = h_shift
        self.s_shift = s_shift
        self.waveform_cachedir = wave_cache_dir
        self.handcraft_cachedir = hand_cache_dir
        self.spectrogram_cachedir = spec_cache_dir

        # waveform augmentation
        wave_transforms = []
        if w_shift:
            wave_transforms.append(Shift(min_fraction=-0.5, max_fraction=0.5, p=1.0))

        if len(wave_transforms) == 0:
            self.wave_transform = None
        else:
            self.wave_transform = Compose(wave_transforms)

        # handcraft augmentation
        # hcraft_transforms = []
        # if h_shift:
        #     hcraft_transforms.append(TimeShift())
        #
        # if len(hcraft_transforms) == 0:
        #     self.hcraft_transform = None
        # else:
        #     self.hcraft_transform = tv_transforms.Compose(hcraft_transforms)
        self.hcraft_transform = None

        # spectrogram augmentation
        spec_transforms = []
        if s_shift:
            spec_transforms.append(TimeShift())

        if len(spec_transforms) == 0:
            self.spec_transform = None
        else:
            self.spec_transform = tv_transforms.Compose(spec_transforms)

    def __len__(self):
        return len(self.data_arr)

    def __getitem__(self, idx):
        item = self.data_arr[idx]
        person_label = item['person_label']
        label_encoded = int(person_label[1:])-1
        dataset_dir = self.data_dir

        wave = load_numpy(os.path.join(dataset_dir, item['filename_wave']))
        hcraft = load_numpy(os.path.join(dataset_dir, item['filename_hcraft']))
        spec = load_numpy(os.path.join(dataset_dir, item['filename_spec']))

        # Add transforms
        if self.wave_transform is not None:
            wave = self.wave_transform(wave, sample_rate=16000)

        if self.hcraft_transform is not None:
            hcraft = self.hcraft_transform(hcraft)

        if self.spec_transform is not None:
            spec = self.spec_transform(spec)

        #         return spec, scene_encoded
        return {"wave": wave, "hcraft": hcraft, "spec": spec}, label_encoded

