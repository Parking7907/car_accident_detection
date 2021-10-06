import torch
import torchaudio
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.data import Dataset
import pytorch_lightning as pl
import numpy as np
import random
from tqdm import tqdm
import os.path
from glob import glob
import soundfile as sf
import warnings
import argparse
from pathlib import Path
from time import time
from datetime import datetime
import pandas as pd
import math

import logging
import pickle
import yaml
from copy import deepcopy
from collections import OrderedDict
import pdb
class StronglyLabeledDataset(Dataset):
    def __init__(self, tsv_read, dataset_dir, return_name, encoder):
        #refer: https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html
        self.dataset_dir = dataset_dir
        self.encoder = encoder
        self.pad_to = encoder.audio_len * encoder.sr
        self.return_name = return_name

        #construct clip dictionary with filename = {path, events} where events = {label, onset and offset}
        clips = {}
        for _, row in tsv_read.iterrows():
            if row["filename"] not in clips.keys():
                clips[row["filename"]] = {"path": os.path.join(dataset_dir, row["filename"]), "events": []}
            if not np.isnan(row["onset"]):
                clips[row["filename"]]["events"].append({"event_label": row["event_label"],
                                                         "onset": row["onset"],
                                                         "offset": row["offset"]})
        self.clips = clips #dictionary for each clip
        self.clip_list = list(clips.keys()) # list of all clip names

    def __len__(self):
        return len(self.clip_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.clip_list[idx]
        clip = self.clips[filename]
        path = clip["path"]

        # get wav
        wav, pad_mask = waveform_modification(path, self.pad_to, self.encoder)

        # get labels
        events = clip["events"]
        if not len(events): #label size = [frames, nclass]
            label = torch.zeros(self.encoder.n_frames, len(self.encoder.labels)).float()
        else:
            label = self.encoder.encode_strong_df(pd.DataFrame(events))
            label = torch.from_numpy(label).float()
        label = label.transpose(0, 1)

        # return
        out_args = [wav, label, pad_mask, idx]
        if self.return_name:
            out_args.extend([filename, path])
        return out_args


def waveform_modification(filepath, pad_to, encoder):
    wav, _ = sf.read(filepath)
    wav = to_mono(wav)  # 모노채널로바꾸기
    wav, pad_mask = pad_wav(wav, pad_to, encoder)
    wav = torch.from_numpy(wav).float()
    wav = normalize_wav(wav)
    return wav, pad_mask

def normalize_wav(wav):
    return wav / (torch.max(torch.max(wav), -torch.min(wav)) + 1e-10)


def to_mono(wav, rand_ch=False):
    if wav.ndim > 1:
        if rand_ch:
            ch_idx = np.random.randint(0, wav.shape[-1] - 1)
            wav = wav[:, ch_idx]
        else:
            wav = np.mean(wav, axis=-1)
    return wav


def pad_wav(wav, pad_to, encoder):
    if len(wav) < pad_to:
        pad_from = len(wav)
        wav = np.pad(wav, (0, pad_to - len(wav)), mode="constant")
    else:
        wav = wav[:pad_to]
        pad_from = pad_to
    pad_idx = np.ceil(encoder._time_to_frame(pad_from / encoder.sr))
    pad_mask = torch.arange(encoder.n_frames) >= pad_idx # size = n_frame, [0, 0, 0, 0, 0, ..., 0, 1, ..., 1]
    return wav, pad_mask


def setmelspectrogram(feature_cfg):
    return torchaudio.transforms.MelSpectrogram(sample_rate=feature_cfg["sample_rate"],
                                                n_fft=feature_cfg["n_window"],
                                                win_length=feature_cfg["n_window"],
                                                hop_length=feature_cfg["hop_length"],
                                                f_min=feature_cfg["f_min"],
                                                f_max=feature_cfg["f_max"],
                                                n_mels=feature_cfg["n_mels"],
                                                window_fn=torch.hamming_window,
                                                wkwargs={"periodic": False},
                                                power=1) # 1:energy, 2:power

class ConcatDatasetBatchSampler(Sampler):
    def __init__(self, samplers, batch_sizes, epoch=0):
        self.batch_sizes = batch_sizes
        self.samplers = samplers #각 dataset에 대한 sampler들
        self.offsets = [0] + np.cumsum([len(x) for x in self.samplers]).tolist()[:-1] #sampler 길이의 cumulative sum

        self.epoch = epoch
        self.set_epoch(self.epoch)

    def _iter_one_dataset(self, c_batch_size, c_sampler, c_offset):
        batch = []
        for idx in c_sampler:
            batch.append(c_offset + idx)
            if len(batch) == c_batch_size:
                yield batch

    def set_epoch(self, epoch):
        if hasattr(self.samplers[0], "epoch"):
            for s in self.samplers:
                s.set_epoch(epoch)

    def __iter__(self):
        iterators = [iter(i) for i in self.samplers]
        tot_batch = []
        for b_num in range(len(self)): #총 batch number만큼 for loop 돌림
            for samp_idx in range(len(self.samplers)): #각 sampler의 길이만큼 for loop 돌림: [0,1,2]
                c_batch = [] #current batch list생성
                while len(c_batch) < self.batch_sizes[samp_idx]: #current sampler의 batchsize만큼 current_batch에 샘플 집어넣기
                    c_batch.append(self.offsets[samp_idx] + next(iterators[samp_idx]))
                tot_batch.extend(c_batch)
            yield tot_batch
            tot_batch = []

    def __len__(self):
        min_len = float("inf")
        for idx, sampler in enumerate(self.samplers):
            c_len = (len(sampler)) // self.batch_sizes[0]
            min_len = min(c_len, min_len)
        return min_len #synth, weak, unlabeled dataset길이를 각각의 batch_size로 나눠서 제일 작은값을 반환