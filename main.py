import torch
import torchaudio
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.data import Dataset
import pytorch_lightning as pl
from settings import *
from model import *
from dataset import *
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
import scipy

import psds_eval
import sed_eval
from psds_eval import PSDSEval, plot_psd_roc


import logging
import pickle
import yaml
from copy import deepcopy
from collections import OrderedDict
import pdb
########################################################            Configs          ##########################################################
def get_configs(config_dir):
    #get hyperparameters from yaml
    with open(config_dir, "r") as f:
        configs = yaml.safe_load(f)

    train_cfg = configs["training"]
    feature_cfg = configs["feature"]
    train_cfg["batch_sizes"] = configs["generals"]["batch_size"]
    train_cfg["net_pooling"] = feature_cfg["net_subsample"]
    train_cfg["ensemble_dir"] = configs["generals"]["ensemble_dir"]
    general_cfg = configs["generals"]
    save_folder = general_cfg["save_folder"]
    #set best paths
    stud_best_path = os.path.join(save_folder, "best_student.pt")
    tch_best_path = os.path.join(save_folder, "best_teacher.pt")
    train_cfg["best_paths"] = [stud_best_path, tch_best_path]
    return configs, train_cfg, feature_cfg, general_cfg


def get_logger(save_folder):
    logger = logging.getLogger()
    formatter = logging.Formatter('%(message)s')
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(os.path.join(save_folder, "log.txt"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
def get_labeldict():
    return OrderedDict({"crash": 0,
                        "Help": 1,
                        "scream": 2,
                        "tire": 3})

def get_encoder(LabelDict, feature_cfg, audio_len):
    return Encoder(list(LabelDict.keys()),
                   audio_len=audio_len,
                   frame_len=feature_cfg["frame_length"],
                   frame_hop=feature_cfg["hop_length"],
                   net_pooling=feature_cfg["net_subsample"],
                   sr=feature_cfg["sr"])

def get_models(configs, train_cfg):
    net = CRNN(**configs["CRNN"])
    # ema network
    ema_net = deepcopy(net)
    for param in ema_net.parameters():
        param.detach_()

    if train_cfg["multigpu"] and (train_cfg["n_gpu"] > 1):
        net = nn.DataParallel(net)
        ema_net = nn.DataParallel(ema_net)

    net = net.to(train_cfg["device"])
    ema_net = ema_net.to(train_cfg["device"])
    return net, ema_net
def get_scaler(scaler_cfg):
    return Scaler(statistic=scaler_cfg["statistic"], normtype=scaler_cfg["normtype"], dims=scaler_cfg["dims"])
def get_f1calcs(n_class, device):
    stud_f1calc = pl.metrics.classification.F1(n_class, average="macro", multilabel=True, compute_on_step=False)
    tch_f1calc = pl.metrics.classification.F1(n_class, average="macro", multilabel=True, compute_on_step=False)
    return stud_f1calc.to(device), tch_f1calc.to(device)

def get_save_directories(configs, train_cfg, iteration, args):
    general_cfg = configs["generals"]
    save_folder = general_cfg["save_folder"]
    savepsds = general_cfg["savepsds"]

    # set save folder
    if save_folder.count("new_exp") > 0:
        save_folder = save_folder + '_gpu:' + str(args.gpu)
        configs["generals"]["save_folder"] = save_folder
    if not train_cfg["test_only"]:
        if iteration is not None:
            save_folder = save_folder + '_iter:' + str(iteration)
        print("save directory : " + save_folder)
        while not os.path.isdir(save_folder):
            os.mkdir(save_folder)  # saving folder
        with open(os.path.join(save_folder, 'config.yaml'), 'w') as f:
            yaml.dump(configs, f)  # save yaml in the saving folder

    #set best paths
    stud_best_path = os.path.join(save_folder, "best_student.pt")
    tch_best_path = os.path.join(save_folder, "best_teacher.pt")
    train_cfg["best_paths"] = [stud_best_path, tch_best_path]

    # psds folder
    if savepsds:
        if not general_cfg["ensemble_avg"]:
            stud_psds_folder = os.path.join(save_folder, "psds_student")
            tch_psds_folder = os.path.join(save_folder, "psds_teacher")
            psds_folders = [stud_psds_folder, tch_psds_folder]
        else:
            stud_psds_folder = os.path.join(general_cfg["ensemble_dir"], "psds_student")
            tch_psds_folder = os.path.join(general_cfg["ensemble_dir"], "psds_teacher")
            both_psds_folder = os.path.join(general_cfg["ensemble_dir"], "psds_both")
            psds_folders = [stud_psds_folder, tch_psds_folder, both_psds_folder]
    else:
        if not general_cfg["ensemble_avg"]:
            psds_folders = [None, None]
        else:
            psds_folders = [None, None, None]
    train_cfg["psds_folders"] = psds_folders
    return configs, train_cfg

def compute_psds_from_operating_points(
    prediction_dfs,
    ground_truth_file,
    durations_file,
    dtc_threshold=0.5,
    gtc_threshold=0.5,
    cttc_threshold=0.3,
    alpha_ct=0,
    alpha_st=0,
    max_efpr=100,
    save_dir=None,
):

    gt = pd.read_csv(ground_truth_file, sep="\t")
    durations = pd.read_csv(durations_file, sep="\t")
    psds_eval = PSDSEval(
        ground_truth=gt,
        metadata=durations,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold,
    )

    for i, k in enumerate(prediction_dfs.keys()):
        det = prediction_dfs[k]
        # see issue https://github.com/audioanalytic/psds_eval/issues/3
        det["index"] = range(1, len(det) + 1)
        det = det.set_index("index")
        psds_eval.add_operating_point(
            det, info={"name": f"Op {i + 1:02d}", "threshold": k}
        )

    psds_score = psds_eval.psds(alpha_ct=alpha_ct, alpha_st=alpha_st, max_efpr=max_efpr)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        pred_dir = os.path.join(
            save_dir,
            f"predictions_dtc{dtc_threshold}_gtc{gtc_threshold}_cttc{cttc_threshold}",
        )
        os.makedirs(pred_dir, exist_ok=True)
        for k in prediction_dfs.keys():
            prediction_dfs[k].to_csv(
                os.path.join(pred_dir, f"predictions_th_{k:.2f}.tsv"),
                sep="\t",
                index=False,
            )

    #    plot_psd_roc(
    #        psds_score,
    #        filename=os.path.join(save_dir, f"PSDS_ct{alpha_ct}_st{alpha_st}_100.png"),
    #    )

    return psds_score.value
class Scaler(nn.Module):
    def __init__(self, statistic="instance", normtype="minmax", dims=(0, 2), eps=1e-8):
        super(Scaler, self).__init__()
        self.statistic = statistic
        self.normtype = normtype
        self.dims = dims
        self.eps = eps

    def load_state_dict(self, state_dict, strict=True):
        if self.statistic == "dataset":
            super(Scaler, self).load_state_dict(state_dict, strict)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if self.statistic == "dataset":
            super(Scaler, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                                      unexpected_keys, error_msgs)

    def forward(self, input):
        if self.statistic == "dataset":
            if self.normtype == "mean":
                return input - self.mean
            elif self.normtype == "standard":
                std = torch.sqrt(self.mean_squared - self.mean ** 2)
                return (input - self.mean) / (std + self.eps)
            else:
                raise NotImplementedError

        elif self.statistic =="instance":
            if self.normtype == "mean":
                return input - torch.mean(input, self.dims, keepdim=True)
            elif self.normtype == "standard":
                return (input - torch.mean(input, self.dims, keepdim=True)) / (
                        torch.std(input, self.dims, keepdim=True) + self.eps)
            elif self.normtype == "minmax":
                return (input - torch.amin(input, dim=self.dims, keepdim=True)) / (
                    torch.amax(input, dim=self.dims, keepdim=True)
                    - torch.amin(input, dim=self.dims, keepdim=True) + self.eps)
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

class ExponentialWarmup(object):
    def __init__(self, optimizer, max_lr, rampup_length, exponent=-5.0):
        self.optimizer = optimizer
        self.rampup_length = rampup_length
        self.max_lr = max_lr
        self.step_num = 1
        self.exponent = exponent

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _get_lr(self):
        return self.max_lr * self._get_scaling_factor()

    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        self._set_lr(lr)

    # def load_state_dict(self, state_dict):
    #     self.__dict__.update(state_dict)
    #
    # def state_dict(self):
    #     return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def _get_scaling_factor(self):
        if self.rampup_length == 0:
            return 1.0
        else:
            current = np.clip(self.step_num, 0.0, self.rampup_length)
            phase = 1.0 - current / self.rampup_length
            return float(np.exp(self.exponent * phase * phase))

def take_log(feature):
    amp2db = torchaudio.transforms.AmplitudeToDB(stype="amplitude")
    amp2db.amin = 1e-5
    return amp2db(feature).clamp(min=-50, max=80)

def update_ema(net, ema_net, step, ema_factor):
    # update EMA model
    alpha = min(1 - 1 / step, ema_factor)
    for ema_params, params in zip(ema_net.parameters(), net.parameters()):
        ema_params.data.mul_(alpha).add_(params.data, alpha=1 - alpha)
    return ema_net


def compute_per_intersection_macro_f1(
    prediction_dfs,
    ground_truth_file,
    durations_file,
    dtc_threshold=0.5,
    gtc_threshold=0.5,
    cttc_threshold=0.3,
):
    """ Compute F1-score per intersection, using the defautl
    Args:
        prediction_dfs: dict, a dictionary with thresholds keys and predictions dataframe
        ground_truth_file: pd.DataFrame, the groundtruth dataframe
        durations_file: pd.DataFrame, the duration dataframe
        dtc_threshold: float, the parameter used in PSDSEval, percentage of tolerance for groundtruth intersection
            with predictions
        gtc_threshold: float, the parameter used in PSDSEval percentage of tolerance for predictions intersection
            with groundtruth
        gtc_threshold: float, the parameter used in PSDSEval to know the percentage needed to count FP as cross-trigger

    Returns:

    """
    gt = pd.read_csv(ground_truth_file, sep="\t")
    durations = pd.read_csv(durations_file, sep="\t")

    psds = PSDSEval(
        ground_truth=gt,
        metadata=durations,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold,
    )
    psds_macro_f1 = []
    for threshold in prediction_dfs.keys():
        if not prediction_dfs[threshold].empty:
            threshold_f1, _ = psds.compute_macro_f_score(prediction_dfs[threshold])
        else:
            threshold_f1 = 0
        if np.isnan(threshold_f1):
            threshold_f1 = 0.0
        psds_macro_f1.append(threshold_f1)
    psds_macro_f1 = np.mean(psds_macro_f1)
    return psds_macro_f1
class BestModels:
    # Class to keep track of the best student/teacher models and save them after training
    def __init__(self):
        self.stud_best_val_metric = 0.0
        self.tch_best_val_metric = 0.0
        self.stud_best_state_dict = None
        self.tch_best_state_dict = None

    def update(self, train_cfg, logger, val_metrics):
        stud_update = False
        tch_update = False
        if val_metrics[0] > self.stud_best_val_metric:
            self.stud_best_val_metric = val_metrics[0]
            self.stud_best_state_dict = train_cfg["net"].state_dict()
            stud_update = True
            # lr_reduc = 0
        if val_metrics[1] > self.tch_best_val_metric:
            self.tch_best_val_metric = val_metrics[1]
            self.tch_best_state_dict = train_cfg["ema_net"].state_dict()
            tch_update = True
            # lr_reduc = 0

        if train_cfg["epoch"] > int(train_cfg["n_epochs"] * 0.5):
            if stud_update:
                if tch_update:
                    logger.info("     best student & teacher model updated at epoch %d!" % (train_cfg["epoch"] + 1))
                else:
                    logger.info("     best student model updated at epoch %d!" % (train_cfg["epoch"] + 1))
            elif tch_update:
                logger.info("     best teacher model updated at epoch %d!" % (train_cfg["epoch"] + 1))
        return logger

    def get_bests(self, best_paths):
        torch.save(self.stud_best_state_dict, best_paths[0])
        torch.save(self.tch_best_state_dict, best_paths[1])
        return self.stud_best_val_metric, self.tch_best_val_metric
def decode_pred_batch(outputs, weak_preds, filenames, encoder, thresholds, median_filter, decode_weak, pad_idx=None):
    pred_dfs = {}
    for threshold in thresholds:
        pred_dfs[threshold] = pd.DataFrame()
    for batch_idx in range(outputs.shape[0]): #outputs size = [bs, n_class, frames]
        for c_th in thresholds:
            output = outputs[batch_idx]       #outputs size = [n_class, frames]
            if pad_idx is not None:
                true_len = int(output.shape[-1] * pad_idx[batch_idx].item)
                output = output[:true_len]
            output = output.transpose(0, 1).detach().cpu().numpy() #output size = [frames, n_class]
            if decode_weak: # if decode_weak = 1 or 2
                for class_idx in range(weak_preds.size(1)):
                    if weak_preds[batch_idx, class_idx] < c_th:
                        output[:, class_idx] = 0
                    elif decode_weak > 1: # use only weak predictions (weakSED)
                        output[:, class_idx] = 1
            #if decode_weak < 2: # weak prediction masking
                #output = output > c_th
                #output = scipy.ndimage.median_filter(output, (median_filter, 1))
                #output = scipy.ndimage.filters.median_filter(output, (median_filter, 1))
            pred = encoder.decode_strong(output)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = Path(filenames[batch_idx]).stem + ".wav"
            pred_dfs[c_th] = pred_dfs[c_th].append(pred, ignore_index=True)
    return pred_dfs



def compute_sed_eval_metrics(predictions, groundtruth):
    """ Compute sed_eval metrics event based and segment based with default parameters used in the task.
    Args:
        predictions: pd.DataFrame, predictions dataframe
        groundtruth: pd.DataFrame, groundtruth dataframe
    Returns:
        tuple, (sed_eval.sound_event.EventBasedMetrics, sed_eval.sound_event.SegmentBasedMetrics)
    """
    metric_event = event_based_evaluation_df(
        groundtruth, predictions, t_collar=0.200, percentage_of_length=0.2
    )
    metric_segment = segment_based_evaluation_df(
        groundtruth, predictions, time_resolution=1.0
    )

    return metric_event, metric_segment

def log_sedeval_metrics(predictions, ground_truth, save_dir=None):
    """ Return the set of metrics from sed_eval
    Args:
        predictions: pd.DataFrame, the dataframe of predictions.
        ground_truth: pd.DataFrame, the dataframe of groundtruth.
        save_dir: str, path to the folder where to save the event and segment based metrics outputs.

    Returns:
        tuple, event-based macro-F1 and micro-F1, segment-based macro-F1 and micro-F1
    """
    if predictions.empty:
        return 0.0, 0.0, 0.0, 0.0

    gt = pd.read_csv(ground_truth, sep="\t")

    event_res, segment_res = compute_sed_eval_metrics(predictions, gt)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "event_f1.txt"), "w") as f:
            f.write(str(event_res))

        with open(os.path.join(save_dir, "segment_f1.txt"), "w") as f:
            f.write(str(segment_res))

    return (
        event_res.results()["class_wise_average"]["f_measure"]["f_measure"],
        event_res.results()["overall"]["f_measure"]["f_measure"],
        segment_res.results()["class_wise_average"]["f_measure"]["f_measure"],
        segment_res.results()["overall"]["f_measure"]["f_measure"],
    )  # return also segment measures

def event_based_evaluation_df(
    reference, estimated, t_collar=0.200, percentage_of_length=0.2
):
    """ Calculate EventBasedMetric given a reference and estimated dataframe

    Args:
        reference: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
            reference events
        estimated: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
            estimated events to be compared with reference
        t_collar: float, in seconds, the number of time allowed on onsets and offsets
        percentage_of_length: float, between 0 and 1, the percentage of length of the file allowed on the offset
    Returns:
         sed_eval.sound_event.EventBasedMetrics with the scores
    """

    evaluated_files = reference["filename"].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=classes,
        t_collar=t_collar,
        percentage_of_length=percentage_of_length,
        empty_system_output_handling="zero_score",
    )

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(
            reference, fname
        )
        estimated_event_list_for_current_file = get_event_list_current_file(
            estimated, fname
        )

        event_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file,
        )

    return event_based_metric
def segment_based_evaluation_df(reference, estimated, time_resolution=1.0):
    """ Calculate SegmentBasedMetrics given a reference and estimated dataframe

        Args:
            reference: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
                reference events
            estimated: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
                estimated events to be compared with reference
            time_resolution: float, the time resolution of the segment based metric
        Returns:
             sed_eval.sound_event.SegmentBasedMetrics with the scores
        """
    evaluated_files = reference["filename"].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    segment_based_metric = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=classes, time_resolution=time_resolution
    )

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(
            reference, fname
        )
        estimated_event_list_for_current_file = get_event_list_current_file(
            estimated, fname
        )

        segment_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file,
        )

    return segment_based_metric
#############################################               ENCODER                   #####################################
class Encoder:
    def __init__(self, labels, audio_len, frame_len, frame_hop, net_pooling=1, sr=16000):
        if type(labels) in [np.ndarray, np.array]:            labels = labels.tolist()
        self.labels = labels
        self.audio_len = audio_len
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.sr = sr
        self.net_pooling = net_pooling
        n_samples = self.audio_len * self.sr
        self.n_frames = int(math.ceil(n_samples/2/self.frame_hop)*2 / self.net_pooling)

    def _time_to_frame(self, time):
        sample = time * self.sr
        frame = sample / self.frame_hop
        return np.clip(frame / self.net_pooling, a_min=0, a_max=self.n_frames)

    def _frame_to_time(self, frame):
        time = frame * self.net_pooling * self.frame_hop / self.sr
        return np.clip(time, a_min=0, a_max=self.audio_len)

    def encode_strong_df(self, events_df):
        # from event dict, generate strong label tensor sized as [n_frame, n_class]
        true_labels = np.zeros((self.n_frames, len(self.labels)))
        for _, row in events_df.iterrows():
            if not pd.isna(row['event_label']):
                label_idx = self.labels.index(row["event_label"])
                onset = int(self._time_to_frame(row["onset"]))           #버림 -> 해당 time frame에 걸쳐있으면 true
                offset = int(np.ceil(self._time_to_frame(row["offset"])))  #올림 -> 해당 time frame에 걸쳐있으면 true
                true_labels[onset:offset, label_idx] = 1
        return true_labels

    def encode_weak(self, events):
        # from event dict, generate weak label tensor sized as [n_class]
        labels = np.zeros((len(self.labels)))
        if len(events) == 0:
            return labels
        else:
            for event in events:
                labels[self.labels.index(event)] = 1
            return labels

    def decode_strong(self, outputs):
        #from the network output sized [n_frame, n_class], generate the label/onset/offset lists
        pred = []
        for i, label_column in enumerate(outputs.T):  #outputs size = [n_class, frames]
            change_indices = self.find_contiguous_regions(label_column)
            for row in change_indices:
                onset = self._frame_to_time(row[0])
                offset = self._frame_to_time(row[1])
                onset = np.clip(onset, a_min=0, a_max=self.audio_len)
                offset = np.clip(offset, a_min=0, a_max=self.audio_len)
                pred.append([self.labels[i], onset, offset])
        return pred

    def decode_weak(self, outputs):
        result_labels = []
        for i, value in enumerate(outputs):
            if value == 1:
                result_labels.append(self.labels[i])
        return result_labels

    def find_contiguous_regions(self, array):
        #find at which frame the label changes in the array
        change_indices = np.logical_xor(array[1:], array[:-1]).nonzero()[0]
        #shift indices to focus the frame after
        change_indices += 1
        if array[0]:
            #if first element of array is True(1), add 0 in the beggining
            #change_indices = np.append(0, change_indices)
            change_indices = np.r_[0, change_indices]
        if array[-1]:
            #if last element is True, add the length of array
            change_indices = np.r_[change_indices, array.size]
        #reshape the result into two columns
        return change_indices.reshape((-1, 2))

################################                      DATASET                       #######################################
def get_mt_datasets(configs, train_cfg):
    general_cfg = configs["generals"]
    encoder = train_cfg["encoder"]
    dataset_cfg = configs["dataset"]
    batch_size_val = general_cfg["batch_size_val"]
    num_workers = general_cfg["num_workers"]
    batch_sizes = general_cfg["batch_size"]
    synthdataset_cfg = configs["synth_dataset"]

    synth_train_tsv = synthdataset_cfg["synth_train_tsv"]
    synth_train_df = pd.read_csv(synth_train_tsv, sep="\t")
    synth_valid_dir = synthdataset_cfg["synth_val_folder"]
    synth_valid_tsv = synthdataset_cfg["synth_val_tsv"]
    synth_valid_df = pd.read_csv(synth_valid_tsv, sep="\t")
    synth_valid_dur = synthdataset_cfg["synth_val_dur"]
    synth_train_dataset = StronglyLabeledDataset(synth_train_df, synthdataset_cfg["synth_train_folder"], False, encoder)
    synth_vaild_dataset = StronglyLabeledDataset(synth_valid_df, synth_valid_dir, True, encoder)
    if not general_cfg["test_on_public_eval"]:
        test_tsv = dataset_cfg["test_tsv"]
        test_df = pd.read_csv(test_tsv, sep="\t")
        test_dur = dataset_cfg["test_dur"]
        test_dataset = StronglyLabeledDataset(test_df, dataset_cfg["test_folder"], True, encoder)
    else:
        test_tsv = dataset_cfg["pubeval_tsv"]
        test_df = pd.read_csv(test_tsv, sep="\t")
        test_dur = dataset_cfg["pubeval_dur"]
        test_dataset = StronglyLabeledDataset(test_df, dataset_cfg["pubeval_folder"], True, encoder)
    # build dataloaders
    # get train dataset
    train_data = synth_train_dataset
    train_samplers = [torch.utils.data.RandomSampler(x) for x in train_data]
    #train_batch_sampler = ConcatDatasetBatchSampler(train_samplers, batch_sizes)
    train_cfg["trainloader"] = DataLoader(train_data, num_workers=num_workers) #batch_sampler=train_batch_sampler
    # get valid dataset
    valid_dataset = synth_vaild_dataset
    train_cfg["validloader"] = DataLoader(valid_dataset, batch_size=batch_size_val, num_workers=num_workers)
    # get test dataset
    train_cfg["testloader"] = DataLoader(test_dataset, batch_size=batch_size_val, num_workers=num_workers)
    train_cfg["train_tsvs"] = [synth_train_df, synth_train_tsv]
    train_cfg["valid_tsvs"] = [synth_valid_dir, synth_valid_tsv, synth_valid_dur]
    train_cfg["test_tsvs"] = [test_tsv, test_dur]
    return train_cfg

#####################################################################################################################
########################################################            MAIN&PARAMETERS          ##########################################################


def main(iteration=None):
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--model', default=1, type=int, help='selection of model setting from the paper')
    parser.add_argument('--gpu', default=1, type=int, help='selection of gpu when you run separate trainings on single server')
    args = parser.parse_args()
    #set configurations
    configs, train_cfg, feature_cfg, general_cfg = get_configs(config_dir="./config.yaml")
    #set save directories
    configs, train_cfg = get_save_directories(configs, train_cfg, iteration, args)
    #set logger
    logger = get_logger(configs["generals"]["save_folder"])
    #torch information
    logger.info("date & time of start is : " + str(datetime.now()).split('.')[0])
    logger.info("torch version is: " + str(torch.__version__))
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    train_cfg["device"] = device
    logger.info("device: " + str(device))
    train_cfg["n_gpu"] = torch.cuda.device_count()
    logger.info("model selection: model %d" % args.model)
    logger.info("number of GPUs: " + str(train_cfg["n_gpu"]))

    #seed
    torch.random.manual_seed(train_cfg["seed"])
    if device == 'cuda':
        torch.cuda.manual_seed_all(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])
    random.seed(train_cfg["seed"])

    #do not show warning
    if not configs["generals"]["warn"]:
        warnings.filterwarnings("ignore")

    #class label dictionary
    LabelDict = get_labeldict()

    #set encoder
    train_cfg["encoder"] = get_encoder(LabelDict, feature_cfg, feature_cfg["audio_max_len"])
    train_cfg["encoder300"] = get_encoder(LabelDict, feature_cfg, 300)

    #set Dataloaders
    train_cfg = get_mt_datasets(configs, train_cfg)

    logger.info('Set_Networks!')
    #set network
    
    train_cfg["net"], train_cfg["ema_net"] = get_models(configs, train_cfg)
    #logger.info("Total Trainable Params: %d" % count_parameters(train_cfg["net"])) #print number of learnable parameters in the model

    logger.info('Set_feature!')
    #set feature
    train_cfg["feat_ext"] = setmelspectrogram(feature_cfg).to(device)

    #set scaler
    train_cfg["scaler"] = get_scaler(configs["scaler"])

    #set f1 calculators
    train_cfg["f1calcs"] = get_f1calcs(len(LabelDict), device)
    logger.info('Set_Done!')
    #loss function, optimizer, scheduler
    if train_cfg["afl_loss"] is None:
        train_cfg["criterion_class"] = nn.BCELoss().cuda()
    else:
        train_cfg["criterion_class"] = AsymmetricalFocalLoss(train_cfg["afl_loss"][0], train_cfg["afl_loss"][1])
    train_cfg["criterion_cons"] = nn.MSELoss().cuda()
    train_cfg["optimizer"] = optim.Adam(train_cfg["net"].parameters(), 1e-3, betas=(0.9, 0.999))
    warmup_steps = train_cfg["n_epochs_warmup"] * len(train_cfg["trainloader"])
    train_cfg["scheduler"] = ExponentialWarmup(train_cfg["optimizer"], configs["opt"]["lr"], warmup_steps)
    printing_epoch, printing_test = get_printings()
    
    if not (train_cfg["test_only"] or configs["generals"]["ensemble_avg"]):
        logger.info('training starts!')
        start_time = time()
        history = History()
        bestmodels = BestModels()
        for train_cfg["epoch"] in range(train_cfg["n_epochs"]):
            epoch_time = time()
            #training
            train_return = train(train_cfg)
            val_return = validation(train_cfg)
            #save best model when best validation metrics occur
            val_metrics = history.update(train_return, val_return)
            #logger.info(printing_epoch % ((train_cfg["epoch"] + 1,) + train_return + val_return + (time() - epoch_time,)))
            logger = bestmodels.update(train_cfg, logger, val_metrics)

        #save model parameters & history dictionary
        logger.info("        best student/teacher val_metrics: %.3f / %.3f" % bestmodels.get_bests(train_cfg["best_paths"]))
        history.save(os.path.join(configs["generals"]["save_folder"], "history.pickle"))
        logger.info("   training took %.2f mins" % ((time()-start_time)/60))
    
    ##############################                        TEST                        ##############################
    if not configs["generals"]["ensemble_avg"]:
        logger.info("   test starts!")

        # test on best model
        train_cfg["net"].load_state_dict(torch.load(train_cfg["best_paths"][0]))
        train_cfg["ema_net"].load_state_dict(torch.load(train_cfg["best_paths"][1]))
        test_returns = test(train_cfg)
        logger.info(printing_test % test_returns)

    ##############################                TEST ENSEMBLE AVERAGE               ##############################
    else:
        logger.info("   ensemble test starts!")
        train_cfg = get_ensemble_models(train_cfg)
        test_returns = test_ensemble(train_cfg)
        logger.info("      ensemble test result is out!"
                    "\n      [student] psds1: %.4f, psds2: %.4f"
                    "\n                event_macro_f1: %.3f, event_micro_f1: %.3f, "
                    "\n                segment_macro_f1: %.3f, segment_micro_f1: %.3f, intersection_f1: %.3f"
                    "\n      [teacher] psds1: %.4f, psds2: %.4f"
                    "\n                event_macro_f1: %.3f, event_micro_f1: %.3f, "
                    "\n                segment_macro_f1: %.3f, segment_micro_f1: %.3f, intersection_f1: %.3f"
                    "\n      [ both ]  psds1: %.4f, psds2: %.4f"
                    "\n                event_macro_f1: %.3f, event_micro_f1: %.3f, "
                    "\n                segment_macro_f1: %.3f, segment_micro_f1: %.3f, intersection_f1: %.3f"
                    % test_returns)


    logger.info("date & time of end is : " + str(datetime.now()).split('.')[0])
    logging.shutdown()
    print("<"*30 + "DONE!" + ">"*30)
########################################################                TRAIN                ##########################################################

def train(train_cfg):
    train_cfg["net"].train()
    train_cfg["ema_net"].train()
    total_loss, class_strong_loss, cons_strong_loss = 0.0, 0.0, 0.0
    strong_bs, weak_bs, _ = train_cfg["batch_sizes"]
    n_train = len(train_cfg["trainloader"])
    #pdb.set_trace()
    tk0 = tqdm(train_cfg["trainloader"], total=n_train, leave=False, desc="training processing")
    for _, (wavs, labels, pad_mask, _) in enumerate(tk0, 0):
        wavs, labels = wavs.to(train_cfg["device"]), labels.to(train_cfg["device"]) # labels size = [bs, n_class, frames]
        features = train_cfg["feat_ext"](wavs)  # features size = [bs, freqs, frames]
        batch_num = features.size(0)
        mask_strong = torch.zeros(batch_num).to(features).bool()
        mask_strong[:strong_bs] = 1                     # mask_strong size = [bs]
        mask_weak = torch.zeros(batch_num).to(features).bool()
        mask_weak[strong_bs:(strong_bs + weak_bs)] = 1  # mask_weak size = [bs]
        labels_weak = (torch.sum(labels[mask_weak], -1) > 0).float() # labels_weak size = [bs, n_class] (weak data only)
        features = train_cfg["feat_ext"](wavs)
        logmels_features = train_cfg["scaler"](take_log(features))
        # model predictions
        train_cfg["optimizer"].zero_grad()                              # strong prediction size = [bs, n_class, frames]
        strong_pred_stud, weak_pred_stud = train_cfg["net"](logmels_features)     # weak prediction size = [bs, n_class]
        with torch.no_grad():
            strong_pred_tch, weak_pred_tch = train_cfg["ema_net"](logmels_features)

        # classification losses                    # strong masked label size = [bs_strong, n_class, frames]
        loss_class_strong = train_cfg["criterion_class"](strong_pred_stud[mask_strong],
                                                         labels[mask_strong])
        #loss_class_weak = train_cfg["criterion_class"](weak_pred_stud[mask_weak], labels_weak)

        # consistency losses
        loss_cons_strong = train_cfg["criterion_cons"](strong_pred_stud, strong_pred_tch.detach())
        #loss_cons_weak = train_cfg["criterion_cons"](weak_pred_stud, weak_pred_tch.detach())
        w_cons = train_cfg["w_cons_max"] * train_cfg["scheduler"]._get_scaling_factor()

        if not train_cfg["trainweak_only"]:
            loss_total = loss_class_strong 
                #+ train_cfg["w_weak"] * loss_class_weak + \ w_cons * (loss_cons_strong + train_cfg["w_weak_cons"] * loss_cons_weak)
        #else:
        #    loss_total = train_cfg["w_weak"] * loss_class_weak + w_cons * train_cfg["w_weak_cons"] * loss_cons_weak
        loss_total.backward()
        train_cfg["optimizer"].step()
        train_cfg["scheduler"].step()

        # update EMA model
        train_cfg["ema_net"] = update_ema(train_cfg["net"], train_cfg["ema_net"], train_cfg["scheduler"].step_num,
                                          train_cfg["ema_factor"])

        total_loss += loss_total.item()
        class_strong_loss += loss_class_strong.item()
        #class_weak_loss += loss_class_weak.item()
        cons_strong_loss += loss_cons_strong.item()
        #cons_weak_loss = loss_cons_weak.item()

    total_loss /= n_train
    class_strong_loss /= n_train
    #class_weak_loss /= n_train
    cons_strong_loss /= n_train
    #cons_weak_loss /= n_train
    return total_loss, class_strong_loss, cons_strong_loss

def validation(train_cfg):
    encoder = train_cfg["encoder"]
    train_cfg["net"].eval()
    train_cfg["ema_net"].eval()
    n_valid = len(train_cfg["validloader"])
    val_stud_buffer = {k: pd.DataFrame() for k in train_cfg["val_thresholds"]}
    val_tch_buffer = {k: pd.DataFrame() for k in train_cfg["val_thresholds"]}
    synth_valid_dir, synth_valid_tsv, synth_valid_dur = train_cfg["valid_tsvs"]
    with torch.no_grad():
        tk1 = tqdm(train_cfg["validloader"], total=n_valid, leave=False, desc="validation processing")
        for _, (wavs, labels, pad_mask, indexes, filenames, paths) in enumerate(tk1, 0):
            wavs, labels = wavs.to(train_cfg["device"]), labels.to(train_cfg["device"]) # labels size = [bs, n_class, frames]
            features = train_cfg["feat_ext"](wavs)  # features size = [bs, freqs, frames]
            logmels = train_cfg["scaler"](take_log(features))

            strong_pred_stud, weak_pred_stud = train_cfg["net"](logmels)
            strong_pred_tch, weak_pred_tch = train_cfg["ema_net"](logmels)

            if not train_cfg["trainweak_withstrong"]:
                mask_strong = (torch.tensor([str(Path(x).parent) == str(Path(synth_valid_dir)) for x in paths])
                               .to(logmels).bool())
            else:
                #mask_weak = torch.ones(labels.size(0)).to(logmels).bool()
                mask_strong = torch.zeros(labels.size(0)).to(logmels).bool()


            #if torch.any(mask_weak):
            #    labels_weak = (torch.sum(labels[mask_weak], -1) > 0).int()  # labels_weak size = [bs, n_class]
            #    #accumulate f1score for weak labels
            #    #pdb.set_trace()
            #    train_cfg["f1calcs"][0](weak_pred_stud[mask_weak], labels_weak)
            #    train_cfg["f1calcs"][1](weak_pred_tch[mask_weak], labels_weak)

            if torch.any(mask_strong):
                #decoded_stud/tch_strong for intersection f1 score
                paths_strong = [x for x in paths if Path(x).parent == Path(synth_valid_dir)]
                stud_pred_dfs = decode_pred_batch(strong_pred_stud[mask_strong], weak_pred_stud[mask_strong],
                                                  paths_strong, encoder, list(val_stud_buffer.keys()),
                                                  train_cfg["median_window"], train_cfg["decode_weak_valid"])
                tch_pred_dfs = decode_pred_batch(strong_pred_tch[mask_strong], weak_pred_tch[mask_strong],
                                                 paths_strong, encoder, list(val_tch_buffer.keys()),
                                                 train_cfg["median_window"], train_cfg["decode_weak_valid"])
                for th in val_stud_buffer.keys():
                    val_stud_buffer[th] = val_stud_buffer[th].append(stud_pred_dfs[th], ignore_index=True)
                for th in val_tch_buffer.keys():
                    val_tch_buffer[th] = val_tch_buffer[th].append(tch_pred_dfs[th], ignore_index=True)

    #stud_weak_f1 = train_cfg["f1calcs"][0].compute()
    #tch_weak_f1 = train_cfg["f1calcs"][1].compute()
    stud_intersection_f1 = compute_per_intersection_macro_f1(val_stud_buffer, synth_valid_tsv, synth_valid_dur)
    tch_intersection_f1 = compute_per_intersection_macro_f1(val_tch_buffer, synth_valid_tsv, synth_valid_dur)
    #if not train_cfg["trainweak_only"]:
    stud_val_metric = stud_intersection_f1
    tch_val_metric = tch_intersection_f1
    return stud_val_metric, tch_val_metric
    
    #else:
    #    return stud_weak_f1.item(), tch_weak_f1.item()



########################################################################################################################
#                                                         TEST                                                         #
########################################################################################################################

def test(train_cfg):
    f = open("./output.txt", 'w')
    #print(Test)
    encoder = train_cfg["encoder"]
    psds_folders = train_cfg["psds_folders"]
    thresholds = np.arange(1 / (train_cfg["n_test_thresholds"] * 2), 1, 1 / train_cfg["n_test_thresholds"])
    train_cfg["net"].eval()
    train_cfg["ema_net"].eval()
    test_tsv, test_dur = train_cfg["test_tsvs"]
    with torch.no_grad():
        stud_test_psds_buffer = {k: pd.DataFrame() for k in thresholds}
        tch_test_psds_buffer = {k: pd.DataFrame() for k in thresholds}
        stud_test_f1_buffer = pd.DataFrame()
        tch_test_f1_buffer = pd.DataFrame()
        tk2 = tqdm(train_cfg["testloader"], total=len(train_cfg["testloader"]), leave=False, desc="test processing")
        for _, (wavs, labels, pad_mask, indexes, filenames, paths) in enumerate(tk2, 0):
            wavs, labels = wavs.to(train_cfg["device"]), labels.to(train_cfg["device"]) # labels size = [bs, n_class, frames]
            features = train_cfg["feat_ext"](wavs)  # features size = [bs, freqs, frames]
            logmels = train_cfg["scaler"](take_log(features))

            stud_preds, weak_stud_preds = train_cfg["net"](logmels)
            tch_preds, weak_tch_preds = train_cfg["ema_net"](logmels)
            #pdb.set_trace()
            #0~ 12 Class 중에서 Best만 찍는 코드 짜기
            for i in range(len(filenames)):
                best = torch.max(tch_preds[i],0)[1]
                best_acc = torch.max(tch_preds[i],0)[0]
                best_list = best.tolist()
                best_acc_list = best_acc.tolist()
                new = ', '.join(str(e) for e in best_list)
                new_acc = ', '.join(str(e) for e in best_acc_list)
                        
                n_data = filenames[i] +  " of Best :" + new + "\n" + " Best_acc :" + new_acc +  "\n\n"
                f.write(n_data)
            stud_pred_dfs = decode_pred_batch(stud_preds, weak_stud_preds, paths, encoder,
                                              list(stud_test_psds_buffer.keys()), train_cfg["median_window"],
                                              train_cfg["decode_weak_test"])
            tch_pred_dfs = decode_pred_batch(tch_preds, weak_tch_preds, paths, encoder,
                                             list(tch_test_psds_buffer.keys()), train_cfg["median_window"],
                                             train_cfg["decode_weak_test"])
            for th in stud_test_psds_buffer.keys():
                stud_test_psds_buffer[th] = stud_test_psds_buffer[th].append(stud_pred_dfs[th], ignore_index=True)
            for th in tch_test_psds_buffer.keys():
                tch_test_psds_buffer[th] = tch_test_psds_buffer[th].append(tch_pred_dfs[th], ignore_index=True)
            stud_pred_df_halfpoint = decode_pred_batch(stud_preds, weak_stud_preds, paths, encoder, [0.5],
                                                       train_cfg["median_window"], train_cfg["decode_weak_test"])
            tch_pred_df_halfpoint = decode_pred_batch(tch_preds, weak_tch_preds, paths, encoder, [0.5],
                                                      train_cfg["median_window"], train_cfg["decode_weak_test"])
            stud_test_f1_buffer = stud_test_f1_buffer.append(stud_pred_df_halfpoint[0.5], ignore_index=True)
            tch_test_f1_buffer = tch_test_f1_buffer.append(tch_pred_df_halfpoint[0.5], ignore_index=True)
    f.close()

    # calculate psds
    psds1_kwargs = {"dtc_threshold": 0.7, "gtc_threshold": 0.7, "alpha_ct": 0, "alpha_st": 1}
    psds2_kwargs = {"dtc_threshold": 0.1, "gtc_threshold": 0.1, "cttc_threshold": 0.3, "alpha_ct": 0.5, "alpha_st": 1}
    stud_psds1 = compute_psds_from_operating_points(stud_test_psds_buffer, test_tsv, test_dur, save_dir=psds_folders[0],
                                                    **psds1_kwargs)
    stud_psds2 = compute_psds_from_operating_points(stud_test_psds_buffer, test_tsv, test_dur, save_dir=psds_folders[0],
                                                    **psds2_kwargs)
    tch_psds1 = compute_psds_from_operating_points(tch_test_psds_buffer, test_tsv, test_dur, save_dir=psds_folders[1],
                                                   **psds1_kwargs)
    tch_psds2 = compute_psds_from_operating_points(tch_test_psds_buffer, test_tsv, test_dur, save_dir=psds_folders[1],
                                                   **psds2_kwargs)
    s_evt_ma_f1, s_evt_mi_f1, s_seg_ma_f1, s_seg_mi_f1 = log_sedeval_metrics(stud_test_f1_buffer,
                                                                             test_tsv, psds_folders[0])
    s_inter_f1 = compute_per_intersection_macro_f1({"0.5": stud_test_f1_buffer}, test_tsv, test_dur)
    t_evt_ma_f1, t_evt_mi_f1, t_seg_ma_f1, t_seg_mi_f1 = log_sedeval_metrics(tch_test_f1_buffer,
                                                                             test_tsv, psds_folders[1])
    t_inter_f1 = compute_per_intersection_macro_f1({"0.5": tch_test_f1_buffer}, test_tsv, test_dur)
    return stud_psds1, stud_psds2, s_evt_ma_f1, s_evt_mi_f1, s_seg_ma_f1, s_seg_mi_f1, s_inter_f1, \
           tch_psds1, tch_psds2, t_evt_ma_f1, t_evt_mi_f1, t_seg_ma_f1, t_seg_mi_f1, t_inter_f1


if __name__ == "__main__":
    n_repeat = 1
    for iter in range(n_repeat):
        #main(iter)
        main()





