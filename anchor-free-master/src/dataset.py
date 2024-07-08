# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import json
import logging
from typing import Tuple
import numpy as np
import pandas as pd
import pickle
import mindspore.dataset as ds
from src.utils import ioa_with_anchors, iou_with_anchors
import random
import math
from ipdb import set_trace
from tqdm import tqdm

log = logging.getLogger(__name__)


def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data

def load_pickle_feature(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def collect_gt(video_info, gt_list):
    video_labels = video_info['annotations']  # the measurement is second, not frame
    video_second = float(video_info['duration'])
    for j in range(len(video_labels)):
        tmp_info = video_labels[j]
        tmp_start = (tmp_info['segment'][0] / video_second)
        tmp_end = (tmp_info['segment'][1] / video_second)
        if tmp_start < tmp_end:
            gt_list.append([tmp_start, tmp_end])

class VideoDataset():
    def __init__(self, cfg, subset="train"):
        self.temporal_scale = cfg.model.temporal_scale
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = cfg.mode

        self.feature_path = cfg.data.feature_path
        self.video_anno_path = cfg.data.video_anno

        if self.mode in 'training':
            self.proposals_path = cfg.data.train_proposals_path
        else:
            self.proposals_path = cfg.data.test_proposals_path

        self._getDatasetDict()

    def _getDatasetDict(self):
        ignore_train_videos = ['S8GtH2Zayds', 'saZkh1Xacp0', 'uRCf7b3qk0I', 'ukyFvye2yK0', 'VsZiOEzQqyI', 'KhkQyn-WblM']
        anno_database = load_json(self.video_anno_path)
        anno_database = anno_database['database']
        self.video_dict = {}
        self.dirty_instance_cnt = 0
        self.dirty_video_cnt = 0
        gt_list = []
        for video_name, anno in tqdm(anno_database.items(), total=len(anno_database)):
            video_subset = anno['subset']
            if 'train' in video_subset:
                collect_gt(anno, gt_list)
            if self.subset not in video_subset:
                continue
            if video_name in ignore_train_videos:
                self.dirty_video_cnt += 1
                continue
            if self.mode in "training":
                video_info = self._filter_dirty_data(anno)
            else:
                video_info = anno
            if video_info is None:
                self.dirty_video_cnt += 1
                continue
            self.video_dict[video_name] = video_info
        self.video_list = sorted(list(self.video_dict.keys()))
        print("%s subset video numbers: %d, drop instance: %d drop video: %d" % (self.subset,
                                                                                 len(self.video_list),
                                                                                 self.dirty_instance_cnt,
                                                                                 self.dirty_video_cnt))
        return gt_list


    def _filter_dirty_data(self, anno):
        new_anno = {"annotations": [],
                    "duration": anno["duration"],
                    "subset": anno["subset"]}
        for a in anno["annotations"]:
            if (a['segment'][1] - a['segment'][0]) > 1:
                new_anno["annotations"].append(a)
            else:
                self.dirty_instance_cnt += 1
        if len(new_anno["annotations"]) > 0:
            return new_anno
        else:
            return None

    def __getitem__(self, index):
        video_data, video_duration, video_name = self._load_feature(index)
        xmin, xmax, score, iou, ioa, gt_xmin, gt_xmax = self._load_proposals(video_name, video_duration)
        meta = {}
        if self.mode in 'training':
            topk = 64
        else:
            topk = score.shape[0]
            # topk = 100
        feature, proposals, gt_boxes, feature_len, temporal_mask = self._sample_data_for_tcanet(video_data,
                                                                                                 xmin, xmax,
                                                                                                 topk,
                                                                                                 video_duration,
                                                                                                 gt_xmin, gt_xmax)
        meta["features"] = feature.astype(np.float32)
        meta["gt_boxes"] = gt_boxes.astype(np.float32)
        meta["proposals"] = proposals.astype(np.float32)
        meta["feature_len"] = np.array([feature_len])
        meta["video_duration"] = np.array([video_duration]).astype(np.float32)
        meta['temporal_mask'] = temporal_mask
        score = np.array(score)
        if self.mode in 'training':
            meta["score"] = np.pad(score,(0, 100-len(score)),'constant')
        else:
            meta["score"] = score
        meta["len_score"] = np.array([len(score)])
        return meta["features"], meta["gt_boxes"], meta["proposals"], meta["feature_len"],meta["video_duration"], meta['temporal_mask'], meta["score"], meta["len_score"]

    def _load_feature(self, index):
        video_name =  self.video_list[index]
        video_info = self.video_dict[video_name]
        video_duration = float(video_info['duration'])
        # load features
        feats = np.load(self.feature_path + "v_" + video_name + ".npy").astype(np.float32)
        # T x C -> C x T
        feats = feats.transpose()
        video_data = np.expand_dims(feats, axis=0)
        return video_data, video_duration, video_name

    def _load_proposals(self, video_name, duration):
        mode = 1
        if mode == 1:
            df = pd.read_csv(os.path.join(self.proposals_path, video_name + ".csv"))
            xmin = df.xmin.values[:]
            xmax = df.xmax.values[:]
            score = df.score.values[:]
            iou = df.iou.values[:]
            ioa = df.ioa.values[:]
            if self.mode in "training":
                gt_xmin = df.gt_xmin.values[:]
                gt_xmax = df.gt_xmax.values[:]
            else:
                gt_xmin = np.zeros_like(iou)
                gt_xmax = np.zeros_like(iou)
            return xmin, xmax, score, iou, ioa, gt_xmin, gt_xmax
        elif mode == 2:
            xmin = self.anchors[:, 0] * duration
            xmax = self.anchors[:, 1] * duration
            score = np.ones_like(xmin)
            iou = np.zeros_like(xmin)
            ioa = np.zeros_like(xmin)
            gt_xmin = np.zeros_like(xmin)
            gt_xmax = np.zeros_like(xmin)
            return xmin, xmax, score, iou, ioa, gt_xmin, gt_xmax
        elif mode == 3:
            df = pd.read_csv(os.path.join(self.proposals_path, video_name + ".csv"))
            xmin = df.xmin.values[:]
            xmax = df.xmax.values[:]
            score = df.score.values[:]
            xmin_score = df.xmin_score.values[:]
            xmax_score = df.xmax_score.values[:]
            score = np.stack([score, xmin_score, xmax_score], axis=1)
            iou = df.iou.values[:]
            ioa = df.ioa.values[:]
            if self.mode in "training":
                gt_xmin = df.gt_xmin.values[:]
                gt_xmax = df.gt_xmax.values[:]
            else:
                gt_xmin = np.zeros_like(iou)
                gt_xmax = np.zeros_like(iou)
            return xmin, xmax, score, iou, ioa, gt_xmin, gt_xmax

    def _sample_data_for_tcanet(self, feature, xmin, xmax, topk, video_duration, gt_xmin, gt_xmax):
        if topk > xmin.shape[0]:
            rel_topk = xmin.shape[0]
        else:
            rel_topk = topk

        feature_len = feature.shape[2]
        # if self.mode in 'training':
        t_max = self.temporal_scale
        if feature.shape[2] <= t_max:
            full_feature = np.zeros((feature.shape[1], t_max))
            assert full_feature.shape[1] >= feature.shape[2]
        else:
            full_feature = np.zeros((feature.shape[1], feature.shape[2]))
        temporal_mask = np.ones((full_feature.shape[1],), dtype=np.bool)
        full_feature[:, :feature.shape[2]] = feature[0, :, :]
        temporal_mask[:feature.shape[2]] = False

        proposals = np.zeros((topk, 3))
        gt_boxes = np.zeros((topk, 2))
        proposals[:rel_topk, 0] = xmin[:rel_topk]
        proposals[:rel_topk, 1] = xmax[:rel_topk]
        gt_boxes[:rel_topk, 0] = gt_xmin[:rel_topk]
        gt_boxes[:rel_topk, 1] = gt_xmax[:rel_topk]
        return full_feature, proposals, gt_boxes, feature_len, temporal_mask

    def __len__(self):
        return len(self.video_list)


def createDataset(cfg, mode) -> Tuple[ds.BatchDataset, dict]:
    """create dataset"""

    subset = mode
    if mode == "eval":
        subset = 'validation'
    data = VideoDataset(cfg, subset)

    if mode == 'train':
        columns = ["features", "gt_boxes", "proposals", "feature_len", "video_duration", 'temporal_mask', "score", "len_score"]
    else:
        columns = ["features", "gt_boxes", "proposals", "feature_len", "video_duration", 'temporal_mask', "score", "len_score"]
    shuffle = mode == "train"
    drop_remainder = mode == "train"
    dataset = ds.GeneratorDataset(data, column_names=columns,
                                  shuffle=shuffle, num_parallel_workers=cfg.data.threads, max_rowsize=32)

    dataset = dataset.batch(cfg[mode].batch_size, drop_remainder=drop_remainder)
    return dataset
