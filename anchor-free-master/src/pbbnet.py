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

import math
import logging
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import ipdb
from src.tmm import TemporalEncoder                                            
from src.ppbm import TemporalBoundaryRegressor
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

class PBBNet(nn.Cell):
    def __init__(self, cfg) -> None:
        super(PBBNet, self).__init__()

        self.input_dim = cfg.model.feat_dim
        self.mode = cfg.mode
        self.lgte_num = cfg.model.lgte_num
        self.hidden_dim_1d = 512

        self.x_1d_b_f = nn.SequentialCell(
            nn.Conv1d(self.input_dim,
                      self.hidden_dim_1d,
                      kernel_size=3,
                      pad_mode='pad',
                      padding=1,
                      has_bias=True,
                      group=4),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim_1d,
                      self.hidden_dim_1d,
                      kernel_size=3,
                      pad_mode='pad',
                      padding=1,
                      has_bias=True,
                      group=4),
            nn.ReLU()
        )

        self.tbr1 = TemporalBoundaryRegressor(cfg)
        self.tbr2 = TemporalBoundaryRegressor(cfg)
        self.tbr3 = TemporalBoundaryRegressor(cfg)

        self.lgtes = nn.CellList(
            [TemporalEncoder(self.hidden_dim_1d, 0.1) for i in range(self.lgte_num)])

    def construct(self, features, video_second, proposals, gt_boxes, temporal_mask):
        training = self.mode in 'training'
        return self.process(features, gt_boxes, proposals, video_second, training)

    def process(self, features, gt_boxes, proposals, video_sec, training):
        features = self.x_1d_b_f(features)
        for layer in self.lgtes:
            features = layer(features)

        batch_size = proposals.shape[0]
        proposals_num = proposals.shape[1]
        for i in range(batch_size):
            proposals[i, :, 2] = i
        proposals = proposals.view(batch_size * proposals_num, 3)
        proposals_select = proposals[:, 0:2].sum(1) > 0
        proposals = proposals[proposals_select, :]

        batch_idx = proposals[:, 2].long()
        features = features[batch_idx]
        video_sec = video_sec[batch_idx].float()
        if training:
            gt_boxes = gt_boxes.view(batch_size * proposals_num, 2)
            gt_boxes = gt_boxes[proposals_select, :]

        preds_iou1, proposals1, rloss1, iloss1 = self.tbr1(proposals, features, video_sec, gt_boxes, 0.5, training)
        preds_iou2, proposals2, rloss2, iloss2 = self.tbr2(proposals1, features, video_sec, gt_boxes, 0.6, training)
        preds_iou3, proposals3, rloss3, iloss3 = self.tbr3(proposals2, features, video_sec, gt_boxes, 0.7, training)

        if training:
            loss_meta = {"rloss1": rloss1, "rloss2": rloss2, "rloss3": rloss3,
                         "iloss1": iloss1, "iloss2": iloss2, "iloss3": iloss3,
                         "total_loss": rloss1 + rloss2 + rloss3 + iloss1 + iloss2 + iloss3}
            if ops.isnan(loss_meta["total_loss"]):
                ipdb.set_trace()
            return loss_meta
        else:
            preds_meta = {"proposals1": proposals1, "proposals2": proposals2, "proposals3": proposals3,
                          "iou1": preds_iou1.view(-1), "iou2": preds_iou2.view(-1), "iou3": preds_iou3.view(-1)}
            return preds_meta


class PBBNetWithLossCell(nn.Cell):
    def __init__(self, net, loss, auto_prefix=False, flags=None):
        super(PBBNetWithLossCell, self).__init__()
        self.network = net
        self.loss = loss

    def construct(self, features, gt_boxes, proposals, feature_len, video_duration, temporal_mask, score, len_score):
        loss_meta = self.network(features, video_duration, proposals, gt_boxes, temporal_mask)
        loss = self.loss(loss_meta)
        return loss


class PBBNetWithEvalCell(nn.Cell):
    def __init__(self, net, auto_prefix=False, flags=None):
        super(PBBNetWithEvalCell, self).__init__()
        self.network = net

    def construct(self, features, gt_boxes, proposals, feature_len, video_duration, temporal_mask, score, len_score):
        preds_meta = self.network(features, video_duration, proposals, gt_boxes, temporal_mask)
        return preds_meta["proposals1"], preds_meta["proposals2"], preds_meta["proposals3"], \
                preds_meta["iou1"], preds_meta["iou2"], preds_meta["iou3"], \
                proposals, video_duration, score
