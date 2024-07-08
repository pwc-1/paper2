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
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

class AN_classifier(nn.Cell):
    def __init__(self, cfg) -> None:
        super(AN_classifier, self).__init__()
        self.tscale = cfg.temporal_scale
        self.prop_boundary_ratio = cfg.prop_boundary_ratio
        self.num_sample = cfg.num_sample
        self.num_sample_perbin = cfg.num_sample_perbin
        self.feat_dim = cfg.feat_dim

        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512

        # self._get_interp1d_mask()

        # Base Module
        self.x_1d_b = nn.SequentialCell(
            nn.Conv1d(self.feat_dim,
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

        # Temporal Evaluation Module
        self.x_1d_s = nn.SequentialCell(
            nn.Conv1d(self.hidden_dim_1d,
                      self.hidden_dim_1d,
                      kernel_size=3,
                      pad_mode='pad',
                      padding=1,
                      has_bias=True,
                      group=4),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim_1d,
                      out_channels=1,
                      has_bias=True,
                      kernel_size=1),
            nn.Sigmoid()
        )

        self.x_1d_e = nn.SequentialCell(
            nn.Conv1d(self.hidden_dim_1d,
                      self.hidden_dim_1d,
                      kernel_size=3,
                      pad_mode='pad',
                      padding=1,
                      has_bias=True,
                      group=4),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim_1d,
                      out_channels=1,
                      has_bias=True,
                      kernel_size=1),
            nn.Sigmoid()
        )

        self.x_1d_a = nn.SequentialCell(
            nn.Conv1d(self.hidden_dim_1d,
                      self.hidden_dim_1d,
                      kernel_size=3,
                      pad_mode='pad',
                      padding=1,
                      has_bias=True,
                      group=4),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim_1d,
                      out_channels=1,
                      has_bias=True,
                      kernel_size=1),
            nn.Sigmoid()
        )



        self.fc = nn.SequentialCell(
            nn.Dense(300, 100),
            nn.ReLU(),
            nn.Dense(100, 25),
            nn.ReLU(),
            nn.Dense(25, 1),
            nn.Sigmoid()
            )


        self.stack = ops.Stack()
        self.cat = ops.Concat(1)
        self.reshape = ops.Reshape()
        self.unsqueeze = ops.ExpandDims()
        self.repeat = ops.repeat_elements
        self.batmul = ops.BatchMatMul()

    def construct(self, x):
        base_feature = self.x_1d_b(x)
        start = self.x_1d_s(base_feature)
        end = self.x_1d_e(base_feature)
        action = self.x_1d_a(base_feature)
        confidence_map = ops.cat([start, end, action], 1) #bs, 3, 100
        num_class = self.fc(confidence_map.reshape(-1, 300))

        return num_class, start.squeeze(1), end.squeeze(1), action.squeeze(1)


class AN_classifier_WithLossCell(nn.Cell):
    def __init__(self, net, loss, auto_prefix=False, flags=None):
        super(AN_classifier_WithLossCell, self).__init__(auto_prefix, flags)
        self.network = net
        self.loss = loss


    def construct(self, features, match_score_start, match_score_end, match_score_action, match_num_action):
        
        num_class, start, end, action = self.network(features)
        loss = self.loss(start, end, action, num_class, match_score_start, match_score_end, match_score_action, match_num_action)
        return loss

class AN_classifier_WithEvalCell(nn.Cell):
    def __init__(self, net, auto_prefix=False, flags=None):
        super(AN_classifier_WithEvalCell, self).__init__(auto_prefix, flags)
        self.network = net

    def construct(self, features, num_action):
        num_class, start, end, action = self.network(features)
        num_action = num_action.reshape(-1)
        num_class = num_class.reshape(-1)
        if num_action == 1:
            label = Tensor([[0,1]])
        else:
            label = Tensor([[1,0]])
        if num_class > 0.3:
            pred = Tensor([[0,1]])
        else:
            pred = Tensor([[1,0]])
        # print(num_class.shape, num_action.shape)
        return pred, label
