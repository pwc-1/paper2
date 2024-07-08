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
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

class AFAN(nn.Cell):
    def __init__(self, cfg) -> None:
        super(AFAN, self).__init__()
        self.tscale = cfg.temporal_scale
        self.prop_boundary_ratio = cfg.prop_boundary_ratio
        self.num_sample = cfg.num_sample
        self.num_sample_perbin = cfg.num_sample_perbin
        self.feat_dim = cfg.feat_dim

        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512

        self._get_interp1d_mask()

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


        # Proposal Evaluation Module
        self.x_1d_p = nn.SequentialCell(
            nn.Conv1d(self.hidden_dim_1d,
                      self.hidden_dim_1d,
                      kernel_size=3,
                      pad_mode='pad',
                      has_bias=True,
                      padding=1),
            nn.ReLU()
        )
        self.x_3d_p = nn.SequentialCell(
            nn.Conv3d(self.hidden_dim_1d,
                      self.hidden_dim_3d,
                      kernel_size=(self.num_sample, 1, 1),
                      has_bias=True,
                      stride=(self.num_sample, 1, 1)),
            nn.ReLU()
        )
        self.x_2d_p = nn.SequentialCell(
            nn.Conv2d(self.hidden_dim_3d,
                      self.hidden_dim_2d,
                      has_bias=True,
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim_2d,
                      self.hidden_dim_2d,
                      kernel_size=3,
                      pad_mode='pad',
                      has_bias=True,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim_2d,
                      self.hidden_dim_2d,
                      kernel_size=3,
                      pad_mode='pad',
                      has_bias=True,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim_2d,
                      out_channels=2,
                      has_bias=True,
                      kernel_size=1),
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
        start = self.x_1d_s(base_feature).squeeze(1)
        end = self.x_1d_e(base_feature).squeeze(1)
        confidence_map = self.x_1d_p(base_feature)
        confidence_map = self._boundary_matching_layer(confidence_map)
        confidence_map = self.x_3d_p(confidence_map).squeeze(2)
        confidence_map = self.x_2d_p(confidence_map)
        return confidence_map, start, end

    def _boundary_matching_layer(self, x):
        input_size = x.shape
        sample_mask = self.repeat(self.unsqueeze(self.sample_mask, 0), input_size[0], axis=0)
        out = self.batmul(x, sample_mask).reshape(input_size[0], \
                                                  input_size[1], \
                                                  self.num_sample, \
                                                  self.tscale, \
                                                  self.tscale)
        return out

    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask

    def _get_interp1d_mask(self):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []
        for end_index in range(self.tscale):
            mask_mat_vector = []
            for start_index in range(self.tscale):
                if start_index <= end_index:
                    p_xmin = start_index
                    p_xmax = end_index + 1
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin, sample_xmax, self.tscale, self.num_sample,
                        self.num_sample_perbin)
                else:
                    p_mask = np.zeros([self.tscale, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)
        self.sample_mask = Tensor(mask_mat).view(self.tscale, -1)
        ops.stop_gradient(self.sample_mask)

class AFANWithLossCell(nn.Cell):
    def __init__(self, net, net_expert_1, net_expert_2, loss, auto_prefix=False, flags=None):
        super(AFANWithLossCell, self).__init__(auto_prefix, flags)
        self.network = net
        self.expert_1 = net_expert_1
        self.expert_2 = net_expert_2
        self.loss = loss


    def construct(self, features, confidence_score, match_score_start, match_score_end, num_action):
        confidence_map, start, end = self.network(features)
        confidence_map_1, start_1, end_1 = self.expert_1(features)
        confidence_map_2, start_2, end_2 = self.expert_2(features)
        loss = self.loss(confidence_map, start, end,
                        confidence_map_1, start_1, end_1,
                        confidence_map_2, start_2, end_2,
                        confidence_score, match_score_start, match_score_end, num_action)
        return loss


class AFANWithEvalCell(nn.Cell):
    def __init__(self, net, an_classifier, auto_prefix=False, flags=None):
        super(AFANWithEvalCell, self).__init__(auto_prefix, flags)
        self.network = net
        self.an_classifier = an_classifier

    def double_f(self, featrue):
        x = featrue[:,:,0].view(-1, 400, 1)
        for i in range(50):
            x = ops.cat([x, featrue[:,:,i].view(-1, 400, 1), featrue[:,:,i].view(-1, 400, 1)], axis =2)
        _, double_featrue = x.view(-1, 400, 101).split([1,100],axis=2)
        return double_featrue

    def construct(self, features):
        confidence_map, start, end = self.network(features)
        num_class, _, _, _ = self.an_classifier(features)

        feature_1, feature_2 = features.split([50,50], axis=2)
        input_data_1 = self.double_f(feature_1)
        input_data_2 = self.double_f(feature_2)
        confidence_map_1, start_1, end_1 = self.network(input_data_1)
        confidence_map_2, start_2, end_2 = self.network(input_data_2)


        start_scores = (start, start_1, start_2)
        end_scores = (end, end_1, end_2)
        clr_confidence = (confidence_map[:, 1], confidence_map_1[:, 1], confidence_map_2[:, 1])
        reg_confidence = (confidence_map[:, 0], confidence_map_1[:, 0], confidence_map_2[:, 0])
        num_class = num_class.reshape(-1)
        return start_scores, end_scores, clr_confidence, reg_confidence, num_class
