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

import logging
import numpy as np
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from mindspore import dtype as mstype
log = logging.getLogger(__name__)



class Bi_Loss(nn.LossBase):
    def __init__(self, reduction='mean'):
        super(Bi_Loss, self).__init__(reduction)
        self.reduce_sum = ops.ReduceSum()
        self.reduce_mean = ops.ReduceMean()
        self.log = ops.Log()

    def construct(self, logits, labels):
        pmask = self.cast(labels > 0.5, mstype.float32)
        num_entries = pmask.size
        num_positive = self.reduce_sum(pmask)
        ratio = num_entries / num_positive
        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio
        epsilon = 1e-6
        loss_pos = coef_1 * self.log(logits + epsilon) * pmask
        loss_neg = coef_0 * self.log(1.0 - logits + epsilon) * (1.0 - pmask)
        loss = -1 * self.reduce_mean(loss_pos + loss_neg)
        return loss


class AN_classifier_Loss(nn.Cell):
    def __init__(self, mode='train'):
        super(AN_classifier_Loss, self).__init__()
        self.stack = ops.Stack()
        self.unstack = ops.Unstack(axis=1)
        self.slice = ops.Slice()
        self.bi_loss = Bi_Loss()
        self.mode = mode

    def construct(self, pred_start, pred_end, pred_action, pred_num, gt_start, gt_end, gt_action, gt_num):
        start_loss = self.bi_loss(pred_start, gt_start)
        end_loss = self.bi_loss(pred_end, gt_end)
        action_loss = self.bi_loss(pred_action, gt_action)
        tem_loss = start_loss + end_loss + action_loss
        
        criterion =  nn.BCEWithLogitsLoss()

        class_num_aciton_loss = criterion(pred_num, gt_num.float())

        loss = tem_loss + 10 * class_num_aciton_loss
        return loss, tem_loss, 10 * class_num_aciton_loss

