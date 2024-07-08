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


class PBBNet_Loss(nn.Cell):
    def __init__(self, mode='train'):
        super(PBBNet_Loss, self).__init__()
        self.mode = mode

    def construct(self, loss_meta):
        if self.mode == "train":
            return loss_meta["total_loss"]
        return loss_meta
