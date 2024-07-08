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
"""eval"""

import logging
import matplotlib
import numpy as np
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.callback import TimeMonitor, SummaryCollector
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.an_classifier import AN_classifier, AN_classifier_WithEvalCell
from src.config import config as cfg
from src.dataset import createDataset

matplotlib.use('Agg')

logging.basicConfig()
logger = logging.getLogger(__name__)

logger.info("Training configuration:\n\v%s\n\v", (cfg.__str__()))

logger.setLevel(cfg.log_level)

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.platform)

    #datasets
    eval_dataset, tem_train_dict = createDataset(cfg, mode='eval')
    batch_num = eval_dataset.get_dataset_size()

    #network
    network = AN_classifier(cfg.model)
    logger.info("Network created")


    #checkpoint
    param_dict = load_checkpoint(cfg.eval.checkpoint)
    load_param_into_net(network, param_dict)

    # train net
    eval_net = AN_classifier_WithEvalCell(network)

    #accuracy
    model = Model(eval_net,
                  eval_network=eval_net,
                  metrics={"accuracy"}, 
                  loss_fn=None)

    results = model.eval(valid_dataset=eval_dataset)
    print(results)

    # #precision
    # model = Model(eval_net,
    #               eval_network=eval_net,
    #               metrics={"precision"}, 
    #               loss_fn=None)

    # results = model.eval(valid_dataset=eval_dataset)
    # print(results)

    # #recall
    # model = Model(eval_net,
    #               eval_network=eval_net,
    #               metrics={"recall"}, 
    #               loss_fn=None)

    # results = model.eval(valid_dataset=eval_dataset)
    # print(results)


    

