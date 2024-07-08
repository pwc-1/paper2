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

import shutil
import json
import logging
import os
import ipdb
from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm
import mindspore.ops as ops
from mindspore.nn import Metric
from src.utils import dump_metric_result

logger = logging.getLogger(__name__)

def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data


class PBBNetMetric(Metric):
    def __init__(self, cfg, subset='validation'):
        self.tscale = cfg.model.temporal_scale
        self.subset = subset  # 'train', 'validation', 'train_val'
        self.video_anno_path = cfg.data.video_anno
        self.postpr_config = cfg
        self.get_dataset_dict()
        self.output_path = cfg.eval.output_path
        self.pbar_update = tqdm(total=len(self.video_list),\
                                postfix="\n")
        self.pbar_update.set_description("Collecting BMN metrics")
        self.clear()
        self.unstack = ops.Unstack(axis=0) # loss unpack
        self.reduce_mean = ops.ReduceMean()
        self.cast = ops.Cast()
        self.sample_counter = 0
        self.results_pairs = []
        self.threads = cfg.eval.threads
        self.env_setup()


    def env_setup(self):
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path, exist_ok=True)

    def get_dataset_dict(self):
        anno_database = load_json(self.video_anno_path)
        anno_database = anno_database['database']
        self.video_dict = {}

        for video_name, anno in tqdm(anno_database.items(), total=len(anno_database)):
            video_subset = anno['subset']
            if self.subset not in video_subset:
                continue
            video_info = anno
            self.video_dict[video_name] = video_info
        self.video_list = sorted(list(self.video_dict.keys()))

    def clear(self):
        logger.info('Resetting %s metrics...', self.subset)
        self.sample_counter = 0
        self.pbar_update.reset()
        self.results_pairs = []

    def update(self, *fetch_list):
        cur_batch_size = 1
        proposals1, proposals2, proposals3, iou1, iou2, iou3, proposals, video_duration, score = fetch_list
        iou1 = iou1.asnumpy()
        iou2 = iou2.asnumpy()
        iou3 = iou3.asnumpy()
        
        video_duration = video_duration.asnumpy().item()
        proposals1 = proposals1.asnumpy() / video_duration
        proposals2 = proposals2.asnumpy() / video_duration
        proposals3 = proposals3.asnumpy() / video_duration
        score = score.asnumpy()
        # ipdb.set_trace()
        new_props = np.stack(
            [proposals1[:,0], proposals2[:,0], proposals2[:,0],
            proposals1[:,1], proposals2[:,1], proposals2[:,1],
            score[0], iou1, iou2, iou3],axis=1)

        video_name = self.video_list[self.sample_counter]
        col_name = ["xmin1", "xmin2", "xmin3", 
                    "xmax1", "xmax2", "xmax3",
                    "ori_score", "pred_iou1", "preds_iou2", "preds_iou3"]
        new_df = pd.DataFrame(new_props, columns=col_name)
        new_df.to_csv(os.path.join(self.output_path, video_name + ".csv"), index=False)
        self.pbar_update.update(cur_batch_size)
        self.sample_counter += cur_batch_size

    def eval(self):
        logger.info("Dumping results...")
        self.dump_results()

    def dump_results(self):
        with Pool(self.threads) as p:
            p.map(dump_metric_result, self.results_pairs)
