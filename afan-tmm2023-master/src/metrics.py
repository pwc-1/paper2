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
from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm
import mindspore.ops as ops
from mindspore.nn import Metric
from src.utils import dump_metric_result
from src.postprocessing import post_processing
import ipdb
logger = logging.getLogger(__name__)

class AFANMetric(Metric):
    def __init__(self, cfg, subset='train_val'):
        self.tscale = cfg.model.temporal_scale
        self.subset = subset  # 'train', 'validation', 'train_val'
        self.anno_file = cfg.data.video_annotations
        self.postpr_config = cfg
        self.get_dataset_dict()
        self.output_path = cfg.eval.output_path
        self.pbar_update = tqdm(total=len(self.video_list),\
                                postfix="\n")
        self.pbar_update.set_description("Collecting AFAN metrics")
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
        annos = json.load(open(self.anno_file))
        self.video_dict = {}
        for video_name in annos.keys():
            video_subset = annos[video_name]["subset"]
            if self.subset == "train_val":
                if "train" in video_subset or "validation" in video_subset:
                    self.video_dict[video_name] = annos[video_name]
            else:
                if self.subset in video_subset:
                    self.video_dict[video_name] = annos[video_name]
        self.video_list = list(self.video_dict.keys())
        self.video_list.sort()

    def clear(self):
        logger.info('Resetting %s metrics...', self.subset)
        self.sample_counter = 0
        self.pbar_update.reset()
        self.results_pairs = []

    def update(self, *fetch_list):
        cur_batch_size = fetch_list[-1].shape[0]
        all_s_score, all_e_score, all_clr_conf, all_reg_conf, pred_num = fetch_list

        new_props = []
        s_score, s_score_1, s_score_2 = all_s_score[0].asnumpy(), all_s_score[1].asnumpy(), all_s_score[2].asnumpy()
        e_score, e_score_1, e_score_2 = all_e_score[0].asnumpy(), all_e_score[1].asnumpy(), all_e_score[2].asnumpy()
        clr_conf, clr_conf_1, clr_conf_2 = all_clr_conf[0].asnumpy(), all_clr_conf[1].asnumpy(), all_clr_conf[2].asnumpy()
        reg_conf, reg_conf_1, reg_conf_2 = all_reg_conf[0].asnumpy(), all_reg_conf[1].asnumpy(), all_reg_conf[2].asnumpy()
        p_num = pred_num.asnumpy()

        # ipdb.set_trace()

        if p_num[0] > 0.5:
            s_score = s_score[0]
            e_score = e_score[0]
            clr_conf = clr_conf[0]
            reg_conf = reg_conf[0]
            for idx in range(self.tscale):
                for jdx in range(self.tscale):
                    start_index = idx
                    end_index = jdx + 1
                    if start_index < end_index < self.tscale:
                        xmin = start_index / self.tscale
                        xmax = end_index / self.tscale
                        xmin_score = s_score[start_index]
                        xmax_score = e_score[end_index]
                        clr_score = clr_conf[idx, jdx]
                        reg_score = reg_conf[idx, jdx]
                        score = xmin_score * xmax_score * clr_score * reg_score
                        new_props.append([xmin, xmax, xmin_score, xmax_score, clr_score, reg_score, score])
        else:
            s_score = s_score[0]
            e_score = e_score[0]
            clr_conf = clr_conf[0]
            reg_conf = reg_conf[0]
            clr_conf_1 = clr_conf_1[0]
            reg_conf_1 = reg_conf_1[0]
            clr_conf_2 = clr_conf_2[0]
            reg_conf_2 = reg_conf_2[0]
            for idx in range(self.tscale):
                for jdx in range(self.tscale):
                    start_index = idx
                    end_index = jdx + 1
                    if start_index < end_index and  end_index<self.tscale :
                        xmin = start_index / self.tscale
                        xmax = end_index / self.tscale
                        if start_index < 50 and end_index < 50:
                            xmin_score = s_score[start_index]
                            xmax_score = e_score[end_index]
                            clr_score = clr_conf_1[2*idx, 2*jdx + 1]
                            reg_score = reg_conf_1[2*idx, 2*jdx + 1]

                        elif start_index < 50 and end_index >= 50:
                            xmin_score = s_score[start_index]
                            xmax_score = e_score[end_index]
                            clr_score = clr_conf[idx, jdx]
                            reg_score = reg_conf[idx, jdx]

                        elif start_index >= 50 and end_index >= 50:
                            xmin_score = s_score[start_index]
                            xmax_score = e_score[end_index]
                            clr_score = clr_conf_2[2*(idx-50), 2*(jdx-50) + 1]
                            reg_score = reg_conf_2[2*(idx-50), 2*(jdx-50) + 1]
                        score = xmin_score * xmax_score * clr_score * reg_score

                        new_props.append([xmin, xmax, xmin_score, xmax_score, clr_score, reg_score, score])

        new_props = np.stack(new_props)

        video_name = self.video_list[self.sample_counter]
        col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_socre", "score"]
        new_df = pd.DataFrame(new_props, columns=col_name)
        new_df.to_csv(os.path.join(self.output_path, video_name + ".csv"), index=False)
        # print(self.sample_counter, os.path.join(self.output_path, video_name + ".csv"))
        self.pbar_update.update(cur_batch_size)
        self.sample_counter += cur_batch_size

    def eval(self):
        logger.info("Dumping results...")
        logger.info("start generate proposals of %s subset", (self.subset))
        logger.warning("NOT DONE WAIT FOR THE NEXT LOGGER MESSAGE")
        ipdb.set_trace()
        post_processing(self.postpr_config)
        logger.info("finish generate proposals of %s subset", (self.subset))
