
"""eval"""
import os
import logging
import matplotlib
import numpy as np
import sys

import matplotlib.pyplot as plt
import json
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.callback import TimeMonitor, SummaryCollector
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.pbbnet import PBBNet, PBBNetWithEvalCell
from src.config import config as cfg
from src.dataset import createDataset
from src.metrics import PBBNetMetric
from src.postprocessing import post_processing
from src.evaluation.eval_proposal import ANETproposal
from src.evaluation.eval_detection import ANETdetection
matplotlib.use('Agg')

logging.basicConfig()
logger = logging.getLogger(__name__)

logger.info("Training configuration:\n\v%s\n\v", (cfg.__str__()))

logger.setLevel(cfg.log_level)


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def write_results(log):
    with open("./ap_result.txt", "w") as f:
        f.write(log)

def run_detection_evaluation(ground_truth_filename, detection_filename,
                             tiou_thresholds=np.linspace(0.5, 0.95, 10),
                             subset='validation', assign_class=None):
    anet_detection = ANETdetection(ground_truth_filename, detection_filename,
                                   tiou_thresholds=tiou_thresholds,
                                   subset=subset, verbose=True, check_status=False,
                                   assign_class=assign_class)
    anet_detection.evaluate()


def plot_metric(opt, average_nr_proposals, average_recall, recall, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    fn_size = 14
    plt.figure(num=None, figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)

    colors = ['k', 'r', 'yellow', 'b', 'c', 'm', 'b', 'pink', 'lawngreen', 'indigo']
    area_under_curve = np.zeros_like(tiou_thresholds)
    for i in range(recall.shape[0]):
        area_under_curve[i] = np.trapz(recall[i], average_nr_proposals)

    for idx, tiou in enumerate(tiou_thresholds[::2]):
        ax.plot(average_nr_proposals, recall[2 * idx, :], color=colors[idx + 1],
                label="tiou=[" + str(tiou) + "], area=" + str(int(area_under_curve[2 * idx] * 100) / 100.),
                linewidth=4, linestyle='--', marker=None)
    # Plots Average Recall vs Average number of proposals.
    ax.plot(average_nr_proposals, average_recall, color=colors[0],
            label="tiou = 0.5:0.05:0.95," + " area=" + str(
                int(np.trapz(average_recall, average_nr_proposals) * 100) / 100.),
            linewidth=4, linestyle='-', marker=None)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[-1]] + handles[:-1], [labels[-1]] + labels[:-1], loc='best')

    plt.ylabel('Average Recall', fontsize=fn_size)
    plt.xlabel('Average Number of Proposals per Video', fontsize=fn_size)
    plt.grid(b=True, which="both")
    plt.ylim([0, 1.0])
    plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
    plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
    plt.savefig(opt["save_fig_path"])


def evaluation_detection(cfg, assign_class=None):
    run_detection_evaluation(cfg.data.video_anno,
                            cfg.postprocessing.detection_result_file,
                             subset='validation',
                             assign_class=assign_class)





if __name__ == '__main__':
    context.set_context(mode=context.PYNATIVE_MODE, device_target=cfg.platform, save_graphs=False)

    #datasets
    cfg.mode = 'validation'
    eval_dataset = createDataset(cfg, mode='eval')
    batch_num = eval_dataset.get_dataset_size()

    #network
    network = PBBNet(cfg)

    #checkpoint
    param_dict = load_checkpoint(cfg.eval.checkpoint)
    load_param_into_net(network, param_dict)

    # train net
    eval_net = PBBNetWithEvalCell(network)

    # metrics
    metric = PBBNetMetric(cfg, subset="validation")

    #models
    model = Model(eval_net,
                  eval_network=eval_net,
                  metrics={"pbbnet_metric": metric},
                  loss_fn=None)

    #callbacks
    time_cb = TimeMonitor(data_size=batch_num)
    summary_collector = SummaryCollector(summary_dir=cfg.summary_save_dir, collect_freq=1)

    cbs = [time_cb]

    model.eval(valid_dataset=eval_dataset,
               callbacks=cbs,
               dataset_sink_mode=False)

    logger.info("Evaluatiom started")

    post_processing(cfg)
    evaluation_detection(cfg)
