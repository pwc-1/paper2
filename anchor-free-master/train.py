
"""train"""
import os
import logging
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.model import Model
import mindspore.nn as nn
from mindspore.nn import Adam, ExponentialDecayLR
from mindspore.train.summary.summary_record import SummaryRecord
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from src.pbbnet import PBBNet, PBBNetWithLossCell
from src.loss import PBBNet_Loss
from src.config import config as cfg
from src.dataset import createDataset
from src.callbacks import CustomLossMonitor
import ipdb

if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger(__name__)

    logger.info("Training configuration:\n\v%s\n\v", (cfg.__str__()))

    logger.setLevel(cfg.log_level)

    context.set_context(mode=context.PYNATIVE_MODE, device_target=cfg.platform, save_graphs=False)

    #datasets
    cfg.mode = 'train'
    train_dataset = createDataset(cfg, mode='train')
    batch_num = train_dataset.get_dataset_size()
    
    #network
    network = PBBNet(cfg)
    
    #lr
    lr = ExponentialDecayLR(cfg.train.learning_rate,
                            cfg.train.step_gamma,
                            cfg.train.step_size * batch_num,
                            is_stair=True)

    #optimizer
    optim = Adam(params=network.trainable_params(),
               learning_rate=lr,
               weight_decay=cfg.train.weight_decay)

    #loss
    loss = PBBNet_Loss(mode="train")

    # train net
    train_net = PBBNetWithLossCell(network, loss)

    #models
    model = Model(network=train_net,
                  optimizer=optim,
                  loss_fn=None,)
    #callbacks
    ckpt_save_dir = cfg.ckpt.save_dir
    config_ck = CheckpointConfig(save_checkpoint_steps=batch_num,
                                 keep_checkpoint_max=cfg.ckpt.keep_checkpoint_max)

    ckpoint_cb = ModelCheckpoint(prefix="pbbnet_auto_",
                                 directory=ckpt_save_dir,
                                 config=config_ck)
    time_cb = TimeMonitor(data_size=batch_num)
    loss_cb = LossMonitor()

    cbs = [time_cb, ckpoint_cb, loss_cb]

    #train
    with SummaryRecord(cfg.summary_save_dir) as summary_record:
        loss_cb = CustomLossMonitor(summary_record=summary_record,
                                    mode="train")

        cbs += [loss_cb]
        logger.info("Start training model")
        model.train(epoch=cfg.train.epoch,
                    train_dataset=train_dataset,
                    callbacks=cbs,
                    dataset_sink_mode=False)

    logger.info("Exp. %s - PBBNet train success", (cfg.experiment_name))
