import os
import numpy as np

import mindspore
import mindspore.ops as ops
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import Model, Tensor, Parameter
from mindspore import dtype as mstype
from mindspore.nn import Cell, Dense, SoftmaxCrossEntropyWithLogits, Momentum, TrainOneStepCell, WithLossCell
from mindspore import ParameterTuple
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore import context, DatasetHelper, connect_network_with_dataset

from data.ccpg import CCPG_DataSet
from losses.crossentropy_loss import CrossEntropyLoss
from losses.triplet_loss import TripletLoss
from model.ogbase_aug import OGBASE_AUG

CCPG_path = "./data/opengait_data_128"
train_dataset = CCPG_DataSet(CCPG_path, mode="Train")
dataset4train = ds.GeneratorDataset(train_dataset, ["silh_data", "id_label", "cloth_label", "view_label"], shuffle=True)
dataset4train = dataset4train.batch(64)

class Compute_Loss(nn.LossBase):
    def construct(self, logits, labels):
        # print(logits, labels)
        ce_criterion = CrossEntropyLoss()
        tr_criterion = TripletLoss(margin=0.3)
        loss_ce = ce_criterion(logits[1], labels)
        loss_tr = tr_criterion(logits[0], labels)
        return loss_ce + loss_tr


# net_opt = nn.Momentum(network.trainable_params(), 0.1, 0.9)

    
class TrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, data, label):
        weights = self.weights
        loss = self.network(data, label)
        grads = self.grad(self.network, weights)(data, label)
        # print(grads)
        # for name, param in self.network.trainable_params():
        #     print(f"Param {name}: Value {param.asnumpy()}")
        return ops.depend(loss, self.optimizer(grads))
    
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
network = OGBASE_AUG()
net_loss = Compute_Loss()
# net_opt = nn.Momentum(network.trainable_params(), 0.005, 0.9)
net_opt = nn.Adam(network.trainable_params(), 1.5, 0.1)
# net_opt = RiemannianAdam(network.trainable_params(), 0.01, 0.9)
net = WithLossCell(network, net_loss)
net = TrainOneStepCell(net, net_opt)
network.set_train()
print("============== Starting Training ==============")
epochs = 500


# from mindspore import context
# from mindspore.parallel import ParallelMode

# # 设置运行环境和设备
# context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
# context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL)
# dp_model = nn.DataParallel(net, loss_repeated_times=1)


for epoch in range(epochs):
    for inputs in dataset4train:
        output = net(inputs[0].astype(mindspore.float32)/255., inputs[1])
        print("epoch: {0}/{1}, losses: {2}".format(epoch + 1, epochs, output.asnumpy().mean(), flush=True))

    if epoch % 10 == 0:
        mindspore.save_checkpoint(network,
                            os.path.join("/home/liweijia/run_mind/cvpr2023", str(epoch + 1)+"train.ckpt"))
    # break

print("============== Training Finished ==============")