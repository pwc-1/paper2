import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P
import mindspore.numpy as np
from mindspore.ops.operations import Equal


class TripletLoss(nn.Cell):
    def __init__(self, margin, loss_term_weight=1.0):
        super(TripletLoss, self).__init__(loss_term_weight)
        self.margin = margin
        self.permute = mindspore.ops.Transpose()
        self.expand_dims = mindspore.ops.ExpandDims()
        self.logical_not = mindspore.ops.LogicalNot()
        self.cast = mindspore.ops.Cast()
        self.equal = Equal()

    def construct(self, embeddings, labels):
        # embeddings: [n, c, p], label: [n]
        embeddings = self.permute(embeddings, (2, 0, 1))

        ref_embed, ref_label = embeddings, labels
        dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]
        mean_dist = dist.mean((1, 2))  # [p]
        ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)
        dist_diff = (ap_dist - an_dist).view(dist.shape[0], -1)
        loss = ops.ReLU()(dist_diff + self.margin)

        hard_loss = mindspore.ops.max(loss, -1)[0]
        loss_avg, loss_num = self.AvgNonZeroReducer(loss)

        return loss_avg

    def AvgNonZeroReducer(self, loss):
        loss_float = self.cast(loss, mindspore.float32)
        loss_bool = P.NotEqual()(loss_float, 0)
        loss_sum = P.ReduceSum(keep_dims=False)(loss_bool, -1)
        loss_num = self.cast(loss_sum, mindspore.float32)
        return loss_sum, loss_num

    def ComputeDistance(self, x, y):
        x = ops.expand_dims(x, 2)  # [p, n_x, 1, c]
        y = ops.expand_dims(y, 3)  # [p, n_y, c, 1]
        inner = P.BatchMatMul()(x, y)  # [p, n_x, 1, 1]
        dist = ops.squeeze(inner, 3)  # [p, n_x, 1]
        dist = ops.sqrt(ops.ReLU()(dist))  # [p, n_x, 1]
        return dist

    def Convert2Triplets(self, row_labels, clo_label, dist):
        row_labels = self.expand_dims(self.cast(row_labels, mindspore.float32), 1)  # [n_r, 1]
        matches = self.logical_not(self.equal(row_labels, self.expand_dims(self.cast(clo_label, mindspore.float32), 0)))  # [n_r, n_c]
        matches_int = self.cast(matches, mindspore.int32)
        diffenc = self.logical_not(matches)  # [n_r, n_c]
        diffenc_int = self.cast(diffenc, mindspore.int32)
        p, n, _ = dist.shape
        ap_dist = dist[:, matches_int].view(p, n, -1, 1)
        an_dist = dist[:, diffenc_int].view(p, n, 1, -1)
        return ap_dist, an_dist