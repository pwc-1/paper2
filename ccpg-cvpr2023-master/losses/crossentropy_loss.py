import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

class CrossEntropyLoss(nn.Cell):
    def __init__(self, scale=16, label_smooth=True, eps=0.1, loss_term_weight=1.0):
        super(CrossEntropyLoss, self).__init__(loss_term_weight)
        self.scale = scale
        self.label_smooth = label_smooth
        self.eps = eps

    def construct(self, logits, labels):
        """
            logits: [n, c, p]
            labels: [n]
        """
        n, c, p = logits.shape
        logits = logits.astype(mindspore.float32)
        # labels = labels.unsqueeze(1).astype(mindspore.int32)
        reshape = ops.Reshape()
        labels = reshape(labels, (labels.shape[0], 1)).astype(mindspore.int32)
        # labels = reshape(labels, (labels.shape[0], 1, 1)).astype(mindspore.int32)
        # labels = labels.repeat(1, 1, p)
        labels = mindspore.ops.tile(labels, (1, p))
        criterion = nn.CrossEntropyLoss(label_smoothing=self.eps)
        # criterion = nn.SoftmaxCrossEntropyWithLogits(label_smoothing=self.eps)
        # loss = criterion(logits*self.scale, labels.repeat(1, p))
        loss = criterion(logits*self.scale, labels)
        return loss