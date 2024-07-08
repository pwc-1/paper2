import mindspore
import mindspore.nn as nn
from mindspore import Parameter, Tensor
from mindspore.common.initializer import Normal, XavierUniform, initializer
from mindspore.ops import operations as P

class OGBASE_AUG(nn.Cell):
    def __init__(self):
        super(OGBASE_AUG, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, pad_mode='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, pad_mode='same')
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, pad_mode='same')
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, pad_mode='same')
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, pad_mode='same')
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad_mode='same')
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, pad_mode='same')
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, pad_mode='same')
        self.fc = nn.Dense(256, 256, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_neck = SeparateBNNecks(parts_num=31, in_channels=256, class_num=200)

    def use_HorizontalPoolingPyramid(self, x):
        op = mindspore.ops.Concat(axis=2)
        bin_num = [16, 8, 4, 2, 1]
        n, c = x.shape[:2]
        features = []
        for b in bin_num:
            z = x.view(n, c, b, -1)
            z = mindspore.ops.cast(z, mindspore.float32)
            # z = mindspore.ops.max(z, -1)[0] + mindspore.ops.mean(z, -1)
            z_max, _ = mindspore.ops.max(z, -1)
            z_mean = mindspore.ops.mean(z, -1)
            z = z_max + z_mean
            features.append(z)
        return op(tuple(features))
    
    def construct(self, x):
        # reshape = mindspore.ops.Reshape()
        b, t, h, w = x.shape
        x = mindspore.ops.expand_dims(x, 1)
        x = x.view(-1, 1, h, w)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.max_pool2d(self.relu(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        _, c, h, w = x.shape
        x = x.view(b, t, c, h, w)
        x = mindspore.ops.max(x, axis=1)[0]
        # return x


        x_0 = self.use_HorizontalPoolingPyramid(x)
        # x_0 = self.fc(x_0)
        # x_a, x_b = self.bn_neck(x_0)
        x_b = self.bn_neck(x_0)

        return x_0, x_b
        # return x_b

class SeparateBNNecks(nn.Cell):
    def __init__(self, parts_num, in_channels, class_num=100, norm=True, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm
        self.fc_bin = Parameter(initializer(
            XavierUniform(), [parts_num, in_channels, class_num], mindspore.float32))
        # self.fc_bin = Parameter(initializer(
        #     XavierUniform(), [in_channels, parts_num*class_num], mindspore.float32))
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            self.bn1d = [nn.BatchNorm1d(in_channels) for _ in range(parts_num)]
        self.parallel_BN1d = parallel_BN1d
        self.matmul = P.BatchMatMul()
        self.reshape = P.Reshape()
        self.permute = mindspore.ops.Transpose()
        self.normalize = mindspore.ops.L2Normalize(axis=-1)

    def construct(self, x):
        n, c, p = P.Shape()(x)
        if self.parallel_BN1d:
            x = self.reshape(x, (n, -1))
            x = self.bn1d(x)
            x = self.reshape(x, (n, c, p))
        else:
            x = P.Concat(2)([bn(x[:, :, i]) for i, bn in enumerate(self.bn1d)])
        
        # feature_reshaped = self.reshape(feature, (self.class_num, -1))
        # fc_bin_reshaped = self.reshape(self.fc_bin, (self.p * c, self.class_num))
        feature = self.permute(x, (2, 0, 1))
        feature_reshaped = self.normalize(feature)
        # feature_reshaped = self.reshape(feature_reshaped, (31*16, 256))
        # self.fc_bin = self.reshape(self.fc_bin, (256, 31*200))
        if self.norm:
            logits = self.matmul(feature_reshaped, self.normalize(self.fc_bin))
        else:
            logits = self.matmul(feature_reshaped, self.fc_bin)

        # return self.permute(feature, (1, 2, 0)), self.permute(logits, (1, 2, 0))
        return self.permute(logits, (1, 2, 0))