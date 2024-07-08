import mindspore.nn as nn
from mindspore.common.initializer import Normal
import mindspore.ops as ops
import mindspore.common.initializer as initializer
import math as math

class Network(nn.Cell):
    def __init__(self, num_classes=10, num_channel=1, include_top=True):
        super(Network, self).__init__()
        self.include_top = include_top
        self.conv1 = nn.Conv2d(num_channel, 32, 5, pad_mode='same')
        self.conv2 = nn.Conv2d(32, 64, 5, pad_mode='same')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2,pad_mode='same')

        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc1 = nn.Dense(8 * 8 * 64, 1024, weight_init=Normal(0.02))
            self.fc2 = nn.Dense(1024, num_classes, weight_init=Normal(0.02))

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if self.include_top:
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
        return x
    
class BasicBlock(nn.Cell):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,pad_mode='pad',
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,pad_mode='pad',
                               padding=1)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,pad_mode='pad',
                               padding=0) or None
    def construct(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = nn.Dropout(keep_prob=self.droprate)(out)
        out = self.conv2(out)
        return ops.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Cell):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.SequentialCell(*layers)
    def construct(self, x):
        return self.layer(x)

class WideResNet(nn.Cell):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor, 64*widen_factor*10]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,pad_mode='pad',
                               padding=1)        
        self.dropout = nn.Dropout(0.3)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd blockop
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU()

        # self.ID_mat = torch.eye(num_classes).cuda()

        self.fc = nn.Dense(nChannels[3], num_classes)
        self.fc.weight.requires_grad = False        # Freezing the weights during training
        self.nChannels = nChannels[3]

        for cell in self.cells():
            if isinstance(cell, nn.Conv2d):
                n = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                cell.weight.set_data(initializer.initializer(initializer.Normal(math.sqrt(2. / n),0),cell.weight.shape,cell.weight.dtype))
                # cell.weight.data.normal_(0, ops.sqrt(2. / n))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(initializer.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(initializer.initializer("zeros", cell.beta.shape, cell.beta.dtype))
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer.initializer(
                    initializer.Orthogonal(gain=1.),
                    cell.weight.shape, cell.weight.dtype))
    def construct(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        # out = self.relu(self.bn1(out))
        # out = self.relu(out)
        out = ops.avg_pool2d(out, 8)
        out_feat = out.view(-1, self.nChannels)
        out = ops.L2Normalize( axis=1)(out_feat)
        out = ops.abs(self.fc(out))
        return out
