import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor
import mindspore

class Non_local(nn.Cell):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = in_channels // reduc_ratio

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, pad_mode="valid")

        self.W = nn.SequentialCell([
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, pad_mode="valid"),
            nn.BatchNorm2d(self.in_channels),
        ])
        self.W[1].gamma.set_data(Tensor(0.0, mindspore.float32))
        self.W[1].beta.set_data(Tensor(0.0, mindspore.float32))

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, pad_mode="valid")

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, pad_mode="valid")

    def construct(self, x):
        batch_size, _, height, width = P.Shape()(x)
        g_x = self.g(x)
        g_x = P.Reshape()(g_x, (batch_size, self.inter_channels, -1))
        g_x = P.Transpose()(g_x, (0, 2, 1))

        theta_x = self.theta(x)
        theta_x = P.Reshape()(theta_x, (batch_size, self.inter_channels, -1))
        theta_x = P.Transpose()(theta_x, (0, 2, 1))
        phi_x = self.phi(x)
        phi_x = P.Reshape()(phi_x, (batch_size, self.inter_channels, -1))
        f = P.MatMul()(theta_x, phi_x)
        N = P.Shape()(f)[-1]
        f_div_C = f / N

        y = P.MatMul()(f_div_C, g_x)
        y = P.Transpose()(y, (0, 2, 1))
        y = P.Reshape()(y, (batch_size, self.inter_channels, height, width))
        W_y = self.W(y)
        z = W_y + x
        return z

