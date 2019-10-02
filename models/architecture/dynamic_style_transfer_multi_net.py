from torch import nn
import torch
from .main_net import MainNet
from .tuning_blocks import TuningBlockList


class DynamicStyleTransferMulti(nn.Module):
    def __init__(self):
        self.name = 'DynamicStyleTransferMulti'
        super(DynamicStyleTransferMulti, self).__init__()
        self.main = MainNet()
        self.tuning_blocks_lower = TuningBlockList()
        self.tuning_blocks_higher = TuningBlockList()
        self.activations = [0] * 10
        self.relu = self.main.relu
        self.device = torch.device("cuda")

    def forward(self, x, alpha=0):
        if alpha == 0 or self.tuning_blocks_higher.activated_blocks == 0:
            return self.main(x)
        out = self.relu(self.main.in1(self.main.conv1(x)))

        if alpha > 0:
            out = self.relu(self.main.in2(self.main.conv2(out + alpha * self.activations[0] * self.tuning_blocks_higher(out, 0))))
        else:
            out = self.relu(self.main.in2(self.main.conv2(out - alpha * self.activations[0] * self.tuning_blocks_lower(out, 0))))

        if alpha > 0:
            out = self.relu(self.main.in3(self.main.conv3(out + alpha * self.activations[1] * self.tuning_blocks_higher(out, 1))))
        else:
            out = self.relu(self.main.in3(self.main.conv3(out - alpha * self.activations[1] * self.tuning_blocks_lower(out, 1))))

        if alpha > 0:
            out = self.main.res1(out + alpha * self.activations[2] * self.tuning_blocks_higher(out, 2))
        else:
            out = self.main.res1(out - alpha * self.activations[2] * self.tuning_blocks_lower(out, 2))

        if alpha > 0:
            out = self.main.res2(out + alpha * self.activations[3] * self.tuning_blocks_higher(out, 3))
        else:
            out = self.main.res2(out - alpha * self.activations[3] * self.tuning_blocks_lower(out, 3))

        if alpha > 0:
            out = self.main.res3(out + alpha * self.activations[4] * self.tuning_blocks_higher(out, 4))
        else:
            out = self.main.res3(out - alpha * self.activations[4] * self.tuning_blocks_lower(out, 4))

        if alpha > 0:
            out = self.main.res4(out + alpha * self.activations[5] * self.tuning_blocks_higher(out, 5))
        else:
            out = self.main.res4(out - alpha * self.activations[5] * self.tuning_blocks_lower(out, 5))

        if alpha > 0:
            out = self.main.res5(out + alpha * self.activations[6] * self.tuning_blocks_higher(out, 6))
        else:
            out = self.main.res5(out - alpha * self.activations[6] * self.tuning_blocks_lower(out, 6))

        if alpha > 0:
            out = self.relu(self.main.in4(self.main.deconv1(out + alpha * self.activations[7] * self.tuning_blocks_higher(out, 7))))

        else:
            out = self.relu(self.main.in4(self.main.deconv1(out - alpha * self.activations[7] * self.tuning_blocks_lower(out, 7))))



        if alpha > 0:
            out = self.relu(self.main.in5(self.main.deconv2(out + alpha * self.activations[8] * self.tuning_blocks_higher(out, 8))))
        else:
            out = self.relu(self.main.in5(self.main.deconv2(out - alpha * self.activations[8] * self.tuning_blocks_lower(out, 8))))

        if alpha > 0:
            out = self.main.deconv3(out + alpha * self.activations[9] * self.tuning_blocks_higher(out, 9))
        else:
            out = self.main.deconv3(out - alpha * self.activations[9] * self.tuning_blocks_lower(out, 9))

        return out

    def activation(self, layer):
        self.activations[layer] = 1


