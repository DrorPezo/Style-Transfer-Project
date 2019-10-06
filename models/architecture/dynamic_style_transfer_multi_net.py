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
        self.activated_blocks = 0

    def forward(self, x, alpha_0=0, alpha_1=None, alpha_2=None):
        if alpha_1 is None or alpha_2 is None:
            alpha_1 = alpha_0
            alpha_2 = alpha_0
        if alpha_0 == 0 and alpha_1 == 0 and alpha_2 == 0:
            return self.main(x)

        alpha = [0] * 10
        if self.activated_blocks > 0:
            a0 = self.activations.index(1)
            alpha[a0] = alpha_0
        if self.activated_blocks > 1:
            a1 = self.activations.index(1, a0+1)
            alpha[a1] = alpha_1
        if self.activated_blocks > 2:
            a2 = self.activations.index(1, a1+1)
            alpha[a2] = alpha_2

        out = self.relu(self.main.in1(self.main.conv1(x)))

        if alpha[0] > 0:
            out = self.relu(self.main.in2(self.main.conv2(out + alpha[0] * self.activations[0] * self.tuning_blocks_higher(out, 0))))
        else:
            out = self.relu(self.main.in2(self.main.conv2(out - alpha[0] * self.activations[0] * self.tuning_blocks_lower(out, 0))))

        if alpha[1] > 0:
            out = self.relu(self.main.in3(self.main.conv3(out + alpha[1] * self.activations[1] * self.tuning_blocks_higher(out, 1))))
        else:
            out = self.relu(self.main.in3(self.main.conv3(out - alpha[1] * self.activations[1] * self.tuning_blocks_lower(out, 1))))

        if alpha[2] > 0:
            out = self.main.res1(out + alpha[2] * self.activations[2] * self.tuning_blocks_higher(out, 2))
        else:
            out = self.main.res1(out - alpha[2] * self.activations[2] * self.tuning_blocks_lower(out, 2))

        if alpha[3] > 0:
            out = self.main.res2(out + alpha[3] * self.activations[3] * self.tuning_blocks_higher(out, 3))
        else:
            out = self.main.res2(out - alpha[3] * self.activations[3] * self.tuning_blocks_lower(out, 3))

        if alpha[4] > 0:
            out = self.main.res3(out + alpha[4] * self.activations[4] * self.tuning_blocks_higher(out, 4))
        else:
            out = self.main.res3(out - alpha[4] * self.activations[4] * self.tuning_blocks_lower(out, 4))

        if alpha[5] > 0:
            out = self.main.res4(out + alpha[5] * self.activations[5] * self.tuning_blocks_higher(out, 5))
        else:
            out = self.main.res4(out - alpha[5] * self.activations[5] * self.tuning_blocks_lower(out, 5))

        if alpha[6] > 0:
            out = self.main.res5(out + alpha[5] * self.activations[6] * self.tuning_blocks_higher(out, 6))
        else:
            out = self.main.res5(out - alpha[5] * self.activations[6] * self.tuning_blocks_lower(out, 6))

        if alpha[7] > 0:
            out = self.relu(self.main.in4(self.main.deconv1(out + alpha[7] * self.activations[7] * self.tuning_blocks_higher(out, 7))))

        else:
            out = self.relu(self.main.in4(self.main.deconv1(out - alpha[7] * self.activations[7] * self.tuning_blocks_lower(out, 7))))

        if alpha[8] > 0:
            out = self.relu(self.main.in5(self.main.deconv2(out + alpha[8] * self.activations[8] * self.tuning_blocks_higher(out, 8))))
        else:
            out = self.relu(self.main.in5(self.main.deconv2(out - alpha[8] * self.activations[8] * self.tuning_blocks_lower(out, 8))))

        if alpha[9] > 0:
            out = self.main.deconv3(out + alpha[9] * self.activations[9] * self.tuning_blocks_higher(out, 9))
        else:
            out = self.main.deconv3(out - alpha[9] * self.activations[9] * self.tuning_blocks_lower(out, 9))

        return out

    def activation(self, layer):
        if self.activations[layer] == 0:
            self.activated_blocks += 1
            self.activations[layer] = 1




