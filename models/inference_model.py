from models.base_model import BaseModel
from models.architecture.dynamic_style_transfer_net import DynamicStyleTransfer
from models.architecture.dynamic_style_transfer_dual_net import DynamicStyleTransferDual
from models.architecture.dynamic_style_transfer_multi_net import DynamicStyleTransferMulti
import torch
from collections import OrderedDict


class InferenceModel(BaseModel):
    def __init__(self, opt, set_net_version=None):
        super(InferenceModel, self).__init__(opt)
        self.tb0_dict = OrderedDict()
        self.tb1_dict = OrderedDict()
        self.tb2_dict = OrderedDict()
        if set_net_version is None:
            self.network_version = opt.network_version
            if self.network_version is 'dual':
                self.net = DynamicStyleTransferDual().to(self.device)
            elif self.network_version is 'multi':
                self.net = DynamicStyleTransferMulti().to(self.device)
            elif self.network_version is 'normal':
                self.net = DynamicStyleTransfer(opt.layer_num).to(self.device)
        elif set_net_version == 'dual':
            self.network_version = set_net_version
            self.net = DynamicStyleTransferDual().to(self.device)
        elif set_net_version == 'normal':
            self.network_version = set_net_version
            self.net = DynamicStyleTransfer(opt.layer_num).to(self.device)
        elif set_net_version == 'multi':
            self.network_version = set_net_version
            self.net = DynamicStyleTransferMulti().to(self.device)

    def forward_and_recover(self, input_batch, alpha_0=0, alpha_1=None, alpha_2=None):
        output_batch = self.net(input_batch, alpha_0=alpha_0, alpha_1=alpha_1, alpha_2=alpha_2)
        return self.recover_tensor(output_batch)

    def multi_forward_and_recover(self, input_batch, alpha):
        output_batch = self.net(input_batch, alpha)
        return self.recover_tensor(output_batch)

    def load_network(self, net_path):
        self.model = torch.load(net_path)
        self.net.load_state_dict(self.model)
        print("Model's state_dict:")
        for key, value in self.net.state_dict().items():
            # print(key)
            if 'tuning_block0' in key:
                self.tb0_dict['key'] = value

            elif 'tuning_block1' in key:
                self.tb1_dict['key'] = value

            elif 'tuning_block2' in key:
                self.tb2_dict['key'] = value

    def load_multi_network(self, net_path):
        self.net.main.load_state_dict(torch.load(net_path))

    def load_block(self, activated_layer, block_path):
        self.net.activation(activated_layer)
        self.net.tuning_blocks_higher.insert_block(activated_layer, block_path)
        self.net.tuning_blocks_lower.insert_block(activated_layer, block_path)






