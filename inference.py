import os
from models.inference_model import InferenceModel
from models.architecture.tuning_blocks import TuningBlockModule, TuningBlock
import config
import utils.utils as utils
import torch
import PIL.Image as Image
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse


# inference configurations #
block = 'block1'
save_blocks = False
load_block = True
block_size = 128
tuning_block = None
network_name = 'feathers_waterfall'
use_saved_config = True  # use the configuration saved at training time (if saved)
set_net_version = 'multi'  # None/normal/dual/multi, set to None if you want to use saved config file
alpha_0s = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # alpha_0 values for normal version
alpha_1s = [0, 1]  # alpha_1 values for normal version (if None alpha_0=alpha_1=alpha_2)
alpha_2s = alpha_1s  # alpha_2 values for normal version (if None alpha_0=alpha_1=alpha_2)
alpha_0s_dual = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1] + alpha_0s  # alpha_0 values for dual version
alpha_1s_dual = [-1, 0, 1]  # alpha_1 values for dual version (if None alpha_0=alpha_1=alpha_2)
alpha_2s_dual = alpha_1s_dual  # alpha_2 values for dual version (if None alpha_0=alpha_1=alpha_2)
# ------------------------ #

parser = argparse.ArgumentParser()
parser.add_argument('--network_name', default=network_name)
parser.add_argument('--use_saved_config', default=use_saved_config, type=lambda x:bool(utils.str2bool(x)))
parser.add_argument('--set_net_version', default=set_net_version)
inference_opt = parser.parse_args()
set_net_version = inference_opt.set_net_version
network_name = inference_opt.network_name
use_saved_config = inference_opt.use_saved_config
if set_net_version == 'None':
    set_net_version = None

networks_path = os.path.join('trained_nets', network_name)
block_path0H = os.path.join('trained_nets', 'feathers_conv3_4', 'checkpoints', 'block1_dynamic_net.pth')
block_path0L = os.path.join('trained_nets', 'feathers_conv3_4_0', 'checkpoints', 'block1_dynamic_net.pth')
block_path1H = os.path.join('trained_nets', 'feathers_res3_2', 'checkpoints', 'block0_dynamic_net.pth')
block_path1L = os.path.join('trained_nets', 'feathers_res3_2_0', 'checkpoints', 'block0_dynamic_net.pth')
block_path2H = os.path.join('trained_nets', 'feathers_deconv3_1', 'checkpoints', 'block2_dynamic_net.pth')
block_path2L = os.path.join('trained_nets', 'feathers_deconv3_1_0', 'checkpoints', 'block2_dynamic_net.pth')

model_path = os.path.join(networks_path, 'model_dir', 'orginal_main_net.pth')# change to 'dynamic_net.pth'
config_path = os.path.join(networks_path, 'config.txt')
inference_images_path = os.path.join('images', 'inference_images')
save_path = os.path.join('results', 'inference_results', network_name)
if not os.path.exists(save_path):
    utils.make_dirs(save_path)

opt = config.get_configurations(parser=parser)
if use_saved_config:
    if os.path.exists(config_path):
        utils.read_config_and_arrange_opt(config_path, opt)
        # set_net_version = None
    else:
        raise ValueError('config_path does not exists')
elif set_net_version is None:
    raise ValueError('id use_saved_config=False you must set set_net_version!=None')


if set_net_version != 'multi':
    dynamic_model = InferenceModel(opt, set_net_version=set_net_version)
    dynamic_model.load_network(model_path)
    for key, value in dynamic_model.net.state_dict().items():
        print(key)

    inference_images_list = list(os.listdir(inference_images_path))
    inference_images_list.sort()

    to_tensor = transforms.ToTensor()
    to_pil_image = transforms.ToPILImage()

    if set_net_version == 'dual' or (use_saved_config and opt.network_version == 'dual'):
        alpha_0s = alpha_0s_dual
        alpha_1s = alpha_1s_dual
        alpha_2s = alpha_2s_dual

    for image_name in inference_images_list:
        input_image = Image.open(os.path.join(inference_images_path, image_name))
        h = input_image.height
        w = input_image.width
        input_image = input_image.resize((int(h/2),int(w/2)))
        input_tensor = to_tensor(input_image).to(dynamic_model.device)
        input_tensor = dynamic_model.normalize(input_tensor)
        input_tensor = input_tensor.expand(1, -1, -1, -1)
        save_name = image_name.split('.')[0]
        for alpha_0 in tqdm(alpha_0s):
            for alpha_1 in alpha_1s:
                for alpha_2 in alpha_2s:
                    output_tensor = dynamic_model.forward_and_recover(input_tensor.requires_grad_(False), alpha_0=alpha_0, alpha_1=alpha_1, alpha_2=alpha_2)
                    output_image = to_pil_image(output_tensor.clamp(min=0.0, max=1).cpu().squeeze(dim=0))
                    if alpha_1 is not None and alpha_2 is not None:
                        output_image.save(os.path.join(save_path, '%s_%3f_%3f_%3f.png' % (save_name, alpha_0, alpha_1, alpha_2)))
                    else:
                        output_image.save(os.path.join(save_path, '%s_%3f.png' % (save_name, alpha_0)))

if set_net_version == 'multi':
    dynamic_model = InferenceModel(opt, set_net_version=set_net_version)
    dynamic_model.load_multi_network(model_path)
    # Add here all the desired blocks
    dynamic_model.load_block(1, block_path0H, block_path0L)
    dynamic_model.load_block(4, block_path1H, block_path1L)
    dynamic_model.load_block(9, block_path2H, block_path2L)
    # for key, value in dynamic_model.net.state_dict().items():
    #     print(key)

    inference_images_list = list(os.listdir(inference_images_path))
    inference_images_list.sort()

    to_tensor = transforms.ToTensor()
    to_pil_image = transforms.ToPILImage()
    # Need to update the loop to operate on all added blocks
    for image_name in inference_images_list:
        input_image = Image.open(os.path.join(inference_images_path, image_name))
        h = input_image.height
        w = input_image.width
        input_image = input_image.resize((int(h/2),int(w/2)))
        input_tensor = to_tensor(input_image).to(dynamic_model.device)
        input_tensor = dynamic_model.normalize(input_tensor)
        input_tensor = input_tensor.expand(1, -1, -1, -1)
        save_name = image_name.split('.')[0]

        for alpha_0 in tqdm(alpha_0s_dual):
            for alpha_1 in alpha_1s_dual:
                for alpha_2 in alpha_2s_dual:
                    output_tensor = dynamic_model.forward_and_recover(input_tensor.requires_grad_(False), alpha_0=alpha_0, alpha_1=alpha_1, alpha_2=alpha_2)
                    output_image = to_pil_image(output_tensor.clamp(min=0.0, max=1).cpu().squeeze(dim=0))
                    if alpha_1 is not None and alpha_2 is not None:
                        output_image.save(os.path.join(save_path, '%s_%3f_%3f_%3f.png' % (save_name, alpha_0, alpha_1, alpha_2)))
                    else:
                        output_image.save(os.path.join(save_path, '%s_%3f.png' % (save_name, alpha_0)))

