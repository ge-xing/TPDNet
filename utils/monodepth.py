
import os
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

import networks.monodepth_networks as mononetworks
# from utils import download_model_if_doesnt_exist


def load_depth_model():
    model_name = "mono+stereo_640x192"

    # download_model_if_doesnt_exist(model_name)
    encoder_path = os.path.join("weight", model_name, "encoder.pth")
    depth_decoder_path = os.path.join("weight", model_name, "depth.pth")

    # LOADING PRETRAINED MODEL
    encoder = mononetworks.ResnetEncoder(18, False)
    depth_decoder = mononetworks.DepthDecoderFeature(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    # feed_height = loaded_dict_enc['height']
    # feed_width = loaded_dict_enc['width']
    # print(feed_height, feed_width)

    loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
    depth_decoder.load_state_dict(loaded_dict)

    encoder.eval()
    depth_decoder.eval();
    return encoder, depth_decoder


import torch

from PIL import Image

encoder, depth_decoder = load_depth_model()

def preprocess(image_path, image_size=384):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    image = transforms.ToTensor()(image)
    return image

def get_depth(image, encoder, depth_decoder, ):
    # You need to input an image and get the corresponding depth information
    # image B C W H
    with torch.no_grad():
        features = encoder(image)
        outputs = depth_decoder(features)

    # out = outputs[("disp", 0)]
    multi_scale_feature = []
    for k, v in outputs.items():
        multi_scale_feature.append(v)

    return multi_scale_feature 
