import imageio
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
import torch
import random
import torch.nn as nn
from torch.nn import functional as F
import math

def interpolate_pos_embed(model, checkpoint_pos_embed):
    """
    Interpolate the positional embedding from the checkpoint to the required size for the model.
    """
    # Dimensions of the pos_embed in the checkpoint
    embedding_size = checkpoint_pos_embed.shape[-1]
    num_patches_ckpt = checkpoint_pos_embed.shape[-2] - 1  # -1 for the class token

    # Dimensions of the pos_embed in the current model
    num_patches_model = model.pos_embed.shape[-2] - 1  # -1 for the class token
    
    if num_patches_ckpt == num_patches_model:
        return checkpoint_pos_embed

    print(f"Interpolating pos_embed from {num_patches_ckpt} patches to {num_patches_model} patches")
    
    # Separate class token y patch embeddings
    class_pos_embed = checkpoint_pos_embed[:, 0]
    patch_pos_embed = checkpoint_pos_embed[:, 1:]
    
    h_ckpt = w_ckpt = int(num_patches_ckpt ** 0.5)
    h_model = w_model = int(num_patches_model ** 0.5)
    
    # Reshape for interpolation
    patch_pos_embed = patch_pos_embed.reshape(1, h_ckpt, w_ckpt, embedding_size).permute(0, 3, 1, 2)
    
    # Interpolate using bicubic
    patch_pos_embed = torch.nn.functional.interpolate(
        patch_pos_embed, 
        size=(h_model, w_model), 
        mode='bicubic', 
        align_corners=False
    )
    
    # Reshape back
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
    
    # Concatenate class token with interpolated patches
    pos_embed = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    
    return pos_embed


def freeze_backbone_layers(model, freeze=True):
    for name, param in model.named_parameters():
        if "head" not in name:  
            param.requires_grad = not freeze


def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False 


def saveImage(filename, image):
    imageTMP = np.clip(image * 255.0, 0, 255).astype('uint8')
    imageio.imwrite(filename, imageTMP)


def save_rgb (img, filename):
    
    img = np.clip(img, 0., 1.)
    if np.max(img) <= 1:
        img = img * 255
    
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)


def load_img (filename, norm=True,):
    img = np.array(Image.open(filename))
    if norm:   
        img = img / 255.
        img = img.astype(np.float32)
    return img


def plot_all (images, figsize=(20,10), axis='off', title=None):
    nplots = len(images)
    fig, axs = plt.subplots(1,nplots, figsize=figsize, dpi=80,constrained_layout=True)
    for i in range(nplots):
        axs[i].imshow(images[i])
        axs[i].axis(axis)
    plt.show()


def generate_gif (img_paths, out_path="results/testmodel/", duration=500, plot_imgs=True):
    """
    Generate a gif from a list of images
    """
    imgs = [Image.open(path) for path in img_paths]

    if plot_imgs:
        for img in imgs:
            plt.imshow(np.array(img))
            plt.show()

    imgs[0].save(os.path.join(out_path, f"evol.gif"),
               save_all=True, append_images=imgs[1:], optimize=False, duration=duration, loop=0)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


########## MODEL ##########

def count_params(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params


def load_weights(model, old_weights):
    '''
    Loads the weights of a pretrained model, picking only the weights that are
    in the new model.
    '''
    new_weights = model.state_dict()
    new_weights.update({k: v for k, v in old_weights.items() if k in new_weights})
    
    model.load_state_dict(new_weights, strict= False)
    loaded_keys = set(old_weights.keys()).intersection(new_weights.keys())
    not_loaded_keys = set(old_weights) - loaded_keys
    if not_loaded_keys:
        for key in not_loaded_keys:
            print(f"Key '{key}' not loaded")
    return model


def freeze_parameters(model,
                      substring:str, substring2:str,
                      Net:bool = False):
    
    #Net atribute in case you want to only freeze some layers
    if Net:
        for name, param in model.named_parameters():
            if substring in name:
                param.requires_grad = True 

            elif substring2 in name:
                param.requires_grad = True 

            else: 
                param.requires_grad = False

        return model
    else:
        for name, param in model.named_parameters():
            param.requires_grad = False  

        return model
  
    
def unfreezing_scheduler(model, epoch, total_epochs):
    '''
    Unfreezes Convnext_v2 (timm_version) backbone layer progressively during training.
    '''
    for param in model.parameters():
        param.requires_grad = False

    for param in model.head.parameters():
            param.requires_grad = True

    if epoch >= 0:
        for param in model.stages[3].parameters():
            param.requires_grad = True

    if epoch > (total_epochs * 0.33):
        for param in model.stages[2].parameters():
            param.requires_grad = True

    if epoch > (total_epochs * 0.67):
        for param in model.stages[1].parameters():
            param.requires_grad = True

    
    