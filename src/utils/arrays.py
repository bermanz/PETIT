import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def save_image(image_in, image_path, aspect_ratio=1.0, nc_out=3):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    if isinstance(image_in, np.ndarray):
        image_pil = Image.fromarray(image_in)
    elif isinstance(image_in, torch.Tensor):
        image_pil = transforms.ToPILImage()(image_in)
    elif isinstance(image_in, Image.Image):
        image_pil = image_in
    else:
        raise Exception("Didn't expect this datatype. Need to implement!")
    if nc_out > 1:
        h, w, _ = image_in.shape
    else:
        h, w = image_in.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def tens2arr(input_tens:torch.Tensor):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        dtype (type)        --  the desired type of the converted numpy array
    """
    if input_tens.ndim == 2:
        tens_trans = input_tens
    else:
        # re-map the chanels dimension into it's native position in numpy arrays
        if input_tens.ndim == 4:
            ch_ax = 1
        elif input_tens.ndim == 3:
            ch_ax = 0
        else:
            raise Exception("The function doesn't support this number of dimensions. Need to implement according to use-case!")
        tens_trans = input_tens.moveaxis(ch_ax, -1)
    arr_np = tens_trans.cpu().numpy().squeeze()  # convert it into a numpy array
    return arr_np


def normalize_arr(arr_in: np.ndarray, mean: float = 0.0, std: float = 1.0):
    """Normalizes the array to end up with the desired mean and std"""
    arr_normed = (arr_in - arr_in.mean()) / arr_in.std()  # this results in a zero mean array with an std of 1
    return (arr_normed * std) + mean
