import platform
import sys
from dataclasses import dataclass
from enum import Enum, auto
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
from hydra.core.config_store import ConfigStore
from utils.deep import NetPhase

rgb2grey = np.array([0.2989, 0.587, 0.1141])
TAU2_RAD_RES = 2**14  # TAU2's radiometric resolution


def get_num_threads():
    if platform.system() == "Windows":
        num_threads = 0
    else:
        num_threads = cpu_count()

    return num_threads


class LwirChannel(Enum):
    """An enumeration for Flir's LWIR channels obtained by applying bandpass filters (or not) in nano-meter."""

    pan = 0
    nm8000 = 8000
    nm9000 = 9000
    nm10000 = 10000
    nm11000 = 11000
    nm12000 = 12000

    def __str__(self) -> str:
        return str(self.value) + "nm" if self.value > 0 else "pan"


class RGB(Enum):
    red = 0  # panchromatic
    green = 1
    blue = 2  # validation


class Domains(Enum):
    pan = auto()  # panchromatic
    mono = auto()


class InOutComb(Enum):

    """The operation which is performed to combine the output of the pan->color generator with the panchromatic input"""

    none = auto()  # no combination
    addition = auto()  # apply addition between input and output
    hadamard = auto()  # apply hadamard multiplication between output and input
    hadamard_bias = auto()  # apply hadamard multiplication between output 1st layer and input, and add to outputs 2nd layer


class DataNorm(Enum):
    const = auto()  # constant normalization (by the resolution of the sensor)
    min_max = (
        auto()
    )  # based on the dataset's dynamic range (x - min(D_x) / (max(D_x)-min(D_x)))
    mean_std = (
        auto()
    )  # based on the dataset's mean and pseudo-std (x - mean(D_x) / std(D_x))


@dataclass(frozen=False)
class Optimizer:
    beta1: float = 0.5  # momentum term of adam
    lr: float = 5e-4  # initial learning rate for adam


@dataclass(frozen=False)
class Scheduler:
    lr_policy: str = (
        "linear"  # learning rate policy. [linear | step | plateau | cosine]
    )
    lr_decay_iters: int = 50  # multiply by a gamma every lr_decay_iters iterations
    n_epochs: int = 50  # number of epochs with the initial learning rate
    n_epochs_decay: int = 50  # number of epochs to linearly decay learning rate to zero


@dataclass(frozen=False)
class Loss:
    lambda_GAN: float = 1.0  # weight for cycle loss (A -> B -> A)')
    lambda_identity: float = 1.0  # was 0.5 by default. use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
    lambda_gp: float = (
        10.0  # weight for the gradient penalty (only relevant if gan_mode=="wgangp")
    )
    lambda_NCE: float = 1.0


@dataclass(frozen=False)
class DataLoader:
    num_threads: int = get_num_threads()  # threads for loading data
    batch_size: int = 2  # input batch size
    load_size: int = 256
    crop_size: int = 256
    max_dataset_size: int = int(
        1e10
    )  # Maximum number of samples allowed per Dataset. If the Dataset directory contains more than max_dataset_size, only a subset is loaded.
    no_flip: bool = True  # if specified, do not flip the images for data augmentation
    preprocess: str = "crop"  # scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    norm_method: DataNorm = DataNorm.mean_std
    norm_mean: float = 0.0
    norm_std: float = 1.0
    fpa_scale: float = 50  # scaling factor for the fpa temperature


@dataclass(frozen=False)
class Paths:
    dataroot: str = Path(sys.path[0]).parent / "data"


@dataclass(frozen=False)
class Network:
    # model parameters
    init_gain: float = 0.02  # scaling factor for normal, xavier and orthogonal.')
    gan_mode: str = "lsgan"  # the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.
    ngf: int = 64  # # of gen filters in the last conv layer
    ndf: int = 64  # # of discrim filters in the first conv layer
    netD: str = "basic"  # specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
    netG: str = "resnet_6blocks"  # specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128| stylegan2 | smallstylegan2 | resnet_cat]')
    netF: str = "mlp_sample"  # how to downsample the feature map ['sample', 'reshape', 'mlp_sample']
    netF_nc: int = 256
    n_layers_D: int = 3  # only used if netD==n_layers
    D_patch_size: int = 70
    norm: str = "instance"  # instance normalization or batch normalization [instance | batch | none]')
    init_type: str = (
        "normal"  # network initialization [normal | xavier | kaiming | orthogonal]')
    )
    dataset_mode: str = "unaligned"  # chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
    direction: str = "pan_to_mono"
    pool_size: int = 50  # the size of image buffer that stores previously generated images. TODO: understand this
    input_nc: int = 1  # number of color-channels in the input space
    output_nc: int = 1  # number of color-channels in the output space
    no_dropout: bool = True  # no dropout for the generator
    in_out_comb: InOutComb = (
        InOutComb.none
    )  # See InOutComb enumeration docstring for explanation
    n_critic: int = 5  # the number of consequent critic training iterations between generator trainings (relevant only when gan_mode==wgangp)
    no_antialias: bool = True  # whether to use resize-convolution layers instead of transposed-convolution
    no_antialias_up: bool = True  # whether to use resize-convolution layers instead of transposed-convolution
    stylegan2_G_num_downsampling: int = 1
    padding_type: str = "zeros"
    nce_layers: str = "0,4,8,12,16"  # compute NCE loss on which layers
    num_patches: int = 256
    flip_equivariance: bool = False  # Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT
    nce_idt: bool = True  # use NCE loss for identity mapping: NCE(G(Y), Y))
    nce_includes_all_negatives_from_minibatch: bool = False
    nce_T: float = 0.07


@dataclass(frozen=False)
class Thermal:
    is_fpa_input: bool = True
    is_physical_model: bool = True  # controls whether to use the physical model output to enhance the predicted output


@dataclass(frozen=False)
class Visualizations:
    display_freq: int = np.iinfo(
        np.uint64
    ).max  # 400 frequency of showing training results on screen
    loss_print_freq: int = 100  # frequency of showing training results on console')
    is_tensorboard_active: bool = True


@dataclass(frozen=False)
class SaveRate:
    save_latest_freq: int = 5000  # frequency of saving the latest results
    save_epoch_freq: int = 5  # frequency of saving checkpoints at the end of epochs
    clean_freq: int = 50  # rate of clearing all saved checkpoints (but the latest)
    save_by_iter: bool = True  # whether saves model by iteration


@dataclass(frozen=False)
class BaseConfig:
    """Sets the base configurations for the training and test of the colorization network."""

    # General Parameters:
    model: str = "CUT"  # ["CycleGan" | "CUT"]
    phase: NetPhase = NetPhase.train
    wl: LwirChannel = LwirChannel.nm9000
    gpu_ids: int = 0  # e.g. (0) \ (0,1,2). use -1 for CPU')
    verbose: bool = True  # if specified, print more debugging information
    load_iter: int = 0  # which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]. otherwise, the code will load models by [epoch]')
    epoch: str = "best"  # which epoch to load? set to latest to use latest cached model
    is_random: bool = False  # if False, all numbers are generated using a default seed and all algorithms are deterministic
    network: Network = Network()
    data_loader: DataLoader = DataLoader()
    paths: Paths = Paths()
    thermal: Thermal = Thermal()


@dataclass(frozen=False)
class TrainConfig(BaseConfig):
    """Sets the base configurations for the training set."""

    continue_train: bool = False  # continue training: load the latest model
    epoch_count: int = 0  # the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq> training parameters
    optimizer: Optimizer = Optimizer()
    scheduler: Scheduler = Scheduler()
    loss: Loss = Loss()
    visualizations: Visualizations = Visualizations()
    save_rate: SaveRate = SaveRate()
    use_regression_model: bool = False  # whether the network learns the residual from the classical colorization model's output or the complete output.
    # network.no_dropout = False TODO: uncomment to check how dropout affects training


@dataclass(frozen=False)
class TestConfig(BaseConfig):
    """Sets the base configurations for the test set."""

    aspect_ratio: float = 1.0  # aspect ratio of result images
    # num_test: none # how many test images to run')
    phase: NetPhase = NetPhase.test


## Store the config in hydra
cs = ConfigStore.instance()
cs.store(name="config", node=TrainConfig)

if __name__ == "__main__":
    ...
