import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import networks
import torch
from configs import LwirChannel
from normalizer import Normalizer
from physical_model import PanToMono as PhysicalModel
from utils.deep import NetPhase


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(
        self,
        config,
        path_to_models: Path = Path("models"),
        path_to_data: Path = Path("data"),
    ):
        """Initialize the BaseModel class.

        Parameters:
            config (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """

        self.config = config
        self.gpu_ids = (
            config.gpu_ids
            if isinstance(config.gpu_ids, (list, tuple))
            else [config.gpu_ids]
        )
        self.phase = config.phase
        self.device = (
            torch.device("cuda:{}".format(self.gpu_ids[0]))
            if self.gpu_ids
            else torch.device("cpu")
        )  # get device name: CPU or GPU
        self.iter_loss = {}
        self.agg_loss = {}
        self.loss_log = {}
        self.visuals = {}
        self.model_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

        # init data normalizers:
        norm_method = config.data_loader.norm_method
        mean = config.data_loader.norm_mean
        std = config.data_loader.norm_std
        self.pan_norm = Normalizer(
            path_to_data, LwirChannel.pan, norm_method, mean, std
        )
        self.mono_norm = Normalizer(
            path_to_data, self.config.wl, norm_method, mean, std
        )

        if self.config.thermal.is_physical_model:
            # init physical model:
            self.physical_model = PhysicalModel(dtype=torch.Tensor)
            self.physical_model.load(path_to_models / "coefficients.npz")
            self.physical_model.to(self.device)

            # init affine correction network:
            self.netPW = networks.PixelWiseAffine()
            self.netPW.to(self.device)

            self.model_names.append("PW")

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pan_to_mono = self.config.network.direction == "pan_to_mono"
        if "mono" in input.keys():
            in_pan = input["pan" if pan_to_mono else "mono"]
            in_mono = input["mono" if pan_to_mono else "pan"]

            pan_real = in_pan["img"].to(self.device)
            mono_real = in_mono["img"].to(self.device)

            self.pan_real = self.pan_norm.normalize(pan_real)
            self.mono_real = self.mono_norm.normalize(mono_real)

            pan_fpa = in_pan["fpa"].type(torch.float).to(self.device)
            mono_fpa = in_mono["fpa"].type(torch.float).to(self.device)
        elif "img" in input.keys():
            pan_real = input["img"].to(self.device)

            self.pan_real = self.pan_norm.normalize(pan_real)
            self.mono_real = None

            pan_fpa = input["fpa"].type(torch.float).to(self.device)
            mono_fpa = torch.full_like(pan_fpa, fill_value=20, dtype=torch.float).to(
                self.device
            )  # default of 20C TODO: make this a parameter

        # predict monochromatic image using physical model:
        if self.config.thermal.is_fpa_input:
            self.pan_fpa = pan_fpa / self.config.data_loader.fpa_scale
            self.mono_fpa = mono_fpa / self.config.data_loader.fpa_scale

        if self.config.thermal.is_physical_model:
            mono_phys, _ = self.physical_model.predict(
                pan_real.squeeze(), t_fpa_pan=pan_fpa, t_fpa_mono=mono_fpa
            )
            mono_phys_norm = (
                self.mono_norm.normalize(mono_phys).type(torch.float).unsqueeze(axis=1)
            )

            self.mono_phys = mono_phys_norm.to(self.device)

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, config):
        """Load and print networks; create schedulers

        Parameters:
            config (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        is_train = self.phase == NetPhase.train
        if is_train:
            self.schedulers = [
                networks.get_scheduler(optimizer, config)
                for optimizer in self.optimizers
            ]
        if not is_train or config.continue_train:
            load_suffix = (
                "%d" % config.load_iter if config.load_iter > 0 else config.epoch
            )
            self.load_networks(load_suffix)

    def set_phase(self, phase: NetPhase):
        """sets the network's phase according to the desired logic"""
        self.phase = phase
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                if phase in [NetPhase.val, NetPhase.test]:
                    net.eval()
                elif phase == NetPhase.train:
                    net.train()
                else:
                    raise Exception("Not a valid phase!")

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]["lr"]
        for scheduler in self.schedulers:
            if self.config.scheduler.lr_policy == "plateau":
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]["lr"]
        logger = logging.getLogger("PETIT")
        logger.info("learning rate %.7f -> %.7f" % (old_lr, lr))

    def get_current_visuals(self):
        pass

    def update_agg_loss(self):
        self.agg_loss += self.iter_loss

    def update_loss(self, epoch, loader_len: int):
        for domain in self.agg_loss.keys().get_level_values(0).unique():
            for loss in self.agg_loss.keys().get_level_values(1).unique():
                self.loss_log.loc[epoch, (self.phase.name, domain, loss)] = (
                    self.agg_loss.loc[(domain, loss)] / loader_len
                )

        # reset agg loss:
        self.agg_loss.loc[:] = 0
        return self.iter_loss

    def save_networks(self, epoch, base_path):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        logger = logging.getLogger("PETIT")
        logger.info("saving the model state")

        for sub_model in self.model_names:
            if isinstance(sub_model, str):
                target_path = base_path / sub_model
                if not target_path.is_dir():
                    target_path.mkdir(parents=True)
                net = getattr(self, "net" + sub_model)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(
                        net.cpu().state_dict(), target_path / (str(epoch) + ".pth")
                    )
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), target_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "running_mean" or key == "running_var"
            ):
                if getattr(module, key) is None:
                    state_dict.pop(".".join(keys))
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "num_batches_tracked"
            ):
                state_dict.pop(".".join(keys))
        else:
            self.__patch_instance_norm_state_dict(
                state_dict, getattr(module, key), keys, i + 1
            )

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        logger = logging.getLogger("PETIT")
        for name in self.model_names:
            if isinstance(name, str):
                load_path = os.path.join("models", name, str(epoch) + ".pth")
                net = getattr(self, "net" + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                logger.info("loading the model from %s" % load_path)
                state_dict = torch.load(load_path)
                if hasattr(state_dict, "_metadata"):
                    del state_dict._metadata

                net.load_state_dict(state_dict)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def plot_losses(self, tens_board):
        pass

    def gen_vis_grid(self):
        pass

    def save_res_log(self, target_dir: Path):
        pass

    def rec_image(self, tens: torch.Tensor, domain: str = "pan", fmt="npy"):
        img = tens.cpu().numpy().squeeze()
        if fmt == "npy":
            normalizer = self.__getattribute__(f"{domain}_norm")
            img_den = normalizer.denormalize(img)
        else:  # uint8 format
            img_max = img.max(axis=(1, 2))
            img_min = img.min(axis=(1, 2))
            img_den = (255 * ((img.T - img_min) / (img_max - img_min)).T).astype(
                "uint8"
            )

        return img_den
