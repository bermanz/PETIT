import logging
import time
from itertools import chain
from pathlib import Path

import networks
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
from base_model import BaseModel
from matplotlib import pyplot as plt
from patchnce import PatchNCELoss
from torchvision import transforms
from utils.arrays import tens2arr
from utils.deep import NetPhase
from utils.visualizations import add_text_to_image, full_dynamic_range


class CUTModel(BaseModel):
    """This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def __init__(self, config):
        BaseModel.__init__(self, config)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.config = config
        self.loss_names = ["G_GAN", "D_real", "D_fake", "G", "D", "NCE"]
        self.visual_names = ["pan_real", "mono_phys", "mono_fake"]
        if config.phase is not NetPhase.test:
            self.visual_names.append("mono_real")
        if config.network.nce_idt and config.phase == NetPhase.train:
            self.loss_names += ["NCE_Y"]
            # self.visual_names += ["mono_idt"]

        self.iter_loss = pd.Series(
            0, index=self.loss_names, dtype=float
        )  # logs the current iterations loss
        self.agg_loss = self.iter_loss.copy()
        phases_types = ["train", "val"]
        if config.phase is not NetPhase.test:
            n_epochs_tot = config.scheduler.n_epochs + config.scheduler.n_epochs_decay
            cols = pd.MultiIndex.from_product([phases_types, self.loss_names])
            self.loss_log = pd.DataFrame(
                index=range(n_epochs_tot), columns=cols, dtype=float
            )
            self.loss_log.index.rename("Epoch", inplace=True)
        self.visuals = {name: None for name in self.visual_names}

        self.nce_layers = [int(i) for i in config.network.nce_layers.split(",")]

        if config.phase == NetPhase.train:
            self.model_names.extend(["G", "F", "D"])
        else:  # during test time, only load G
            self.model_names.append("G")

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(config, self.gpu_ids)
        self.netF = networks.define_F(config, self.gpu_ids)

        if config.phase == NetPhase.train:
            self.netD = networks.define_D(config, self.gpu_ids)

            # define loss functions
            self.criterionGAN = networks.GANLoss(config.network.gan_mode).to(
                self.device
            )
            self.criterionNCE = []

            for _ in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(config).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            if self.config.thermal.is_physical_model:
                g_params = chain(self.netG.parameters(), self.netPW.parameters())
            else:
                g_params = self.netG.parameters()
            self.optimizer_G = torch.optim.Adam(
                g_params, lr=config.optimizer.lr, betas=(config.optimizer.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(),
                lr=config.optimizer.lr,
                betas=(config.optimizer.beta1, 0.999),
            )

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["pan"]["img"].size(0) // max(len(self.gpu_ids), 1)
        self.set_input(data)
        self.pan_real = self.pan_real[:bs_per_gpu]
        self.mono_real = self.mono_real[:bs_per_gpu]
        self.forward()  # compute fake images: G(A)
        if self.config.phase == NetPhase.train:
            self.calc_loss_D().backward()  # calculate gradients for D
            self.calc_loss_G().backward()  # calculate graidents for G
            if self.config.loss.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(
                    self.netF.parameters(),
                    lr=self.config.optimizer.lr,
                    betas=(self.config.optimizer.beta1, 0.999),
                )
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self, is_train_gen: bool):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.calc_loss_D()
        self.loss_D.backward()
        self.optimizer_D.step()

        if is_train_gen:
            # update G
            self.set_requires_grad(self.netD, False)
            self.optimizer_G.zero_grad()
            if self.config.network.netF == "mlp_sample" and hasattr(
                self, "optimizer_F"
            ):
                self.optimizer_F.zero_grad()
            freq = (
                self.config.network.n_critic
                if self.config.network.gan_mode == "wgangp"
                else 1
            )
            self.loss_G = self.calc_loss_G(freq)
            self.loss_G.backward()
            self.optimizer_G.step()

            if self.config.network.netF == "mlp_sample":
                self.optimizer_F.step()

        self.update_agg_loss()

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = (
            torch.cat((self.pan_real, self.mono_real), dim=0)
            if self.config.network.nce_idt and self.config.phase == NetPhase.train
            else self.pan_real
        )
        if self.config.network.flip_equivariance:
            self.flipped_for_equivariance = self.config.isTrain and (
                np.random.random() < 0.5
            )
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        if self.config.thermal.is_fpa_input:
            if self.config.network.nce_idt and self.config.phase == NetPhase.train:
                self.fake = self.netG(
                    self.real,
                    src_fpa=torch.cat([self.pan_fpa, self.mono_fpa]),
                    tgt_fpa=torch.cat([self.mono_fpa, self.mono_fpa]),
                )
            else:
                self.fake = self.netG(
                    self.real, src_fpa=self.pan_fpa, tgt_fpa=self.mono_fpa
                )
        else:
            self.fake = self.netG(self.real)

        # Add calibration-based estimator to estimated monochromatic channel:
        if self.config.thermal.is_physical_model:
            self.mono_fake = self.fake[: self.pan_real.size(0)] + self.netPW(
                self.mono_phys
            )
        else:
            self.mono_fake = self.fake[: self.pan_real.size(0)]

        if self.config.network.nce_idt:
            self.mono_idt = self.fake[self.pan_real.size(0) :]

    def calc_loss_D(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.mono_fake.detach()
        real = self.mono_real
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.mono_real)
        self.loss_D_real = self.criterionGAN(self.pred_real, True).mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        # Add gradient loss (if wgan is used)
        if (
            self.phase == NetPhase.train
            and self.config.network.gan_mode == "wgangp"
            and self.config.loss.lambda_gp > 0
        ):
            gradient_penalty, _ = networks.cal_gradient_penalty(
                self.netD,
                real,
                fake,
                device="cuda",
                lambda_gp=self.config.loss.lambda_gp,
            )
            self.loss_D += gradient_penalty

        # log losses:
        self.iter_loss.loc["D"] = self.loss_D.item()
        self.iter_loss.loc["D_real"] = self.loss_D_real.item()
        self.iter_loss.loc["D_fake"] = self.loss_D_fake.item()

        return self.loss_D

    def calc_loss_G(self, freq: int = 1):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.mono_fake
        # First, G(A) should fake the discriminator
        if self.config.loss.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = (
                self.criterionGAN(pred_fake, True).mean() * self.config.loss.lambda_GAN
            )
        else:
            self.loss_G_GAN = 0.0

        if self.config.loss.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.pan_real, self.mono_fake)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.config.network.nce_idt and self.config.loss.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.mono_real, self.mono_idt)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both

        # log losses:
        self.iter_loss.loc["G"] = self.loss_G.item() * freq
        self.iter_loss.loc["G_GAN"] = self.loss_G_GAN.item() * freq
        self.iter_loss.loc["NCE"] = self.loss_NCE.item() * freq
        if self.config.network.nce_idt and self.config.loss.lambda_NCE > 0.0:
            self.iter_loss.loc["NCE_Y"] = self.loss_NCE_Y.item() * freq

        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)

        if self.config.thermal.is_fpa_input:
            feat_q = self.netG(
                tgt,
                self.nce_layers,
                encode_only=True,
                src_fpa=self.mono_fpa,
                tgt_fpa=self.mono_fpa,
            )
        else:
            feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.config.network.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        if self.config.thermal.is_fpa_input:
            feat_k = self.netG(
                src,
                self.nce_layers,
                encode_only=True,
                src_fpa=self.pan_fpa,
                tgt_fpa=self.mono_fpa,
            )
        else:
            feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(
            feat_k, self.config.network.num_patches, None
        )
        feat_q_pool, _ = self.netF(feat_q, self.config.network.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(
            feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers
        ):
            loss = crit(f_q, f_k) * self.config.loss.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def print_current_losses(self, epoch, iters, t_iter_start):
        """print current losses on console; also save the losses to the disk
        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        losses = self.iter_loss
        iter_time = time.time() - t_iter_start
        message_header = "epoch: %d, iters: %d, iter_dur: %.3f " % (
            epoch,
            iters,
            iter_time,
        )
        loss_names_msg = losses.keys().tolist()
        loss_vals = ["{:.3e}".format(val) for val in losses.values]
        loss_vals_msg = " ".join(loss_vals)

        message = "\n".join(
            [message_header, " ".join(loss_names_msg), loss_vals_msg, ""]
        )
        logger = logging.getLogger("PETIT")
        logger.debug(message)

    def plot_losses(self, tens_board, epoch):
        losses = self.loss_log.loc[epoch]

        # plot all individual loss components:
        losses_types = losses.keys().get_level_values(1).unique()
        for loss_type in losses_types:
            title = loss_type
            scalars = losses.loc[(pd.IndexSlice[:], loss_type)]
            tens_board.writer.add_scalars(title, scalars, epoch)

    def update_loss(self, epoch, loader_len: int):
        for loss in self.agg_loss.keys():
            self.loss_log.loc[epoch, (self.phase.name, loss)] = (
                self.agg_loss.loc[loss] / loader_len
            )

        # reset agg loss:
        self.agg_loss.loc[:] = 0
        return self.iter_loss

    def gen_vis_grid(self):
        visuals = self.get_current_visuals()
        vis_list = []
        for vis_name, vis in visuals.items():
            vis_np = tens2arr(vis)
            if (
                vis.shape[0] > 1
            ):  # the batch size is > 1, hence visualize only the first sample from the batch:
                vis_np = vis_np[0]
            vis_full_dr = full_dynamic_range(vis_np)
            vis_labeled = add_text_to_image(vis_full_dr, vis_name, fill="cyan")
            vis_list.append(transforms.ToTensor()(vis_labeled))
        vis_grid = torchvision.utils.make_grid(vis_list, nrow=len(vis_list))
        return vis_grid

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        for vis_type in self.visual_names:
            self.visuals[vis_type] = self.__getattribute__(vis_type)[
                :, 0, ...
            ].unsqueeze(axis=1)
        return self.visuals

    def save_res_log(self, target_dir: Path):
        target_dir.mkdir(parents=True, exist_ok=True)
        losses = self.loss_log[~(self.loss_log.isnull()).all(axis=1)]
        losses.to_excel(target_dir / "loss_log.xls")

        # melt table:
        losses.reset_index(level=0, inplace=True)
        losses_melt = losses.melt(id_vars=["Epoch"], var_name=["phase", "loss"])

        # render and save facet grids:
        grid = sns.FacetGrid(
            losses_melt,
            col="loss",
            hue="phase",
            col_wrap=3,
            sharey=False,
            height=5,
            aspect=1.5,
        )
        grid.map_dataframe(sns.lineplot, x="Epoch", y="value")
        grid.add_legend()
        grid.savefig(target_dir / "losses.png")
        plt.close()
