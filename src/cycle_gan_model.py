import itertools
from collections import namedtuple
from pathlib import Path

import networks
import pandas as pd
import seaborn as sns
import torch
import torchvision
from base_model import BaseModel
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
from utils.arrays import tens2arr
from utils.deep import NetPhase
from utils.visualizations import add_text_to_image, full_dynamic_range

DiscOut = namedtuple("DiscOut", ["D_real", "D_fake"])
DomainState = namedtuple("DomainState", ["real", "fake", "cycle", "identity"])


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    def __init__(self, config):
        """Initialize the CycleGAN class.

        Parameters:
            config (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        BaseModel.__init__(self, config)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_loss>
        domain_types = ["pan", "mono"]
        losses_types = ["D", "G", "cycle", "identity", "pred_real", "pred_fake"]
        idx = pd.MultiIndex.from_product([domain_types, losses_types])
        self.iter_loss = pd.Series(
            0, index=idx, dtype=float
        )  # logs the current iterations loss
        self.agg_loss = self.iter_loss.copy()
        phases_types = ["train", "val"]
        try:
            n_epochs_tot = config.scheduler.n_epochs + config.scheduler.n_epochs_decay
        except AttributeError:
            n_epochs_tot = 0
        cols = pd.MultiIndex.from_product([phases_types, domain_types, losses_types])
        self.loss_log = pd.DataFrame(
            index=range(n_epochs_tot), columns=cols, dtype=float
        )
        self.loss_log.index.rename("Epoch", inplace=True)
        visual_prot = {name: None for name in ["real", "fake", "cycle", "identity"]}
        self.visuals = {"pan": visual_prot.copy(), "mono": visual_prot.copy()}
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.

        if self.phase == NetPhase.train:
            self.model_names.extend(["G_pan", "G_mono", "D_pan", "D_mono"])
        else:  # during test time, only load Gs
            self.model_names.extend(["G_pan", "G_mono"])

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_pan (G), G_mono (F), D_pan (D_Y), D_mono (D_X)
        self.in_out_comb = config.network.in_out_comb
        self.netG_mono = networks.define_G(config, self.gpu_ids)
        self.netG_pan = networks.define_G(config, self.gpu_ids)
        if self.phase == NetPhase.train:  # define discriminators
            self.netD_pan = networks.define_D(config, self.gpu_ids)
            self.netD_mono = networks.define_D(config, self.gpu_ids)

            if (
                config.loss.lambda_identity > 0.0
            ):  # only works when input and output images have the same number of channels
                assert config.network.input_nc == config.network.output_nc
            # define loss functions
            # define GAN loss.
            self.criterionGAN = networks.GANLoss(config.network.gan_mode).to(
                self.device
            )
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if self.config.thermal.is_physical_model:
                self.optimizer_G = torch.optim.Adam(
                    itertools.chain(
                        self.netG_pan.parameters(),
                        self.netG_mono.parameters(),
                        self.netPW.parameters(),
                    ),
                    lr=config.optimizer.lr,
                    betas=(config.optimizer.beta1, 0.999),
                )
            else:
                self.optimizer_G = torch.optim.Adam(
                    itertools.chain(
                        self.netG_pan.parameters(), self.netG_mono.parameters()
                    ),
                    lr=config.optimizer.lr,
                    betas=(config.optimizer.beta1, 0.999),
                )
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(
                    self.netD_pan.parameters(), self.netD_mono.parameters()
                ),
                lr=config.optimizer.lr,
                betas=(config.optimizer.beta1, 0.999),
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def _forward(self, pan_fpa, mono_fpa):
        # Pan to Mono:
        self.mono_fake_deep = self.netG_mono(
            self.pan_real, src_fpa=pan_fpa, tgt_fpa=mono_fpa
        )
        if self.config.thermal.is_physical_model:
            self.mono_fake = self.mono_fake_deep + self.netPW(self.mono_phys)
        else:
            self.mono_fake = self.mono_fake_deep
        self.pan_cycle = self.netG_pan(
            self.mono_fake, src_fpa=mono_fpa, tgt_fpa=pan_fpa
        )  # G_pan(G_mono(A))
        self.pan_identity = self.netG_pan(
            self.pan_real, src_fpa=pan_fpa, tgt_fpa=pan_fpa
        )

        # Mono to Pan:
        self.pan_fake_deep = self.netG_pan(
            self.mono_real, src_fpa=mono_fpa, tgt_fpa=pan_fpa
        )  # G_pan(B)
        self.pan_fake = self.pan_fake_deep
        self.mono_cycle = self.netG_mono(
            self.pan_fake, src_fpa=pan_fpa, tgt_fpa=mono_fpa
        )  # G_mono(G_pan(B))
        self.mono_identity = self.netG_mono(
            self.mono_real, src_fpa=mono_fpa, tgt_fpa=mono_fpa
        )

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        # Outputs of Pan:
        if self.config.thermal.is_fpa_input:
            self._forward(pan_fpa=self.pan_fpa, mono_fpa=self.mono_fpa)
        else:
            self._forward(pan_fpa=None, mono_fpa=None)

    def calc_loss_G(self, freq: int = 1):
        lambda_idt = self.config.loss.lambda_identity
        lambda_GAN = self.config.loss.lambda_GAN
        lambda_GAN = self.config.loss.lambda_GAN

        if lambda_idt > 0:
            loss_pan_identity = (
                self.criterionIdt(self.pan_identity, self.pan_real)
                * lambda_GAN
                * lambda_idt
            )
            loss_mono_identity = (
                self.criterionIdt(self.mono_identity, self.mono_real)
                * lambda_GAN
                * lambda_idt
            )
        else:
            self.mono_identity = self.pan_identity = torch.zeros_like(self.pan_real)
            loss_pan_identity = torch.tensor(0)
            loss_mono_identity = torch.tensor(0)

        logits_mono_fake = self.netD_mono(self.mono_fake)
        loss_mono_G = self.criterionGAN(logits_mono_fake, True)

        logits_pan_fake = self.netD_pan(self.pan_fake)
        loss_pan_G = self.criterionGAN(logits_pan_fake, True)

        loss_pan_cycle = self.criterionCycle(self.pan_cycle, self.pan_real)

        loss_mono_cycle = self.criterionCycle(self.mono_cycle, self.mono_real)

        pan_tot_G = loss_pan_G + lambda_GAN * (
            loss_pan_cycle + loss_pan_identity * lambda_idt
        )
        mono_tot_G = loss_mono_G + lambda_GAN * (
            loss_mono_cycle + loss_mono_identity * lambda_idt
        )
        self.loss_tot_G = pan_tot_G + mono_tot_G

        # log losses:
        self.iter_loss.loc[("pan", "G")] = loss_pan_G.item() * freq
        self.iter_loss.loc[("pan", "cycle")] = loss_pan_cycle.item() * freq
        self.iter_loss.loc[("pan", "identity")] = loss_pan_identity.item() * freq
        self.iter_loss.loc[("mono", "G")] = loss_mono_G.item() * freq
        self.iter_loss.loc[("mono", "cycle")] = loss_mono_cycle.item() * freq
        self.iter_loss.loc[("mono", "identity")] = loss_mono_identity.item() * freq

    def calc_loss_D(self):
        loss_mono_D, pred_mono_real, pred_mono_fake = self.forward_D_basic(
            self.netD_mono, self.mono_real, self.mono_fake
        )
        loss_pan_D, pred_pan_real, pred_pan_fake = self.forward_D_basic(
            self.netD_pan, self.pan_real, self.pan_fake
        )
        self.loss_D_tot = loss_pan_D + loss_mono_D

        # log losses:
        self.iter_loss.loc[("pan", "D")] = loss_pan_D.item()
        self.iter_loss.loc[("pan", "pred_real")] = pred_pan_real.item()
        self.iter_loss.loc[("pan", "pred_fake")] = pred_pan_fake.item()
        self.iter_loss.loc[("mono", "D")] = loss_mono_D.item()
        self.iter_loss.loc[("mono", "pred_real")] = pred_mono_real.item()
        self.iter_loss.loc[("mono", "pred_fake")] = pred_mono_fake.item()

    def forward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """

        # Real
        logits_real = netD(real)
        loss_D_real = self.criterionGAN(logits_real, True)
        # Fake
        logits_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(logits_fake, False)

        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        # Add gradient loss (if wgan is used)
        if (
            self.phase == NetPhase.train
            and self.config.network.gan_mode == "wgangp"
            and self.config.loss.lambda_gp > 0
        ):
            gradient_penalty, _ = networks.cal_gradient_penalty(
                netD,
                real,
                fake.detach(),
                device="cuda",
                lambda_gp=self.config.loss.lambda_gp,
            )
            loss_D += gradient_penalty

        # compute predictions of descriminator outputs for debug:
        sigmoid = nn.Sigmoid()
        preds_real = sigmoid(logits_real).mean()
        preds_fake = sigmoid(logits_fake).mean()
        return loss_D, preds_real, preds_fake

    def optimize_parameters(self, is_train_gen: bool):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

        # Optimize D:
        self.set_requires_grad([self.netD_pan, self.netD_mono], True)
        self.calc_loss_D()
        self.optimizer_D.zero_grad()  # set D_pan and D_mono's gradients to zero
        self.loss_D_tot.backward()  # calculate gradients for D_pan
        self.optimizer_D.step()  # update D_pan and D_mono's weights

        if is_train_gen:
            # Optimize G:
            self.set_requires_grad([self.netD_pan, self.netD_mono], False)
            freq = (
                self.config.network.n_critic
                if self.config.network.gan_mode == "wgangp"
                else 1
            )
            self.calc_loss_G(freq)
            self.optimizer_G.zero_grad()  # set G_mono and G_pan's gradients to zero
            self.loss_tot_G.backward()  # calculate gradients for G_mono and G_pan
            self.optimizer_G.step()  # update G_mono and G_pan's weights

        self.update_agg_loss()

    def get_current_losses(self) -> pd.DataFrame:
        losses = self.iter_loss
        loss_names_msg = ["{}_{:<15}".format(wl, name) for wl, name in losses.keys()]
        loss_vals = ["{:.3e}".format(val) for val in losses.values]
        loss_vals_msg = " ".join(["{:<15}".format(val) for val in loss_vals])
        return "\n".join([" ".join(loss_names_msg), loss_vals_msg, ""])

    def plot_losses(self, tens_board, epoch):
        losses = self.loss_log.loc[epoch]
        loss_params = self.config.loss
        lambda_idt = loss_params.lambda_identity
        lambda_cycle = {"pan": loss_params.lambda_GAN, "mono": loss_params.lambda_GAN}

        # plot all individual loss components:
        losses_types = losses.keys().get_level_values(2).unique()
        domains = losses.keys().get_level_values(1).unique()

        def get_scalars(domain, loss_type):
            return losses.loc[pd.IndexSlice[:], domain, loss_type]

        total_losses = {}
        for domain in domains:
            for loss_type in losses_types:
                title = "_".join([loss_type, domain])
                scalars = get_scalars(domain, loss_type)
                tens_board.writer.add_scalars(title, scalars, epoch)

            # weighted-loss:
            weighted_G = get_scalars(domain, "G") * 1.0
            weighted_cycle = get_scalars(domain, "cycle") * lambda_cycle[domain]
            weighted_identity = (
                get_scalars(domain, "identity") * lambda_cycle[domain] * lambda_idt
            )
            weighted_tot = weighted_G + weighted_cycle + weighted_identity
            weighted_losses = {
                "G": weighted_G,
                "cycle": weighted_cycle,
                "identity": weighted_identity,
                "total": weighted_tot,
            }
            for loss_type, loss in weighted_losses.items():
                tens_board.writer.add_scalars(
                    "_".join(["Weighted", loss_type]), loss, epoch
                )
            total_losses[domain] = weighted_tot

    def update_loss(self, epoch, loader_len: int):
        for domain in self.agg_loss.keys().get_level_values(0).unique():
            for loss in self.agg_loss.keys().get_level_values(1).unique():
                self.loss_log.loc[epoch, (self.phase.name, domain, loss)] = (
                    self.agg_loss.loc[(domain, loss)] / loader_len
                )

        # reset agg loss:
        self.agg_loss.loc[:] = 0
        return self.iter_loss

    def gen_vis_grid(self):
        visuals = self.get_current_visuals()
        vis_list = []
        fake_idx = []
        for i, (domain, domain_vis) in enumerate(visuals.items()):
            for j, (vis_name, vis) in enumerate(domain_vis.items()):
                vis_np = tens2arr(vis)
                if (
                    vis.shape[0] > 1
                ):  # the batch size is > 1, hence visualize only the first sample from the batch:
                    vis_np = vis_np[0]
                vis_full_dr = full_dynamic_range(vis_np)
                if vis_name == "fake":
                    fake_idx.append(i * len(domain_vis) + j)
                fill_color = "cyan" if domain == "mono" else "yellow"
                vis_labeled = add_text_to_image(vis_full_dr, domain, fill=fill_color)
                rb_anchor_idx = [dim - 5 for dim in vis_full_dr.shape]
                vis_labeled = add_text_to_image(
                    vis_labeled, vis_name, origin_idx=tuple(rb_anchor_idx), anchor="rb"
                )
                vis_list.append(transforms.ToTensor()(vis_labeled))

        # swap the order of the fake images to align the visuals by scene:
        vis_list[fake_idx[0]], vis_list[fake_idx[-1]] = (
            vis_list[fake_idx[-1]],
            vis_list[fake_idx[0]],
        )

        vis_grid = torchvision.utils.make_grid(
            vis_list, nrow=len(vis_list) // len(visuals)
        )
        return vis_grid

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        for wl, vis_dicts in self.visuals.items():
            for vis_type in vis_dicts.keys():
                self.visuals[wl][vis_type] = self.__getattribute__(
                    "_".join((wl, vis_type))
                )
        return self.visuals

    def save_res_log(self, target_dir: Path):
        target_dir.mkdir(parents=True, exist_ok=True)
        losses = self.loss_log[~(self.loss_log.isnull()).all(axis=1)]
        losses.to_excel(target_dir / "loss_log.xls")

        # melt table:
        losses.reset_index(level=0, inplace=True)
        losses_melt = losses.melt(
            id_vars=["Epoch"], var_name=["phase", "domain", "loss"]
        )

        # render and save facet grids:
        for domain, loss_data in losses_melt.groupby("domain"):
            grid = sns.FacetGrid(
                loss_data,
                col="loss",
                hue="phase",
                col_wrap=3,
                sharey=False,
                height=5,
                aspect=1.5,
            )
            grid.map_dataframe(sns.lineplot, x="Epoch", y="value")
            grid.add_legend()
            grid.savefig(target_dir / f"{domain}.png")
            plt.close()
