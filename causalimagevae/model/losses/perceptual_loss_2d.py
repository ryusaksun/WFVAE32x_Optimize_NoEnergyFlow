import torch
from torch import nn
import torch.nn.functional as F
from .lpips import LPIPS
from .discriminator import NLayerDiscriminator, weights_init
from ..modules.wavelet import HaarWaveletTransform2D


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real))
        + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def l1(x, y):
    return torch.abs(x - y)


def l2(x, y):
    return torch.pow((x - y), 2)


class LPIPSWithDiscriminator(nn.Module):
    """LPIPS loss with discriminator for 2D images"""
    def __init__(
        self,
        disc_start,
        logvar_init=0.0,
        kl_weight=1.0,
        pixelloss_weight=1.0,
        perceptual_weight=1.0,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_loss="hinge",
        learn_logvar: bool = False,
        wavelet_weight=0.01,
        loss_type: str = "l1",
    ):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.wavelet_weight = wavelet_weight
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.logvar = nn.Parameter(
            torch.full((), logvar_init), requires_grad=learn_logvar
        )
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels, ndf=64, n_layers=disc_num_layers, use_actnorm=use_actnorm
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.loss_func = l1 if loss_type == "l1" else l2

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        layer = last_layer if last_layer is not None else self.last_layer[0]

        nll_grads = torch.autograd.grad(nll_loss, layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-6)
        d_weight = torch.clamp(d_weight, 0.0, 1e6).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        inputs,
        reconstructions,
        posteriors,
        optimizer_idx,
        global_step,
        split="train",
        weights=None,
        last_layer=None,
        wavelet_coeffs=None,
    ):
        bs = inputs.shape[0]
        if optimizer_idx == 0:  # Generator
            # For images: (B, C, H, W) - no need to rearrange
            rec_loss = self.loss_func(inputs, reconstructions)
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs, reconstructions)
                rec_loss = rec_loss + self.perceptual_weight * p_loss
            nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
            weighted_nll_loss = nll_loss
            if weights is not None:
                weighted_nll_loss = weights * nll_loss
            weighted_nll_loss = (
                torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            )
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
            kl_loss = posteriors.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            # Wavelet loss for 2D
            if wavelet_coeffs:
                # wavelet_coeffs = (enc_coeffs, dec_coeffs)
                # enc_coeffs order: large → small (e.g., 256×256, 128×128, 64×64)
                # dec_coeffs order: small → large (e.g., 64×64, 128×128, 256×256)
                # Need to reverse dec_coeffs to match spatial dimensions
                enc_coeffs, dec_coeffs = wavelet_coeffs
                dec_coeffs_reversed = list(reversed(dec_coeffs))
                wl_loss = 0
                for enc_c, dec_c in zip(enc_coeffs, dec_coeffs_reversed):
                    wl_loss += torch.sum(l1(enc_c, dec_c)) / bs
            else:
                wl_loss = torch.tensor(0.0, device=inputs.device)

            # GAN loss
            if global_step >= self.discriminator_iter_start:
                logits_fake = self.discriminator(reconstructions)
                g_loss = -torch.mean(logits_fake)
                if self.disc_factor > 0.0:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                else:
                    d_weight = torch.tensor(1.0, device=inputs.device)
            else:
                d_weight = torch.tensor(0.0, device=inputs.device)
                g_loss = torch.tensor(0.0, device=inputs.device, requires_grad=True)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            loss = (
                weighted_nll_loss
                + self.kl_weight * kl_loss
                + d_weight * disc_factor * g_loss
                + self.wavelet_weight * wl_loss
            )

            log = {
                f"{split}/total_loss": loss.clone().detach().mean(),
                f"{split}/logvar": self.logvar.detach(),
                f"{split}/kl_loss": kl_loss.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/d_weight": d_weight.detach(),
                f"{split}/disc_factor": torch.tensor(disc_factor),
                f"{split}/g_loss": g_loss.detach().mean(),
            }
            log[f"{split}/p_loss"] = p_loss.detach().mean() if self.perceptual_weight > 0 else torch.tensor(0.0)
            if self.wavelet_weight > 0:
                log[f"{split}/wl_loss"] = wl_loss.detach().mean()
            return loss, log

        if optimizer_idx == 1:  # Discriminator
            logits_real = self.discriminator(inputs.detach())
            logits_fake = self.discriminator(reconstructions.detach())

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                f"{split}/disc_loss": d_loss.clone().detach().mean(),
                f"{split}/logits_real": logits_real.detach().mean(),
                f"{split}/logits_fake": logits_fake.detach().mean(),
            }
            return d_loss, log
