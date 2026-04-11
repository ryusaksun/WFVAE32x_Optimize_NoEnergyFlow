import torch
from torch import nn
import torch.nn.functional as F
from .lpips import LPIPS
from .discriminator import NLayerDiscriminator, weights_init
from .multiscale_discriminator import MultiscaleDiscriminator
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
    """LPIPS loss with discriminator for 2D images.

    Supports two discriminator variants via ``disc_type``:

    - ``"single"`` — original single-scale PatchGAN (`NLayerDiscriminator`).
      Backward-compatible default behavior for legacy experiments.
    - ``"multiscale"`` — pix2pixHD multi-scale PatchGAN with feature-matching
      loss (see `multiscale_discriminator.py`). Enables ``feat_match_weight``.
    """
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
        adaptive_weight_clamp: float = 1e5,
        # --- multi-scale discriminator + feature matching (pix2pixHD) ---
        disc_type: str = "single",
        num_D: int = 3,
        n_layers_D: int = 3,
        feat_match_weight: float = 10.0,
        # --- discriminator normalization switch (Miyato et al. 2018) ---
        disc_norm: str = "bn",
    ):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        assert disc_type in ["single", "multiscale"]
        assert disc_norm in ["bn", "sn", "in", "none"], (
            f"disc_norm must be one of bn|sn|in|none, got {disc_norm!r}"
        )
        self.wavelet_weight = wavelet_weight
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.logvar = nn.Parameter(
            torch.full((), logvar_init), requires_grad=learn_logvar
        )

        self.disc_type = disc_type
        self.num_D = num_D
        self.n_layers_D = n_layers_D
        self.feat_match_weight = feat_match_weight
        self.disc_norm = disc_norm
        if disc_type == "multiscale":
            self.discriminator = MultiscaleDiscriminator(
                input_nc=disc_in_channels, ndf=64,
                n_layers=n_layers_D, num_D=num_D,
                disc_norm=disc_norm,
            ).apply(weights_init)
        else:
            self.discriminator = NLayerDiscriminator(
                input_nc=disc_in_channels, ndf=64,
                n_layers=disc_num_layers, use_actnorm=use_actnorm,
                disc_norm=disc_norm,
            ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.loss_func = l1 if loss_type == "l1" else l2
        self.adaptive_weight_clamp = adaptive_weight_clamp

    # ------------------------------------------------------------------
    # Multi-scale discriminator helpers
    # ------------------------------------------------------------------
    def _multiscale_g_loss(self, pred_list):
        """Average of -mean(final_logit) across all scales. Returns a scalar."""
        losses = [-torch.mean(p[-1]) for p in pred_list]
        return sum(losses) / len(losses)

    def _multiscale_d_loss(self, pred_real, pred_fake):
        """Per-scale hinge/vanilla disc loss, averaged across scales."""
        losses = [self.disc_loss(r[-1], f[-1]) for r, f in zip(pred_real, pred_fake)]
        return sum(losses) / len(losses)

    def _feature_match_loss(self, pred_fake, pred_real):
        """pix2pixHD feature-matching loss: L1 on intermediate discriminator features.

        Matches all layers except the final logit. Already multiplied by
        ``feat_match_weight``.
        """
        feat_weights = 4.0 / (self.n_layers_D + 1)
        D_weights = 1.0 / self.num_D
        loss = pred_fake[0][0].new_zeros(())
        for i in range(self.num_D):
            # len(pred_fake[i]) == n_layers + 2; skip the final element (logit)
            for j in range(len(pred_fake[i]) - 1):
                loss = loss + D_weights * feat_weights * F.l1_loss(
                    pred_fake[i][j], pred_real[i][j].detach()
                )
        return loss * self.feat_match_weight

    @staticmethod
    def _logits_mean(pred_list):
        """Aggregate final logits across scales into a scalar (for logging)."""
        return torch.stack([p[-1].detach().mean() for p in pred_list]).mean()

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        layer = last_layer if last_layer is not None else self.last_layer[0]

        nll_grads = torch.autograd.grad(nll_loss, layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, layer, retain_graph=True)[0]

        # 暂存梯度范数供 log dict 使用
        self._last_nll_grads_norm = torch.norm(nll_grads).detach()
        self._last_g_grads_norm = torch.norm(g_grads).detach()

        d_weight = self._last_nll_grads_norm / (self._last_g_grads_norm + 1e-6)
        d_weight = torch.clamp(d_weight, 0.0, self.adaptive_weight_clamp).detach()
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
            pixel_rec_loss = self.loss_func(inputs, reconstructions)
            rec_loss = pixel_rec_loss
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

            # Per-channel KL for monitoring posterior collapse
            kl_per_ch = posteriors.kl_per_channel().mean(0)  # [C]
            active_channels = (kl_per_ch > 0.01).sum().item()

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

            # GAN loss (+ optional feature-matching loss for multiscale path)
            fm_loss = torch.zeros((), device=inputs.device)
            if global_step >= self.discriminator_iter_start:
                if self.disc_type == "multiscale":
                    pred_fake = self.discriminator(reconstructions)
                    g_loss = self._multiscale_g_loss(pred_fake)
                    if self.feat_match_weight > 0:
                        # pred_real is only needed as a frozen L1 target — wrap in
                        # no_grad to free D's intermediate activations (~8-12 GB @
                        # 1024px). Detach at L1 call still keeps gradient from
                        # flowing back through pred_real.
                        with torch.no_grad():
                            pred_real = self.discriminator(inputs.detach())
                        fm_loss = self._feature_match_loss(pred_fake, pred_real)
                else:
                    logits_fake = self.discriminator(reconstructions)
                    g_loss = -torch.mean(logits_fake)

                if self.disc_factor > 0.0:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                else:
                    d_weight = torch.tensor(1.0, device=inputs.device)
                    self._last_nll_grads_norm = torch.tensor(0.0, device=inputs.device)
                    self._last_g_grads_norm = torch.tensor(0.0, device=inputs.device)
            else:
                d_weight = torch.tensor(0.0, device=inputs.device)
                g_loss = torch.tensor(0.0, device=inputs.device, requires_grad=True)
                self._last_nll_grads_norm = torch.tensor(0.0, device=inputs.device)
                self._last_g_grads_norm = torch.tensor(0.0, device=inputs.device)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            loss = (
                weighted_nll_loss
                + self.kl_weight * kl_loss
                + d_weight * disc_factor * g_loss
                + self.wavelet_weight * wl_loss
                + fm_loss
            )

            log = {
                f"{split}/total_loss": loss.clone().detach().mean(),
                f"{split}/logvar": self.logvar.detach(),
                f"{split}/kl_loss": kl_loss.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": pixel_rec_loss.detach().mean(),
                f"{split}/d_weight": d_weight.detach(),
                f"{split}/disc_factor": torch.tensor(disc_factor, device=inputs.device),
                f"{split}/g_loss": g_loss.detach().mean(),
            }
            log[f"{split}/p_loss"] = p_loss.detach().mean() if self.perceptual_weight > 0 else torch.tensor(0.0, device=inputs.device)
            log[f"{split}/nll_grads_norm"] = self._last_nll_grads_norm
            log[f"{split}/g_grads_norm"] = self._last_g_grads_norm
            log[f"{split}/fm_loss"] = fm_loss.detach().mean()
            if self.wavelet_weight > 0:
                log[f"{split}/wl_loss"] = wl_loss.detach().mean()
            log[f"{split}/kl_per_channel"] = kl_per_ch.detach().cpu()
            log[f"{split}/active_channels"] = torch.tensor(active_channels, device=inputs.device)
            return loss, log

        if optimizer_idx == 1:  # Discriminator
            if self.disc_type == "multiscale":
                pred_real = self.discriminator(inputs.detach())
                pred_fake = self.discriminator(reconstructions.detach())
                d_loss_raw = self._multiscale_d_loss(pred_real, pred_fake)
                logits_real_mean = self._logits_mean(pred_real)
                logits_fake_mean = self._logits_mean(pred_fake)
            else:
                logits_real = self.discriminator(inputs.detach())
                logits_fake = self.discriminator(reconstructions.detach())
                d_loss_raw = self.disc_loss(logits_real, logits_fake)
                logits_real_mean = logits_real.detach().mean()
                logits_fake_mean = logits_fake.detach().mean()

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            d_loss = disc_factor * d_loss_raw

            log = {
                f"{split}/disc_loss": d_loss.clone().detach().mean(),
                f"{split}/logits_real": logits_real_mean,
                f"{split}/logits_fake": logits_fake_mean,
            }
            return d_loss, log
