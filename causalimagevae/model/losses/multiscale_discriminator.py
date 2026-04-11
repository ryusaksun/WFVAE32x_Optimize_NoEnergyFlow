"""Multi-scale PatchGAN discriminator + feature extraction, ported from pix2pixHD.

Reference: https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py (MIT License)

Paper: Wang et al., "High-Resolution Image Synthesis and Semantic Manipulation with
Conditional GANs" (CVPR 2018, arXiv:1711.11585).

Unlike the single-scale `NLayerDiscriminator` in `discriminator.py`, this module:
1. Exposes each conv block as a separate `self.modelN` so intermediate feature maps
   can be collected during forward (for feature-matching loss).
2. Wraps `num_D` independent discriminators at progressively downsampled spatial
   scales (via AvgPool2d), returning a List[List[Tensor]] output.

The single-scale discriminator in `discriminator.py` is untouched — both coexist
and are selected via `LPIPSWithDiscriminator(disc_type=...)`.
"""

import numpy as np
import torch.nn as nn

from .discriminator import _get_disc_norm_layer, _maybe_sn


class NLayerDiscriminatorWithFeat(nn.Module):
    """PatchGAN discriminator that exposes intermediate layer outputs.

    Port of pix2pixHD `NLayerDiscriminator` with `getIntermFeat=True`.
    Layer layout (for n_layers=3): model0 (input conv), model1/model2 (stride-2
    middle), model3 (stride-1 penultimate), model4 (1-channel logit).
    Total submodules = n_layers + 2.

    Forward returns a list `[out0, out1, ..., outN]` where `outN` is the final
    logit and `out0..out{N-1}` are intermediate activations used for the
    feature-matching loss.

    ``disc_norm`` selects the normalization mode (``bn`` default / ``sn`` /
    ``in`` / ``none``). See ``discriminator._get_disc_norm_layer`` for details.
    """

    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3,
                 disc_norm: str = "bn"):
        super().__init__()
        self.n_layers = n_layers
        self.disc_norm = disc_norm

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        use_bias = disc_norm != "bn"

        sequence = [[
            _maybe_sn(
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                disc_norm,
            ),
            nn.LeakyReLU(0.2, True),
        ]]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                _maybe_sn(
                    nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    disc_norm,
                ),
                _get_disc_norm_layer(disc_norm, nf),
                nn.LeakyReLU(0.2, True),
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            _maybe_sn(
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                disc_norm,
            ),
            _get_disc_norm_layer(disc_norm, nf),
            nn.LeakyReLU(0.2, True),
        ]]
        sequence += [[
            _maybe_sn(
                nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw),
                disc_norm,
            ),
        ]]

        # Register each block as its own attribute so we can fetch intermediate outputs.
        for n in range(len(sequence)):
            setattr(self, "model" + str(n), nn.Sequential(*sequence[n]))

    def forward(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            block = getattr(self, "model" + str(n))
            res.append(block(res[-1]))
        return res[1:]  # drop the input itself; last element is the final logit


class MultiscaleDiscriminator(nn.Module):
    """Multi-scale PatchGAN discriminator (pix2pixHD).

    Runs `num_D` independent `NLayerDiscriminatorWithFeat` instances on the input
    at progressively downsampled spatial scales. Used with the feature-matching
    loss for more stable high-resolution adversarial training.

    Forward returns `List[List[Tensor]]`:
      - outer length = `num_D` (one entry per scale, ordered from full-res to
        smallest)
      - inner length = `n_layers + 2` (intermediate features + final logit)
      - `result[i][-1]` is the scale-`i` final PatchGAN logit (for GAN loss)
      - `result[i][:-1]` are the intermediate features (for feature-matching loss)
    """

    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3,
                 num_D: int = 3, disc_norm: str = "bn"):
        super().__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.disc_norm = disc_norm

        for i in range(num_D):
            netD = NLayerDiscriminatorWithFeat(
                input_nc=input_nc, ndf=ndf, n_layers=n_layers, disc_norm=disc_norm,
            )
            # Flatten each scale's submodules into this module so DDP sees them.
            for j in range(n_layers + 2):
                setattr(self, f"scale{i}_layer{j}", getattr(netD, f"model{j}"))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def _single_forward(self, scale_idx: int, x):
        res = [x]
        for j in range(self.n_layers + 2):
            block = getattr(self, f"scale{scale_idx}_layer{j}")
            res.append(block(res[-1]))
        return res[1:]

    def forward(self, x):
        # pix2pixHD indexes in reverse so `result[0]` is the full-resolution scale
        # and `result[num_D-1]` is the most downsampled. We follow the same convention.
        result = []
        input_downsampled = x
        for i in range(self.num_D):
            scale_idx = self.num_D - 1 - i
            result.append(self._single_forward(scale_idx, input_downsampled))
            if i != (self.num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result
