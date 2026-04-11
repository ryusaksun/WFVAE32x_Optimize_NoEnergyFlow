import functools
import torch.nn as nn
from torch.nn.utils import spectral_norm as _spectral_norm
from ..modules.normalize import ActNorm


def _get_disc_norm_layer(disc_norm: str, channels: int) -> nn.Module:
    """Return the norm layer inserted between Conv and activation.

    - ``bn``   → BatchNorm2d (legacy default, batch-statistics dependent)
    - ``in``   → InstanceNorm2d (batch-size independent, per-sample normalization)
    - ``sn``   → Identity (spectral norm operates on weights, not activations)
    - ``none`` → Identity (pure conv + activation, ablation)
    """
    if disc_norm == "bn":
        return nn.BatchNorm2d(channels)
    elif disc_norm == "in":
        return nn.InstanceNorm2d(channels, affine=True)
    elif disc_norm in ("sn", "none"):
        return nn.Identity()
    else:
        raise ValueError(f"unknown disc_norm={disc_norm!r}")


def _maybe_sn(module: nn.Module, disc_norm: str) -> nn.Module:
    """Wrap a Conv/Linear module with spectral_norm when disc_norm == 'sn'.

    Uses PyTorch's official ``torch.nn.utils.spectral_norm`` (power iteration
    with one step per forward, u/v buffers persisted).
    """
    if disc_norm == "sn":
        return _spectral_norm(module)
    return module


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # spectral_norm 包装后 weight 是 computed property，weight_orig 才是真正的可学参数
        if hasattr(m, 'weight_orig'):
            nn.init.normal_(m.weight_orig.data, 0.0, 0.02)
        elif hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def weights_init_conv(m):
    if hasattr(m, 'conv'):
        m = m.conv
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

    Supports four normalization modes via ``disc_norm``:
        - ``bn``   : BatchNorm2d on middle layers (legacy default)
        - ``sn``   : torch.nn.utils.spectral_norm on every Conv2d, Identity instead of BN
        - ``in``   : InstanceNorm2d on middle layers (batch-size independent)
        - ``none`` : no normalization (Identity), pure Conv + LeakyReLU

    ``use_actnorm=True`` takes priority over ``disc_norm`` and forces ActNorm for
    backward compatibility with legacy experiments.
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False, disc_norm: str = "bn"):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)   -- the number of channels in input images
            ndf (int)        -- the number of filters in the last conv layer
            n_layers (int)   -- the number of conv layers in the discriminator
            use_actnorm      -- legacy path: if True, forces ActNorm and ignores disc_norm
            disc_norm (str)  -- one of {"bn", "sn", "in", "none"}
        """
        super(NLayerDiscriminator, self).__init__()

        kw = 4
        padw = 1

        if use_actnorm:
            # Legacy path: ActNorm, ignores disc_norm
            norm_layer = ActNorm
            use_bias = True  # ActNorm has its own affine, but we keep bias for legacy parity
            sequence = [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True),
            ]
            nf_mult = 1
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2 ** n, 8)
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw,
                              stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                ]
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n_layers, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw,
                          stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
            ]
            self.main = nn.Sequential(*sequence)
            return

        # New path: switchable disc_norm
        # BN has affine params → conv bias redundant; SN/IN/none need conv bias.
        use_bias = disc_norm != "bn"

        sequence = [
            _maybe_sn(
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                disc_norm,
            ),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                _maybe_sn(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw,
                              stride=2, padding=padw, bias=use_bias),
                    disc_norm,
                ),
                _get_disc_norm_layer(disc_norm, ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            _maybe_sn(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw,
                          stride=1, padding=padw, bias=use_bias),
                disc_norm,
            ),
            _get_disc_norm_layer(disc_norm, ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            _maybe_sn(
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
                disc_norm,
            ),
        ]
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)
