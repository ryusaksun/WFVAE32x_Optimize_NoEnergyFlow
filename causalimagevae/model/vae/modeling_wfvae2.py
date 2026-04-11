from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from ..modules import (
    ResnetBlock2D,
    Conv2d,
    HaarWaveletTransform2D,
    InverseHaarWaveletTransform2D,
    Normalize,
    nonlinearity,
    AttnBlock2D,
    Attention2DFix,
)
from ..registry import ModelRegistry
from ..modeling_videobase import VideoBaseAE
from ..utils.module_utils import resolve_str_to_obj
from ..utils.distrib_utils import DiagonalGaussianDistribution
from ..modeling_output import AutoencoderKLOutput, DecoderOutput, ForwardOutput
from diffusers.configuration_utils import register_to_config


class WFDownBlock(nn.Module):
    """Wavelet Flow Down Block for image processing (2D only)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        energy_flow_size: int = 128,
        num_res_blocks: int = 2,
        res_block: nn.Module = ResnetBlock2D,
        dropout: float = 0.0,
        norm_type: str = "groupnorm",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.energy_flow_size = energy_flow_size

        # 2D Haar wavelet: 4 coefficients × 3 RGB = 12 channels
        self.wavelet_transform = HaarWaveletTransform2D()
        self.in_flow_conv = Conv2d(
            12, energy_flow_size, kernel_size=3, stride=1, padding=1
        )
        self.res_block = nn.Sequential(
            *[res_block(in_channels=in_channels, out_channels=in_channels, dropout=dropout, norm_type=norm_type) for _ in range(num_res_blocks - 1)]
        )
        self.conv_down = Conv2d(in_channels, out_channels // 4, kernel_size=3, stride=1, padding=1)
        self.out_res_block = res_block(in_channels=out_channels + energy_flow_size, out_channels=out_channels, dropout=dropout, norm_type=norm_type)

    def _down_shortcut(self, x):
        """Non-parametric shortcut: [B, in_ch, H, W] → [B, out_ch, H/2, W/2]
        Space-to-channel (pixel_unshuffle) + channel averaging (DC-AE).
        Requires 4*in_channels divisible by out_channels.
        """
        x = F.pixel_unshuffle(x, 2)  # [B, 4*in_ch, H/2, W/2]
        B, C, H, W = x.shape
        out_ch = self.out_channels
        if C == out_ch:
            return x
        elif C > out_ch:
            return x.reshape(B, out_ch, C // out_ch, H, W).mean(dim=2)
        else:
            return x.repeat(1, out_ch // C, 1, 1)

    def forward(self, x, w):
        x = self.res_block(x)
        shortcut = self._down_shortcut(x)
        x = self.conv_down(x)
        x = F.pixel_unshuffle(x, 2)
        x = x + shortcut

        coeffs = self.wavelet_transform(w[:, :3])
        w = self.in_flow_conv(coeffs)

        x = torch.concat([x, w], dim=1)
        return self.out_res_block(x), coeffs


class WFUpBlock(nn.Module):
    """Wavelet Flow Up Block for image processing (2D only)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        energy_flow_size: int = 128,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        norm_type: str = "groupnorm",
        res_block: nn.Module = ResnetBlock2D
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.energy_flow_size = energy_flow_size
        assert num_res_blocks >= 2, "num res block too small"

        # 2D processing
        self.branch_conv = ResnetBlock2D(in_channels=in_channels, out_channels=in_channels + energy_flow_size, dropout=dropout, norm_type=norm_type)
        self.out_flow_conv = nn.Sequential(
            ResnetBlock2D(in_channels=energy_flow_size, out_channels=energy_flow_size, dropout=dropout, norm_type=norm_type),
            Conv2d(in_channels=energy_flow_size, out_channels=12, kernel_size=3, stride=1, padding=1)
        )
        self.inverse_wavelet_transform = InverseHaarWaveletTransform2D()
        self.res_block = nn.Sequential(
            *[res_block(in_channels=in_channels, out_channels=in_channels, dropout=dropout, norm_type=norm_type) for _ in range(num_res_blocks - 2)]
        )
        self.conv_up = Conv2d(in_channels, out_channels * 4, kernel_size=3, stride=1, padding=1)
        self.out_res_block = res_block(in_channels=out_channels, out_channels=out_channels, dropout=dropout, norm_type=norm_type)

    def _up_shortcut(self, x):
        """Non-parametric shortcut: [B, in_ch, H, W] → [B, out_ch, 2H, 2W]
        Channel duplicating + pixel_shuffle (DC-AE style).
        Requires out_channels * 4 divisible by in_channels.
        """
        repeats = self.out_channels * 4 // self.in_channels
        x = x.repeat_interleave(repeats, dim=1)  # [B, out_ch*4, H, W]
        x = F.pixel_shuffle(x, 2)                 # [B, out_ch, 2H, 2W]
        return x

    def forward(self, x, w=None):
        x = self.branch_conv(x)

        coeffs = self.out_flow_conv(x[:, -self.energy_flow_size:])
        if w is not None:
            coeffs = torch.cat([coeffs[:, :3] + w, coeffs[:, 3:]], dim=1)
        w = self.inverse_wavelet_transform(coeffs)

        x = self.res_block(x[:, :-self.energy_flow_size])
        shortcut = self._up_shortcut(x)
        x = self.conv_up(x)
        x = F.pixel_shuffle(x, 2)
        x = x + shortcut

        return self.out_res_block(x), w, coeffs


class Encoder(VideoBaseAE):

    @register_to_config
    def __init__(
        self,
        latent_dim: int = 16,
        num_resblocks: Union[int, List[int]] = 2,
        energy_flow_size: int = 64,
        dropout: float = 0.0,
        norm_type: str = "groupnorm",
        base_channels: List[int] = [128, 256, 512],
        mid_layers_type: List[str] = ["ResnetBlock2D", "Attention2DFix", "ResnetBlock2D"],
    ) -> None:
        super().__init__()
        num_down_stages = len(base_channels) - 1
        if isinstance(num_resblocks, int):
            num_resblocks = [num_resblocks] * num_down_stages
        assert len(num_resblocks) == num_down_stages, (
            f"encoder num_resblocks length {len(num_resblocks)} must match "
            f"number of down stages {num_down_stages}"
        )

        # 2D Haar: 12 channels (4 coeffs × 3 RGB)
        self.wavelet_transform_in = HaarWaveletTransform2D()
        self.conv_in = Conv2d(12, base_channels[0], kernel_size=3, stride=1, padding=1)

        self.down_blocks = nn.ModuleList()
        for idx in range(num_down_stages):
            down_block = WFDownBlock(
                in_channels=base_channels[idx],
                out_channels=base_channels[idx+1],
                energy_flow_size=energy_flow_size,
                num_res_blocks=num_resblocks[idx],
                res_block=ResnetBlock2D,
                dropout=dropout,
                norm_type=norm_type
            )
            self.down_blocks.append(down_block)

        # Mid
        mid_layers = []
        for mid_layer_type in mid_layers_type:
            if "Attn" in mid_layer_type or "Attention" in mid_layer_type:
                mid_layers.append(resolve_str_to_obj(mid_layer_type)(
                    in_channels=base_channels[-1], norm_type=norm_type,
                ))
            else:
                mid_layers.append(resolve_str_to_obj(mid_layer_type)(
                    in_channels=base_channels[-1], out_channels=base_channels[-1], dropout=dropout, norm_type=norm_type,
                ))
        self.mid = nn.Sequential(*mid_layers)

        self.norm_out = Normalize(base_channels[-1], norm_type=norm_type)
        self.conv_out = Conv2d(
            base_channels[-1], latent_dim * 2, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        coeffs = self.wavelet_transform_in(x)
        h = self.conv_in(coeffs)

        inter_coeffs = []
        for down_block in self.down_blocks:
            h, coeffs = down_block(h, coeffs)
            inter_coeffs.append(coeffs)

        h = self.mid(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h, inter_coeffs


class Decoder(VideoBaseAE):

    @register_to_config
    def __init__(
        self,
        latent_dim: int = 16,
        num_resblocks: Union[int, List[int]] = 2,
        dropout: float = 0.0,
        energy_flow_size: int = 128,
        norm_type: str = "groupnorm",
        base_channels: List[int] = [128, 256, 512],
        mid_layers_type: List[str] = ["ResnetBlock2D", "Attention2DFix", "ResnetBlock2D"],
    ) -> None:
        super().__init__()
        self.energy_flow_size = energy_flow_size

        num_up_stages = len(base_channels) - 1
        if isinstance(num_resblocks, int):
            num_resblocks = [num_resblocks] * num_up_stages
        assert len(num_resblocks) == num_up_stages, (
            f"decoder num_resblocks length {len(num_resblocks)} must match "
            f"number of up stages {num_up_stages}"
        )

        self.conv_in = Conv2d(
            latent_dim, base_channels[-1], kernel_size=3, stride=1, padding=1
        )

        mid_layers = []
        for mid_layer_type in mid_layers_type:
            if "Attn" in mid_layer_type or "Attention" in mid_layer_type:
                mid_layers.append(resolve_str_to_obj(mid_layer_type)(
                    in_channels=base_channels[-1],
                    norm_type=norm_type,
                ))
            else:
                mid_layers.append(resolve_str_to_obj(mid_layer_type)(
                    in_channels=base_channels[-1],
                    out_channels=base_channels[-1],
                    dropout=dropout,
                    norm_type=norm_type,
                ))
        self.mid = nn.Sequential(*mid_layers)

        self.up_blocks = nn.ModuleList()
        for stage_id, idx in enumerate(range(len(base_channels) - 1, 0, -1)):
            up_block = WFUpBlock(
                in_channels=base_channels[idx],
                out_channels=base_channels[idx-1],
                energy_flow_size=energy_flow_size,
                num_res_blocks=num_resblocks[stage_id],
                res_block=ResnetBlock2D,
                dropout=dropout,
                norm_type=norm_type
            )
            self.up_blocks.append(up_block)

        # Out
        self.norm_out = Normalize(base_channels[0], norm_type=norm_type)
        self.conv_out = Conv2d(base_channels[0], 12, kernel_size=3, stride=1, padding=1)
        self.inverse_wavelet_transform_out = InverseHaarWaveletTransform2D()

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid(h)
        inter_coeffs = []
        w = None
        for up_block in self.up_blocks:
            h, w, coeffs = up_block(h, w)
            inter_coeffs.append(coeffs)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h = torch.cat([h[:, :3] + w, h[:, 3:]], dim=1)
        dec = self.inverse_wavelet_transform_out(h)
        return dec, inter_coeffs


@ModelRegistry.register("WFIVAE2")
class WFIVAE2Model(VideoBaseAE):
    """Wavelet Flow Image VAE 2 - Pure image processing model."""

    @register_to_config
    def __init__(
        self,
        latent_dim: int = 16,
        base_channels: List[int] = [128, 256, 512],
        decoder_base_channels: Optional[List[int]] = None,
        encoder_num_resblocks: Union[int, List[int]] = 2,
        encoder_energy_flow_size: int = 128,
        decoder_num_resblocks: Union[int, List[int]] = 3,
        decoder_energy_flow_size: int = 128,
        dropout: float = 0.0,
        norm_type: str = "groupnorm",
        mid_layers_type: List[str] = ["ResnetBlock2D", "Attention2DFix", "ResnetBlock2D"],
        scale: List[float] = [0.18215] * 16,
        shift: List[float] = [0] * 16,
    ) -> None:
        super().__init__()
        self.use_tiling = False
        self.use_quant_layer = False

        if decoder_base_channels is None:
            decoder_base_channels = base_channels

        self.encoder = Encoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            num_resblocks=encoder_num_resblocks,
            energy_flow_size=encoder_energy_flow_size,
            dropout=dropout,
            norm_type=norm_type,
            mid_layers_type=mid_layers_type
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            base_channels=decoder_base_channels,
            num_resblocks=decoder_num_resblocks,
            energy_flow_size=decoder_energy_flow_size,
            dropout=dropout,
            norm_type=norm_type,
            mid_layers_type=mid_layers_type
        )

    def get_encoder(self):
        return [self.encoder]

    def get_decoder(self):
        return [self.decoder]

    def encode(self, x):
        h, coeffs = self.encoder(x)
        posterior = DiagonalGaussianDistribution(h)
        return AutoencoderKLOutput(latent_dist=posterior, extra_output=coeffs)

    def decode(self, z):
        dec, coeffs = self.decoder(z)
        return DecoderOutput(sample=dec, extra_output=coeffs)

    def forward(self, input, sample_posterior=True):
        encode_output = self.encode(input)
        posterior, enc_coeffs = (
            encode_output.latent_dist,
            encode_output.extra_output,
        )

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        decode_output = self.decode(z)
        dec, dec_coeffs = decode_output.sample, decode_output.extra_output

        return ForwardOutput(
            sample=dec,
            latent_dist=posterior,
            extra_output=(enc_coeffs, dec_coeffs),
        )

    def get_last_layer(self):
        if hasattr(self.decoder.conv_out, "conv"):
            return self.decoder.conv_out.conv.weight
        else:
            return self.decoder.conv_out.weight

    def init_from_ckpt(self, path, ignore_keys=None):
        if ignore_keys is None:
            ignore_keys = []
        sd = torch.load(path, map_location="cpu", weights_only=False)
        print("init from " + path)

        if (
            "ema_state_dict" in sd
            and len(sd["ema_state_dict"]) > 0
            and os.environ.get("NOT_USE_EMA_MODEL", "0") != "1"
        ):
            print("Load from ema model!")
            sd = sd["ema_state_dict"]
            sd = {key.replace("module.", ""): value for key, value in sd.items()}
        elif "state_dict" in sd:
            print("Load from normal model!")
            if "gen_model" in sd["state_dict"]:
                sd = sd["state_dict"]["gen_model"]
            else:
                sd = sd["state_dict"]

        keys = list(sd.keys())

        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
