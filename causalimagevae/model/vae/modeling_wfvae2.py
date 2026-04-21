from typing import List, Literal, Optional, Union

BlockType = Literal["dcae", "classic"]
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
    Upsample,
    Downsample,
)
from ..registry import ModelRegistry
from ..modeling_videobase import VideoBaseAE
from ..utils.module_utils import resolve_str_to_obj
from ..utils.distrib_utils import DiagonalGaussianDistribution
from ..modeling_output import AutoencoderKLOutput, DecoderOutput, ForwardOutput
from diffusers.configuration_utils import register_to_config


class WFDownBlock(nn.Module):
    """Wavelet Flow Down Block for image processing (2D only).

    When ``use_energy_flow=False``, the mid-layer wavelet inflow (wavelet_transform
    + in_flow_conv) is dropped and ``out_res_block`` takes the main trunk alone
    (no concat with energy-flow channels). forward still returns a tuple for
    interface symmetry with the ef=True path.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        energy_flow_size: int = 128,
        num_res_blocks: int = 2,
        res_block: nn.Module = ResnetBlock2D,
        dropout: float = 0.0,
        norm_type: str = "groupnorm",
        use_eca: bool = False,
        use_energy_flow: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.energy_flow_size = energy_flow_size
        self.use_energy_flow = use_energy_flow

        if use_energy_flow:
            # 2D Haar wavelet: 4 coefficients × 3 RGB = 12 channels
            self.wavelet_transform = HaarWaveletTransform2D()
            self.in_flow_conv = Conv2d(
                12, energy_flow_size, kernel_size=3, stride=1, padding=1
            )
        self.res_block = nn.Sequential(
            *[res_block(in_channels=in_channels, out_channels=in_channels, dropout=dropout, norm_type=norm_type, use_eca=use_eca) for _ in range(num_res_blocks - 1)]
        )
        self.conv_down = Conv2d(in_channels, out_channels // 4, kernel_size=3, stride=1, padding=1)
        out_res_in = out_channels + energy_flow_size if use_energy_flow else out_channels
        self.out_res_block = res_block(in_channels=out_res_in, out_channels=out_channels, dropout=dropout, norm_type=norm_type, use_eca=use_eca)

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

    def forward(self, x, w=None):
        x = self.res_block(x)
        shortcut = self._down_shortcut(x)
        x = self.conv_down(x)
        x = F.pixel_unshuffle(x, 2)
        x = x + shortcut

        if self.use_energy_flow:
            coeffs = self.wavelet_transform(w[:, :3])
            flow = self.in_flow_conv(coeffs)
            x = torch.concat([x, flow], dim=1)
            return self.out_res_block(x), coeffs
        return self.out_res_block(x), None


class WFDownBlockClassic(nn.Module):
    """Classic-style Wavelet Flow Down Block.

    Spatial reduction delegates to the ``Downsample`` module from
    ``causalimagevae.model.modules.updownsample``; no DCAE residual shortcut.
    Signature matches ``WFDownBlock`` so the ``Encoder`` can switch between the
    two purely via ``block_type``. ``use_energy_flow=False`` drops the wavelet
    inflow and keeps ``out_res_block``'s input at ``in_channels``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        energy_flow_size: int = 128,
        num_res_blocks: int = 2,
        res_block: nn.Module = ResnetBlock2D,
        dropout: float = 0.0,
        norm_type: str = "groupnorm",
        use_eca: bool = False,
        use_energy_flow: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.energy_flow_size = energy_flow_size
        self.use_energy_flow = use_energy_flow

        if use_energy_flow:
            self.wavelet_transform = HaarWaveletTransform2D()
            self.in_flow_conv = Conv2d(
                12, energy_flow_size, kernel_size=3, stride=1, padding=1
            )
        self.res_block = nn.Sequential(
            *[res_block(in_channels=in_channels, out_channels=in_channels, dropout=dropout, norm_type=norm_type, use_eca=use_eca) for _ in range(num_res_blocks - 1)]
        )
        # Classic: 保持通道不变,通道变换在 out_res_block 里完成
        self.down = Downsample(in_channels=in_channels, out_channels=in_channels)
        out_res_in = in_channels + energy_flow_size if use_energy_flow else in_channels
        self.out_res_block = res_block(
            in_channels=out_res_in,
            out_channels=out_channels,
            dropout=dropout,
            norm_type=norm_type,
            use_eca=use_eca,
        )

    def forward(self, x, w=None):
        x = self.res_block(x)
        x = self.down(x)  # 空间 /2,通道不变

        if self.use_energy_flow:
            coeffs = self.wavelet_transform(w[:, :3])
            flow = self.in_flow_conv(coeffs)
            x = torch.concat([x, flow], dim=1)  # [in_ch + ef, H/2, W/2]
            return self.out_res_block(x), coeffs
        return self.out_res_block(x), None


class WFUpBlock(nn.Module):
    """Wavelet Flow Up Block for image processing (2D only).

    When ``use_energy_flow=False``: ``branch_conv`` keeps the channel count
    (no ``+energy_flow_size`` expansion), ``out_flow_conv`` /
    ``inverse_wavelet_transform`` are not created, and forward skips the
    wavelet-residual path. The returned tuple keeps ``(x, None, None)`` to
    match the ef=True signature.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        energy_flow_size: int = 128,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        norm_type: str = "groupnorm",
        res_block: nn.Module = ResnetBlock2D,
        use_eca: bool = False,
        use_energy_flow: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.energy_flow_size = energy_flow_size
        self.use_energy_flow = use_energy_flow
        assert num_res_blocks >= 2, "num res block too small"

        # 2D processing
        branch_out = in_channels + energy_flow_size if use_energy_flow else in_channels
        self.branch_conv = ResnetBlock2D(in_channels=in_channels, out_channels=branch_out, dropout=dropout, norm_type=norm_type, use_eca=use_eca)
        if use_energy_flow:
            self.out_flow_conv = nn.Sequential(
                ResnetBlock2D(in_channels=energy_flow_size, out_channels=energy_flow_size, dropout=dropout, norm_type=norm_type, use_eca=use_eca),
                Conv2d(in_channels=energy_flow_size, out_channels=12, kernel_size=3, stride=1, padding=1)
            )
            self.inverse_wavelet_transform = InverseHaarWaveletTransform2D()
        self.res_block = nn.Sequential(
            *[res_block(in_channels=in_channels, out_channels=in_channels, dropout=dropout, norm_type=norm_type, use_eca=use_eca) for _ in range(num_res_blocks - 2)]
        )
        self.conv_up = Conv2d(in_channels, out_channels * 4, kernel_size=3, stride=1, padding=1)
        self.out_res_block = res_block(in_channels=out_channels, out_channels=out_channels, dropout=dropout, norm_type=norm_type, use_eca=use_eca)

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

        if self.use_energy_flow:
            coeffs = self.out_flow_conv(x[:, -self.energy_flow_size:])
            if w is not None:
                coeffs = torch.cat([coeffs[:, :3] + w, coeffs[:, 3:]], dim=1)
            w = self.inverse_wavelet_transform(coeffs)
            x = x[:, :-self.energy_flow_size]
        else:
            coeffs = None
            w = None

        x = self.res_block(x)
        shortcut = self._up_shortcut(x)
        x = self.conv_up(x)
        x = F.pixel_shuffle(x, 2)
        x = x + shortcut

        return self.out_res_block(x), w, coeffs


class WFUpBlockClassic(nn.Module):
    """Classic-style Wavelet Flow Up Block.

    Spatial expansion delegates to the ``Upsample`` module from
    ``causalimagevae.model.modules.updownsample``; no DCAE residual shortcut.
    Signature matches ``WFUpBlock`` so the ``Decoder`` can switch between the
    two purely via ``block_type``. ``use_energy_flow=False`` keeps
    ``branch_conv`` channels constant and drops the wavelet outflow path.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        energy_flow_size: int = 128,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        norm_type: str = "groupnorm",
        res_block: nn.Module = ResnetBlock2D,
        use_eca: bool = False,
        use_energy_flow: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.energy_flow_size = energy_flow_size
        self.use_energy_flow = use_energy_flow
        assert num_res_blocks >= 2, "num res block too small"

        branch_out = in_channels + energy_flow_size if use_energy_flow else in_channels
        self.branch_conv = ResnetBlock2D(
            in_channels=in_channels,
            out_channels=branch_out,
            dropout=dropout,
            norm_type=norm_type,
            use_eca=use_eca,
        )
        if use_energy_flow:
            self.out_flow_conv = nn.Sequential(
                ResnetBlock2D(
                    in_channels=energy_flow_size,
                    out_channels=energy_flow_size,
                    dropout=dropout,
                    norm_type=norm_type,
                    use_eca=use_eca,
                ),
                Conv2d(in_channels=energy_flow_size, out_channels=12, kernel_size=3, stride=1, padding=1),
            )
            self.inverse_wavelet_transform = InverseHaarWaveletTransform2D()
        self.res_block = nn.Sequential(
            *[res_block(in_channels=in_channels, out_channels=in_channels, dropout=dropout, norm_type=norm_type, use_eca=use_eca) for _ in range(num_res_blocks - 2)]
        )
        # Classic: 保持通道不变,通道变换在 out_res_block 里完成
        self.up = Upsample(in_channels=in_channels, out_channels=in_channels)
        self.out_res_block = res_block(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            norm_type=norm_type,
            use_eca=use_eca,
        )

    def forward(self, x, w=None):
        x = self.branch_conv(x)

        if self.use_energy_flow:
            coeffs = self.out_flow_conv(x[:, -self.energy_flow_size:])
            if w is not None:
                coeffs = torch.cat([coeffs[:, :3] + w, coeffs[:, 3:]], dim=1)
            w = self.inverse_wavelet_transform(coeffs)
            x = x[:, :-self.energy_flow_size]
        else:
            coeffs = None
            w = None

        x = self.res_block(x)
        x = self.up(x)  # 空间 *2,通道不变

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
        use_io_shortcut: bool = False,
        use_eca: bool = False,
        block_type: BlockType = "dcae",
        use_energy_flow: bool = True,
    ) -> None:
        super().__init__()
        assert block_type in ("dcae", "classic"), f"unknown block_type: {block_type}"
        num_down_stages = len(base_channels) - 1
        if isinstance(num_resblocks, int):
            num_resblocks = [num_resblocks] * num_down_stages
        assert len(num_resblocks) == num_down_stages, (
            f"encoder num_resblocks length {len(num_resblocks)} must match "
            f"number of down stages {num_down_stages}"
        )

        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.use_io_shortcut = use_io_shortcut
        self.use_energy_flow = use_energy_flow
        if use_io_shortcut:
            assert base_channels[-1] % (2 * latent_dim) == 0, (
                f"Encoder I/O shortcut 要求 base_channels[-1]={base_channels[-1]} "
                f"能被 2*latent_dim={2*latent_dim} 整除"
            )

        # 2D Haar: 12 channels (4 coeffs × 3 RGB) — W^(1) input-side wavelet
        # is retained regardless of use_energy_flow, since it drives the 2×
        # spatial compression at the encoder entry (not the mid-layer pathway).
        self.wavelet_transform_in = HaarWaveletTransform2D()
        self.conv_in = Conv2d(12, base_channels[0], kernel_size=3, stride=1, padding=1)

        down_block_cls = WFDownBlock if block_type == "dcae" else WFDownBlockClassic
        self.down_blocks = nn.ModuleList()
        for idx in range(num_down_stages):
            down_block = down_block_cls(
                in_channels=base_channels[idx],
                out_channels=base_channels[idx+1],
                energy_flow_size=energy_flow_size,
                num_res_blocks=num_resblocks[idx],
                res_block=ResnetBlock2D,
                dropout=dropout,
                norm_type=norm_type,
                use_eca=use_eca,
                use_energy_flow=use_energy_flow,
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
                    in_channels=base_channels[-1], out_channels=base_channels[-1], dropout=dropout, norm_type=norm_type, use_eca=use_eca,
                ))
        self.mid = nn.Sequential(*mid_layers)

        self.norm_out = Normalize(base_channels[-1], norm_type=norm_type)
        self.conv_out = Conv2d(
            base_channels[-1], latent_dim * 2, kernel_size=3, stride=1, padding=1
        )

    def _out_shortcut(self, h):
        """非参数 shortcut: [B, base_channels[-1], H, W] -> [B, 2*latent_dim, H, W].
        按通道分组取均值 (DC-AE / HunyuanVAE 风格)。零参数。
        """
        B, C, H, W = h.shape
        out_ch = 2 * self.latent_dim
        return h.reshape(B, out_ch, C // out_ch, H, W).mean(dim=2)

    def forward(self, x):
        coeffs = self.wavelet_transform_in(x)
        h = self.conv_in(coeffs)

        if self.use_energy_flow:
            inter_coeffs = []
            for down_block in self.down_blocks:
                h, coeffs = down_block(h, coeffs)
                inter_coeffs.append(coeffs)
        else:
            inter_coeffs = None
            for down_block in self.down_blocks:
                h, _ = down_block(h)

        h = self.mid(h)

        shortcut = self._out_shortcut(h) if self.use_io_shortcut else None
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if shortcut is not None:
            h = h + shortcut

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
        use_io_shortcut: bool = False,
        use_eca: bool = False,
        block_type: BlockType = "dcae",
        use_energy_flow: bool = True,
    ) -> None:
        super().__init__()
        assert block_type in ("dcae", "classic"), f"unknown block_type: {block_type}"
        self.energy_flow_size = energy_flow_size

        num_up_stages = len(base_channels) - 1
        if isinstance(num_resblocks, int):
            num_resblocks = [num_resblocks] * num_up_stages
        assert len(num_resblocks) == num_up_stages, (
            f"decoder num_resblocks length {len(num_resblocks)} must match "
            f"number of up stages {num_up_stages}"
        )

        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.use_io_shortcut = use_io_shortcut
        self.use_energy_flow = use_energy_flow
        if use_io_shortcut:
            assert base_channels[-1] % latent_dim == 0, (
                f"Decoder I/O shortcut 要求 base_channels[-1]={base_channels[-1]} "
                f"能被 latent_dim={latent_dim} 整除"
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
                    use_eca=use_eca,
                ))
        self.mid = nn.Sequential(*mid_layers)

        up_block_cls = WFUpBlock if block_type == "dcae" else WFUpBlockClassic
        self.up_blocks = nn.ModuleList()
        for stage_id, idx in enumerate(range(len(base_channels) - 1, 0, -1)):
            up_block = up_block_cls(
                in_channels=base_channels[idx],
                out_channels=base_channels[idx-1],
                energy_flow_size=energy_flow_size,
                num_res_blocks=num_resblocks[stage_id],
                res_block=ResnetBlock2D,
                dropout=dropout,
                norm_type=norm_type,
                use_eca=use_eca,
                use_energy_flow=use_energy_flow,
            )
            self.up_blocks.append(up_block)

        # Out
        self.norm_out = Normalize(base_channels[0], norm_type=norm_type)
        self.conv_out = Conv2d(base_channels[0], 12, kernel_size=3, stride=1, padding=1)
        self.inverse_wavelet_transform_out = InverseHaarWaveletTransform2D()

    def _in_shortcut(self, z):
        """非参数 shortcut: [B, latent_dim, H, W] -> [B, base_channels[-1], H, W].
        按通道 repeat_interleave (DC-AE / HunyuanVAE 风格)。零参数。
        """
        repeats = self.base_channels[-1] // self.latent_dim
        return z.repeat_interleave(repeats, dim=1)

    def forward(self, z):
        h = self.conv_in(z)
        if self.use_io_shortcut:
            h = h + self._in_shortcut(z)
        h = self.mid(h)
        if self.use_energy_flow:
            inter_coeffs = []
            w = None
            for up_block in self.up_blocks:
                h, w, coeffs = up_block(h, w)
                inter_coeffs.append(coeffs)
            h = self.norm_out(h)
            h = nonlinearity(h)
            h = self.conv_out(h)
            h = torch.cat([h[:, :3] + w, h[:, 3:]], dim=1)
        else:
            inter_coeffs = None
            for up_block in self.up_blocks:
                h, _, _ = up_block(h)
            h = self.norm_out(h)
            h = nonlinearity(h)
            h = self.conv_out(h)
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
        use_io_shortcut: bool = False,
        use_eca: bool = False,
        block_type: BlockType = "dcae",
        use_energy_flow: bool = True,
        scale: List[float] = [0.18215] * 16,
        shift: List[float] = [0] * 16,
    ) -> None:
        super().__init__()
        self.use_tiling = False
        self.use_quant_layer = False

        assert block_type in ("dcae", "classic"), f"unknown block_type: {block_type}"
        assert isinstance(use_energy_flow, bool), (
            f"use_energy_flow must be bool, got {type(use_energy_flow).__name__}"
        )
        self.use_energy_flow = use_energy_flow

        # Classic 路径下 I/O shortcut 的通道 repeat 语义不成立,软降级 + warning
        if block_type == "classic" and use_io_shortcut:
            print(
                "[WFIVAE2Model] block_type='classic' is incompatible with "
                "use_io_shortcut=True; forcing use_io_shortcut=False. "
                "(Classic 路径下 I/O shortcut 的通道 repeat 语义不成立。)"
            )
            use_io_shortcut = False
            # 同步写回 config,使保存的 config.json 反映真实运行状态
            self.register_to_config(use_io_shortcut=False)

        if decoder_base_channels is None:
            decoder_base_channels = base_channels

        self.encoder = Encoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            num_resblocks=encoder_num_resblocks,
            energy_flow_size=encoder_energy_flow_size,
            dropout=dropout,
            norm_type=norm_type,
            mid_layers_type=mid_layers_type,
            use_io_shortcut=use_io_shortcut,
            use_eca=use_eca,
            block_type=block_type,
            use_energy_flow=use_energy_flow,
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            base_channels=decoder_base_channels,
            num_resblocks=decoder_num_resblocks,
            energy_flow_size=decoder_energy_flow_size,
            dropout=dropout,
            norm_type=norm_type,
            mid_layers_type=mid_layers_type,
            use_io_shortcut=use_io_shortcut,
            use_eca=use_eca,
            block_type=block_type,
            use_energy_flow=use_energy_flow,
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

        # When use_energy_flow=False, enc_coeffs/dec_coeffs are None. Collapse
        # the pair to a single None so downstream code (`if extra_output:` in
        # train loop and `if wavelet_coeffs:` in loss) skip WL loss correctly
        # — a `(None, None)` tuple would be truthy and mis-trigger the branch.
        extra_output = (enc_coeffs, dec_coeffs) if enc_coeffs is not None else None

        return ForwardOutput(
            sample=dec,
            latent_dist=posterior,
            extra_output=extra_output,
        )

    def get_last_layer(self):
        if hasattr(self.decoder.conv_out, "conv"):
            return self.decoder.conv_out.conv.weight
        else:
            return self.decoder.conv_out.weight

    def _warn_block_type_mismatch(self, state_dict):
        """Detect whether a ckpt's block_type structure matches this model's.

        Uses the current model's own ``state_dict().keys()`` as ground truth:
        keys present in the ckpt but unknown to the model (unexpected) and
        keys expected by the model but absent from the ckpt (missing) are
        typical signatures of a DCAE↔classic mix-up.
        A single warning is printed when **most** of one side overlaps with
        the other — enough to diagnose accidental mix-ups without being
        noisy on merged / partial checkpoints.
        """
        model_keys = set(self.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys
        # Heuristic: if both "missing" and "unexpected" are non-trivial and
        # touch WFDown/WFUp sub-modules, it's almost certainly a block_type mismatch.
        block_missing = [k for k in missing if ".down_blocks." in k or ".up_blocks." in k]
        block_unexpected = [k for k in unexpected if ".down_blocks." in k or ".up_blocks." in k]
        if block_missing and block_unexpected:
            expected = self.config.get("block_type", "dcae")
            print(
                f"[WARN] block_type='{expected}' but checkpoint has {len(block_unexpected)} "
                f"incompatible WFDown/WFUp keys (e.g. {sorted(block_unexpected)[:2]}) and is "
                f"missing {len(block_missing)} expected keys — you probably loaded a ckpt "
                f"trained with the other block_type."
            )

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

        # 检测 block_type mismatch,给出可读的诊断
        self._warn_block_type_mismatch(sd)

        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
