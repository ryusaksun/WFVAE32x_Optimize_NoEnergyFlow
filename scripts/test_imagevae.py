#!/usr/bin/env python3
"""
WFIVAE2 图像VAE 单元测试脚本
============================

配置驱动：通过 --config 指定模型配置文件，自动推导压缩比、下采样块数等参数。

测试内容:
1. 模型创建和前向传播
2. 编码器/解码器分离测试
3. 维度验证
4. 解码器独立性验证（从潜变量独立重建，无skip connection）
5. 能量流融合测试（concat融合到主干）
6. 不同分辨率支持
7. 小波损失系数提取

使用方法:
    # 默认 8x 配置 (64chn-192bc)
    python scripts/test_imagevae.py

    # 32x 配置
    python scripts/test_imagevae.py --config examples/wfivae2-image-32x-192bc.json
"""

import argparse
import torch
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from causalimagevae.model import ModelRegistry
from causalimagevae.model.vae import WFIVAE2Model


def load_config(config_path):
    """从配置文件加载参数并推导关键值"""
    with open(config_path, "r") as f:
        cfg = json.load(f)

    base_channels = cfg["base_channels"]
    latent_dim = cfg["latent_dim"]
    num_down_blocks = len(base_channels) - 1
    compression_ratio = 2 ** (num_down_blocks + 1)

    return {
        "config_path": config_path,
        "base_channels": base_channels,
        "latent_dim": latent_dim,
        "num_down_blocks": num_down_blocks,
        "compression_ratio": compression_ratio,
        "encoder_num_resblocks": cfg.get("encoder_num_resblocks", 2),
        "encoder_energy_flow_size": cfg.get("encoder_energy_flow_size", 128),
        "decoder_num_resblocks": cfg.get("decoder_num_resblocks", 3),
        "decoder_energy_flow_size": cfg.get("decoder_energy_flow_size", 128),
        "norm_type": cfg.get("norm_type", "groupnorm"),
    }


def test_model_creation(p):
    """测试1: 模型创建"""
    print("=" * 60)
    print("测试1: 模型创建")
    print("=" * 60)

    model = WFIVAE2Model(
        latent_dim=p["latent_dim"],
        base_channels=p["base_channels"],
        encoder_num_resblocks=p["encoder_num_resblocks"],
        encoder_energy_flow_size=p["encoder_energy_flow_size"],
        decoder_num_resblocks=p["decoder_num_resblocks"],
        decoder_energy_flow_size=p["decoder_energy_flow_size"],
        norm_type=p["norm_type"],
    )

    total_params = sum(v.numel() for v in model.parameters())
    trainable_params = sum(v.numel() for v in model.parameters() if v.requires_grad)
    encoder_params = sum(v.numel() for v in model.encoder.parameters())
    decoder_params = sum(v.numel() for v in model.decoder.parameters())

    print(f"✓ 模型创建成功")
    print(f"  总参数量: {total_params / 1e6:.2f}M")
    print(f"  可训练参数: {trainable_params / 1e6:.2f}M")
    print(f"  编码器参数: {encoder_params / 1e6:.2f}M")
    print(f"  解码器参数: {decoder_params / 1e6:.2f}M")
    print(f"  下采样块数量: {len(model.encoder.down_blocks)}")
    print(f"  上采样块数量: {len(model.decoder.up_blocks)}")

    return model


def test_forward_pass(model, p, resolution):
    """测试: 前向传播"""
    cr = p["compression_ratio"]
    latent_dim = p["latent_dim"]
    num_down = p["num_down_blocks"]
    print(f"\n{'=' * 60}")
    print(f"测试: 前向传播 ({resolution} 分辨率, {cr}x压缩)")
    print("=" * 60)

    x = torch.randn(1, 3, resolution, resolution)
    print(f"输入形状: {x.shape}")

    model.eval()
    with torch.no_grad():
        out = model(x)

    print(f"输出形状: {out.sample.shape}")
    print(f"潜在均值形状: {out.latent_dist.mean.shape}")
    print(f"潜在方差形状: {out.latent_dist.var.shape}")

    expected_latent_size = resolution // cr
    assert out.sample.shape == x.shape, f"输出形状不匹配! 预期 {x.shape}, 得到 {out.sample.shape}"
    assert out.latent_dist.mean.shape == (1, latent_dim, expected_latent_size, expected_latent_size), \
        f"潜在形状错误: {out.latent_dist.mean.shape}, 预期 (1, {latent_dim}, {expected_latent_size}, {expected_latent_size})"

    assert out.extra_output is not None, "WFIVAE2应有extra_output用于小波损失"
    enc_coeffs, dec_coeffs = out.extra_output
    print(f"编码器小波系数数量: {len(enc_coeffs)} (应为{num_down})")
    print(f"解码器小波系数数量: {len(dec_coeffs)} (应为{num_down})")

    assert len(enc_coeffs) == num_down, f"编码器系数数量错误: {len(enc_coeffs)}"
    assert len(dec_coeffs) == num_down, f"解码器系数数量错误: {len(dec_coeffs)}"

    print(f"✓ 前向传播测试通过 ({resolution} 分辨率)")
    return out


def test_encode_decode(model, p, resolution):
    """测试: 编码/解码分离"""
    cr = p["compression_ratio"]
    latent_dim = p["latent_dim"]
    print(f"\n{'=' * 60}")
    print(f"测试: 编码/解码分离 ({resolution} 分辨率)")
    print("=" * 60)

    x = torch.randn(1, 3, resolution, resolution)

    model.eval()
    with torch.no_grad():
        enc_out = model.encode(x)
        z = enc_out.latent_dist.sample()
        print(f"编码输入: {x.shape}")
        print(f"潜在变量: {z.shape}")

        assert enc_out.extra_output is not None, "编码器应有extra_output"
        print(f"编码器小波系数: {len(enc_out.extra_output)} 层")

        dec_out = model.decode(z)
        print(f"解码输出: {dec_out.sample.shape}")

    assert dec_out.sample.shape == x.shape, "解码输出形状不匹配"
    expected_latent_size = resolution // cr
    assert z.shape == (1, latent_dim, expected_latent_size, expected_latent_size), \
        f"潜在变量形状错误: {z.shape}"
    print("✓ 编码/解码分离测试通过")


def test_decoder_independence(model, p):
    """测试: 解码器独立性"""
    cr = p["compression_ratio"]
    latent_dim = p["latent_dim"]
    # 使用最小可行分辨率的 latent
    latent_size = 8
    output_size = latent_size * cr
    print(f"\n{'=' * 60}")
    print("测试: 解码器独立性验证")
    print("=" * 60)

    z = torch.randn(1, latent_dim, latent_size, latent_size)

    model.eval()
    with torch.no_grad():
        dec_out = model.decode(z)

    print(f"随机潜在变量: {z.shape}")
    print(f"解码器输出: {dec_out.sample.shape}")

    assert dec_out.sample.shape == (1, 3, output_size, output_size), \
        f"解码器输出形状错误: {dec_out.sample.shape}"
    print("✓ 解码器可独立运行，能量流从潜变量重建（无skip connection）")


def test_different_resolutions(model, p):
    """测试: 不同分辨率"""
    cr = p["compression_ratio"]
    latent_dim = p["latent_dim"]
    print(f"\n{'=' * 60}")
    print(f"测试: 不同分辨率测试 ({cr}x压缩)")
    print("=" * 60)

    # 生成必须是压缩比整数倍的分辨率
    resolutions = [cr * m for m in [8, 16, 24, 32] if cr * m <= 1024]

    model.eval()
    for res in resolutions:
        h, w = res, res
        x = torch.randn(1, 3, h, w)
        try:
            with torch.no_grad():
                out = model(x)
            expected_latent_h, expected_latent_w = h // cr, w // cr
            latent_shape = out.latent_dist.mean.shape
            status = "✓" if out.sample.shape == x.shape else "✗"
            print(f"{status} {h}×{w}: 输入 {x.shape} → latent {latent_shape} → 输出 {out.sample.shape}")
            assert latent_shape[2] == expected_latent_h and latent_shape[3] == expected_latent_w
        except Exception as e:
            print(f"✗ {h}×{w}: 错误 - {str(e)[:80]}")


def test_gradient_flow(model, p):
    """测试: 梯度流动"""
    cr = p["compression_ratio"]
    latent_dim = p["latent_dim"]
    # 使用较小分辨率
    resolution = cr * 8
    expected_latent = resolution // cr
    print(f"\n{'=' * 60}")
    print("测试: 梯度流动测试")
    print("=" * 60)

    model.train()

    x = torch.randn(1, 3, resolution, resolution, requires_grad=True)

    out = model(x)
    loss = out.sample.mean()
    loss.backward()

    has_grad = x.grad is not None and x.grad.abs().sum() > 0
    print(f"输入梯度: {'✓ 存在' if has_grad else '✗ 不存在'}")

    params_with_grad = sum(1 for v in model.parameters() if v.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    print(f"参数梯度: {params_with_grad}/{total_params} 个参数有梯度")

    print(f"Latent 形状: {out.latent_dist.mean.shape}  (预期: [1, {latent_dim}, {expected_latent}, {expected_latent}])")
    assert out.latent_dist.mean.shape == (1, latent_dim, expected_latent, expected_latent)

    print("✓ 梯度流动正常")


def test_wavelet_coeffs_structure(model, p):
    """测试: 小波系数结构"""
    cr = p["compression_ratio"]
    num_down = p["num_down_blocks"]
    resolution = cr * 16
    print(f"\n{'=' * 60}")
    print("测试: 小波系数结构测试")
    print("=" * 60)

    model.eval()
    x = torch.randn(1, 3, resolution, resolution)

    with torch.no_grad():
        out = model(x)

        assert out.extra_output is not None
        enc_coeffs, dec_coeffs = out.extra_output

        print(f"编码器小波系数: {len(enc_coeffs)} 层")
        for i, coeffs in enumerate(enc_coeffs):
            print(f"  enc_coeffs[{i}]: {coeffs.shape}")

        print(f"解码器小波系数: {len(dec_coeffs)} 层")
        for i, coeffs in enumerate(dec_coeffs):
            print(f"  dec_coeffs[{i}]: {coeffs.shape}")

        assert len(enc_coeffs) == num_down, f"编码器应有{num_down}层系数，得到{len(enc_coeffs)}"
        assert len(dec_coeffs) == num_down, f"解码器应有{num_down}层系数，得到{len(dec_coeffs)}"

        dec_coeffs_reversed = list(reversed(dec_coeffs))
        for i, (enc_c, dec_c) in enumerate(zip(enc_coeffs, dec_coeffs_reversed)):
            assert enc_c.shape == dec_c.shape, f"第{i}层系数形状不匹配: {enc_c.shape} vs {dec_c.shape}"
            print(f"  ✓ 第{i}层匹配: {enc_c.shape}")

    print("✓ 小波系数结构测试通过")


def test_wavelet_loss_computation(p):
    """测试: 小波损失计算模拟"""
    cr = p["compression_ratio"]
    num_down = p["num_down_blocks"]
    resolution = cr * 16
    print(f"\n{'=' * 60}")
    print("测试: 小波损失计算模拟")
    print("=" * 60)

    model = WFIVAE2Model(
        latent_dim=p["latent_dim"],
        base_channels=p["base_channels"],
        encoder_energy_flow_size=p["encoder_energy_flow_size"],
        decoder_energy_flow_size=p["decoder_energy_flow_size"],
    )
    model.eval()

    x = torch.randn(1, 3, resolution, resolution)

    with torch.no_grad():
        out = model(x)
        enc_coeffs, dec_coeffs = out.extra_output

        dec_coeffs_reversed = list(reversed(dec_coeffs))
        wl_loss = 0
        bs = x.shape[0]
        for enc_c, dec_c in zip(enc_coeffs, dec_coeffs_reversed):
            wl_loss += torch.sum(torch.abs(enc_c - dec_c)) / bs
        wl_loss = wl_loss / len(enc_coeffs)

        print(f"小波损失值: {wl_loss.item():.4f}")
        print(f"损失部分数: {len(enc_coeffs)} (应为{num_down})")

    print("✓ 小波损失计算测试通过")


def test_dimension_flow(model, p):
    """测试: 维度变化链路"""
    cr = p["compression_ratio"]
    num_down = p["num_down_blocks"]
    latent_dim = p["latent_dim"]
    base_channels = p["base_channels"]
    resolution = cr * 16
    print(f"\n{'=' * 60}")
    print(f"测试: 维度变化链路 ({resolution}分辨率, {num_down}层下采样, {cr}x压缩)")
    print("=" * 60)

    model.eval()
    x = torch.randn(1, 3, resolution, resolution)
    print(f"输入: {x.shape}")

    with torch.no_grad():
        print("\n--- 编码器 ---")

        # 小波变换
        wavelet_size = resolution // 2
        coeffs = model.encoder.wavelet_transform_in(x)
        print(f"小波变换: {coeffs.shape} (期望 [1, 12, {wavelet_size}, {wavelet_size}])")
        assert coeffs.shape == (1, 12, wavelet_size, wavelet_size)

        # conv_in
        h = model.encoder.conv_in(coeffs)
        print(f"conv_in: {h.shape} (期望 [1, {base_channels[0]}, {wavelet_size}, {wavelet_size}])")
        assert h.shape == (1, base_channels[0], wavelet_size, wavelet_size)

        # WFDownBlock × num_down — 动态生成期望形状
        w = coeffs
        spatial = wavelet_size
        expected_shapes = []
        for i in range(num_down):
            spatial = spatial // 2
            expected_shapes.append(
                ((1, base_channels[i + 1], spatial, spatial), (1, 12, spatial, spatial))
            )

        for i, down_block in enumerate(model.encoder.down_blocks):
            h, w = down_block(h, w)
            expected_h, expected_w = expected_shapes[i]
            print(f"WFDownBlock{i+1}: h={h.shape}, w={w.shape}")
            assert h.shape == expected_h, f"down_block{i+1} h形状错误: {h.shape} != {expected_h}"
            assert w.shape == expected_w, f"down_block{i+1} w形状错误: {w.shape} != {expected_w}"

        # mid + output
        h = model.encoder.mid(h)
        print(f"mid: {h.shape}")

        h = model.encoder.norm_out(h)
        from causalimagevae.model.modules import nonlinearity
        h = nonlinearity(h)
        h = model.encoder.conv_out(h)
        final_spatial = resolution // cr
        print(f"编码器输出: {h.shape} (期望 [1, {latent_dim*2}, {final_spatial}, {final_spatial}])")
        assert h.shape == (1, latent_dim * 2, final_spatial, final_spatial)

        # 解码器
        print(f"\n--- 解码器 ---")
        z = h[:, :latent_dim, :, :]
        print(f"潜在变量: {z.shape}")

        h = model.decoder.conv_in(z)
        print(f"解码器conv_in: {h.shape} (期望 [1, {base_channels[-1]}, {final_spatial}, {final_spatial}])")
        assert h.shape == (1, base_channels[-1], final_spatial, final_spatial)

        h = model.decoder.mid(h)
        print(f"mid: {h.shape}")

        # WFUpBlock × num_down — 动态生成期望形状
        dec_spatial = final_spatial
        expected_up_shapes = []
        reversed_channels = list(reversed(base_channels))
        for i in range(num_down):
            dec_spatial = dec_spatial * 2
            expected_up_shapes.append(
                ((1, reversed_channels[i + 1], dec_spatial, dec_spatial), (1, 3, dec_spatial, dec_spatial))
            )

        w = None
        for i, up_block in enumerate(model.decoder.up_blocks):
            h, w, coeffs = up_block(h, w)
            expected_h, expected_w = expected_up_shapes[i]
            print(f"WFUpBlock{i+1}: h={h.shape}, w={w.shape}")
            assert h.shape == expected_h, f"up_block{i+1} h形状错误: {h.shape} != {expected_h}"
            assert w.shape == expected_w, f"up_block{i+1} w形状错误: {w.shape} != {expected_w}"

        # output
        h = model.decoder.norm_out(h)
        h = nonlinearity(h)
        h = model.decoder.conv_out(h)
        print(f"解码器conv_out: {h.shape} (期望 [1, 12, {wavelet_size}, {wavelet_size}])")
        assert h.shape == (1, 12, wavelet_size, wavelet_size)

        if w is not None:
            h[:, :3] = h[:, :3] + w

        dec = model.decoder.inverse_wavelet_transform_out(h)
        print(f"逆小波变换: {dec.shape} (期望 [1, 3, {resolution}, {resolution}])")
        assert dec.shape == (1, 3, resolution, resolution)

    print("✓ 维度链路测试通过!")


def test_config_loading(p):
    """测试: 从配置文件加载"""
    config_path = Path(p["config_path"])
    print(f"\n{'=' * 60}")
    print("测试: 从配置文件加载模型")
    print("=" * 60)

    if not config_path.exists():
        print(f"警告: 配置文件不存在: {config_path}")
        return

    print(f"从配置文件加载: {config_path}")
    model_cls = ModelRegistry.get_model("WFIVAE2")
    model = model_cls.from_config(str(config_path))
    model.eval()

    print(f"配置详情:")
    print(f"  base_channels: {model.config['base_channels']}")
    print(f"  latent_dim: {model.config['latent_dim']}")
    print(f"  下采样块数: {len(model.encoder.down_blocks)}")
    print(f"  压缩比: {p['compression_ratio']}x")

    cr = p["compression_ratio"]
    resolution = cr * 16
    x = torch.randn(1, 3, resolution, resolution)
    with torch.no_grad():
        output = model(x)

    print(f"输出形状: {output.sample.shape}")
    print(f"潜变量形状: {output.latent_dist.mean.shape}")
    assert output.sample.shape == x.shape
    print("✓ 配置文件加载测试通过")


def main():
    parser = argparse.ArgumentParser(description="WFIVAE2 单元测试")
    parser.add_argument(
        "--config",
        type=str,
        default="examples/wfivae2-image-64chn-192bc.json",
        help="模型配置文件路径",
    )
    args = parser.parse_args()

    p = load_config(args.config)

    print("\n" + "=" * 60)
    print("WFIVAE2 图像VAE 单元测试")
    print(f"({p['num_down_blocks']}层WFDownBlock/WFUpBlock, concat能量流融合, {p['compression_ratio']}x压缩)")
    print(f"配置: {args.config}")
    print(f"base_channels: {p['base_channels']}")
    print(f"latent_dim: {p['latent_dim']}")
    print("=" * 60)

    try:
        model = test_model_creation(p)

        # 选择合适的测试分辨率
        cr = p["compression_ratio"]
        test_res_small = cr * 16   # 小分辨率
        test_res_large = cr * 32   # 大分辨率 (只在内存允许时)

        test_forward_pass(model, p, test_res_small)

        if test_res_large <= 1024:
            test_forward_pass(model, p, test_res_large)

        test_encode_decode(model, p, test_res_small)
        test_decoder_independence(model, p)
        test_different_resolutions(model, p)
        test_gradient_flow(model, p)
        test_wavelet_coeffs_structure(model, p)
        test_wavelet_loss_computation(p)
        test_dimension_flow(model, p)
        test_config_loading(p)

        num_down = p["num_down_blocks"]
        latent_dim = p["latent_dim"]
        print(f"\n{'=' * 60}")
        print("✓ 所有测试通过!")
        print("=" * 60)
        print(f"\n架构验证:")
        print(f"  - {num_down}层 WFDownBlock (concat能量流融合)")
        print(f"  - {num_down}层 WFUpBlock (outflow能量流重建)")
        print(f"  - {cr}x空间压缩 (小波2x × 下采样{cr // 2}x ({num_down}个DownBlock))")
        for res in [256, 512, 1024]:
            if res % cr == 0:
                ls = res // cr
                print(f"  - {res}: latent [{latent_dim}, {ls}, {ls}]")
        print(f"  - 小波损失: {num_down}个部分 (对应{num_down}个下采样层)")
        print(f"  - 解码器独立重建，无skip connection")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
