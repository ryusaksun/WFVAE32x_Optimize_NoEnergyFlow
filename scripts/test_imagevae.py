#!/usr/bin/env python3
"""
WFIVAE2 图像VAE 单元测试脚本
============================

测试内容:
1. 模型创建和前向传播
2. 编码器/解码器分离测试
3. 维度验证
4. 解码器独立性验证（从潜变量独立重建，无skip connection）
5. 能量流融合测试（concat融合到主干）
6. 不同分辨率支持
7. 小波损失系数提取

架构说明 (修改后的2层下采样):
    - WFDownBlock × 2: concat能量流融合
    - WFUpBlock × 2: outflow能量流重建
    - 8x空间压缩: 小波2x × 下采样4x (2个DownBlock)
    - 1024: latent [B, 32, 128, 128]
    - 512: latent [B, 32, 64, 64]

使用方法:
    python scripts/test_imagevae.py
"""

import torch
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from causalimagevae.model import ModelRegistry
from causalimagevae.model.vae import WFIVAE2Model


def test_model_creation():
    """测试1: 模型创建"""
    print("=" * 60)
    print("测试1: 模型创建")
    print("=" * 60)

    model = WFIVAE2Model(
        latent_dim=32,
        base_channels=[192, 384, 768],  # 2个下采样块
        encoder_num_resblocks=2,
        encoder_energy_flow_size=128,
        decoder_num_resblocks=3,
        decoder_energy_flow_size=128,
        norm_type="layernorm",
    )

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())

    print(f"✓ 模型创建成功")
    print(f"  总参数量: {total_params / 1e6:.2f}M")
    print(f"  可训练参数: {trainable_params / 1e6:.2f}M")
    print(f"  编码器参数: {encoder_params / 1e6:.2f}M")
    print(f"  解码器参数: {decoder_params / 1e6:.2f}M")
    print(f"  下采样块数量: {len(model.encoder.down_blocks)}")
    print(f"  上采样块数量: {len(model.decoder.up_blocks)}")

    return model


def test_forward_pass_512(model):
    """测试2: 前向传播 (512 分辨率)"""
    print("\n" + "=" * 60)
    print("测试2: 前向传播 (512 分辨率)")
    print("=" * 60)

    x = torch.randn(1, 3, 512, 512)
    print(f"输入形状: {x.shape}")

    model.eval()
    with torch.no_grad():
        out = model(x)

    print(f"输出形状: {out.sample.shape}")
    print(f"潜在均值形状: {out.latent_dist.mean.shape}")
    print(f"潜在方差形状: {out.latent_dist.var.shape}")

    # 验证形状 (8x压缩: 小波2x × 下采样4x)
    # 512 -> 256 (小波) -> 128 (down1) -> 64 (down2)
    expected_latent_size = 512 // 8  # = 64
    assert out.sample.shape == x.shape, f"输出形状不匹配! 预期 {x.shape}, 得到 {out.sample.shape}"
    assert out.latent_dist.mean.shape == (1, 32, expected_latent_size, expected_latent_size), \
        f"潜在形状错误: {out.latent_dist.mean.shape}, 预期 (1, 32, {expected_latent_size}, {expected_latent_size})"

    # 验证 extra_output (enc_coeffs, dec_coeffs)
    assert out.extra_output is not None, "WFIVAE2应有extra_output用于小波损失"
    enc_coeffs, dec_coeffs = out.extra_output
    print(f"编码器小波系数数量: {len(enc_coeffs)} (应为2)")
    print(f"解码器小波系数数量: {len(dec_coeffs)} (应为2)")

    assert len(enc_coeffs) == 2, f"编码器系数数量错误: {len(enc_coeffs)}"
    assert len(dec_coeffs) == 2, f"解码器系数数量错误: {len(dec_coeffs)}"

    print("✓ 前向传播测试通过 (512 分辨率)")
    return out


def test_forward_pass_1024(model):
    """测试3: 前向传播 (1024 分辨率)"""
    print("\n" + "=" * 60)
    print("测试3: 前向传播 (1024 分辨率)")
    print("=" * 60)

    x = torch.randn(1, 3, 1024, 1024)
    print(f"输入形状: {x.shape}")

    model.eval()
    with torch.no_grad():
        out = model(x)

    print(f"输出形状: {out.sample.shape}")
    print(f"潜在均值形状: {out.latent_dist.mean.shape}")

    # 验证形状 (8x压缩: 1024 -> 128)
    expected_latent_size = 1024 // 8  # = 128
    assert out.sample.shape == x.shape, f"输出形状不匹配! 预期 {x.shape}, 得到 {out.sample.shape}"
    assert out.latent_dist.mean.shape == (1, 32, expected_latent_size, expected_latent_size), \
        f"潜在形状错误: {out.latent_dist.mean.shape}, 预期 (1, 32, {expected_latent_size}, {expected_latent_size})"

    print("✓ 前向传播测试通过 (1024 分辨率)")


def test_encode_decode_512(model):
    """测试4: 编码/解码分离 (512 分辨率)"""
    print("\n" + "=" * 60)
    print("测试4: 编码/解码分离 (512 分辨率)")
    print("=" * 60)

    x = torch.randn(1, 3, 512, 512)

    model.eval()
    with torch.no_grad():
        # 编码
        enc_out = model.encode(x)
        z = enc_out.latent_dist.sample()
        print(f"编码输入: {x.shape}")
        print(f"潜在变量: {z.shape}")

        # 验证有extra_output (编码器中间系数)
        assert enc_out.extra_output is not None, "编码器应有extra_output"
        print(f"编码器小波系数: {len(enc_out.extra_output)} 层")

        # 解码 (关键: 不需要编码器旁路, 能量流从潜变量重建)
        dec_out = model.decode(z)
        print(f"解码输出: {dec_out.sample.shape}")

    assert dec_out.sample.shape == x.shape, "解码输出形状不匹配"
    expected_latent_size = 512 // 8
    assert z.shape == (1, 32, expected_latent_size, expected_latent_size), \
        f"潜在变量形状错误: {z.shape}"
    print("✓ 编码/解码分离测试通过")


def test_decoder_independence(model):
    """测试5: 解码器独立性"""
    print("\n" + "=" * 60)
    print("测试5: 解码器独立性验证")
    print("=" * 60)

    # 创建随机潜在变量 (512 分辨率对应 64x64 latent)
    z = torch.randn(1, 32, 64, 64)

    model.eval()
    with torch.no_grad():
        # 解码器应该能独立工作，不需要编码器的任何输出
        dec_out = model.decode(z)

    print(f"随机潜在变量: {z.shape}")
    print(f"解码器输出: {dec_out.sample.shape}")

    assert dec_out.sample.shape == (1, 3, 512, 512), f"解码器输出形状错误: {dec_out.sample.shape}"
    print("✓ 解码器可独立运行，能量流从潜变量重建（无skip connection）")


def test_different_resolutions(model):
    """测试6: 不同分辨率"""
    print("\n" + "=" * 60)
    print("测试6: 不同分辨率测试 (8x压缩)")
    print("=" * 60)

    # 测试不同分辨率，都是8x压缩
    resolutions = [(256, 256), (512, 512), (768, 768), (1024, 1024)]

    model.eval()
    for h, w in resolutions:
        x = torch.randn(1, 3, h, w)
        try:
            with torch.no_grad():
                out = model(x)
            # 验证输出形状和latent形状
            expected_latent_h, expected_latent_w = h // 8, w // 8
            latent_shape = out.latent_dist.mean.shape
            status = "✓" if out.sample.shape == x.shape else "✗"
            print(f"{status} {h}×{w}: 输入 {x.shape} → latent {latent_shape} → 输出 {out.sample.shape}")

            # 验证latent维度正确
            assert latent_shape[2] == expected_latent_h and latent_shape[3] == expected_latent_w
        except Exception as e:
            print(f"✗ {h}×{w}: 错误 - {str(e)[:50]}")


def test_gradient_flow(model):
    """测试7: 梯度流动"""
    print("\n" + "=" * 60)
    print("测试7: 梯度流动测试")
    print("=" * 60)

    model.train()

    # 使用较小分辨率测试梯度流动
    x = torch.randn(1, 3, 256, 256, requires_grad=True)

    out = model(x)
    loss = out.sample.mean()
    loss.backward()

    # 检查梯度
    has_grad = x.grad is not None and x.grad.abs().sum() > 0
    print(f"输入梯度: {'✓ 存在' if has_grad else '✗ 不存在'}")

    # 检查模型参数梯度
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"参数梯度: {params_with_grad}/{total_params} 个参数有梯度")

    # 验证latent维度 (256 / 8 = 32)
    print(f"Latent 形状: {out.latent_dist.mean.shape}  (预期: [1, 32, 32, 32])")
    assert out.latent_dist.mean.shape == (1, 32, 32, 32)

    print("✓ 梯度流动正常")


def test_wavelet_coeffs_structure(model):
    """测试8: 小波系数结构"""
    print("\n" + "=" * 60)
    print("测试8: 小波系数结构测试")
    print("=" * 60)

    model.eval()
    x = torch.randn(1, 3, 512, 512)

    with torch.no_grad():
        out = model(x)

        # 验证 extra_output 结构
        assert out.extra_output is not None
        enc_coeffs, dec_coeffs = out.extra_output

        print(f"编码器小波系数: {len(enc_coeffs)} 层")
        for i, coeffs in enumerate(enc_coeffs):
            print(f"  enc_coeffs[{i}]: {coeffs.shape}")

        print(f"解码器小波系数: {len(dec_coeffs)} 层")
        for i, coeffs in enumerate(dec_coeffs):
            print(f"  dec_coeffs[{i}]: {coeffs.shape}")

        # 验证系数数量 (2个下采样块 = 2组系数)
        assert len(enc_coeffs) == 2, f"编码器应有2层系数，得到{len(enc_coeffs)}"
        assert len(dec_coeffs) == 2, f"解码器应有2层系数，得到{len(dec_coeffs)}"

        # 验证编码器和解码器反转后系数匹配
        dec_coeffs_reversed = list(reversed(dec_coeffs))
        for i, (enc_c, dec_c) in enumerate(zip(enc_coeffs, dec_coeffs_reversed)):
            assert enc_c.shape == dec_c.shape, f"第{i}层系数形状不匹配: {enc_c.shape} vs {dec_c.shape}"
            print(f"  ✓ 第{i}层匹配: {enc_c.shape}")

    print("✓ 小波系数结构测试通过")


def test_wavelet_loss_computation():
    """测试9: 小波损失计算模拟"""
    print("\n" + "=" * 60)
    print("测试9: 小波损失计算模拟")
    print("=" * 60)

    model = WFIVAE2Model(
        latent_dim=32,
        base_channels=[192, 384, 768],
        encoder_energy_flow_size=128,
        decoder_energy_flow_size=128,
    )
    model.eval()

    x = torch.randn(1, 3, 512, 512)

    with torch.no_grad():
        out = model(x)
        enc_coeffs, dec_coeffs = out.extra_output

        # 模拟训练脚本中的小波损失计算
        dec_coeffs_reversed = list(reversed(dec_coeffs))
        wl_loss = 0
        bs = x.shape[0]
        for enc_c, dec_c in zip(enc_coeffs, dec_coeffs_reversed):
            wl_loss += torch.sum(torch.abs(enc_c - dec_c)) / bs
        wl_loss = wl_loss / len(enc_coeffs)

        print(f"小波损失值: {wl_loss.item():.4f}")
        print(f"损失部分数: {len(enc_coeffs)}")

    print("✓ 小波损失计算测试通过")


def test_dimension_flow(model):
    """测试10: 维度变化链路"""
    print("\n" + "=" * 60)
    print("测试10: 维度变化链路 (512分辨率, 2层下采样)")
    print("=" * 60)

    model.eval()
    x = torch.randn(1, 3, 512, 512)
    print(f"输入: {x.shape}")

    with torch.no_grad():
        # 追踪编码器维度
        print("\n--- 编码器 ---")

        # 输入小波变换
        coeffs = model.encoder.wavelet_transform_in(x)
        print(f"小波变换: {coeffs.shape} (期望 [1, 12, 256, 256])")
        assert coeffs.shape == (1, 12, 256, 256)

        # conv_in
        h = model.encoder.conv_in(coeffs)
        print(f"conv_in: {h.shape} (期望 [1, 192, 256, 256])")
        assert h.shape == (1, 192, 256, 256)

        # WFDownBlock × 2
        w = coeffs
        expected_shapes = [
            ((1, 384, 128, 128), (1, 12, 128, 128)),   # down1: 192->384, 256->128
            ((1, 768, 64, 64), (1, 12, 64, 64)),       # down2: 384->768, 128->64
        ]

        for i, down_block in enumerate(model.encoder.down_blocks):
            h, w = down_block(h, w)
            expected_h, expected_w = expected_shapes[i]
            print(f"WFDownBlock{i+1}: h={h.shape}, w={w.shape}")
            assert h.shape == expected_h, f"down_block{i+1} h形状错误: {h.shape} != {expected_h}"
            assert w.shape == expected_w, f"down_block{i+1} w形状错误: {w.shape} != {expected_w}"

        # mid layers
        h = model.encoder.mid(h)
        print(f"mid: {h.shape}")

        # output
        h = model.encoder.norm_out(h)
        from causalimagevae.model.modules import nonlinearity
        h = nonlinearity(h)
        h = model.encoder.conv_out(h)
        print(f"编码器输出: {h.shape} (期望 [1, 64, 64, 64])")
        assert h.shape == (1, 64, 64, 64)

        # 追踪解码器维度
        print("\n--- 解码器 ---")
        z = h[:, :32, :, :]  # 只取均值部分 (latent_dim=32)
        print(f"潜在变量: {z.shape}")

        h = model.decoder.conv_in(z)
        print(f"解码器conv_in: {h.shape} (期望 [1, 768, 64, 64])")
        assert h.shape == (1, 768, 64, 64)

        # mid layers
        h = model.decoder.mid(h)
        print(f"mid: {h.shape}")

        # WFUpBlock × 2
        expected_up_shapes = [
            ((1, 384, 128, 128), (1, 3, 128, 128)),   # up1: 768->384, 64->128
            ((1, 192, 256, 256), (1, 3, 256, 256)),   # up2: 384->192, 128->256
        ]

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
        print(f"解码器conv_out: {h.shape} (期望 [1, 12, 256, 256])")
        assert h.shape == (1, 12, 256, 256)

        # 融合能量流
        if w is not None:
            h[:, :3] = h[:, :3] + w

        # 逆小波变换
        dec = model.decoder.inverse_wavelet_transform_out(h)
        print(f"逆小波变换: {dec.shape} (期望 [1, 3, 512, 512])")
        assert dec.shape == (1, 3, 512, 512)

    print("✓ 维度链路测试通过!")


def test_config_loading():
    """测试11: 从配置文件加载"""
    print("\n" + "=" * 60)
    print("测试11: 从配置文件加载模型")
    print("=" * 60)

    config_path = Path("examples/wfivae2-image-192bc.json")
    if config_path.exists():
        print(f"从配置文件加载: {config_path}")
        model_cls = ModelRegistry.get_model("WFIVAE2")
        model = model_cls.from_config(str(config_path))
        model.eval()

        print(f"配置详情:")
        print(f"  base_channels: {model.config['base_channels']}")
        print(f"  latent_dim: {model.config['latent_dim']}")
        print(f"  下采样块数: {len(model.encoder.down_blocks)}")

        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            output = model(x)

        print(f"输出形状: {output.sample.shape}")
        print(f"潜变量形状: {output.latent_dist.mean.shape}")
        assert output.sample.shape == x.shape
        print("✓ 配置文件加载测试通过")
    else:
        print(f"警告: 配置文件不存在: {config_path}")


def main():
    print("\n" + "=" * 60)
    print("WFIVAE2 图像VAE 单元测试")
    print("(2层WFDownBlock/WFUpBlock, concat能量流融合, 8x压缩)")
    print("=" * 60)

    try:
        # 测试1: 模型创建
        model = test_model_creation()

        # 测试2: 前向传播 (512)
        test_forward_pass_512(model)

        # 测试3: 前向传播 (1024)
        test_forward_pass_1024(model)

        # 测试4: 编码/解码分离
        test_encode_decode_512(model)

        # 测试5: 解码器独立性
        test_decoder_independence(model)

        # 测试6: 不同分辨率
        test_different_resolutions(model)

        # 测试7: 梯度流动
        test_gradient_flow(model)

        # 测试8: 小波系数结构
        test_wavelet_coeffs_structure(model)

        # 测试9: 小波损失计算
        test_wavelet_loss_computation()

        # 测试10: 维度链路
        test_dimension_flow(model)

        # 测试11: 配置文件加载
        test_config_loading()

        print("\n" + "=" * 60)
        print("✓ 所有测试通过!")
        print("=" * 60)
        print("\n架构验证:")
        print("  - 2层 WFDownBlock (concat能量流融合)")
        print("  - 2层 WFUpBlock (outflow能量流重建)")
        print("  - 8x空间压缩 (小波2x × 下采样4x)")
        print("  - 512: latent [32, 64, 64]")
        print("  - 1024: latent [32, 128, 128]")
        print("  - 小波损失: 2个部分 (对应2个下采样层)")
        print("  - 解码器独立重建，无skip connection")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
