#!/usr/bin/env python3
"""Draw detailed WF-VAE 32x + DC-AE Shortcut architecture diagram — v7."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})

C = {
    'input':    '#2E7D32', 'wavelet':  '#E65100', 'conv':     '#1565C0',
    'resblock': '#6A1B9A', 'down':     '#C62828', 'up':       '#AD1457',
    'attn':     '#00838F', 'concat':   '#D84315', 'split':    '#4E342E',
    'latent':   '#37474F', 'flow':     '#F57F17', 'mid':      '#283593',
    'norm':     '#558B2F', 'add':      '#BF360C', 'output':   '#2E7D32',
    'bg_enc':   '#F3E5F5', 'bg_dec':   '#E3F2FD', 'bg_block': '#FFF8E1',
    'bg_mid':   '#E8EAF6', 'bg_res':   '#FAFAFA',
    'shortcut': '#00695C',  # DC-AE shortcut color (teal)
}


def box(ax, x, y, w, h, text, color, fs=8, tc='white', alpha=0.92, lw=0.8, zorder=4):
    p = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.06",
                        lw=lw, edgecolor='#424242', facecolor=color, alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=tc,
            fontweight='bold', zorder=zorder+1)


def region(ax, x, y, w, h, label, color, fs=8, alpha=0.22, ls='--', lw=1.2,
           label_pos='tl', label_color='#616161'):
    p = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.15",
                        lw=lw, edgecolor='#9E9E9E', facecolor=color, alpha=alpha,
                        zorder=1, linestyle=ls)
    ax.add_patch(p)
    if label:
        lx = x - w/2 + 0.2 if label_pos == 'tl' else x
        ha = 'left' if label_pos == 'tl' else 'center'
        ax.text(lx, y + h/2 - 0.25, label, ha=ha, va='top', fontsize=fs,
                color=label_color, fontweight='bold', fontstyle='italic', zorder=2)


def arr(ax, x1, y1, x2, y2, color='#424242', lw=1.0, ms=8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, mutation_scale=ms), zorder=3)


def darr(ax, x1, y1, x2, y2, color='#999', lw=0.8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, mutation_scale=7,
                                linestyle='dashed'), zorder=3)


def dim(ax, x, y, text, fs=7.5, color='#555555'):
    ax.text(x, y, text, ha='left', va='center', fontsize=fs, color=color, zorder=5)


def spatial_label(ax, x, y, text, fs=7.5, color='#9E9E9E'):
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=color,
            fontweight='bold', zorder=2,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='#BDBDBD',
                      lw=0.5, alpha=0.8))


def note(ax, x, y, text, fs=7, color='#888'):
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=color,
            fontstyle='italic', zorder=5)


def shortcut_arrow(ax, x, y_top, y_bot, label, color=None):
    """Draw a curved DC-AE shortcut arrow on the left side of a block."""
    if color is None:
        color = C['shortcut']
    ax.annotate('', xy=(x, y_bot), xytext=(x, y_top),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.0,
                                connectionstyle='arc3,rad=-0.35', linestyle='-',
                                mutation_scale=10), zorder=7)
    mid_y = (y_top + y_bot) / 2
    ax.text(x - 1.5, mid_y, label, ha='center', va='center', fontsize=6.5,
            color=color, fontweight='bold', rotation=90, zorder=7,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=color, lw=0.8, alpha=0.9))


# Output directory
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.dirname(OUT_DIR)  # project root

# ═════════════════════════════════════════════════════════════
#  FIGURE 1 — Full Architecture (256px input)
# ═════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(30, 56))
ax.set_xlim(-7, 29)
ax.set_ylim(-88, 4)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('white')

ax.text(11, 3, 'WF-VAE2  32×  +  DC-AE Residual Shortcut', ha='center', va='center',
        fontsize=22, fontweight='bold', color='#212121')
ax.text(11, 2.0,
        '256px input  |  base_channels = [256, 512, 512, 1024, 1024]  |  latent_dim = 64  |  enc_rb = [2,3,9,2]  |  dec_rb = [3,4,12,2]  |  total params = 478.09 M',
        ha='center', va='center', fontsize=10, color='#757575')
ax.text(11, 1.2,
        'DC-AE shortcut: pixel_unshuffle/shuffle + channel avg/dup  (zero parameters)',
        ha='center', va='center', fontsize=10, color=C['shortcut'], fontweight='bold')

MX = 6.0;  FX = 16.0
BW = 4.8;  BH = 0.7;  FBW = 4.2;  FBH = 0.6;  DX = 2.7
SX = -3.5
BLOCK_GAP = 0.8
AW_MAIN = 1.2;  AW_FLOW = 1.0;  AW_SEC = 0.8

y = 0.0
enc_start = y + 0.5

ax.text(SX, enc_start - 0.1, 'Spatial\nResolution', ha='center', va='center',
        fontsize=8, fontweight='bold', color='#9E9E9E')

# ── Input (256px) ──
y -= 1.8
box(ax, MX, y, BW, BH, 'Input RGB Image', C['input'], fs=10)
dim(ax, MX+DX, y, '[3, 256, 256]')
spatial_label(ax, SX, y, '256×256', fs=8)

# ── HaarWavelet ──
y -= 1.4
arr(ax, MX, y+1.1, MX, y+0.45, lw=AW_MAIN)
box(ax, MX, y, BW, BH, 'HaarWaveletTransform2D', C['wavelet'], fs=9)
dim(ax, MX+DX, y, '[12, 128, 128]')
spatial_label(ax, SX, y, '128×128', fs=8, color=C['wavelet'])
box(ax, FX, y, FBW, FBH, 'coeffs₀', C['flow'], fs=8.5, tc='#333')
arr(ax, MX+BW/2+0.1, y, FX-FBW/2-0.1, y, color=C['flow'], lw=AW_FLOW)
dim(ax, FX+FBW/2+0.2, y, '[12, 128, 128]')
note(ax, (MX+FX)/2, y+0.5, 'non-parametric  2× compression', fs=7, color=C['wavelet'])

# ── conv_in ──
y -= 1.4
arr(ax, MX, y+1.1, MX, y+0.45, lw=AW_MAIN)
box(ax, MX, y, BW, BH, 'Conv3×3  (12 → 256)', C['conv'], fs=9)
dim(ax, MX+DX, y, '[256, 128, 128]')

# ── 4 × WFDownBlock (with DC-AE shortcut) ──
# (in_ch, out_ch, energy_flow_size, sp_in, sp_out, num_rb)
blocks = [
    (256,  512,   128, '128×128', '64×64', 2),
    (512,  512,   128, '64×64',   '32×32', 3),
    (512,  1024,  128, '32×32',   '16×16', 9),
    (1024, 1024,  128, '16×16',   '8×8',   2),
]

enc_coeffs_ys = []
SC_X = MX - BW/2 - 0.3  # shortcut arrow x position

for i, (ic, oc, ef, sp_in, sp_out, nrb) in enumerate(blocks):
    ybt = y - 0.7
    block_top_y = y - 0.5  # for shortcut arrow start

    y -= 1.6
    arr(ax, MX, y+1.3, MX, y+0.45, lw=AW_MAIN)
    box(ax, MX, y, BW, BH, f'ResnetBlock2D ({ic}) ×{nrb-1}', C['resblock'], fs=8)
    dim(ax, MX+DX, y, f'[{ic}, {sp_in}]')

    y -= 1.3
    arr(ax, MX, y+1.0, MX, y+0.45, lw=AW_MAIN)
    box(ax, MX, y, BW, BH, 'Downsample (stride=2)', C['down'], fs=8)
    dim(ax, MX+DX, y, f'[{ic}, {sp_out}]')
    spatial_label(ax, SX, y, sp_out, fs=7.5)

    # Flow
    fy = y + 0.6
    arr(ax, FX, fy+2.0, FX, fy+0.4, color=C['flow'], lw=AW_FLOW)
    box(ax, FX, fy, FBW, FBH, 'HaarWavelet2D (RGB)', C['wavelet'], fs=7.5)
    dim(ax, FX+FBW/2+0.2, fy, f'[12, {sp_out}]')

    fy -= 0.9
    arr(ax, FX, fy+0.65, FX, fy+0.4, color=C['flow'], lw=AW_FLOW)
    box(ax, FX, fy, FBW, FBH, f'in_flow_conv (12→{ef})', C['conv'], fs=7.5)
    dim(ax, FX+FBW/2+0.2, fy, f'[{ef}, {sp_out}]')

    # Concat
    y -= 1.4
    arr(ax, MX, y+1.1, MX+0.5, y+0.45, lw=AW_MAIN)
    cx = (MX + FX) / 2 - 0.5
    cw = FX - MX + 1.5
    box(ax, cx, y, cw, BH, f'Concat  dim=1   [{ic} + {ef} = {ic+ef}]', C['concat'], fs=8)
    arr(ax, FX, fy-0.35, cx+cw/2-0.5, y+0.4, color=C['flow'], lw=AW_FLOW)
    dim(ax, cx+cw/2+0.3, y, f'[{ic+ef}, {sp_out}]')

    y -= 1.3
    arr(ax, cx, y+1.0, MX, y+0.45, lw=AW_MAIN)
    box(ax, MX, y, BW, BH, f'ResnetBlock2D ({ic+ef}→{oc})', C['resblock'], fs=8)
    # Add ⊕ symbol to indicate shortcut addition
    box(ax, MX-BW/2-0.5, y, 0.5, 0.5, '⊕', C['shortcut'], fs=11, tc='white', alpha=0.95)
    dim(ax, MX+DX, y, f'[{oc}, {sp_out}]')

    box(ax, FX, y, FBW, FBH, f'enc_coeffs[{i}]', C['flow'], fs=7.5, tc='#333')
    arr(ax, FX, fy-0.35-0.5, FX, y+0.35, color=C['flow'], lw=AW_SEC)
    enc_coeffs_ys.append(y)

    # DC-AE shortcut arrow
    block_bot_y = y + 0.1
    shortcut_arrow(ax, SC_X, block_top_y, block_bot_y,
                   f'pixel_unshuffle\n+ avg 2 groups\n({ic}→{oc})')

    ybb = y - 0.7
    region(ax, (MX+FX)/2, (ybt+ybb)/2, FX-MX+7, ybt-ybb,
           f'WFDownBlock {i+1}   ({ic} → {oc})   num_rb={nrb}', C['bg_block'], fs=9)

    y -= BLOCK_GAP

# ── Encoder Mid (1024) ──
y -= 1.2
arr(ax, MX, y+1.0+BLOCK_GAP, MX, y+0.45, lw=AW_MAIN)
mt = y+0.55
box(ax, MX, y, BW, BH, 'ResnetBlock2D (1024)', C['resblock'], fs=8.5)
y -= 1.0
arr(ax, MX, y+0.7, MX, y+0.45, lw=AW_MAIN)
box(ax, MX, y, BW, BH, 'Attention2DFix (1024)', C['attn'], fs=8.5)
y -= 1.0
arr(ax, MX, y+0.7, MX, y+0.45, lw=AW_MAIN)
box(ax, MX, y, BW, BH, 'ResnetBlock2D (1024)', C['resblock'], fs=8.5)
mb = y-0.5
region(ax, MX, (mt+mb)/2, BW+2, mt-mb, 'Encoder Mid', C['bg_mid'], fs=9)
spatial_label(ax, SX, (mt+mb)/2, '8×8', fs=7.5, color=C['mid'])

y -= 1.2
arr(ax, MX, y+0.9, MX, y+0.45, lw=AW_MAIN)
box(ax, MX, y, BW, BH, 'LayerNorm → SiLU', C['norm'], fs=8.5, tc='#333')
y -= 1.0
arr(ax, MX, y+0.7, MX, y+0.45, lw=AW_MAIN)
box(ax, MX, y, BW, BH, 'Conv3×3 (1024 → 128)', C['conv'], fs=8.5)
dim(ax, MX+DX, y, '[128, 8, 8]  →  split(μ, σ)')

enc_end = y - 0.7
region(ax, (MX+FX)/2, (enc_start+enc_end)/2, 24, enc_start-enc_end,
       'ENCODER', C['bg_enc'], fs=15, alpha=0.10, ls='-', lw=2.0, label_color='#7B1FA2')

# ═══════ LATENT ═══════
y -= 2.0
arr(ax, MX, y+1.7, MX, y+0.6, lw=AW_MAIN)
latent_y = y
box(ax, 11, y, 13, 1.0,
    'Latent :    z = μ + σ · ε              z  shape = [64, 8, 8]',
    C['latent'], fs=11)
arr(ax, MX+BW/2+0.1, y, 11-6.5-0.1, y, color=C['latent'], lw=1.5)
spatial_label(ax, SX, y, '8×8', fs=8, color=C['latent'])

# ══════════════════════
#   DECODER
# ══════════════════════
dec_start = y - 0.8

no_skip_y = (latent_y + dec_start) / 2 - 1.5
ax.text(11, no_skip_y,
        '⚠  Decoder is fully independent — NO skip connections from Encoder.\n'
        'All information must pass through the latent bottleneck.',
        ha='center', va='center', fontsize=9, color='#C62828', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE', edgecolor='#C62828',
                  lw=1.2, alpha=0.9), zorder=6)

y -= 2.5
arr(ax, 11, y+2.0, MX, y+0.45, lw=AW_MAIN)
box(ax, MX, y, BW, BH, 'Conv3×3 (64 → 1024)', C['conv'], fs=9)
dim(ax, MX+DX, y, '[1024, 8, 8]')

# Decoder Mid (1024)
y -= 1.4
arr(ax, MX, y+1.1, MX, y+0.45, lw=AW_MAIN)
mt2 = y+0.55
box(ax, MX, y, BW, BH, 'ResnetBlock2D (1024)', C['resblock'], fs=8.5)
y -= 1.0
arr(ax, MX, y+0.7, MX, y+0.45, lw=AW_MAIN)
box(ax, MX, y, BW, BH, 'Attention2DFix (1024)', C['attn'], fs=8.5)
y -= 1.0
arr(ax, MX, y+0.7, MX, y+0.45, lw=AW_MAIN)
box(ax, MX, y, BW, BH, 'ResnetBlock2D (1024)', C['resblock'], fs=8.5)
mb2 = y-0.5
region(ax, MX, (mt2+mb2)/2, BW+2, mt2-mb2, 'Decoder Mid', C['bg_mid'], fs=9)

# ── 4 × WFUpBlock (with DC-AE shortcut) ──
# (in_ch, out_ch, energy_flow_size, sp_in, sp_out, num_rb)
up_blocks = [
    (1024, 1024, 128, '8×8',     '16×16',   3),
    (1024,  512, 128, '16×16',   '32×32',   4),
    (512,   512, 128, '32×32',   '64×64',   12),
    (512,   256, 128, '64×64',   '128×128', 2),
]

last_w_fy = None
dec_coeffs_ys = []

for i, (ic, oc, ef, sp_in, sp_out, nrb) in enumerate(up_blocks):
    ybt = y - 0.7
    block_top_y = y - 0.5

    y -= 1.6
    arr(ax, MX, y+1.3, MX, y+0.45, lw=AW_MAIN)
    box(ax, MX, y, BW, BH, f'ResnetBlock2D ({ic}→{ic+ef})', C['resblock'], fs=8)
    dim(ax, MX+DX, y, f'[{ic+ef}, {sp_in}]')

    y -= 1.2
    arr(ax, MX, y+0.9, MX, y+0.45, lw=AW_MAIN)
    box(ax, MX, y, BW, BH, f'Split  [{ic} | {ef}]', C['split'], fs=8)

    # Flow
    fy = y
    arr(ax, MX+BW/2+0.1, fy, FX-FBW/2-0.1, fy, color=C['flow'], lw=AW_FLOW)
    box(ax, FX, fy, FBW, FBH, f'energy flow  [{ef}ch]', C['flow'], fs=7.5, tc='#333')

    fy -= 1.0
    arr(ax, FX, fy+0.7, FX, fy+0.4, color=C['flow'], lw=AW_FLOW)
    box(ax, FX, fy, FBW, FBH, 'ResBlock + Conv (128→12)', C['conv'], fs=7.5)
    dim(ax, FX+FBW/2+0.2, fy, f'[12, {sp_in}]')

    dc_x = FX + 5.0
    box(ax, dc_x, fy, 2.5, 0.5, f'dec_coeffs[{i}]', C['flow'], fs=7, tc='#333')
    arr(ax, FX+FBW/2+0.1, fy, dc_x-1.25-0.1, fy, color=C['flow'], lw=AW_SEC)
    dec_coeffs_ys.append(fy)

    fy -= 1.0
    arr(ax, FX, fy+0.7, FX, fy+0.4, color=C['flow'], lw=AW_FLOW)
    if i == 0:
        box(ax, FX, fy, FBW, FBH, 'coeffs  (w=None)', C['flow'], fs=7.5, tc='#333')
    else:
        box(ax, FX, fy, FBW, FBH, 'LL[:3] += w  (prev block)', C['add'], fs=7.5)

    fy -= 1.0
    arr(ax, FX, fy+0.7, FX, fy+0.4, color=C['flow'], lw=AW_FLOW)
    box(ax, FX, fy, FBW, FBH, 'InverseHaarWavelet2D', C['wavelet'], fs=8)
    dim(ax, FX+FBW/2+0.2, fy, f'→ w [3, {sp_out}]', color=C['flow'])
    last_w_fy = fy

    if i < len(up_blocks) - 1:
        arr(ax, FX, fy-0.35, FX, fy-2.0, color=C['flow'], lw=AW_FLOW)
        ax.text(FX+0.4, fy-1.1, 'w ↓', fontsize=7.5, color=C['flow'], fontweight='bold')

    # Main path
    y -= 1.3
    arr(ax, MX, y+1.0, MX, y+0.45, lw=AW_MAIN)
    rb_count = max(nrb - 2, 0)
    rb_label = f'ResnetBlock2D ({ic}) ×{rb_count}' if rb_count > 0 else f'res_block: empty (num-2=0)'
    box(ax, MX, y, BW, BH, rb_label, C['resblock'], fs=8)
    dim(ax, MX+DX, y, f'[{ic}, {sp_in}]')

    y -= 1.3
    arr(ax, MX, y+1.0, MX, y+0.45, lw=AW_MAIN)
    box(ax, MX, y, BW, BH, 'Upsample (interp×2 + Conv)', C['up'], fs=8)
    dim(ax, MX+DX, y, f'[{ic}, {sp_out}]')
    spatial_label(ax, SX, y, sp_out, fs=7.5)

    y -= 1.3
    arr(ax, MX, y+1.0, MX, y+0.45, lw=AW_MAIN)
    box(ax, MX, y, BW, BH, f'ResnetBlock2D ({ic}→{oc})', C['resblock'], fs=8)
    box(ax, MX-BW/2-0.5, y, 0.5, 0.5, '⊕', C['shortcut'], fs=11, tc='white', alpha=0.95)
    dim(ax, MX+DX, y, f'[{oc}, {sp_out}]')

    # DC-AE shortcut arrow
    block_bot_y = y + 0.1
    shortcut_arrow(ax, SC_X, block_top_y, block_bot_y,
                   f'pixel_shuffle\n+ dup 2×\n({ic}→{oc})')

    ybb = y - 0.7
    region(ax, (MX+FX)/2+1, (ybt+ybb)/2, FX-MX+9, ybt-ybb,
           f'WFUpBlock {i+1}   ({ic} → {oc})   num_rb={nrb}', C['bg_block'], fs=9)

    y -= BLOCK_GAP

# ── Decoder output ──
y -= 0.8
arr(ax, MX, y+0.8+BLOCK_GAP, MX, y+0.45, lw=AW_MAIN)
box(ax, MX, y, BW, BH, 'LayerNorm → SiLU', C['norm'], fs=8.5, tc='#333')

y -= 1.0
arr(ax, MX, y+0.7, MX, y+0.45, lw=AW_MAIN)
box(ax, MX, y, BW, BH, 'Conv3×3 (256 → 12)', C['conv'], fs=8.5)
dim(ax, MX+DX, y, '[12, 128, 128]')

y -= 1.0
arr(ax, MX, y+0.7, MX, y+0.45, lw=AW_MAIN)
box(ax, MX, y, BW, BH, 'h[:,:3] += w  (final LL)', C['add'], fs=8)
arr(ax, FX-FBW/2-0.1, last_w_fy, MX+BW/2+0.1, y, color=C['flow'], lw=1.5)
mid_arr_x = (FX-FBW/2 + MX+BW/2) / 2
mid_arr_y = (last_w_fy + y) / 2
ax.text(mid_arr_x, mid_arr_y+0.35, 'w from WFUpBlock 4', ha='center', fontsize=6.5,
        color=C['flow'], fontstyle='italic',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1))

y -= 1.1
arr(ax, MX, y+0.8, MX, y+0.45, lw=AW_MAIN)
box(ax, MX, y, BW, BH, 'InvHaarWaveletTransform2D', C['wavelet'], fs=9)
dim(ax, MX+DX, y, '[3, 256, 256]')
spatial_label(ax, SX, y, '256×256', fs=8, color=C['wavelet'])

y -= 1.4
arr(ax, MX, y+1.1, MX, y+0.45, lw=AW_MAIN)
box(ax, MX, y, BW, BH, 'Output RGB Image', C['output'], fs=10)
dim(ax, MX+DX, y, '[3, 256, 256]')

dec_end = y - 0.7
region(ax, (MX+FX)/2+1, (dec_start+dec_end)/2, 24, dec_start-dec_end,
       'DECODER', C['bg_dec'], fs=15, alpha=0.10, ls='-', lw=2.0, label_color='#1565C0')

# ═══════════════════════
#  Wavelet Loss
# ═══════════════════════
wlx, wly = 24, -17
ax.text(wlx, wly,
        '    Wavelet Loss    \n\n'
        '  L₁( enc_coeffs[i] ,\n'
        '       dec_coeffs[3−i] )\n\n'
        '  i = 0, 1, 2, 3\n'
        '  4 scale levels  ',
        ha='center', va='center', fontsize=9, color='#BF360C',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3E0', edgecolor='#BF360C', lw=1.5),
        zorder=6)

enc_mid_y = np.mean(enc_coeffs_ys)
dec_mid_y = np.mean(dec_coeffs_ys)
darr(ax, FX+FBW/2+0.1, enc_mid_y, wlx-3.0, wly+1.8, color='#E65100', lw=1.0)
ax.text(FX+FBW/2+0.8, enc_mid_y+0.4, 'enc ×4', fontsize=6.5, color='#E65100', fontstyle='italic')
darr(ax, FX+5.0+1.25+0.1, dec_mid_y, wlx+2.0, wly+1.8, color='#BF360C', lw=1.0)
ax.text(FX+5.0+1.25+0.8, dec_mid_y+0.4, 'dec ×4', fontsize=6.5, color='#BF360C', fontstyle='italic')

# ═══════════════════════
#  DC-AE Shortcut Detail Box
# ═══════════════════════
sx_box_x, sx_box_y = 24, -6
ax.text(sx_box_x, sx_box_y,
        '  DC-AE Residual Shortcut  \n'
        '  (non-parametric, 0 params)\n\n'
        '  Downsample:\n'
        '    pixel_unshuffle(2)\n'
        '    → [4C, H/2, W/2]\n'
        '    → reshape + mean(2 groups)\n'
        '    → [2C, H/2, W/2]\n\n'
        '  Upsample:\n'
        '    pixel_shuffle(2)\n'
        '    → [C/4, 2H, 2W]\n'
        '    → channel duplicate 2×\n'
        '    → [C/2, 2H, 2W]\n\n'
        '  output = main_path + shortcut  ',
        ha='center', va='center', fontsize=8, color=C['shortcut'],
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#E0F2F1',
                  edgecolor=C['shortcut'], lw=1.5),
        zorder=6)

# ═══════════════════════
#  Legend
# ═══════════════════════
legend = [
    ('HaarWavelet / InvWavelet',       C['wavelet']),
    ('Conv2d',                         C['conv']),
    ('ResnetBlock2D',                  C['resblock']),
    ('Downsample (stride-2)',          C['down']),
    ('Upsample (interp + conv)',       C['up']),
    ('Attention2DFix',                 C['attn']),
    ('Concat',                         C['concat']),
    ('Add (residual)',                 C['add']),
    ('Wavelet Energy Flow',            C['flow']),
    ('DC-AE Shortcut (non-param)',     C['shortcut']),
    ('Latent Distribution',            C['latent']),
    ('Norm / Activation',              C['norm']),
]
lx = 24
ly = sx_box_y - 7.5
ax.text(lx, ly, 'Legend', ha='center', fontsize=11, fontweight='bold', color='#333')
for k, (label, color) in enumerate(legend):
    yy = ly - 0.7 - k*0.6
    p = FancyBboxPatch((lx-2.0, yy-0.2), 0.7, 0.4, boxstyle="round,pad=0.03",
                        facecolor=color, edgecolor='#424242', lw=0.5, alpha=0.9, zorder=4)
    ax.add_patch(p)
    ax.text(lx-1.2, yy, label, ha='left', va='center', fontsize=8, color='#333', zorder=5)


out_path_1 = os.path.join(OUT_DIR, 'wfvae2_32x_dcae_architecture.png')
plt.savefig(out_path_1, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_path_1}")
plt.close()


# ═══════════════════════════════════════════════════════════════
#  FIGURE 2 — WFDownBlock & WFUpBlock Detail (with DC-AE shortcut)
# ═══════════════════════════════════════════════════════════════

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 20))
fig2.patch.set_facecolor('white')
for a in [ax1, ax2]:
    a.set_aspect('equal')
    a.axis('off')

BW2 = 5.0;  FW2 = 4.8;  BH2 = 0.8

# ══════════════════════
#  WFDownBlock Detail
# ══════════════════════
ax1.set_xlim(-4, 22)
ax1.set_ylim(0, 20)
ax1.set_title('WFDownBlock  (example: 256 → 512, stage 0, num_rb=2)  with DC-AE Shortcut',
              fontsize=14, fontweight='bold', pad=15)

MX1, FX1 = 5.5, 15.0
region(ax1, 9, 10, 22, 18, '', C['bg_block'], alpha=0.12, fs=1)

ax1.text(MX1, 19.5, 'Main Trunk', ha='center', fontsize=10, fontweight='bold', color=C['resblock'])
ax1.text(FX1, 19.5, 'Wavelet Energy Flow', ha='center', fontsize=10, fontweight='bold', color=C['flow'])

y = 18.5
box(ax1, MX1, y, BW2, BH2, 'Input  x', '#757575', fs=10)
dim(ax1, MX1+BW2/2+0.3, y, '[256, 128, 128]', fs=9)
box(ax1, FX1, y, FW2, BH2, 'Input  coeffs', C['flow'], fs=10, tc='#333')
dim(ax1, FX1+FW2/2+0.3, y, '[12, 128, 128]', fs=9)

# DC-AE shortcut path (left side)
sc_x = MX1 - BW2/2 - 1.5
sc_top_y = y - 0.5
ax1.text(sc_x, y, 'DC-AE\nShortcut', ha='center', va='center', fontsize=8,
         color=C['shortcut'], fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#E0F2F1',
                   edgecolor=C['shortcut'], lw=1.0, alpha=0.9), zorder=6)

y -= 2.0
arr(ax1, MX1, y+1.6, MX1, y+0.5, lw=AW_MAIN)
box(ax1, MX1, y, BW2, BH2, 'res_block: ResBlock(256→256) ×(num_rb-1)=1', C['resblock'], fs=8.5)
dim(ax1, MX1+BW2/2+0.3, y, '[256, 128, 128]', fs=8.5)

y -= 2.0
arr(ax1, MX1, y+1.6, MX1, y+0.5, lw=AW_MAIN)
box(ax1, MX1, y, BW2, BH2+0.15, 'conv_down Conv(256→128,s=1)\n+ pixel_unshuffle(2)', C['down'], fs=8.5)
dim(ax1, MX1+BW2/2+0.3, y, '[512, 64, 64]', fs=8.5)

# Flow path
fy = 14.5
arr(ax1, FX1, fy+2.1, FX1, fy+0.5, color=C['flow'], lw=AW_FLOW)
box(ax1, FX1, fy, FW2, BH2, 'HaarWavelet2D( coeffs[:,:3] )', C['wavelet'], fs=8.5)
dim(ax1, FX1+FW2/2+0.3, fy, '[12, 64, 64]', fs=8)
note(ax1, FX1, fy+1.0, 'only RGB (first 3 channels)', fs=7.5, color=C['wavelet'])

fy -= 2.0
arr(ax1, FX1, fy+1.6, FX1, fy+0.5, color=C['flow'], lw=AW_FLOW)
box(ax1, FX1, fy, FW2, BH2, 'in_flow_conv:  Conv3×3 (12 → 128)', C['conv'], fs=9)
dim(ax1, FX1+FW2/2+0.3, fy, '[128, 64, 64]', fs=8)

# Concat
y -= 2.8
cx1 = (MX1 + FX1) / 2
cw1 = 8.0
arr(ax1, MX1, y+2.4, cx1-2.0, y+0.55, lw=AW_MAIN)
arr(ax1, FX1, fy-0.5, cx1+2.0, y+0.55, color=C['flow'], lw=AW_FLOW)
box(ax1, cx1, y, cw1, BH2, 'torch.concat( [x, w],  dim=1 )', C['concat'], fs=10.5)
dim(ax1, cx1+cw1/2+0.3, y, '[512+128 = 640, 64, 64]', fs=8.5)

y -= 2.0
arr(ax1, cx1, y+1.6, cx1, y+0.5, lw=AW_MAIN)
box(ax1, cx1, y, cw1, BH2, 'out_res_block: ResBlock (640 → 512)', C['resblock'], fs=10.5)
dim(ax1, cx1+cw1/2+0.3, y, '[512, 64, 64]', fs=8.5)

# ⊕ addition with shortcut (note: ⊕ already happened inside DCAE stage before concat)
y -= 2.0
arr(ax1, cx1, y+1.6, cx1, y+0.5, lw=AW_MAIN)
box(ax1, cx1, y, cw1, BH2, 'Output h = out_res_block(concat)', C['resblock'], fs=10.5)
dim(ax1, cx1+cw1/2+0.3, y, '[512, 64, 64]', fs=8.5)

# Shortcut path boxes on the left
sc_y1 = 15.0
box(ax1, sc_x, sc_y1, 3.8, 0.7, 'pixel_unshuffle(2)', C['shortcut'], fs=8)
dim(ax1, sc_x-1.9, sc_y1-0.6, '[1024, 64, 64]', fs=7.5, color=C['shortcut'])
arr(ax1, sc_x, sc_top_y-0.5, sc_x, sc_y1+0.45, color=C['shortcut'], lw=1.8)

sc_y2 = sc_y1 - 2.0
box(ax1, sc_x, sc_y2, 3.8, 0.7, 'reshape + mean\n(2 groups → 512)', C['shortcut'], fs=8)
dim(ax1, sc_x-1.9, sc_y2-0.6, '[512, 64, 64]', fs=7.5, color=C['shortcut'])
arr(ax1, sc_x, sc_y1-0.45, sc_x, sc_y2+0.45, color=C['shortcut'], lw=1.8)

# Connect shortcut to ⊕
arr(ax1, sc_x+1.9+0.1, sc_y2-0.5, cx1-cw1/2-0.1, y, color=C['shortcut'], lw=2.0)

# Outputs
y -= 2.0
arr(ax1, cx1-2.0, y+1.6, MX1, y+0.5, lw=AW_MAIN)
arr(ax1, cx1+2.0, y+1.6, FX1, y+0.5, color=C['flow'], lw=AW_FLOW)
box(ax1, MX1, y, BW2, BH2, 'Output  h', '#616161', fs=10)
dim(ax1, MX1+BW2/2+0.3, y, '[512, 64, 64]', fs=9)
box(ax1, FX1, y, FW2, BH2, 'Output  coeffs', C['flow'], fs=10, tc='#333')
dim(ax1, FX1+FW2/2+0.3, y, '[12, 64, 64]', fs=9)

note(ax1, FX1, y-0.65, '↓ passed as input to next WFDownBlock', fs=8, color=C['flow'])


# ══════════════════════
#  WFUpBlock Detail
# ══════════════════════
ax2.set_xlim(-4, 24)
ax2.set_ylim(0, 20)
ax2.set_title('WFUpBlock  (example: 1024 → 512, stage 1, num_rb=4)  with DC-AE Shortcut',
              fontsize=14, fontweight='bold', pad=15)

MX2, FX2 = 6.0, 15.5
region(ax2, 10, 10, 24, 18.5, '', C['bg_block'], alpha=0.12, fs=1)

ax2.text(MX2, 19.5, 'Main Trunk', ha='center', fontsize=10, fontweight='bold', color=C['resblock'])
ax2.text(FX2, 19.5, 'Wavelet Energy Flow', ha='center', fontsize=10, fontweight='bold', color=C['flow'])

ax2.text(10, 0.5,
         '⚠  No skip connections from Encoder — decoder reconstructs energy flow entirely from latent.',
         ha='center', va='center', fontsize=8, color='#C62828', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#C62828',
                   lw=1.0, alpha=0.9), zorder=6)

y = 18.5
box(ax2, MX2, y, BW2, BH2, 'Input  x', '#757575', fs=10)
dim(ax2, MX2+BW2/2+0.3, y, '[1024, 16, 16]', fs=9)
box(ax2, FX2, y, FW2, BH2, 'Input  w  (None or [3,H,W])', C['flow'], fs=8.5, tc='#333')

# DC-AE shortcut path (left side)
sc_x2 = MX2 - BW2/2 - 1.5
sc_top_y2 = y - 0.5
ax2.text(sc_x2, y, 'DC-AE\nShortcut', ha='center', va='center', fontsize=8,
         color=C['shortcut'], fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#E0F2F1',
                   edgecolor=C['shortcut'], lw=1.0, alpha=0.9), zorder=6)

# branch_conv
y -= 1.7
arr(ax2, MX2, y+1.3, MX2, y+0.5, lw=AW_MAIN)
box(ax2, MX2, y, BW2, BH2, 'branch_conv: ResBlock(1024→1152)', C['resblock'], fs=8)
dim(ax2, MX2+BW2/2+0.3, y, '[1152, 16, 16]', fs=8.5)

# Split
y -= 1.5
arr(ax2, MX2, y+1.1, MX2, y+0.5, lw=AW_MAIN)
box(ax2, MX2, y, BW2, BH2, 'Split  [:1024]  |  [-128:]', C['split'], fs=9)

# Flow
fy = y
arr(ax2, MX2+BW2/2+0.1, fy, FX2-FW2/2-0.1, fy, color=C['flow'], lw=1.5)
note(ax2, (MX2+FX2)/2, fy+0.35, 'last 128 channels →', fs=8, color=C['flow'])
box(ax2, FX2, fy, FW2, FBH+0.15, 'energy flow  [128, 16, 16]', C['flow'], fs=8.5, tc='#333')

fy -= 1.2
arr(ax2, FX2, fy+0.9, FX2, fy+0.5, color=C['flow'], lw=AW_FLOW)
box(ax2, FX2, fy, FW2, BH2, 'ResnetBlock2D (128→128)', C['resblock'], fs=8.5)
dim(ax2, FX2+FW2/2+0.3, fy, '[128, 16, 16]', fs=8)

fy -= 1.2
arr(ax2, FX2, fy+0.9, FX2, fy+0.5, color=C['flow'], lw=AW_FLOW)
box(ax2, FX2, fy, FW2, BH2, 'Conv3×3 (128 → 12)', C['conv'], fs=9)
dim(ax2, FX2+FW2/2+0.3, fy, '[12, 16, 16]', fs=8)

dc2_x = FX2 + 5.0
box(ax2, dc2_x, fy, 2.6, 0.55, 'dec_coeffs', C['flow'], fs=8, tc='#333')
arr(ax2, FX2+FW2/2+0.1, fy, dc2_x-1.3-0.1, fy, color=C['flow'], lw=AW_SEC)
note(ax2, dc2_x, fy-0.5, '→ wavelet loss', fs=7.5, color='#BF360C')

fy -= 1.4
arr(ax2, FX2, fy+1.1, FX2, fy+0.5, color=C['flow'], lw=AW_FLOW)
box(ax2, FX2, fy, FW2, BH2, 'coeffs[:,:3] += w   (if w ≠ None)', C['add'], fs=8)
note(ax2, FX2, fy-0.55, 'LL band residual from\nprevious WFUpBlock', fs=7.5, color=C['add'])

fy -= 1.6
arr(ax2, FX2, fy+1.2, FX2, fy+0.5, color=C['flow'], lw=AW_FLOW)
box(ax2, FX2, fy, FW2, BH2, 'InverseHaarWavelet2D', C['wavelet'], fs=9.5)
dim(ax2, FX2+FW2/2+0.3, fy, '→  w  [3, 32, 32]', fs=9, color=C['flow'])

# Main path
y -= 1.8
arr(ax2, MX2, y+1.4, MX2, y+0.5, lw=AW_MAIN)
note(ax2, MX2-BW2/2-0.3, y, 'first\n1024ch', fs=7.5, color=C['resblock'])
box(ax2, MX2, y, BW2, BH2, 'res_block: ResBlock(1024→1024) ×(num_rb-2)=2', C['resblock'], fs=7.5)
dim(ax2, MX2+BW2/2+0.3, y, '[1024, 16, 16]', fs=8.5)

y -= 1.7
arr(ax2, MX2, y+1.3, MX2, y+0.5, lw=AW_MAIN)
box(ax2, MX2, y, BW2, BH2+0.15, 'conv_up Conv(1024→2048,s=1)\n+ pixel_shuffle(2)', C['up'], fs=8.5)
dim(ax2, MX2+BW2/2+0.3, y, '[512, 32, 32]', fs=8.5)

y -= 1.7
arr(ax2, MX2, y+1.3, MX2, y+0.5, lw=AW_MAIN)
box(ax2, MX2, y, BW2, BH2, '⊕  Add  (main + shortcut)', C['shortcut'], fs=9.5)
dim(ax2, MX2+BW2/2+0.3, y, '[512, 32, 32]', fs=8.5)

# ⊕ addition  (this is now the out_res_block step)
y -= 1.7
arr(ax2, MX2, y+1.3, MX2, y+0.5, lw=AW_MAIN)
box(ax2, MX2, y, BW2, BH2, 'out_res_block: ResBlock(512→512)', C['resblock'], fs=9)
dim(ax2, MX2+BW2/2+0.3, y, '[512, 32, 32]', fs=8.5)

# Shortcut path
sc_y1_2 = 15.5
box(ax2, sc_x2, sc_y1_2, 3.8, 0.7, 'repeat_interleave(2)', C['shortcut'], fs=8)
dim(ax2, sc_x2-1.9, sc_y1_2-0.6, '[2048, 16, 16]', fs=7.5, color=C['shortcut'])
arr(ax2, sc_x2, sc_top_y2-0.5, sc_x2, sc_y1_2+0.45, color=C['shortcut'], lw=1.8)

sc_y2_2 = sc_y1_2 - 2.0
box(ax2, sc_x2, sc_y2_2, 3.8, 0.7, 'pixel_shuffle(2)', C['shortcut'], fs=8)
dim(ax2, sc_x2-1.9, sc_y2_2-0.6, '[512, 32, 32]', fs=7.5, color=C['shortcut'])
arr(ax2, sc_x2, sc_y1_2-0.45, sc_x2, sc_y2_2+0.45, color=C['shortcut'], lw=1.8)

# Connect shortcut to ⊕
arr(ax2, sc_x2+1.9+0.1, sc_y2_2-0.5, MX2-BW2/2-0.1, y, color=C['shortcut'], lw=2.0)

# Outputs
y -= 1.7
arr(ax2, MX2, y+1.3, MX2, y+0.5, lw=AW_MAIN)
box(ax2, MX2, y, BW2, BH2, 'Output  h', '#616161', fs=10)
dim(ax2, MX2+BW2/2+0.3, y, '[512, 32, 32]', fs=9)

box(ax2, FX2, y, FW2, BH2, 'Output  w  (→ next block)', C['flow'], fs=8.5, tc='#333')
dim(ax2, FX2+FW2/2+0.3, y, '[3, 32, 32]', fs=9, color=C['flow'])
arr(ax2, FX2, fy-0.5, FX2, y+0.5, color=C['flow'], lw=AW_FLOW)
note(ax2, FX2, y-0.6, '↓ w → next WFUpBlock  (or final LL add)', fs=7.5, color=C['flow'])


plt.tight_layout(pad=2.5)
out_path_2 = os.path.join(OUT_DIR, 'wfvae2_32x_dcae_blocks_detail.png')
plt.savefig(out_path_2, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_path_2}")
plt.close()

print("\nDone! v7 — with DC-AE residual shortcuts")
