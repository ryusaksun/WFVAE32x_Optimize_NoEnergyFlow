#!/usr/bin/env python3
"""Draw WF-VAE 32x  —  DCAE + no energy flow  —  architecture diagram.

This branch ships exactly one config (`examples/wfivae2-image-32x-192bc.json`,
`use_energy_flow=false`, `block_type="dcae"`). The figure reflects that:

  - W^(1) Haar WT / IWT at encoder entry / decoder exit is retained (drives 2×
    spatial compression). No mid-layer wavelet inflow/outflow pathway.
  - Main trunk only: no right-column `energy flow` pipeline, no `concat [x,w]`,
    no `out_flow_conv`, no per-stage `enc_coeffs` / `dec_coeffs`, no WL loss.
  - `WFDownBlock.out_res_block` in_channels = `out_ch` (not `out_ch+ef`).
  - `WFUpBlock.branch_conv` keeps channels constant (in→in, not in→in+ef).

The `block_type="classic"` code path in `modeling_wfvae2.py` is untouched but
no JSON ships for it; no classic figure is drawn here either.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})

C = {
    'input':    '#2E7D32', 'wavelet':  '#E65100', 'conv':     '#1565C0',
    'resblock': '#6A1B9A', 'down':     '#C62828', 'up':       '#AD1457',
    'attn':     '#00838F', 'latent':   '#37474F', 'mid':      '#283593',
    'norm':     '#558B2F', 'output':   '#2E7D32',
    'bg_enc':   '#F3E5F5', 'bg_dec':   '#E3F2FD', 'bg_block': '#FFF8E1',
    'bg_mid':   '#E8EAF6',
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
    """Curved DC-AE shortcut arrow on the left side of a block."""
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


OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.dirname(OUT_DIR)  # project root

# ═════════════════════════════════════════════════════════════
#  FIGURE 1 — Full Architecture (256px input, DCAE + no-ef)
# ═════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(22, 50))
ax.set_xlim(-7, 22)
ax.set_ylim(-78, 4)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('white')

ax.text(7, 3, 'WF-VAE2  32×  —  DCAE  +  no energy flow', ha='center', va='center',
        fontsize=22, fontweight='bold', color='#212121')
ax.text(7, 2.0,
        '256px input  |  base_channels = [256, 512, 512, 1024, 1024]  |  latent_dim = 64  |  enc_rb = [4,4,4,2]  |  dec_rb = [5,5,5,3]  |  total params = 467.80 M (enc 145.55 / dec 322.25)',
        ha='center', va='center', fontsize=9.5, color='#757575')
ax.text(7, 1.2,
        'DCAE block-level shortcut + I/O bottleneck shortcut (HunyuanVAE) + ECA  |  use_energy_flow=false  →  no mid-layer wavelet inflow/outflow, no WL loss',
        ha='center', va='center', fontsize=9.5, color=C['shortcut'], fontweight='bold')

MX = 6.0
BW = 4.8;  BH = 0.7;  DX = 2.7
SX = -3.5
BLOCK_GAP = 0.8
AW_MAIN = 1.2

y = 0.0
enc_start = y + 0.5

ax.text(SX, enc_start - 0.1, 'Spatial\nResolution', ha='center', va='center',
        fontsize=8, fontweight='bold', color='#9E9E9E')

# ── Input (256px) ──
y -= 1.8
box(ax, MX, y, BW, BH, 'Input RGB Image', C['input'], fs=10)
dim(ax, MX+DX, y, '[3, 256, 256]')
spatial_label(ax, SX, y, '256×256', fs=8)

# ── HaarWavelet (W^(1), input-side, always kept even when ef=false) ──
y -= 1.4
arr(ax, MX, y+1.1, MX, y+0.45, lw=AW_MAIN)
box(ax, MX, y, BW, BH, 'HaarWaveletTransform2D', C['wavelet'], fs=9)
dim(ax, MX+DX, y, '[12, 128, 128]')
spatial_label(ax, SX, y, '128×128', fs=8, color=C['wavelet'])
note(ax, MX, y-0.55, 'W^(1): non-parametric 2× spatial compression (retained even with energy flow OFF)',
     fs=7, color=C['wavelet'])

# ── conv_in ──
y -= 1.6
arr(ax, MX, y+1.3, MX, y+0.45, lw=AW_MAIN)
box(ax, MX, y, BW, BH, 'Conv3×3  (12 → 256)', C['conv'], fs=9)
dim(ax, MX+DX, y, '[256, 128, 128]')

# ── 4 × WFDownBlock (DCAE, no energy flow) ──
# (in_ch, out_ch, sp_in, sp_out, num_rb)
blocks = [
    (256,  512,  '128×128', '64×64', 4),
    (512,  512,  '64×64',   '32×32', 4),
    (512,  1024, '32×32',   '16×16', 4),
    (1024, 1024, '16×16',   '8×8',   2),
]
SC_X = MX - BW/2 - 0.3  # shortcut arrow x position

for i, (ic, oc, sp_in, sp_out, nrb) in enumerate(blocks):
    ybt = y - 0.7
    block_top_y = y - 0.5

    y -= 1.6
    arr(ax, MX, y+1.3, MX, y+0.45, lw=AW_MAIN)
    box(ax, MX, y, BW, BH, f'ResnetBlock2D ({ic}) ×{nrb-1}', C['resblock'], fs=8)
    dim(ax, MX+DX, y, f'[{ic}, {sp_in}]')

    y -= 1.3
    arr(ax, MX, y+1.0, MX, y+0.45, lw=AW_MAIN)
    box(ax, MX, y, BW, BH, f'conv_down Conv({ic}→{oc//4},s=1) + pixel_unshuffle(2)', C['down'], fs=7.5)
    dim(ax, MX+DX, y, f'[{oc}, {sp_out}]')
    spatial_label(ax, SX, y, sp_out, fs=7.5)

    y -= 1.3
    arr(ax, MX, y+1.0, MX, y+0.45, lw=AW_MAIN)
    box(ax, MX, y, BW, BH, f'out_res_block ({oc}→{oc}) +ECA', C['resblock'], fs=8)
    box(ax, MX-BW/2-0.5, y, 0.5, 0.5, '⊕', C['shortcut'], fs=11, tc='white', alpha=0.95)
    dim(ax, MX+DX, y, f'[{oc}, {sp_out}]')

    # DC-AE shortcut arrow
    block_bot_y = y + 0.1
    shortcut_arrow(ax, SC_X, block_top_y, block_bot_y,
                   f'pixel_unshuffle\n+ avg 2 groups\n({ic}→{oc})')

    ybb = y - 0.7
    region(ax, MX+1, (ybt+ybb)/2, BW+5, ybt-ybb,
           f'WFDownBlock {i+1}   ({ic} → {oc})   num_rb={nrb}   (no energy flow)',
           C['bg_block'], fs=9)

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
mb_em2_y = y  # last mid resblock output y (for I/O shortcut source)
mb = y-0.5
region(ax, MX, (mt+mb)/2, BW+2, mt-mb, 'Encoder Mid', C['bg_mid'], fs=9)
spatial_label(ax, SX, (mt+mb)/2, '8×8', fs=7.5, color=C['mid'])

y -= 1.2
arr(ax, MX, y+0.9, MX, y+0.45, lw=AW_MAIN)
box(ax, MX, y, BW, BH, 'LayerNorm → SiLU', C['norm'], fs=8.5, tc='#333')
y -= 1.0
arr(ax, MX, y+0.7, MX, y+0.45, lw=AW_MAIN)
box(ax, MX, y, BW, BH, 'Conv3×3 (1024 → 128)', C['conv'], fs=8.5)
# I/O shortcut ⊕ marker on conv_out
box(ax, MX-BW/2-0.5, y, 0.5, 0.5, '⊕', C['shortcut'], fs=11, tc='white', alpha=0.95)
dim(ax, MX+DX, y, '[128, 8, 8]  →  split(μ, σ)')

# Encoder I/O shortcut bypass: EM2 output → group_avg → ⊕ (bypasses norm+swish+conv_out)
io_sc_x = MX - BW/2 - 2.2
io_sc_y = (mb_em2_y + y) / 2
box(ax, io_sc_x, io_sc_y, 3.6, 0.7, '_out_shortcut\ngroup_avg (1024→128)', C['shortcut'], fs=7.5, tc='white', alpha=0.95)
ax.annotate('', xy=(io_sc_x, io_sc_y+0.4), xytext=(MX-BW/2-0.1, mb_em2_y),
            arrowprops=dict(arrowstyle='->', color=C['shortcut'], lw=1.8,
                            connectionstyle='arc3,rad=-0.2'), zorder=6)
ax.annotate('', xy=(MX-BW/2-0.5-0.25, y), xytext=(io_sc_x, io_sc_y-0.4),
            arrowprops=dict(arrowstyle='->', color=C['shortcut'], lw=1.8,
                            connectionstyle='arc3,rad=-0.2'), zorder=6)
note(ax, io_sc_x, io_sc_y-1.0, 'I/O shortcut (non-parametric)', fs=7, color=C['shortcut'])

enc_end = y - 0.7
region(ax, MX+2, (enc_start+enc_end)/2, 19, enc_start-enc_end,
       'ENCODER', C['bg_enc'], fs=15, alpha=0.10, ls='-', lw=2.0, label_color='#7B1FA2')

# ═══════ LATENT ═══════
y -= 2.0
arr(ax, MX, y+1.7, MX, y+0.6, lw=AW_MAIN)
latent_y = y
box(ax, 9, y, 12, 1.0,
    'Latent :    z = μ + σ · ε              z  shape = [64, 8, 8]',
    C['latent'], fs=11)
arr(ax, MX+BW/2+0.1, y, 9-6-0.1, y, color=C['latent'], lw=1.5)
spatial_label(ax, SX, y, '8×8', fs=8, color=C['latent'])

# ══════════════════════
#   DECODER
# ══════════════════════
dec_start = y - 0.8

no_skip_y = (latent_y + dec_start) / 2 - 1.5
ax.text(9, no_skip_y,
        '⚠  Decoder is fully independent — NO skip connections from Encoder.\n'
        'All information must pass through the latent bottleneck.',
        ha='center', va='center', fontsize=9, color='#C62828', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE', edgecolor='#C62828',
                  lw=1.2, alpha=0.9), zorder=6)

y -= 2.5
arr(ax, 9, y+2.0, MX, y+0.45, lw=AW_MAIN)
box(ax, MX, y, BW, BH, 'Conv3×3 (64 → 1024)', C['conv'], fs=9)
# I/O shortcut ⊕ marker on conv_in
box(ax, MX-BW/2-0.5, y, 0.5, 0.5, '⊕', C['shortcut'], fs=11, tc='white', alpha=0.95)
dim(ax, MX+DX, y, '[1024, 8, 8]')

# Decoder I/O shortcut bypass: z → repeat_interleave → ⊕
io_sc_d_x = MX - BW/2 - 2.2
io_sc_d_y = y + 1.1
box(ax, io_sc_d_x, io_sc_d_y, 3.6, 0.7, '_in_shortcut\nrepeat_interleave (64→1024)', C['shortcut'], fs=7.5, tc='white', alpha=0.95)
ax.annotate('', xy=(io_sc_d_x, io_sc_d_y+0.4), xytext=(9-1.2, y+1.85),
            arrowprops=dict(arrowstyle='->', color=C['shortcut'], lw=1.8,
                            connectionstyle='arc3,rad=0.2'), zorder=6)
ax.annotate('', xy=(MX-BW/2-0.5-0.25, y), xytext=(io_sc_d_x, io_sc_d_y-0.4),
            arrowprops=dict(arrowstyle='->', color=C['shortcut'], lw=1.8,
                            connectionstyle='arc3,rad=-0.2'), zorder=6)
note(ax, io_sc_d_x, io_sc_d_y+1.0, 'I/O shortcut (non-parametric)', fs=7, color=C['shortcut'])

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

# ── 4 × WFUpBlock (DCAE, no energy flow) ──
# (in_ch, out_ch, sp_in, sp_out, num_rb)
up_blocks = [
    (1024, 1024, '8×8',     '16×16',   5),
    (1024,  512, '16×16',   '32×32',   5),
    (512,   512, '32×32',   '64×64',   5),
    (512,   256, '64×64',   '128×128', 3),
]

for i, (ic, oc, sp_in, sp_out, nrb) in enumerate(up_blocks):
    ybt = y - 0.7
    block_top_y = y - 0.5

    y -= 1.6
    arr(ax, MX, y+1.3, MX, y+0.45, lw=AW_MAIN)
    box(ax, MX, y, BW, BH, f'branch_conv ResBlock+ECA ({ic}→{ic})', C['resblock'], fs=8)
    dim(ax, MX+DX, y, f'[{ic}, {sp_in}]')

    rb_count = max(nrb - 2, 0)
    y -= 1.3
    arr(ax, MX, y+1.0, MX, y+0.45, lw=AW_MAIN)
    rb_label = f'ResnetBlock2D ({ic}) ×{rb_count}' if rb_count > 0 else f'res_block: empty (num-2=0)'
    box(ax, MX, y, BW, BH, rb_label, C['resblock'], fs=8)
    dim(ax, MX+DX, y, f'[{ic}, {sp_in}]')

    y -= 1.3
    arr(ax, MX, y+1.0, MX, y+0.45, lw=AW_MAIN)
    box(ax, MX, y, BW, BH, f'conv_up Conv({ic}→{oc*4},s=1) + pixel_shuffle(2)', C['up'], fs=7.5)
    dim(ax, MX+DX, y, f'[{oc}, {sp_out}]')
    spatial_label(ax, SX, y, sp_out, fs=7.5)

    y -= 1.3
    arr(ax, MX, y+1.0, MX, y+0.45, lw=AW_MAIN)
    box(ax, MX, y, BW, BH, f'out_res_block ({oc}→{oc}) +ECA', C['resblock'], fs=8)
    box(ax, MX-BW/2-0.5, y, 0.5, 0.5, '⊕', C['shortcut'], fs=11, tc='white', alpha=0.95)
    dim(ax, MX+DX, y, f'[{oc}, {sp_out}]')

    # DC-AE shortcut arrow
    block_bot_y = y + 0.1
    shortcut_arrow(ax, SC_X, block_top_y, block_bot_y,
                   f'pixel_shuffle\n+ repeat {oc*4//ic}×\n({ic}→{oc})')

    ybb = y - 0.7
    region(ax, MX+1, (ybt+ybb)/2, BW+5, ybt-ybb,
           f'WFUpBlock {i+1}   ({ic} → {oc})   num_rb={nrb}   (no energy flow)',
           C['bg_block'], fs=9)

    y -= BLOCK_GAP

# ── Decoder output ──
y -= 0.8
arr(ax, MX, y+0.8+BLOCK_GAP, MX, y+0.45, lw=AW_MAIN)
box(ax, MX, y, BW, BH, 'LayerNorm → SiLU', C['norm'], fs=8.5, tc='#333')

y -= 1.0
arr(ax, MX, y+0.7, MX, y+0.45, lw=AW_MAIN)
box(ax, MX, y, BW, BH, 'Conv3×3 (256 → 12)', C['conv'], fs=8.5)
dim(ax, MX+DX, y, '[12, 128, 128]')

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
region(ax, MX+2, (dec_start+dec_end)/2, 19, dec_start-dec_end,
       'DECODER', C['bg_dec'], fs=15, alpha=0.10, ls='-', lw=2.0, label_color='#1565C0')

# ═══════════════════════
#  DC-AE Shortcut Detail Box
# ═══════════════════════
sx_box_x, sx_box_y = 18, -22
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
#  No-ef note
# ═══════════════════════
noef_box_y = sx_box_y - 10
ax.text(sx_box_x, noef_box_y,
        '  use_energy_flow = false  \n'
        '  (this branch\'s shipped config)\n\n'
        '  Removed vs. paper WF-VAE:\n'
        '  • in_flow_conv  (each down)\n'
        '  • out_flow_conv  (each up)\n'
        '  • branch_conv +128 expansion\n'
        '  • IWT inside up blocks\n'
        '  • h[:,:3] += w  at conv_out\n'
        '  • enc/dec_coeffs list\n'
        '  • WL loss (CSV col stays 0)\n\n'
        '  Kept: W^(1) HaarWT at entry,\n'
        '  IWT at exit, DCAE & I/O shortcuts.',
        ha='center', va='center', fontsize=8, color='#BF360C',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3E0',
                  edgecolor='#BF360C', lw=1.5),
        zorder=6)

# ═══════════════════════
#  Legend
# ═══════════════════════
legend = [
    ('HaarWavelet / InvWavelet',       C['wavelet']),
    ('Conv2d',                         C['conv']),
    ('ResnetBlock2D  (+ ECA attn)',    C['resblock']),
    ('Downsample (DCAE)',              C['down']),
    ('Upsample (DCAE)',                C['up']),
    ('Attention2DFix',                 C['attn']),
    ('DC-AE + I/O Shortcut (0 param)', C['shortcut']),
    ('Latent Distribution',            C['latent']),
    ('Norm / Activation',              C['norm']),
]
lx = 18
ly = noef_box_y - 10
ax.text(lx, ly, 'Legend', ha='center', fontsize=11, fontweight='bold', color='#333')
for k, (label, color) in enumerate(legend):
    yy = ly - 0.7 - k*0.6
    p = FancyBboxPatch((lx-2.0, yy-0.2), 0.7, 0.4, boxstyle="round,pad=0.03",
                        facecolor=color, edgecolor='#424242', lw=0.5, alpha=0.9, zorder=4)
    ax.add_patch(p)
    ax.text(lx-1.2, yy, label, ha='left', va='center', fontsize=8, color='#333', zorder=5)


out_path_1 = os.path.join(OUT_DIR, 'wfvae2_32x_dcae_noef_architecture.png')
plt.savefig(out_path_1, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_path_1}")
plt.close()


# ═══════════════════════════════════════════════════════════════
#  FIGURE 2 — WFDownBlock & WFUpBlock Detail (DCAE + no energy flow)
# ═══════════════════════════════════════════════════════════════

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 14))
fig2.patch.set_facecolor('white')
for a in [ax1, ax2]:
    a.set_aspect('equal')
    a.axis('off')

BW2 = 5.5;  BH2 = 0.8

# ══════════════════════
#  WFDownBlock Detail
# ══════════════════════
ax1.set_xlim(-5, 15)
ax1.set_ylim(-1, 18)
ax1.set_title('WFDownBlock  (example: 256 → 512, stage 0, num_rb=4)  —  DCAE + no energy flow',
              fontsize=13, fontweight='bold', pad=12)

MX1 = 6.5
region(ax1, 5, 8.5, 18, 18, '', C['bg_block'], alpha=0.12, fs=1)

y = 17.0
box(ax1, MX1, y, BW2, BH2, 'Input  x', '#757575', fs=10)
dim(ax1, MX1+BW2/2+0.3, y, '[256, 128, 128]', fs=9)

# DC-AE shortcut label at top-left
sc_x = MX1 - BW2/2 - 2.0
sc_top_y = y - 0.5
ax1.text(sc_x, y, 'DC-AE\nShortcut', ha='center', va='center', fontsize=8,
         color=C['shortcut'], fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#E0F2F1',
                   edgecolor=C['shortcut'], lw=1.0, alpha=0.9), zorder=6)

y -= 2.2
arr(ax1, MX1, y+1.8, MX1, y+0.5, lw=AW_MAIN)
box(ax1, MX1, y, BW2, BH2, 'res_block: ResBlock+ECA(256→256) ×(num_rb-1)=3', C['resblock'], fs=8.5)
dim(ax1, MX1+BW2/2+0.3, y, '[256, 128, 128]', fs=8.5)

y -= 2.2
arr(ax1, MX1, y+1.8, MX1, y+0.5, lw=AW_MAIN)
box(ax1, MX1, y, BW2, BH2+0.15, 'conv_down Conv(256→128,s=1)\n+ pixel_unshuffle(2)', C['down'], fs=8.5)
dim(ax1, MX1+BW2/2+0.3, y, '[512, 64, 64]', fs=8.5)

# Shortcut path boxes on the left
sc_y1 = 13.5
box(ax1, sc_x, sc_y1, 3.8, 0.7, 'pixel_unshuffle(2)', C['shortcut'], fs=8)
dim(ax1, sc_x-2.0, sc_y1-0.6, '[1024, 64, 64]', fs=7.5, color=C['shortcut'])
arr(ax1, sc_x, sc_top_y-0.5, sc_x, sc_y1+0.45, color=C['shortcut'], lw=1.8)

sc_y2 = sc_y1 - 2.0
box(ax1, sc_x, sc_y2, 3.8, 0.7, 'reshape + mean\n(2 groups → 512)', C['shortcut'], fs=8)
dim(ax1, sc_x-2.0, sc_y2-0.6, '[512, 64, 64]', fs=7.5, color=C['shortcut'])
arr(ax1, sc_x, sc_y1-0.45, sc_x, sc_y2+0.45, color=C['shortcut'], lw=1.8)

# out_res_block (no concat; input channels = out_ch because no energy-flow concat)
y -= 2.5
arr(ax1, MX1, y+2.1, MX1, y+0.5, lw=AW_MAIN)
box(ax1, MX1, y, BW2, BH2, 'out_res_block: ResBlock+ECA (512 → 512)', C['resblock'], fs=9.5)
box(ax1, MX1-BW2/2-0.5, y, 0.5, 0.5, '⊕', C['shortcut'], fs=11, tc='white', alpha=0.95)
dim(ax1, MX1+BW2/2+0.3, y, '[512, 64, 64]', fs=8.5)
note(ax1, MX1, y-0.7, 'in_channels = out_ch (not out_ch+ef — no mid-layer energy flow)',
     fs=7.5, color='#BF360C')

# Connect shortcut to ⊕
arr(ax1, sc_x+1.9+0.1, sc_y2-0.5, MX1-BW2/2-0.1, y, color=C['shortcut'], lw=2.0)

# Output
y -= 2.2
arr(ax1, MX1, y+1.8, MX1, y+0.5, lw=AW_MAIN)
box(ax1, MX1, y, BW2, BH2, 'Output  h', '#616161', fs=10)
dim(ax1, MX1+BW2/2+0.3, y, '[512, 64, 64]', fs=9)
note(ax1, MX1, y-0.7, 'forward returns (h, None) — no inter_coeffs under use_energy_flow=false',
     fs=7.5, color='#616161')


# ══════════════════════
#  WFUpBlock Detail
# ══════════════════════
ax2.set_xlim(-5, 15)
ax2.set_ylim(-1, 18)
ax2.set_title('WFUpBlock  (example: 1024 → 512, stage 1, num_rb=5)  —  DCAE + no energy flow',
              fontsize=13, fontweight='bold', pad=12)

MX2 = 6.5
region(ax2, 5, 8.5, 18, 18, '', C['bg_block'], alpha=0.12, fs=1)

y = 17.0
box(ax2, MX2, y, BW2, BH2, 'Input  x', '#757575', fs=10)
dim(ax2, MX2+BW2/2+0.3, y, '[1024, 16, 16]', fs=9)

sc_x2 = MX2 - BW2/2 - 2.0
sc_top_y2 = y - 0.5
ax2.text(sc_x2, y, 'DC-AE\nShortcut', ha='center', va='center', fontsize=8,
         color=C['shortcut'], fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#E0F2F1',
                   edgecolor=C['shortcut'], lw=1.0, alpha=0.9), zorder=6)

# branch_conv (no +ef expansion)
y -= 2.0
arr(ax2, MX2, y+1.6, MX2, y+0.5, lw=AW_MAIN)
box(ax2, MX2, y, BW2, BH2, 'branch_conv: ResBlock+ECA (1024→1024)', C['resblock'], fs=9)
dim(ax2, MX2+BW2/2+0.3, y, '[1024, 16, 16]', fs=8.5)
note(ax2, MX2, y-0.7, 'in_channels = in_ch (no +128 expansion; no split / out_flow_conv)',
     fs=7.5, color='#BF360C')

# res_block
y -= 2.2
arr(ax2, MX2, y+1.8, MX2, y+0.5, lw=AW_MAIN)
box(ax2, MX2, y, BW2, BH2, 'res_block: ResBlock+ECA(1024→1024) ×(num_rb-2)=3', C['resblock'], fs=8)
dim(ax2, MX2+BW2/2+0.3, y, '[1024, 16, 16]', fs=8.5)

# conv_up
y -= 2.2
arr(ax2, MX2, y+1.8, MX2, y+0.5, lw=AW_MAIN)
box(ax2, MX2, y, BW2, BH2+0.15, 'conv_up Conv(1024→2048,s=1)\n+ pixel_shuffle(2)', C['up'], fs=8.5)
dim(ax2, MX2+BW2/2+0.3, y, '[512, 32, 32]', fs=8.5)

# out_res_block
y -= 2.2
arr(ax2, MX2, y+1.8, MX2, y+0.5, lw=AW_MAIN)
box(ax2, MX2, y, BW2, BH2, 'out_res_block: ResBlock+ECA (512→512)', C['resblock'], fs=9)
box(ax2, MX2-BW2/2-0.5, y, 0.5, 0.5, '⊕', C['shortcut'], fs=11, tc='white', alpha=0.95)
dim(ax2, MX2+BW2/2+0.3, y, '[512, 32, 32]', fs=8.5)

# Shortcut path boxes
sc_y1_2 = 13.5
box(ax2, sc_x2, sc_y1_2, 3.8, 0.7, 'repeat_interleave(2)', C['shortcut'], fs=8)
dim(ax2, sc_x2-2.0, sc_y1_2-0.6, '[2048, 16, 16]', fs=7.5, color=C['shortcut'])
arr(ax2, sc_x2, sc_top_y2-0.5, sc_x2, sc_y1_2+0.45, color=C['shortcut'], lw=1.8)

sc_y2_2 = sc_y1_2 - 2.0
box(ax2, sc_x2, sc_y2_2, 3.8, 0.7, 'pixel_shuffle(2)', C['shortcut'], fs=8)
dim(ax2, sc_x2-2.0, sc_y2_2-0.6, '[512, 32, 32]', fs=7.5, color=C['shortcut'])
arr(ax2, sc_x2, sc_y1_2-0.45, sc_x2, sc_y2_2+0.45, color=C['shortcut'], lw=1.8)

arr(ax2, sc_x2+1.9+0.1, sc_y2_2-0.5, MX2-BW2/2-0.1, y, color=C['shortcut'], lw=2.0)

# Output
y -= 2.2
arr(ax2, MX2, y+1.8, MX2, y+0.5, lw=AW_MAIN)
box(ax2, MX2, y, BW2, BH2, 'Output  h', '#616161', fs=10)
dim(ax2, MX2+BW2/2+0.3, y, '[512, 32, 32]', fs=9)
note(ax2, MX2, y-0.7, 'forward returns (h, None, None) — no wavelet residual / dec_coeffs',
     fs=7.5, color='#616161')


plt.tight_layout(pad=2.0)
out_path_2 = os.path.join(OUT_DIR, 'wfvae2_32x_dcae_noef_blocks_detail.png')
plt.savefig(out_path_2, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_path_2}")
plt.close()


print("\nDone — WF-VAE 32× DCAE + no energy flow architecture diagrams.")
