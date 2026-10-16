"""
    Plot per-keypoint covariance ellipses predicted by IntensityFeatureModel.

    Examples:
        # 200 best keypoints, one per 32px block, 2-sigma ellipses
        python plot_covariances.py --img /home/asrl/datasets/coco/val2017/000000000139.jpg \
            --ckpt ckpts/cov_v0/xfeat_synthetic_5000.pth --n_show 200 --grid 32

        # zoom on a region, show the least certain keypoints, add the uncertainty map
        python plot_covariances.py --img img.png --ckpt model.pth --roi 100 50 300 200 \
            --select least_certain --n_show 30 --std_map
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection

from modules.intensity_model import IntensityFeatureModel


def parse_args():
    p = argparse.ArgumentParser(description='Plot per-keypoint covariance ellipses.')
    p.add_argument('--img', required=True, help='Input image (read as grayscale).')
    p.add_argument('--ckpt', default=None,
                   help='IntensityFeatureModel state_dict. If omitted, loads --base_weights non-strict '
                        '(mu/cov heads stay random -> useful only as a baseline).')
    p.add_argument('--base_weights', default=os.path.join(os.path.dirname(__file__), 'weights/xfeat.pt'))
    p.add_argument('--out', default=None, help='Save figure here instead of showing it.')

    # Detection
    p.add_argument('--top_k', type=int, default=4096, help='Max keypoints detected before filtering.')
    p.add_argument('--min_score', type=float, default=0.05, help='Detection score threshold.')

    # Subset filtering for legibility
    p.add_argument('--roi', type=float, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'), default=None,
                   help='Only keep (and zoom to) keypoints inside this box, in original image px.')
    p.add_argument('--grid', type=int, default=0,
                   help='Keep at most one keypoint (highest score) per GRID x GRID px block. 0 = off.')
    p.add_argument('--select', default='top',
                   choices=['top', 'random', 'most_certain', 'least_certain'],
                   help='How to pick the --n_show keypoints that remain after the filters above.')
    p.add_argument('--n_show', type=int, default=150, help='Number of keypoints to draw. 0 = all.')
    p.add_argument('--seed', type=int, default=0)

    # Drawing
    p.add_argument('--sigma', type=float, default=2.0, help='Ellipse radius in standard deviations.')
    p.add_argument('--exaggerate', type=float, default=1.0,
                   help='Extra scale on ellipse size (covariances are ~1px; raise this on full-image views).')
    p.add_argument('--std_map', action='store_true', help='Add a panel with the per-cell uncertainty map.')
    return p.parse_args()


def load_model(ckpt, base_weights, dev):
    net = IntensityFeatureModel().to(dev).eval()
    if ckpt is not None:
        net.load_state_dict(torch.load(ckpt, map_location=dev))
    else:
        missing, _ = net.load_state_dict(torch.load(base_weights, map_location=dev), strict=False)
        print('[warn] no --ckpt given; untrained heads:', sorted({k.split('.')[0] for k in missing}))
    return net


@torch.inference_mode()
def detect_with_cov(net, img, dev, top_k, min_score):
    """
        img: (H, W) uint8 grayscale.
        Returns keypoints (N,2), covariances (N,2,2), scores (N,) in original image px,
        and the per-cell uncertainty map det(Σ)^(1/4) at coarse (H/8, W/8) resolution.
    """
    H, W = img.shape
    _H, _W = (H // 32) * 32, (W // 32) * 32
    rh, rw = H / _H, W / _W

    x = torch.from_numpy(img).float().to(dev)[None, None] / 255.
    x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)

    _, logits, hmap, mu, L, cov = net(x)

    score = (1 - F.softmax(logits, 1)[:, 64]) * hmap[:, 0]                   # p(kpt in cell) * reliability
    score = score * (score == F.max_pool2d(score[:, None], 3, 1, 1)[:, 0])   # coarse NMS
    score = score[0]
    Hc, Wc = score.shape

    s, idx = score.flatten().topk(min(top_k, Hc * Wc))
    keep = s > min_score
    s, idx = s[keep], idx[keep]

    cell = torch.stack([idx % Wc, idx // Wc], -1).float()
    kps = (cell * 8 + mu[0].reshape(-1, 2)[idx]) * torch.tensor([rw, rh], device=dev)
    S = torch.diag(torch.tensor([rw, rh], device=dev, dtype=cov.dtype))
    covs = S @ cov[0].reshape(-1, 2, 2)[idx] @ S                               # Σ' = S Σ Sᵀ

    std_map = (L[0, ..., 0, 0] * L[0, ..., 1, 1]).sqrt()

    return kps.cpu().numpy(), covs.cpu().numpy(), s.cpu().numpy(), std_map.cpu().numpy()


def mean_std(covs):
    """ Scalar uncertainty per keypoint: det(Σ)^(1/4), i.e. geometric-mean std in px. """
    return np.linalg.det(covs).clip(min=0) ** 0.25


def filter_keypoints(kps, covs, scores, args):
    """ Apply ROI -> grid -> selection, returning the indices to draw. """
    idx = np.arange(len(kps))

    if args.roi is not None:
        x0, y0, x1, y1 = args.roi
        inside = (kps[:, 0] >= x0) & (kps[:, 0] < x1) & (kps[:, 1] >= y0) & (kps[:, 1] < y1)
        idx = idx[inside]

    if args.grid > 0 and len(idx):
        # Best-scoring keypoint per block: sort by score, keep the first hit of each block
        idx = idx[np.argsort(-scores[idx])]
        blocks = (kps[idx] // args.grid).astype(np.int64)
        _, first = np.unique(blocks, axis=0, return_index=True)
        idx = idx[np.sort(first)]

    if args.n_show > 0 and len(idx) > args.n_show:
        if args.select == 'top':
            order = np.argsort(-scores[idx])
        elif args.select == 'random':
            order = np.random.default_rng(args.seed).permutation(len(idx))
        elif args.select == 'most_certain':
            order = np.argsort(mean_std(covs[idx]))
        else:  # least_certain
            order = np.argsort(-mean_std(covs[idx]))
        idx = idx[order[:args.n_show]]

    return idx


def cov_to_ellipses(covs, n_sigma):
    """ Full axis lengths (width = major) and angle in degrees for matplotlib. """
    w, V = np.linalg.eigh(covs)                        # ascending eigenvalues
    w = w.clip(min=0)
    widths = 2 * n_sigma * np.sqrt(w[:, 1])            # major axis
    heights = 2 * n_sigma * np.sqrt(w[:, 0])           # minor axis
    angles = np.degrees(np.arctan2(V[:, 1, 1], V[:, 0, 1]))
    return widths, heights, angles


def plot(img, kps, covs, scores, std_map, args):
    n_panels = 2 if args.std_map else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(9 * n_panels, 9 * img.shape[0] / img.shape[1]),
                             squeeze=False)
    ax = axes[0, 0]
    ax.imshow(img, cmap='gray', interpolation='nearest')

    std = mean_std(covs)
    w, h, a = cov_to_ellipses(covs, args.sigma * args.exaggerate)
    norm = matplotlib.colors.Normalize(vmin=std.min() if len(std) else 0, vmax=std.max() if len(std) else 1)

    ells = EllipseCollection(w, h, a, units='xy', offsets=kps, offset_transform=ax.transData,
                             facecolors='none', edgecolors=plt.cm.plasma(norm(std)), linewidths=1.2)
    ax.add_collection(ells)
    ax.scatter(kps[:, 0], kps[:, 1], s=4, c='lime', zorder=3)

    sm = plt.cm.ScalarMappable(norm=norm, cmap='plasma')
    fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, label='det(Σ)^¼  [px]')

    title = f'{len(kps)} keypoints, {args.sigma:g}σ ellipses'
    if args.exaggerate != 1:
        title += f' (×{args.exaggerate:g} size)'
    if len(std):
        title += f'\nstd median {np.median(std):.2f}px, range [{std.min():.2f}, {std.max():.2f}]'
    ax.set_title(title)

    if args.roi is not None:
        x0, y0, x1, y1 = args.roi
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
    ax.set_axis_off()

    if args.std_map:
        ax2 = axes[0, 1]
        H, W = img.shape
        im = ax2.imshow(std_map, cmap='plasma', extent=(0, W, H, 0), interpolation='nearest')
        ax2.imshow(img, cmap='gray', alpha=0.25, extent=(0, W, H, 0))
        fig.colorbar(im, ax=ax2, fraction=0.035, pad=0.02, label='det(Σ)^¼  [px]')
        ax2.set_title('Per-cell uncertainty (only keypoint cells are supervised)')
        if args.roi is not None:
            ax2.set_xlim(x0, x1)
            ax2.set_ylim(y1, y0)
        ax2.set_axis_off()

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches='tight')
        print('saved', args.out)
    else:
        plt.show()


def main():
    args = parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img = cv2.imread(args.img, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(args.img)

    net = load_model(args.ckpt, args.base_weights, dev)
    kps, covs, scores, std_map = detect_with_cov(net, img, dev, args.top_k, args.min_score)
    idx = filter_keypoints(kps, covs, scores, args)
    print(f'detected {len(kps)} keypoints, drawing {len(idx)}')

    plot(img, kps[idx], covs[idx], scores[idx], std_map, args)


if __name__ == '__main__':
    main()
