import os
import numpy as np
import torch
import torch.nn.functional as F

from modules.xfeat import XFeat
from modules.intensity_model import IntensityFeatureModel


class IntensityXFeat(XFeat):
	"""
		XFeat inference with per-keypoint 2x2 covariance.
		base_weights: XFeat backbone (xfeat.pt), loaded non-strict -> new heads stay random.
		weights:      full IntensityFeatureModel checkpoint (overrides base_weights if given).
	"""

	def __init__(self, weights = None,
				 base_weights = os.path.abspath(os.path.dirname(__file__)) + '/../weights/xfeat.pt',
				 top_k = 4096, detection_threshold = 0.05):
		super().__init__(weights = None, top_k = top_k, detection_threshold = detection_threshold)
		self.net = IntensityFeatureModel().to(self.dev).eval()

		if weights is not None:
			self.net.load_state_dict(torch.load(weights, map_location = self.dev))
		elif base_weights is not None:
			missing, _ = self.net.load_state_dict(torch.load(base_weights, map_location = self.dev), strict = False)
			print('[IntensityXFeat] untrained heads:', sorted({k.split('.')[0] for k in missing}))

	@torch.inference_mode()
	def detectAndCompute(self, x, top_k = None, detection_threshold = None):
		"""
			x: (B,C,H,W) tensor or (H,W[,C]) np array.
			Returns List[Dict]: keypoints (N,2), covariances (N,2,2), scores (N,), descriptors (N,64)
		"""
		if top_k is None: top_k = self.top_k
		if detection_threshold is None: detection_threshold = self.detection_threshold
		x, rh, rw = self.preprocess_tensor(x)
		B, _, _H, _W = x.shape

		feats, logits, hmap, mu, L, cov = self.net(x)
		feats = F.normalize(feats, dim = 1)

		score = (1 - F.softmax(logits, 1)[:, 64]) * hmap[:, 0]                  # (B,Hc,Wc)
		score = score * (score == F.max_pool2d(score[:, None], 3, 1, 1)[:, 0])  # coarse NMS
		Hc, Wc = score.shape[-2:]
		S = torch.diag(torch.tensor([rw, rh], device = self.dev, dtype = cov.dtype))

		out = []
		for b in range(B):
			s, idx = score[b].flatten().topk(min(top_k, Hc * Wc))
			keep = s > detection_threshold
			s, idx = s[keep], idx[keep]

			cell = torch.stack([idx % Wc, idx // Wc], -1).float()
			kps = cell * 8 + mu[b].reshape(-1, 2)[idx]                          # in resized image px
			desc = F.normalize(self.interpolator(feats[b:b+1], kps[None], H = _H, W = _W)[0], dim = -1)
			covs = S @ cov[b].reshape(-1, 2, 2)[idx] @ S                         # Σ' = S Σ S
			kps = kps * torch.tensor([rw, rh], device = self.dev)

			out.append({'keypoints': kps, 'covariances': covs, 'scores': s, 'descriptors': desc})
		return out

	@torch.inference_mode()
	def uncertainty_map(self, x):
		""" (Hc,Wc) per-cell det(Σ)^(1/4) in px, for visualisation. """
		x, _, _ = self.preprocess_tensor(x)
		L = self.net(x)[4]
		return (L[..., 0, 0] * L[..., 1, 1]).sqrt()[0].cpu().numpy()
