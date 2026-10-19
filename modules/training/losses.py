import torch
import torch.nn.functional as F
import math

from modules.dataset.megadepth import megadepth_warper

from modules.training import utils

from third_party.alike_wrapper import extract_alike_kpts, extract_alike_kpts_scores, detectAndCompute

def dual_softmax_loss(X, Y, temp = 0.2):
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    dist_mat = (X @ Y.t()) * temp
    conf_matrix12 = F.log_softmax(dist_mat, dim=1)
    conf_matrix21 = F.log_softmax(dist_mat.t(), dim=1)

    with torch.no_grad():
        conf12 = torch.exp( conf_matrix12 ).max(dim=-1)[0]
        conf21 = torch.exp( conf_matrix21 ).max(dim=-1)[0]
        conf = conf12 * conf21

    target = torch.arange(len(X), device = X.device)

    loss = F.nll_loss(conf_matrix12, target) + \
           F.nll_loss(conf_matrix21, target)

    return loss, conf

def smooth_l1_loss(input, target, beta=2.0, size_average=True):
    diff = torch.abs(input - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.mean() if size_average else loss.sum()

def fine_loss(f1, f2, pts1, pts2, fine_module, ws=7):
    '''
        Compute Fine features and spatial loss
    '''
    C, H, W = f1.shape
    N = len(pts1)

    #Sort random offsets
    with torch.no_grad():
        a = -(ws//2)
        b = (ws//2)
        offset_gt = (a - b) * torch.rand(N, 2, device = f1.device) + b
        pts2_random = pts2 + offset_gt

    #pdb.set_trace()
    patches1 = utils.crop_patches(f1.unsqueeze(0), (pts1+0.5).long(), size=ws).view(C, N, ws * ws).permute(1, 2, 0) #[N, ws*ws, C]
    patches2 = utils.crop_patches(f2.unsqueeze(0), (pts2_random+0.5).long(), size=ws).view(C, N, ws * ws).permute(1, 2, 0)  #[N, ws*ws, C]

    #Apply transformer
    patches1, patches2 = fine_module(patches1, patches2)

    features = patches1.view(N, ws, ws, C)[:, ws//2, ws//2, :].view(N, 1, 1, C) # [N, 1, 1, C]
    patches2 = patches2.view(N, ws, ws, C) # [N, w, w, C]

    #Dot Product
    heatmap_match = (features * patches2).sum(-1)
    offset_coords = utils.subpix_softmax2d(heatmap_match)

    #Invert offset because center crop inverts it
    offset_gt = -offset_gt 

    #MSE
    error = ((offset_coords - offset_gt)**2).sum(-1).mean()

    #error = smooth_l1_loss(offset_coords, offset_gt)

    return error


def alike_distill_loss(kpts, img):
    C, H, W = kpts.shape
    kpts = kpts.permute(1,2,0) 
    img = img.permute(1,2,0).expand(-1,-1,3).cpu().numpy() * 255

    with torch.no_grad():
        alike_kpts = torch.tensor( extract_alike_kpts(img), device=kpts.device )
        labels = torch.ones((H, W), dtype = torch.long, device = kpts.device) * 64 # -> Default is non-keypoint (bin 64)
        offsets = (((alike_kpts/8) - (alike_kpts/8).long())*8).long()
        offsets =  offsets[:, 0] + 8*offsets[:, 1]  # Linear IDX
        labels[(alike_kpts[:,1]/8).long(), (alike_kpts[:,0]/8).long()] = offsets

    kpts = kpts.view(-1,C)
    labels = labels.view(-1)

    mask = labels < 64
    idxs_pos = mask.nonzero().flatten()
    idxs_neg = (~mask).nonzero().flatten()
    perm = torch.randperm(idxs_neg.size(0))[:len(idxs_pos)//32]
    idxs_neg = idxs_neg[perm]
    idxs = torch.cat([idxs_pos, idxs_neg])

    kpts = kpts[idxs]
    labels = labels[idxs]

    with torch.no_grad():
        predicted = kpts.max(dim=-1)[1]
        acc =  (labels == predicted)
        acc = acc.sum() / len(acc)

    kpts = F.log_softmax(kpts)
    loss = F.nll_loss(kpts, labels, reduction = 'mean')

    return loss, acc

def alike_distill_loss_with_conf(kpts, mu, L, img, top_k=512, min_score=0.4, neg_ratio=1/32, edge_margin=0):
    '''
        kpts: (65, Hc, Wc) logits, mu: (Hc, Wc, 2) in [0,8], L: (Hc, Wc, 2, 2), img: (1, H, W) in [0,1]
        edge_margin: drop teacher keypoints within this many px of a lidar no-return pixel (stored as
                     1/255, see extract_lidar_images.py). They sit on the artificial return/no-return
                     boundary (sky line, rig mask) rather than on scene structure. 0 = off (COCO).
        Returns: cls loss, cls acc, gaussian NLL, mean err (px), mean std (px)
    '''
    C, H, W = kpts.shape
    kpts = kpts.permute(1,2,0)

    with torch.no_grad():
        alike_kpts, alike_scores, _ = detectAndCompute(img[None].expand(-1, 3, -1, -1))
        alike_kpts, alike_scores = alike_kpts.to(kpts.device), alike_scores.to(kpts.device)
        keep = alike_scores >= min_score

        if edge_margin > 0 and len(alike_kpts):
            invalid = (img[:1] <= 1.5 / 255).float()[None]                               # (1,1,H,W)
            near_invalid = F.max_pool2d(invalid, 2 * edge_margin + 1, 1, edge_margin)[0, 0] > 0
            xi = alike_kpts[:, 0].round().long().clamp(0, near_invalid.shape[1] - 1)
            yi = alike_kpts[:, 1].round().long().clamp(0, near_invalid.shape[0] - 1)
            keep &= ~near_invalid[yi, xi]

        alike_kpts, alike_scores = alike_kpts[keep], alike_scores[keep]
        alike_kpts = alike_kpts[torch.argsort(-alike_scores)[:top_k]]
        if len(alike_kpts) == 0:
            with torch.enable_grad():
                zero = mu.sum() * 0 + kpts.sum() * 0
            return zero, zero.detach(), zero, zero.detach(), zero.detach()


        
        cell = (alike_kpts / 8).long()
        cell_idx = cell[:, 1] * W + cell[:, 0]

        # Keep one kpt per cell so the class label and continuous target agree
        order = torch.argsort(cell_idx, stable=True)
        cs = cell_idx[order]
        keep = torch.ones_like(cs, dtype=torch.bool)
        keep[1:] = cs[1:] != cs[:-1]
        sel = order[keep]
        alike_kpts, cell_idx = alike_kpts[sel], cell_idx[sel]

        offsets = alike_kpts - 8 * torch.floor(alike_kpts / 8)   # (M,2) continuous, [0,8)
        bins = offsets.long().clamp(0, 7)
        labels = torch.full((H * W,), 64, dtype=torch.long, device=kpts.device)
        labels[cell_idx] = bins[:, 0] + 8 * bins[:, 1]

    kpts = kpts.reshape(-1, C)

    mask = labels < 64
    idxs_pos = mask.nonzero().flatten()
    idxs_neg = (~mask).nonzero().flatten()
    perm = torch.randperm(idxs_neg.size(0))[:int(len(idxs_pos) * neg_ratio)]
    idxs = torch.cat([idxs_pos, idxs_neg[perm]])

    kpts_sel = kpts[idxs]
    labels_sel = labels[idxs]

    with torch.no_grad():
        acc = (kpts_sel.max(dim=-1)[1] == labels_sel).float().mean()

    loss = F.nll_loss(F.log_softmax(kpts_sel, dim=-1), labels_sel)

    # Gaussian NLL on cells that contain a keypoint only
    mu_pos = mu.reshape(-1, 2)[cell_idx]
    L_pos = L.reshape(-1, 2, 2)[cell_idx]
    nll = gaussian_nll_2d(mu_pos, L_pos, offsets)

    with torch.no_grad():
        err_px = (mu_pos - offsets).norm(dim=-1).mean()
        std_px = (L_pos[:, 0, 0] * L_pos[:, 1, 1]).sqrt().mean()   # det(Σ)^(1/4)

    return loss, acc, nll, err_px, std_px


def keypoint_position_loss(kpts1, kpts2, pts1, pts2, softmax_temp = 1.0):
    '''
        Computes coordinate classification loss, by re-interpreting the 64 bins to 8x8 grid and optimizing
        for correct offsets
    '''
    C, H, W = kpts1.shape
    kpts1 = kpts1.permute(1,2,0) * softmax_temp
    kpts2 = kpts2.permute(1,2,0) * softmax_temp

    with torch.no_grad():
        #Generate meshgrid
        x, y = torch.meshgrid(torch.arange(W, device=kpts1.device), torch.arange(H, device=kpts1.device), indexing ='xy')
        xy = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
        xy*=8

        #Generate collision map
        hashmap = torch.ones((H*8, W*8, 2), dtype = torch.long, device = kpts1.device) * -1
        hashmap[(pts1[:,1]).long(), (pts1[:,0]).long(), :] = (pts2).long()

        #Estimate offset of src kpts 
        _, kpts1_offsets = kpts1.max(dim=-1)
        kpts1_offsets_x = kpts1_offsets  % 8
        kpts1_offsets_y = kpts1_offsets // 8
        kpts1_offsets_xy = torch.cat([kpts1_offsets_x.unsqueeze(-1), 
                                      kpts1_offsets_y.unsqueeze(-1)], dim=-1)
        #pdb.set_trace()
        kpts1_coords = xy + kpts1_offsets_xy

        #find src -> tgt pts
        kpts1_coords = kpts1_coords.view(-1,2)
        gt_12 = hashmap[kpts1_coords[:,1], kpts1_coords[:,0]]
        mask_valid = torch.all(gt_12 >= 0, dim=-1)
        gt_12 = gt_12[mask_valid]

        #find offset labels
        labels2 = (gt_12/8) - (gt_12/8).long()
        labels2 = (labels2 * 8).long()
        labels2 = labels2[:, 0] + 8*labels2[:, 1] #linear index
        
    kpts2_selected = kpts2[(gt_12[:, 1]/8).long(), (gt_12[:, 0]/8).long()]        

    kpts1_selected = F.log_softmax(kpts1.view(-1,C)[mask_valid], dim=-1)
    kpts2_selected = F.log_softmax(kpts2_selected, dim=-1)

    #Here we enforce softmax to keep current max on src kps
    with torch.no_grad():
        _, labels1 =  kpts1_selected.max(dim=-1)

    predicted2 = kpts2_selected.max(dim=-1)[1]
    acc =  (labels2 == predicted2)
    acc = acc.sum() / len(acc)

    loss = F.nll_loss(kpts1_selected, labels1, reduction = 'mean') + \
           F.nll_loss(kpts2_selected, labels2, reduction = 'mean')
    
    #pdb.set_trace()

    return loss, acc

def coordinate_classification_loss(coords1, pts1, pts2, conf):
    '''
        Computes the fine coordinate classification loss, by re-interpreting the 64 bins to 8x8 grid and optimizing
        for correct offsets after warp
    '''
    #Do not backprop coordinate warps
    with torch.no_grad():

        coords1_detached = pts1 * 8 

        #find offset
        offsets1_detached = (coords1_detached/8) - (coords1_detached/8).long()
        offsets1_detached = (offsets1_detached * 8).long()
        labels1 = offsets1_detached[:, 0] + 8*offsets1_detached[:, 1]

    #pdb.set_trace()
    coords1_log = F.log_softmax(coords1, dim=-1)

    predicted = coords1.max(dim=-1)[1]
    acc =  (labels1 == predicted)
    acc = acc[conf > 0.1]
    acc = acc.sum() / len(acc)

    loss = F.nll_loss(coords1_log, labels1, reduction = 'none')
    
    #Weight loss by confidence, giving more emphasis on reliable matches
    conf = conf / conf.sum()
    loss = (loss * conf).sum()

    return loss * 2., acc

def keypoint_loss(heatmap, target):
    # Compute L1 loss
    L1_loss = F.l1_loss(heatmap, target)
    return L1_loss * 3.0

def hard_triplet_loss(X,Y, margin = 0.5):

    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    dist_mat = torch.cdist(X, Y, p=2.0)
    dist_pos = torch.diag(dist_mat)
    dist_neg = dist_mat + 100.*torch.eye(*dist_mat.size(), dtype = dist_mat.dtype, 
            device = dist_mat.get_device() if dist_mat.is_cuda else torch.device("cpu"))

    #filter repeated patches on negative distances to avoid weird stuff on gradients
    dist_neg = dist_neg + dist_neg.le(0.01).float()*100.

    #Margin Ranking Loss
    hard_neg = torch.min(dist_neg, 1)[0]

    loss = torch.clamp(margin + dist_pos - hard_neg, min=0.)

    return loss.mean()


def gaussian_nll_2d(mu, L, target):
    '''
        Full-covariance 2D Gaussian NLL. mu, target: (..., 2); L: (..., 2, 2) lower-triangular, Σ = L Lᵀ
    '''
    diff = (target - mu).unsqueeze(-1)                                     # (..., 2, 1)
    z = torch.linalg.solve_triangular(L, diff, upper=False).squeeze(-1)    # L z = diff
    maha = (z ** 2).sum(-1)                                                # rᵀ Σ⁻¹ r
    log_det = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)   # log|Σ|
    return (0.5 * (maha + log_det) + math.log(2 * math.pi)).mean()
