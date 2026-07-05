import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_patches(feat: torch.Tensor, patch_size: int, stride: int):
    """
    Extract a patch bank from a (C, H, W) feature map.
    """
    C, H, W = feat.shape
    patches = F.unfold(feat.unsqueeze(0), kernel_size=patch_size, stride=stride)  # [1, C*p*p, N]
    patches = patches.view(C, patch_size, patch_size, -1)
    return patches


def _safe_int(x):
    return max(1, int(round(x)))


def _num_locs(size, patch, stride):
    # (#locations per dimension for sliding window)
    return (size - patch) // stride + 1


def _lin_to_xy(idx_lin, nW):
    # linear index -> (y, x)
    y = idx_lin // nW
    x = idx_lin % nW
    return y, x


def _xy_to_lin(y, x, nW):
    return y * nW + x


def feature_match_index_downsample_topk(
    feat_input: torch.Tensor,
    feat_ref: torch.Tensor,
    patch_size: int = 3,
    input_stride: int = 1,
    ref_stride: int = 1,
    is_norm: bool = True,
    norm_input: bool = False,
    downsample_ratio: float = 2.0,
    no_overlap_input: bool = True,
    topk: int = 3
):
    """
    Patch matching with optional downsampling.

    Returns:
        idx_topk: Linear indices into the original reference patch grid.
        val_topk: Matching scores for each selected patch.
    """
    assert feat_input.dim() == 3 and feat_ref.dim() == 3, "feat_input/ref shape must be (C,H,W)"
    C, H, W = feat_input.shape
    _, Hr, Wr = feat_ref.shape
    device = feat_input.device

    # Force non-overlapping input windows when requested.
    input_stride_eff = patch_size if no_overlap_input else input_stride

    H_out = _num_locs(H, patch_size, input_stride_eff)
    W_out = _num_locs(W, patch_size, input_stride_eff)

    if downsample_ratio <= 1.0000001:
        return _feature_match_index_nods_topk(
            feat_input, feat_ref,
            patch_size, input_stride_eff, ref_stride,
            is_norm, norm_input,
            topk=topk
        )

    # Match on downsampled features, then remap indices to the original grid.
    ds = float(downsample_ratio)
    H_ds = max(1, int(round(H / ds)))
    W_ds = max(1, int(round(W / ds)))
    Hr_ds = max(1, int(round(Hr / ds)))
    Wr_ds = max(1, int(round(Wr / ds)))

    patch_ds      = _safe_int(patch_size / ds)
    in_stride_ds  = patch_ds if no_overlap_input else _safe_int(input_stride / ds)
    ref_stride_ds = _safe_int(ref_stride / ds)

    feat_in_ds  = F.interpolate(feat_input.unsqueeze(0), size=(H_ds, W_ds),  mode="bilinear", align_corners=False).squeeze(0)
    feat_ref_ds = F.interpolate(feat_ref  .unsqueeze(0), size=(Hr_ds, Wr_ds), mode="bilinear", align_corners=False).squeeze(0)

    idx_topk_ds, val_topk_ds, aux_ds = _feature_match_index_nods_topk(
        feat_in_ds, feat_ref_ds,
        patch_size=patch_ds,
        input_stride=in_stride_ds,
        ref_stride=ref_stride_ds,
        is_norm=is_norm,
        norm_input=norm_input,
        topk=topk,
        return_aux=True
    )
    nH_ref_ds, nW_ref_ds, H_out_ds, W_out_ds = aux_ds

    nH_ref = _num_locs(Hr, patch_size, ref_stride)
    nW_ref = _num_locs(Wr, patch_size, ref_stride)

    flat_idx = idx_topk_ds.reshape(-1)  # (k*H_out_ds*W_out_ds,)
    y_ds, x_ds = _lin_to_xy(flat_idx, nW_ref_ds)
    top_ds_y = y_ds * ref_stride_ds
    top_ds_x = x_ds * ref_stride_ds
    top_y = (top_ds_y.float() * ds).round().long()
    top_x = (top_ds_x.float() * ds).round().long()
    y_ref = (top_y / ref_stride).round().long().clamp_(0, nH_ref - 1)
    x_ref = (top_x / ref_stride).round().long().clamp_(0, nW_ref - 1)
    idx_topk_orig = _xy_to_lin(y_ref, x_ref, nW_ref).view(topk, H_out_ds, W_out_ds)
    val_topk_orig = val_topk_ds

    # Align the downsampled output grid to the full output grid.
    if (H_out_ds, W_out_ds) != (H_out, W_out):
        idx_img = idx_topk_orig.to(torch.float32).unsqueeze(0)  # [1,k,H_ds,W_ds]
        idx_img = F.interpolate(idx_img, size=(H_out, W_out), mode="nearest")
        idx_topk_final = idx_img.squeeze(0).to(torch.long)      # [k,H,W]

        # values
        val_img = val_topk_orig.unsqueeze(0)                    # [1,k,H_ds,W_ds]
        val_img = F.interpolate(val_img, size=(H_out, W_out), mode="nearest")
        val_topk_final = val_img.squeeze(0)                     # [k,H,W]
    else:
        idx_topk_final = idx_topk_orig
        val_topk_final = val_topk_orig

    return idx_topk_final.to(device=device), val_topk_final.to(device=device)


def _feature_match_index_nods_topk(
    feat_input: torch.Tensor,
    feat_ref: torch.Tensor,
    patch_size: int,
    input_stride: int,
    ref_stride: int,
    is_norm: bool,
    norm_input: bool,
    topk: int = 1,
    return_aux: bool = False,
):
    """
    Patch matching without downsampling.

    Returns:
        idx_topk: (k, H_out, W_out)
        val_topk: (k, H_out, W_out)
        Optional aux: (nH_ref, nW_ref, H_out, W_out)
    """
    C, H, W = feat_input.shape
    Cr, Hr, Wr = feat_ref.shape
    assert C == Cr, "Channel mismatch."

    patches_ref = sample_patches(feat_ref, patch_size, ref_stride)  # (C,p,p,N)
    nW_ref = _num_locs(Wr, patch_size, ref_stride)
    nH_ref = _num_locs(Hr, patch_size, ref_stride)

    H_out = _num_locs(H, patch_size, input_stride)
    W_out = _num_locs(W, patch_size, input_stride)

    # Process reference patches in chunks to keep memory bounded.
    batch_size = max(1, int(1024.0**2 * 512 / (H * W)))
    N = patches_ref.shape[-1]

    val_topk = torch.full((topk, H_out, W_out), float("-inf"), device=feat_input.device, dtype=feat_input.dtype)
    idx_topk = torch.full((topk, H_out, W_out), -1, device=feat_input.device, dtype=torch.long)

    for s in range(0, N, batch_size):
        batch = patches_ref[..., s:s + batch_size]  # (C,p,p,B)
        if batch.shape[-1] == 0:
            continue
        if is_norm:
            batch = batch / (batch.norm(p=2, dim=(0, 1, 2)) + 1e-5)
        weight = batch.permute(3, 0, 1, 2).contiguous()     # (B,C,p,p)
        corr = F.conv2d(feat_input.unsqueeze(0), weight, stride=input_stride)  # [1,B,H_out,W_out]
        corr = corr.squeeze(0)  # (B,H_out,W_out)

        B = corr.shape[0]
        vals_cat = torch.cat([val_topk, corr], dim=0)  # ((k+B),H_out,W_out)

        idx_batch = (s + torch.arange(B, device=feat_input.device)).view(B, 1, 1).expand(B, H_out, W_out)
        idx_cat = torch.cat([idx_topk, idx_batch], dim=0)   # ((k+B),H_out,W_out)

        vals, pos = torch.topk(vals_cat, k=topk, dim=0)     # (k,H_out,W_out)
        gather_idx = pos
        idx_cat_exp = idx_cat  # ((k+B),H,W)
        idx_sel = torch.gather(idx_cat_exp, 0, gather_idx)  # (k,H_out,W_out)

        val_topk = vals
        idx_topk = idx_sel

    if norm_input:
        patches_input = sample_patches(feat_input, patch_size, input_stride)  # (C,p,p, H_out*W_out)
        norm = patches_input.norm(p=2, dim=(0, 1, 2)) + 1e-5
        norm = norm.view(H_out, W_out)  # (H_out,W_out)
        val_topk = val_topk / norm.unsqueeze(0)

    if return_aux:
        return idx_topk, val_topk, (nH_ref, nW_ref, H_out, W_out)
    return idx_topk, val_topk


class RefPatchReorderer_TopK(nn.Module):
    """
    Reorder the top-k reference patches and stack them along channels.

    Output shape: (B, K*C, H, W)
    """
    def __init__(self,
                 patch_size=8,
                 stride=8,
                 downsample_ratio=2,
                 is_norm=True,
                 norm_input=True,
                 ref_stride=2,
                 topk=3):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.downsample_ratio = downsample_ratio
        self.is_norm = is_norm
        self.norm_input = norm_input
        self.ref_stride = ref_stride if ref_stride is not None else stride
        self.topk = topk

    def reorder_single(self, feat_in: torch.Tensor, feat_ref: torch.Tensor):
        """
        feat_in  : (C, H, W)
        feat_ref : (C, H, W)
        """
        C, H, W = feat_in.shape
        device, dtype = feat_in.device, feat_in.dtype

        # Match and return top-k reference patch indices.
        # max_idx: (K, H_out, W_out)
        # max_val: (K, H_out, W_out)
        max_idx, max_val = feature_match_index_downsample_topk(
            feat_in, feat_ref,
            patch_size=self.patch_size,
            input_stride=self.stride,
            ref_stride=self.ref_stride,
            is_norm=self.is_norm,
            norm_input=self.norm_input,
            downsample_ratio=self.downsample_ratio,
            no_overlap_input=True,
            topk=3
        )
        
        K, H_out, W_out = max_idx.shape
        
        # patches_ref: (C, p, p, Nref)
        patches_ref = sample_patches(feat_ref, self.patch_size, self.ref_stride)

        # Arrange the selected patches in fold order.
        # (K, H_out, W_out) -> (H_out, W_out, K) -> (H_out*W_out*K,)
        idx_sel = max_idx.permute(1, 2, 0).reshape(-1) 
        
        selected = patches_ref[..., idx_sel]

        selected = selected.view(C, self.patch_size, self.patch_size, H_out * W_out, K)
        
        selected = selected.permute(4, 0, 1, 2, 3)
        
        selected = selected.reshape(K * C, self.patch_size, self.patch_size, H_out * W_out)

        cols = selected.reshape(1, (K * C) * self.patch_size * self.patch_size, -1)

        feat_ref_reordered = F.fold(
            cols,
            output_size=(H, W),
            kernel_size=self.patch_size,
            stride=self.patch_size
        ).squeeze(0)  # (K*C, H, W)

        return {
            'feat_ref_reordered': feat_ref_reordered.to(device=device, dtype=dtype),
            'max_idx': max_idx,   # (K, H_out, W_out)
            'max_val': max_val,   # (K, H_out, W_out)
        }

    def forward(self, feature_in, feature_ref):
        B, C, H, W = feature_in.shape
        reordered_list, idx_list, val_list = [], [], []

        for ind in range(B):
            feat_in  = feature_in[ind]
            feat_ref = feature_ref[ind]
            
            out = self.reorder_single(feat_in, feat_ref)
            
            reordered_list.append(out['feat_ref_reordered'].unsqueeze(0))
            idx_list.append(out['max_idx'])
            val_list.append(out['max_val'])

        reordered_ref = torch.cat(reordered_list, dim=0)
        return reordered_ref, idx_list, val_list
