import torch
import torch.nn as nn
import torch.nn.functional as F

from models.archs.base_arch import ContentExtractor
from models.archs.guided_enhance_arch import GuidedEnhanceNet
from models.archs.patch_reorder_arch import RefPatchReorderer_TopK
from models.archs.swin_transformer_arch import SwinBlock
from models.archs.tsa_fusion_arch import TSAFusionV2


class SimilarWindowCrossAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 embed_dim,
                 num_heads=4,
                 patch_size=8,
                 qkv_bias=True,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 auto_pad=True,
                 return_attn=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.patch_size = patch_size
        self.auto_pad = auto_pad
        self.return_attn = return_attn

        # Normalize each local token before projection.
        self.q_ln  = nn.LayerNorm(in_channels)
        self.kv_ln = nn.LayerNorm(in_channels)

        self.q_proj = nn.Linear(in_channels, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(in_channels, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(in_channels, embed_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(embed_dim, in_channels, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

        self.skip_proj = nn.Linear(in_channels, in_channels, bias=True)

        # self.diag_lambda = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(0.3))

        self.use_cosine = True

    def _unfold_no_overlap(self, x):
        """
        Split a feature map into non-overlapping windows.
        """
        B, C_total, H, W = x.shape
        P = self.patch_size
        
        # Pad to a full window grid when requested.
        pad_h = (P - H % P) % P
        pad_w = (P - W % P) % P
        if self.auto_pad and (pad_h or pad_w):
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            H_pad, W_pad = H + pad_h, W + pad_w
        else:
            H_pad = H - (H - P) % P
            W_pad = W - (W - P) % P
            x = x[:, :, :H_pad, :W_pad]
            
        # Output shape: (B, C_total * P * P, Nwin).
        cols = F.unfold(x, kernel_size=P, stride=P) 
        return cols, (H_pad, W_pad, pad_h if self.auto_pad else 0, pad_w if self.auto_pad else 0)

    def _fold_back(self, cols, H_pad, W_pad):
        P = self.patch_size
        x = F.fold(cols, output_size=(H_pad, W_pad), kernel_size=P, stride=P)
        return x

    def forward(self, x, g):
        """
        x: (B, C, H, W) - Query Image
        g: (B, K*C, H, W) - stacked reference candidates.
        """
        B, C, H, W = x.shape
        B_g, C_g, H_g, W_g = g.shape
        
        # Infer the number of reference candidates from the channel count.
        assert B == B_g, "Batch size mismatch"
        assert C_g % C == 0, f"Ref channels ({C_g}) must be multiple of Input channels ({C})"
        K = C_g // C  # K=3
        
        # -----------------------------------------------------------
        # 2. Window Partition (Unfold)
        # -----------------------------------------------------------
        x_cols, (H_pad, W_pad, pad_h, pad_w) = self._unfold_no_overlap(x) # (B, C*T, Nwin)
        g_cols, _ = self._unfold_no_overlap(g)                            # (B, K*C*T, Nwin)
        
        Nwin = x_cols.shape[-1]
        P = self.patch_size
        T = P * P  # e.g., 64

        # -----------------------------------------------------------
        # 3. Reshape and Expand Tokens
        # -----------------------------------------------------------
        # Query tokens: (B, Nwin, T, C).
        x_tokens = x_cols.transpose(1, 2).contiguous().view(B, Nwin, T, C)
        
        # Key/value tokens: split K candidates and append them along the token axis.
        # (B, Nwin, T, K*C) -> (B, Nwin, T, K, C) -> (B, Nwin, K, T, C) -> (B, Nwin, K*T, C)
        g_tokens = g_cols.transpose(1, 2).contiguous().view(B, Nwin, T, K*C)
        g_tokens = g_tokens.view(B, Nwin, T, K, C).permute(0, 1, 3, 2, 4).contiguous()
        g_tokens = g_tokens.view(B, Nwin, K * T, C) 

        # -----------------------------------------------------------
        # 4. Projection & Attention
        # -----------------------------------------------------------
        x_tokens_n = self.q_ln(x_tokens)     # (B, N, T, C)
        g_tokens_n = self.kv_ln(g_tokens)    # (B, N, K*T, C)
        
        q = self.q_proj(x_tokens_n)          # (B, N, T, E)
        k = self.k_proj(g_tokens_n)          # (B, N, K*T, E)
        v = self.v_proj(g_tokens_n)          # (B, N, K*T, E)
        
        # Multi-head split
        Hh = self.num_heads
        Dh = self.head_dim
        
        q = q.view(B, Nwin, T, Hh, Dh).permute(0, 1, 3, 2, 4)       # (B, N, Hh, T, Dh)
        k = k.view(B, Nwin, K * T, Hh, Dh).permute(0, 1, 3, 2, 4)   # (B, N, Hh, K*T, Dh)
        v = v.view(B, Nwin, K * T, Hh, Dh).permute(0, 1, 3, 2, 4)   # (B, N, Hh, K*T, Dh)
        
        # Cosine Similarity
        if self.use_cosine:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            
        # Attention Matrix: (B, N, Hh, T, K*T)
        # Query length is T; key length is K*T.
        attn = torch.matmul(q, k.transpose(-2, -1)) 
        
        # -----------------------------------------------------------
        # 5. Modified Diagonal Bias (Multi-Candidate)
        # -----------------------------------------------------------
        # Optional diagonal bias for same-position matches across candidates.
        
        # Single-candidate bias: (1, 1, 1, T, T).
        # eye = torch.eye(T, device=attn.device, dtype=attn.dtype).view(1, 1, 1, T, T)
        
        # Repeat across candidates: (1, 1, 1, T, K*T).
        # diag_bias = eye.repeat(1, 1, 1, 1, K) 
        
        # attn = attn + self.diag_lambda * diag_bias
        
        # Softmax & Aggregate
        attn = attn * self.scale
        attn = attn.softmax(dim=-1) # Softmax over K*T dimension
        attn = self.attn_drop(attn)
        
        # Output: (B, N, Hh, T, Dh) -> Aggregated from K*T values
        y = torch.matmul(attn, v) 
        
        # -----------------------------------------------------------
        # 6. Final Projection & Skip
        # -----------------------------------------------------------
        y = y.permute(0, 1, 3, 2, 4).contiguous().view(B, Nwin, T, Hh * Dh)
        y = self.proj(y)
        y = self.proj_drop(y)
        
        x_base = self.skip_proj(x_tokens_n)
        y = x_base + self.alpha * (y - x_base)
        
        # Fold back
        y_cols = y.view(B, Nwin, T * C).transpose(1, 2).contiguous()
        y_img = self._fold_back(y_cols, H_pad, W_pad)
        
        if self.auto_pad and (pad_h or pad_w):
            y_img = y_img[:, :, :H, :W]
            
        if self.return_attn:
            return y_img, attn
            
        return y_img


class LPENet(nn.Module):

    def __init__(self, ngf=128, depths=(4,4), num_heads=(4,4), window_size=8, use_checkpoint=True):
        super(LPENet, self).__init__()
        
        self.content_extractor = ContentExtractor(in_nc=3, out_nc=ngf, nf=16, n_blocks=4)
        
        self.fusion = TSAFusionV2(num_feat=ngf, num_frame=9, in_ch=3, embed_ch=32)
        
        self.enhancer = GuidedEnhanceNet(input_channel=128, c=128, width=128, blk_num=4)
        
        self.reorder = RefPatchReorderer_TopK(patch_size=8,stride=8,downsample_ratio=2,is_norm=True,norm_input=True,ref_stride=2, topk=3)
        self.cross_atten = SimilarWindowCrossAttention(
                in_channels=ngf,embed_dim=ngf,num_heads=4,
                patch_size=8,qkv_bias=True,attn_drop=0.0,
                proj_drop=0.0,auto_pad=True,return_attn=False)
        
        self.body = SwinBlock(embed_dim=ngf,depths=depths, 
                                num_heads=num_heads, window_size=window_size, 
                                use_checkpoint=use_checkpoint)
        self.conv_after_ca = nn.Conv2d(ngf*2, ngf, 3, 1, 1)
        self.conv_after_body = nn.Conv2d(ngf, ngf, 3, 1, 1)
        self.upconv1 = nn.Conv2d(ngf, 64 * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_hr = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv_last = nn.Conv2d(16, 3, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, img_lq, img_ref): 
        """
        Args:
            x (Tensor): the input image of SRNTT.
            maps (dict[Tensor]): the swapped feature maps on relu2_1 and relu1_1. 
          
            depths of the maps are 128 and 64 respectively.
        """
        h, w = img_ref.shape[2:]
        img_lq = F.interpolate(img_lq, size = (h // 2, w // 2), mode='bilinear')
        lq_feat = self.fusion(img_lq)
        
        ref_feat = self.content_extractor(img_ref)
        
        feat_enhanced, out_low = self.enhancer(lq_feat, ref_feat)
        reordered_ref, _, _ = self.reorder(feat_enhanced, ref_feat)
        
        out = self.cross_atten(feat_enhanced, reordered_ref)
        out = self.conv_after_ca(torch.cat([out, feat_enhanced], dim = 1))
        
        out = self.conv_after_body(self.body(out)) + out
        
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        
        return out, out_low
