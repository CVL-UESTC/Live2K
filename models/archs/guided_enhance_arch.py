import torch
import torch.nn as nn
import torch.nn.functional as F

from models.archs.base_arch import DepthwiseConv, LayerNorm2d, SimpleGate


class DynamicNAFBlock(nn.Module):
    """
    NAF-style block with FiLM modulation on both residual branches.
    """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand

        # branch 1
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = DepthwiseConv(dw_channel, 3, 1, 1)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, bias=True),
        )
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, bias=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # branch 2 (FFN)
        ffn_channel = FFN_Expand * c
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, bias=True)
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, x, beta_map, gamma_map):
        # branch 1
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg(y)                      # (B, dw//2, H, W)
        y = y * self.sca(y)
        y = self.conv3(y)
        y = self.dropout1(y)
        y = x + y * beta_map                # FiLM1

        # branch 2
        z = self.conv4(self.norm2(y))
        z = self.sg2(z)
        z = self.conv5(z)
        z = self.dropout2(z)
        out = y + z * gamma_map             # FiLM2
        return out


class CrossAttention(nn.Module):
    def __init__(self, in_channels, head_count=4):
        super().__init__()
        self.head_count = head_count
        self.head_dim = in_channels // head_count

        self.query_proj = nn.Conv2d(in_channels, in_channels, 1)
        self.key_proj = nn.Conv2d(in_channels, in_channels, 1)
        self.value_proj = nn.Conv2d(in_channels, in_channels, 1)

        self.out_proj = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, inp_feat, guide_feat):
        B, C, H, W = inp_feat.shape

        # Query from input feature
        query = self.query_proj(inp_feat).view(B, self.head_count, self.head_dim, H * W).permute(0, 1, 3, 2) # (B, h, N, d)
        # Key, Value from guide feature
        guide_H, guide_W = guide_feat.shape[2:]
        key = self.key_proj(guide_feat).view(B, self.head_count, self.head_dim, guide_H * guide_W) # (B, h, d, M)
        value = self.value_proj(guide_feat).view(B, self.head_count, self.head_dim, guide_H * guide_W).permute(0, 1, 3, 2) # (B, h, M, d)

        # (B, h, N, d) @ (B, h, d, M) -> (B, h, N, M)
        attention_scores = torch.matmul(query, key) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # (B, h, N, M) @ (B, h, M, d) -> (B, h, N, d)
        attended_feat = torch.matmul(attention_weights, value)

        attended_feat = attended_feat.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)

        return self.out_proj(attended_feat)


class GuidedEnhanceNet(nn.Module):
    def __init__(self, input_channel=128, c=128, width=128, blk_num=4):
        """
        Single-scale guided enhancer.

        Args:
            input_channel: Number of input feature channels.
            c: Channel count used for the guidance features.
            width: Internal feature width.
            blk_num: Number of DynamicNAFBlock layers.
        """
        super().__init__()

        self.cross_attention = CrossAttention(in_channels=c)

        self.guidance_processor = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c, c, 3, 1, 1)
        )

        self.intro = nn.Conv2d(in_channels=input_channel, out_channels=width, kernel_size=3, padding=1, bias=True)

        # Single-resolution FiLM-modulated backbone.
        self.blocks = nn.ModuleList([DynamicNAFBlock(width) for _ in range(blk_num)])
        
        # self.ending = nn.Conv2d(in_channels=width, out_channels=3, kernel_size=3, padding=1, bias=True)
        self.ending = nn.Sequential(
            nn.Conv2d(width, 3, 3, 1, 1)
        )

        self.map_head = nn.Conv2d(c, 2 * width, 3, 1, 1)
        self.downsample = nn.Upsample(scale_factor=0.125, mode='bilinear', align_corners=False)
        self.padder_size = 8

    def _make_beta_gamma(self, guidance_feat: torch.Tensor, head: nn.Module, size=None):
        """
        guidance_feat: (B, c, H_g, W_g)
        head: Conv2d(c, 2*width, 3,1,1)
        size: Target spatial size (H, W).
        return: beta_map (tanh), gamma_map (sigmoid)
        """
        if size is not None:
            H, W = size
            if guidance_feat.shape[2] != H or guidance_feat.shape[3] != W:
                guidance_feat = F.interpolate(guidance_feat, size=(H, W), mode='bilinear', align_corners=False)
        maps = head(guidance_feat)
        beta_map, gamma_map = maps.chunk(2, dim=1)
        beta_map = torch.tanh(beta_map)
        gamma_map = torch.sigmoid(gamma_map)
        return beta_map, gamma_map
    def check_image_size(self, x: torch.Tensor):
        """Reflect-pad the input so height and width are divisible by padder_size."""
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if mod_pad_h == 0 and mod_pad_w == 0:
            return x
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')

    def forward(self, inp, guide):
        B, C, h, w = inp.shape
        
        inp   = self.check_image_size(inp)
        guide = self.check_image_size(guide)

        # Build low-resolution guidance features for the FiLM heads.
        feat_guide_low = self.downsample(guide)
        feat_inp_low   = self.downsample(inp)
        attended_feat  = self.cross_attention(feat_inp_low, feat_guide_low)
        base_guidance_feat = self.guidance_processor(attended_feat)  # (B, c, H', W')

        feat = self.intro(inp)
        shortcut = feat
        H, W = feat.shape[2:]
        beta_map, gamma_map = self._make_beta_gamma(base_guidance_feat, self.map_head, (H, W))

        for block in self.blocks:
            feat = block(feat, beta_map, gamma_map)

        feat = feat + shortcut  
        out = self.ending(feat)
        return feat[:, :, :h, :w], out[:, :, :h, :w]
