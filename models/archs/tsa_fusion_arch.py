import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3x3(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, 1, 1, bias=bias)
    def forward(self, x):
        return self.conv(x)


class DepthwiseSeparableConv3D(nn.Module):
    """Lightweight 3D depthwise-separable convolution."""
    def __init__(self, c_in, c_out, k=(3,3,3), stride=1, padding=1):
        super().__init__()
        self.dw = nn.Conv3d(c_in, c_in, kernel_size=k, stride=stride, padding=padding,
                            groups=c_in, bias=True)
        self.pw = nn.Conv3d(c_in, c_out, kernel_size=1, bias=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        x = self.dw(x)
        x = self.act(x)
        x = self.pw(x)
        x = self.act(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, c, r=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, c // r, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.avg(x)
        w = self.fc(w)
        return x * w


class GlobalContextBlock2D(nn.Module):
    """
    Lightweight global-context block.

    It pools a spatial context vector with softmax attention, then uses small
    channel MLPs to produce additive and/or multiplicative modulation.
    """
    def __init__(self, in_channels, reduction=16, add=True, mul=True):
        super().__init__()
        self.add = add
        self.mul = mul
        mid = max(1, in_channels // reduction)

        self.attn = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)

        if self.mul:
            self.transform_mul = nn.Sequential(
                nn.Conv2d(in_channels, mid, 1, bias=True),
                nn.LayerNorm([mid, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, in_channels, 1, bias=True),
                nn.Sigmoid()
            )
        if self.add:
            self.transform_add = nn.Sequential(
                nn.Conv2d(in_channels, mid, 1, bias=True),
                nn.LayerNorm([mid, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, in_channels, 1, bias=True)
            )

    def forward(self, x):
        B, C, H, W = x.shape
        attn_map = self.attn(x).view(B, 1, -1)            # (B,1,HW)
        attn = torch.softmax(attn_map, dim=-1)            # (B,1,HW)

        x_flat = x.view(B, C, -1)                         # (B,C,HW)
        context = torch.bmm(x_flat, attn.transpose(1,2))  # (B,C,1)
        context = context.view(B, C, 1, 1)                # (B,C,1,1)

        out = x
        if self.mul:
            scale = self.transform_mul(context)           # (B,C,1,1), in (0,1)
            out = out * (1.0 + scale)
        if self.add:
            shift = self.transform_add(context)           # (B,C,1,1)
            out = out + shift
        return out


class TemporalPixelAttentionLite(nn.Module):
    """
    Per-pixel temporal attention.

    Input and output shape: (B, T, C, H, W). The reference frame defaults to
    the temporal center frame.
    """
    def __init__(self, c_in, embed_ratio=0.5, ref_index=None):
        super().__init__()
        d = max(1, int(c_in * embed_ratio))
        self.embed = nn.Conv2d(c_in, d, kernel_size=1, bias=True)
        self.ref_index = ref_index
        self.scale = d ** -0.5

    def forward(self, feat):  # (B,T,C,H,W)
        B, T, C, H, W = feat.shape
        ref_idx = self.ref_index if self.ref_index is not None else (T // 2)

        x = feat.reshape(B*T, C, H, W)
        emb = self.embed(x)                        # (B*T, d, H, W)
        d = emb.shape[1]
        emb = F.normalize(emb, dim=1, eps=1e-6)
        emb = emb.view(B, T, d, H, W)

        emb_ref = emb[:, ref_idx]                  # (B, d, H, W)

        # Per-pixel dot-product similarity to the reference frame.
        # sim[b,t,h,w] = <emb[b,t,:,h,w], emb_ref[b,:,h,w]>
        sim = (emb * emb_ref.unsqueeze(1)).sum(dim=2) * self.scale  # (B, T, H, W)

        # softmax over T
        attn = torch.softmax(sim, dim=1).unsqueeze(2)               # (B, T, 1, H, W)

        out = feat * attn                                           # (B, T, C, H, W)
        return out


class SpatioTemporal3DRefiner(nn.Module):
    """
    Refine (B, T, C, H, W) features with 3D convolutions.
    """
    def __init__(self, c_in, c_mid=None, num_blocks=2):
        super().__init__()
        c_mid = c_mid or c_in
        blocks = [DepthwiseSeparableConv3D(c_in, c_mid)]
        for _ in range(num_blocks-1):
            blocks.append(DepthwiseSeparableConv3D(c_mid, c_mid))
        self.net = nn.Sequential(*blocks)
        self.proj = nn.Conv3d(c_mid, c_in, 1, bias=True)

    def forward(self, x):  # (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)
        y = self.net(x)
        y = self.proj(y)
        y = x + y
        y = y.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
        return y


class SpatialAttentionPyramid(nn.Module):
    """
    Pyramid spatial attention with global context and SE modulation.
    """
    def __init__(self, c):
        super().__init__()
        
        self.context_block = GlobalContextBlock2D(in_channels=c, reduction=16, add=True, mul=True)

        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.spa1 = nn.Conv2d(c, c, 1, bias=True)
        self.down = nn.ModuleList([
            nn.MaxPool2d(3, 2, 1),
            nn.AvgPool2d(3, 2, 1)
        ])
        self.mix1 = nn.Conv2d(2*c, c, 1, bias=True)
        self.conv3x3 = Conv3x3(c, c)
        self.l1 = nn.Conv2d(c, c, 1, bias=True)
        self.mix2 = nn.Conv2d(2*c, c, 3, 1, 1, bias=True)
        self.conv3x3_2 = Conv3x3(c, c)

        self.mask = nn.Conv2d(c, c, 3, 1, 1, bias=True)
        self.bias = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c, c, 1, bias=True),
        )
        self.se = SEBlock(c)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.context_block(x)

        attn = self.lrelu(self.spa1(x))
        a1 = self.down[0](attn)  # max
        a2 = self.down[1](attn)  # avg
        attn = self.lrelu(self.mix1(torch.cat([a1, a2], dim=1)))
        attn = self.lrelu(self.conv3x3(attn))

        attn_l1 = self.lrelu(self.l1(attn))
        b1 = self.down[0](attn_l1)
        b2 = self.down[1](attn_l1)
        attn_l = self.lrelu(self.mix2(torch.cat([b1, b2], dim=1)))
        attn_l = self.lrelu(self.conv3x3_2(attn_l))
        attn_l = F.interpolate(attn_l, size=attn.shape[-2:], mode='bilinear', align_corners=False)

        attn = self.lrelu(self.conv3x3(attn) + attn_l)
        attn = self.se(attn)
        attn = F.interpolate(attn, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        m = torch.sigmoid(self.mask(attn))
        a = self.bias(attn)
        out = x * (2.0 * m) + a
        return out


class TSAFusionV2(nn.Module):
    """
    Fuse a stack of frames into a half-resolution feature map.

    Input shape is flattened as (B, T*C, H, W), with T=num_frame.
    Output shape is (B, num_feat, H/2, W/2).
    """
    def __init__(self,
                 num_feat=128,
                 num_frame=9,
                 in_ch=3,
                 embed_ch=32):
        super().__init__()
        self.num_frame = num_frame
        self.center = num_frame // 2

        self.head = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(in_ch * 4, embed_ch, 3, 1, 1),
            nn.LeakyReLU(0.1, True)
        )

        self.temporal_att = TemporalPixelAttentionLite(c_in=embed_ch, embed_ratio=0.5, ref_index=None)

        self.refiner_3d = SpatioTemporal3DRefiner(c_in=embed_ch, c_mid=embed_ch, num_blocks=2)

        self.fuse_1x1 = nn.Conv2d(num_frame * embed_ch, num_feat, 1, 1, bias=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        self.spatial_attn = SpatialAttentionPyramid(num_feat)

    def forward(self, x):
        
        B, _, H, W = x.shape
        x = x.reshape(B, 9, 3, H, W)
        
        B, T, C, H, W = x.shape
        assert T == self.num_frame, f"Expect T={self.num_frame}, got {T}"

        x = x.view(B*T, C, H, W)
        feat = self.head(x)                      # (B*T, 32, H/2, W/2)
        h, w = feat.shape[-2:]
        feat = feat.view(B, T, 32, h, w)         # (B, T, 32, h, w)

        feat = self.temporal_att(feat)            # (B, T, 32, h, w)

        feat = self.refiner_3d(feat)             # (B, T, 32, h, w)

        feat_cat = feat.permute(0, 2, 1, 3, 4).contiguous()     # (B, 32, T, h, w)
        feat_cat = feat_cat.view(B, 32*T, h, w)                 # (B, 32*T, h, w)
        fused = self.act(self.fuse_1x1(feat_cat))               # (B, num_feat, h, w)

        out = self.spatial_attn(fused)                          # (B, num_feat, h, w)
        return out
