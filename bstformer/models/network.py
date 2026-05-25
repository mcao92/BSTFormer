import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce
from operator import mul
import einops

def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,dim=1,keepdim=True)
    return y

def At(y,Phi):
    x = y*Phi
    return x

class FFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv3d(dim, dim, 3,padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, dim, 1),
        )

    def forward(self, x):
        x = einops.rearrange(x,"b d h w c->b c d h w")
        x = self.ffn(x)
        x = einops.rearrange(x,"b c d h w->b d h w c")
        return x

class Conv1x1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(dim, dim, 1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = einops.rearrange(x,"b d h w c->b c d h w")
        x = self.conv(x)
        x = einops.rearrange(x,"b c d h w->b d h w c")
        return x

def grid_partition(x,grid_size):
    gh,gw = grid_size[1],grid_size[2]
    x = einops.rearrange(x," b d (gh h) (gw w) c->(b d h w) (gh gw) c",gh=gh,gw=gw)
    return x

def grid_reverse(windows,grid_size, B, D, H):
    gh = grid_size[1]
    x = einops.rearrange(windows,"(b d h w) (gh gw) c->b d (gh h) (gw w) c",b=B,d=D,h=H//gh,gh=gh)
    return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

def get_window_size(window_size):
    use_window_size = list(window_size)
    return tuple(use_window_size)

class TimeAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        # C=C//2
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SpaceAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, flag=False):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class BSTFormerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=(1,7,7),
                 qkv_bias=True, qk_scale=None,
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.blockattn = SpaceAttention(
            dim//3, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale)

        self.gridattn = SpaceAttention(
            dim//3, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale)

        self.ffn_block = FFN(dim//3)
        self.ffn_grid = FFN(dim//3)
        self.ffn_time = FFN(dim//3)

        self.time_attn = TimeAttention(
            dim//3,num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            )

        self.conv = Conv1x1(dim)

    def st_attention(self, x):
        B, D, H, W, C = x.shape
        window_size = get_window_size(self.window_size)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape

        x1,x2,x3 = torch.chunk(x,chunks=3,dim=-1)
        C=C//3

        windows_x = window_partition(x1, window_size)
        attn_windows = self.blockattn(windows_x)
        attn_windows = attn_windows.view(-1, *(window_size+(C,)))
        windows_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)
        windows_x = x1 + windows_x
        windows_x = self.ffn_block(windows_x)+windows_x

        x2 = x2 + windows_x
        grids_x = grid_partition(x2,window_size)
        grids_x = self.gridattn(grids_x,flag=True)
        grids_x = grid_reverse(grids_x,self.window_size,B,Dp,Hp)
        grids_x = x2+grids_x
        grids_x = self.ffn_grid(grids_x)+grids_x

        x3 = x3 + grids_x
        times_x = einops.rearrange(x3,"b d h w c->(b h w) d c")
        times_x = self.time_attn(times_x)
        times_x = einops.rearrange(times_x,"(b h w) d c -> b d h w c",h=Hp ,w=Wp)
        times_x = x3 + times_x
        times_x = self.ffn_time(times_x)+times_x

        x = torch.cat([windows_x,grids_x,times_x],dim=-1)

        if pad_d1 >0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward(self, x):
        shortcut = x
        x = self.st_attention(x)
        x = self.conv(x)+shortcut
        return x

class BSTFormerLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(1,7,7),
                 qkv_bias=False,
                 qk_scale=None,
                 ):
        super().__init__()
        self.window_size = window_size
        self.depth = depth

        self.blocks = nn.ModuleList([
            BSTFormerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
            )
            for i in range(depth)])

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = einops.rearrange(x, 'b c d h w -> b d h w c')
        for blk in self.blocks:
            x = blk(x)
        x = x.view(B, D, H, W, -1)
        x = einops.rearrange(x, 'b d h w c -> b c d h w')
        return x

class BSTFormer(nn.Module):
    def __init__(self,color_channels,units,dim=48):
        super(BSTFormer, self).__init__()
        self.color_channels = color_channels
        self.conv_first = nn.Sequential(
            nn.Conv3d(1, dim, kernel_size=(3,7,7), stride=1,padding=(1,3,3)),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, dim*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim*2, dim*4, kernel_size=3, stride=(1,2,2), padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv_last = nn.Sequential(
            nn.ConvTranspose3d(dim*4, dim*2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), output_padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim*2, dim, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, color_channels, kernel_size=3, stride=1, padding=1)
        )

        self.layers = nn.ModuleList()
        for i in range(units):
            bstformer_layer = BSTFormerLayer(
                    dim=dim*4,
                    depth=2,
                    num_heads=4,
                    window_size=(1,7,7),
                    qkv_bias=True,
                    qk_scale=None,
            )
            self.layers.append(bstformer_layer)

    def bayer_init(self,y,Phi,Phi_s):
        bayer = [[0,0], [0,1], [1,0], [1,1]]
        b,f,h,w = Phi.shape
        y_bayer = torch.zeros(b,1,h//2,w//2,4).to(y.device)
        Phi_bayer = torch.zeros(b,f,h//2,w//2,4).to(y.device)
        Phi_s_bayer = torch.zeros(b,1,h//2,w//2,4).to(y.device)
        for ib in range(len(bayer)):
            ba = bayer[ib]
            y_bayer[...,ib] = y[:,:,ba[0]::2,ba[1]::2]
            Phi_bayer[...,ib] = Phi[:,:,ba[0]::2,ba[1]::2]
            Phi_s_bayer[...,ib] = Phi_s[:,:,ba[0]::2,ba[1]::2]
        y_bayer = einops.rearrange(y_bayer,"b f h w ba->(b ba) f h w")
        Phi_bayer = einops.rearrange(Phi_bayer,"b f h w ba->(b ba) f h w")
        Phi_s_bayer = einops.rearrange(Phi_s_bayer,"b f h w ba->(b ba) f h w")

        x = At(y_bayer,Phi_bayer)
        yb = A(x,Phi_bayer)
        x = x + At(torch.div(y_bayer-yb,Phi_s_bayer),Phi_bayer)
        x = einops.rearrange(x,"(b ba) f h w->b f h w ba",b=b)
        x_bayer = torch.zeros(b,f,h,w).to(y.device)
        for ib in range(len(bayer)):
            ba = bayer[ib]
            x_bayer[:,:,ba[0]::2, ba[1]::2] = x[...,ib]
        x = x_bayer.unsqueeze(1)
        return x

    def forward(self, y, Phi, Phi_s):
        out_list = []
        xe_list = []
        if self.color_channels==3:
            x = self.bayer_init(y,Phi,Phi_s)
        else:
            x = At(y,Phi)
            yb = A(x,Phi)
            x = x + At(torch.div(y-yb,Phi_s),Phi)

            x_e = x

            x = x.unsqueeze(1)



        out = self.conv_first(x)
        for layer in self.layers:
            out = layer(out)
            # break
        out = self.conv_last(out)

        if self.color_channels!=3:
            out = out.squeeze(1)
        out_list.append(out)
        xe_list.append(x_e)
        # return out_list, x_e
        return out_list
