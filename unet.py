import math # positional embeddingsのsin, cos用
from inspect import isfunction # inspectモジュール
from functools import partial # 関数の引数を一部設定できる便利ツール
 
# PyTorch, 計算関係
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

import matplotlib.pyplot as plt
 
timesteps = 1000
time = torch.arange(timesteps)

def exists(x):
    return x is not None  

def default(val, d):
    """
    If val is not None, returns True. 
    If val is None, returns the result of a function call if d is a function, or returns its value if d is not a function.
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device 
        half_dim = self.dim // 2 # Half of the dim
        embeddings = math.log(10000) / (half_dim - 1) 
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Residual(nn.Module):
  """
  Residual Connection
  """
  def __init__(self, fn):
    super().__init__()
    self.fn = fn
 
  def forward(self, x, *args, **kwargs):
    """
    f(x) + x
    """
    return self.fn(x, *args, **kwargs) + x
  
class UpsampleConv(nn.Module):
  """
  Transposed Convolution for upsample
  """
  def __init__(self, dim):
    super().__init__()
    self.trans_conv = nn.ConvTranspose2d(
      in_channels=dim, 
      out_channels=dim, 
      kernel_size=4, 
      stride=2, 
      padding=1 
    )
 
  def forward(self, x):
    return self.trans_conv(x)
  
class DownsampleConv(nn.Module):
  """
  Convolution for Downsample
  """
  def __init__(self, dim):
    super().__init__()
    self.conv = nn.Conv2d(
      in_channels=dim, 
      out_channels=dim, 
      kernel_size=4, 
      stride=2, 
      padding=1
    )
 
  def forward(self, x):
    return self.conv(x)
  
class ConvBlock(nn.Module):
  def __init__(self, dim, dim_out, groups=8):
    super().__init__()
    self.proj = nn.Conv2d(dim, dim_out, 3, padding=1) # conv
    self.norm = nn.GroupNorm(groups, dim_out) # normalization
    self.act = nn.SiLU() # activation
 
  def forward(self, x):
    x = self.proj(x) # conv
    x = self.norm(x) # normalization
    x = self.act(x) # activation
    return x
  
class ResnetBlock(nn.Module):
  def __init__(self, dim, dim_out, time_emb_dim = None, groups=8):
    super().__init__()
    # 時点情報(time_emb)
    if exists(time_emb_dim):
      self.mlp = (
          nn.Sequential(
            nn.SiLU(), 
            nn.Linear(time_emb_dim, dim_out)
          )
      )
    else:
      self.mlp = (None)
      
    # 画像(x)の処理
    self.block1 = ConvBlock(dim, dim_out, groups=groups)
    self.block2 = ConvBlock(dim_out, dim_out, groups=groups)
    if dim != dim_out: # インプットとアウトプットのサイズが違えばconv
      self.res_conv = nn.Conv2d(dim, dim_out, 1)
    else:
       self.res_conv = nn.Identity()
 
  def forward(self, x, time_emb=None):
    # conv1
    h = self.block1(x) 
     
    # time embの付加
    if exists(self.mlp) and exists(time_emb):
      time_emb = self.mlp(time_emb)
      h = rearrange(time_emb, "b c -> b c 1 1") + h 
 
    # conv 2
    h = self.block2(h)
     
    # conv + 残差結合
    return h + self.res_conv(x)
  
class Unet(nn.Module):
  def __init__(
    self,
    dim,
    init_dim=None,
    out_dim=None,  
    dim_mults=(1, 2, 4, 8),
    channels=3,
    with_time_emb=True,
    resnet_block_groups=8,
  ):
    super().__init__()
 
    self.channels = channels
    init_dim = default(init_dim, dim // 3 * 2)
    self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)
 
    dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
    in_out = list(zip(dims[:-1], dims[1:])) # (input_dim, output_dim)というタプルのリストを作成する

    resnet_block = partial(ResnetBlock, groups=resnet_block_groups)
 
    # time embeddings
    if with_time_emb:
      time_dim = dim
      # time_mlp: pos emb -> Linear -> GELU -> Linear
      self.time_mlp = nn.Sequential(
          SinusoidalPositionEmbeddings(dim),
          nn.Linear(dim, time_dim),
          nn.GELU(),
          nn.Linear(time_dim, time_dim)
      )
    else:
      time_dim = None
      self.time_mlp = None
 
    self.downs = nn.ModuleList([])
    self.ups = nn.ModuleList([])
    num_resolutions = len(in_out) # blockを処理する回数
    
    # ダウンサンプル
    for ind, (dim_in, dim_out) in enumerate(in_out):
      is_last = ind >= (num_resolutions - 1)
 
      self.downs.append(
          nn.ModuleList(
              [
                  resnet_block(dim_in, dim_out, time_emb_dim=time_dim),
                  resnet_block(dim_out, dim_out, time_emb_dim=time_dim),
                  #Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                  DownsampleConv(dim_out) if not is_last else nn.Identity(),
                
              ]
          )
      )
 
    # 中間ブロック
    mid_dim = dims[-1]
    self.mid_block1 = resnet_block(mid_dim, mid_dim, time_emb_dim=time_dim)
    #self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
    self.mid_block2 = resnet_block(mid_dim, mid_dim, time_emb_dim=time_dim)
 
    # アップサンプル
    for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
      is_last = ind >= (num_resolutions - 1)
 
      self.ups.append(
          nn.ModuleList(
              [
                resnet_block(dim_out * 2, dim_in, time_emb_dim=time_dim),
               resnet_block(dim_in, dim_in, time_emb_dim=time_dim),
               #Residual(PreNorm(dim_in, LinearAttention(dim_in))),
               UpsampleConv(dim_in) if not is_last else nn.Identity(),
              ]
          )
      )
    out_dim = default(out_dim, channels)
    self.final_conv = nn.Sequential(
        resnet_block(dim, dim),
        nn.Conv2d(dim, out_dim, 1)
    )
 
  def forward(self, x, time):
    x = self.init_conv(x)
    t = self.time_mlp(time) if exists(self.time_mlp) else None
    h = []
    
    # ダウンサンプル
    for block1, block2, downsample in self.downs:
      x = block1(x, t)
      x = block2(x, t)
      #x = attn(x)
      h.append(x)
      x = downsample(x)
 
 
    # 中間
    x = self.mid_block1(x, t)
    #x = self.mid_attn(x)
    x = self.mid_block2(x, t)
 
    # アップサンプル
    for block1, block2, upsample in self.ups:
      x = torch.cat((x, h.pop()), dim=1) # downsampleで計算したhをくっつける
      x = block1(x, t)
      x = block2(x, t)
      #x = attn(x)
      x = upsample(x)
 
    return self.final_conv(x)
  
if __name__ == "__main__" :
    image_size = 128
    channels = 3
    batch_size = 8
    timesteps = 200
    model = Unet(
        dim=image_size,
        dim_mults=(1, 2, 4, 8),
        channels=channels,
        with_time_emb=True,
        resnet_block_groups=2,
    )
    data = torch.randn((batch_size, channels, image_size, image_size))
    t = torch.randint(0, timesteps, (batch_size,)).long()
    output = model(data, t)
    print(output.size())
