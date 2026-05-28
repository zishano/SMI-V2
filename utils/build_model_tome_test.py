#transform timm vits to versions that can stop feeding forward specific patches

from types import MethodType
import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import Attention,Block
import math
from typing import Callable, Tuple

import torch
import random

def do_nothing(x, mode=None):
    return x

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)##Half-Attention

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge

def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size

def vit_attention_forward(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    # attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    del q, k

    # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B, N, C)
    del v
    x = self.proj(x)
    x = self.proj_drop(x)
    return x,attn,qkv[0].mean(1)

def vit_block_forward(self,x,size,r):
    # x_out,attn_out=self.attn(self.norm1(x))
    x_out,attn_out,metric=self.attn(self.norm1(x))
    # random_value = random.uniform(0, 1)
    # if random_value > 0.1:
    #     r = 2
    # else:
    #     r = 0
    r = r
    if r > 0:
        # Apply ToMe here
        merge, _ = bipartite_soft_matching(
            metric,
            r,
            True,
            False,
        )
        x, _ = merge_wavg(merge, x, size)
        x_out, size = merge_wavg(merge, x_out, size)
    
    x = x + (x_out)
    x = x + (self.mlp(self.norm2(x)))
    return x,attn_out,size

def vit_forward_features(self, x,current_abs_index,next_relative_index):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)##拼接一个维度
        x = x + self.pos_embed
        # sparse    
        if next_relative_index.shape[1]==current_abs_index.shape[1]:
            pass
        else:
            current_abs_index=torch.gather(current_abs_index,1,next_relative_index)##根据索引选择元素
            assert current_abs_index[0][0]==0

        x=torch.gather(x,1,current_abs_index.unsqueeze(-1).expand(-1,-1,x.size(-1)))##序列采样数据

        x = self.pos_drop(x)
        attn_weights = []
        size = None
        # r0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for blk in self.blocks:
            # r = r0.pop(0)
            r = 0
            x, attn, size= blk(x,size,r)
            # x, attn, size= blk(x,size)
            # x, attn = blk(x)
            attn_weights.append(attn)
        x = self.norm(x)[:, 0]

        return x,attn_weights,current_abs_index

def vit_forward(self,x,current_abs_index,next_relative_index):
    x,attn_out,current_abs_index = self.forward_features(x,current_abs_index,next_relative_index)
    x = self.head(x)
    return x,attn_out,current_abs_index





class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B


def build_model_tome(name, Pretrained=True):
    """
    Get a vision transformer model.

    This will insert
    current_abs_index (the absolute index of current patches)
    and next_relative_index  (the relative index of patches to retain)
    to the original input of attention.forward, block.forward/forward_feature, and net.forward

    Currently support almost all quantization in timm.quantization.transformers, including:
    - vit_tiny/small/base/large_patch16/patch32_224/384,
    - deit_tiny/small/base(_distilled)_patch16_224,
    """
    net = timm.create_model(name,pretrained=Pretrained)
    # ##使用tome对模型进行处理
    # import ToMe.tome as tome##
    # tome.patch.timm(net)
    # net.r = 16
    # ##使用tome对模型进行处理
    # print(net)
    for name, module in net.named_modules():
        if isinstance(module, Attention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(vit_attention_forward, module)
        if isinstance(module,Block):
            module.forward = MethodType(vit_block_forward, module)
            net.forward_features = MethodType(vit_forward_features, net)
            net.forward = MethodType(vit_forward, net)

    net = net.cuda()
    net.eval()
    # print(net)
    return net
