from torch import optim
from tqdm import tqdm
import pickle
import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_
from pytorch_pretrained_vit import ViT
import os



'''
Novelty: feeding q into kv iteratively

We use the content sequence to generate the query Q, 
and use the style sequence to generate the key K and the value V

'''



def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


class FeedForward(nn.Module):
    def __init__(self, emb_size, hidden_size, dropout=0.1, add_norm=True):
        super().__init__()
        self.add_norm = add_norm

        self.fc_liner = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.GELU(),
            # nn.Dropout(p=dropout),
            nn.Linear(hidden_size, emb_size),
            nn.Dropout(p=dropout),
        )

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-6)

    def forward(self, x):
        out = self.fc_liner(x)
        if self.add_norm:
            return self.LayerNorm(x + out)
        return out


class CrossAttention(nn.Module):
    def __init__(self, dim=768, num_heads=8, 
                 qkv_bias=False, 
                 proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        assert (
            self.head_dim * num_heads == dim
        ), "Embedding size needs to be divisible by heads"

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self,q,kv): 
        N = kv.shape[0] # Number of samples
        seq_len = kv.shape[1] # Sequence length

        # Split the embedding into self.heads different pieces
        values = self.wv(kv) ## (N, seq_len, embed_size)
        keys = self.wk(kv) ## (N, seq_len, embed_size)
        queries = self.wq(q) ## (N, seq_len, embed_size)

        values = values.view(N, seq_len, self.num_heads, self.head_dim).transpose(1,2) ## (N, heads, seq_len, head_dim)
        keys = keys.view(N, seq_len, self.num_heads, self.head_dim).transpose(1,2) ## (N, heads, seq_len, head_dim)
        queries = queries.view(N, seq_len, self.num_heads, self.head_dim).transpose(1,2) ## (N, heads, seq_len, head_dim)

        # Compute the dot product between queries and keys to get the similarity matrix
        attention = torch.einsum("nhqd,nhkd->nhqk", [queries, keys]) # (QK^T)
        attention = F.softmax(attention / (self.embed_size ** (1 / 2)), dim=-1) ## (1/sqrt(d_k)) * QK^T, (N, heads, seq_len, seq_len)

        # Multiply by values
        out = torch.einsum("nhqk,nhkd->nhqd", [attention, values]) # softmax(QK^T)V, (N, heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(N, seq_len, self.embed_size) ## (N, seq_len, embed_size)

        out = self.proj(out)
        out = self.proj_drop(out)

        return out # batch_size, num_patches, 768


class FusionBlock(nn.Module):
    def __init__(self, styl_embed=768,
                 content_embed=768,
                 hidden_size = 768, 
                 num_heads=12, 
                 norm_layer=nn.LayerNorm,
                 self_attn_model=None,
                 has_mlp = True,
                 curr_layer=0):
        super().__init__()

        self.has_mlp = has_mlp

        self.style_proj = nn.Linear(styl_embed,hidden_size)
        self.content_proj = nn.Linear(content_embed,hidden_size)

        self.style_norm = norm_layer(hidden_size)
        self.content_norm = norm_layer(hidden_size)

        self.cross_attn = CrossAttention(dim=hidden_size,
                              num_heads=num_heads, 
                              proj_drop=0.1)
        self.fusion_norm = norm_layer(hidden_size)
        
        self.self_attn = self_attn_model.transformer.blocks[curr_layer].eval()
    
        if self.has_mlp:
            self.ffn = FeedForward(emb_size=hidden_size,hidden_size=hidden_size*4)


    def forward(self,style_feats,content_feats,mask=None):
        # self_attention
        style_feats = self.self_attn(self.style_norm(style_feats),mask=mask)
        content_feats = self.self_attn(self.content_norm(content_feats),mask=mask)

        fusion_feats = self.cross_attn(q=content_feats,kv=style_feats)
        
        if self.has_mlp:
            fusion_feats = fusion_feats + self.ffn(self.fusion_norm(fusion_feats))

        return fusion_feats,content_feats  # N x num_patches x 768

class CrossStyTr(nn.Module):
    def __init__(
        self,style_embed=768,
        content_embed=768,
        hidden_size = 768,
        num_heads=12,
        img_size = 256,
        norm_layer = nn.LayerNorm,
        device='cuda',
        depth = 12,
        has_mlp = True,
    ):
        super().__init__()

        self.device = device
        # Image encoder specifics
        # ViT default patch embeddings
        self.vit = ViT('B_16_imagenet1k', pretrained=True,image_size=img_size).to(self.device) # construct and load 
        self.vit.fc = None
        freeze_model(self.vit)

        self.new_ps = nn.Conv2d(img_size , img_size , (1,1))
        self.averagepooling = nn.AdaptiveAvgPool2d(9) # change 18 to 9

        self.fusion = nn.ModuleList([
            FusionBlock(styl_embed=style_embed,
                        content_embed = content_embed,
                        self_attn_model = self.vit,
                        hidden_size=hidden_size, 
                        num_heads=num_heads, 
                        norm_layer=norm_layer,
                        has_mlp = has_mlp,
                        curr_layer=curr_layer) for curr_layer in range(depth)])
        
        self.content_cls = nn.Parameter(torch.zeros(1, 1, hidden_size,dtype=torch.float32))
        self.style_cls = nn.Parameter(torch.zeros(1, 1, hidden_size,dtype=torch.float32))

       

    def forward(self,content_img,style_img,mask=None):
        '''
        @param img: (N, 3, 256, 256)
        @od        
        '''
        if content_img.shape != style_img.shape:
            print('content image shape: ',content_img.shape)
            print('style image shape: ',style_img.shape)
            raise ValueError("Error: content_img and style_img must have the same shape")

        N,_,H,W = content_img.shape
        
        # content-aware positional embedding
        content_pool = self.averagepooling(content_img)       
        pos_c = self.new_ps(content_pool)
        pos_embed_c = F.interpolate(pos_c, mode='bilinear',size= style_img.shape[-2:])

        ###flatten NxCxHxW to HWxNxC     
        style_img = style_img.flatten(2).permute(2, 0, 1)
      
        content_img = content_img.flatten(2).permute(2, 0, 1)
        if pos_embed_c is not None:
            pos_embed_c = pos_embed_c.flatten(2).permute(2, 0, 1)
            print('shape of pos_embed_c',pos_embed_c.shape)
        
        content_img = content_img + pos_embed_c

        for blk in self.fusion:
            style_feats,conten_feats = blk(style_feats=style_img,content_feats=content_img)
        
        return style_feats,conten_feats



