from torch import optim
from tqdm import tqdm
import pickle
import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
import torch.nn.functional as F

from function import normal,normal_style
from function import calc_mean_std


from torch.nn.utils import clip_grad_norm_
from pytorch_pretrained_vit import ViT
import os



'''
Novelty: feeding q into kv iteratively

We use the content sequence to generate the query Q, 
and use the style sequence to generate the key K and the value V

'''

# decoder = nn.Sequential(
#     nn.Conv2d(512, 256, (3, 3)), 
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 256, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(256, 128, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 128, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(128, 64, (3, 3)),
#     nn.ReLU(),
#     nn.Upsample(scale_factor=2, mode='nearest'),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 64, (3, 3)),
#     nn.ReLU(),
#     nn.ReflectionPad2d((1, 1, 1, 1)),
#     nn.Conv2d(64, 3, (3, 3)),
# )

decoder = nn.Sequential(
    nn.Conv2d(768, 256, (3, 3), padding=1),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),  # 16x16 -> 32x32
    nn.Conv2d(256, 256, (3, 3), padding=1),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32 -> 64x64
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),  # 64x64 -> 128x128
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),  # 128x128 -> 256x256
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


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
        
        self.embed_size = dim

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

        #self.cross_attn = CrossAttention(dim=hidden_size,
        #                      num_heads=num_heads, 
        #                      proj_drop=0.1)
        self.fusion_norm = norm_layer(hidden_size)
        
        self.self_attn = self_attn_model.transformer.blocks[curr_layer].eval()
    
        if self.has_mlp:
            self.ffn = FeedForward(emb_size=hidden_size,hidden_size=hidden_size*4)


    def forward(self,style_feats,content_feats,mask=None):
        # self_attention
        style_feats = self.self_attn(self.style_norm(style_feats),mask=mask)
        content_feats = self.self_attn(self.content_norm(content_feats),mask=mask)

        style_feats = self.style_proj(style_feats)
        content_feats = self.content_proj(content_feats)
        
        #fusion_feats = self.cross_attn(q=content_feats,kv=style_feats)
        fusion_feats = style_feats + content_feats
        
        fusion_feats = self.fusion_norm(fusion_feats)
        
        if self.has_mlp:
            #fusion_feats = fusion_feats + self.ffn(fusion_feats)
            content_feats = content_feats + self.ffn(self.fusion_norm(fusion_feats))

        return style_feats,content_feats 

class CrossStyTr(nn.Module):
    def __init__(
        self,decoder,
        encoder,
        is_train = True,
        style_embed=768,
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
        self.is_train = is_train
        # Image encoder for calculating loss
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.mse_loss = nn.MSELoss()

        ##############################################################
        # Load a pretrained ViT
        self.vit = ViT('B_16_imagenet1k', pretrained=True,image_size=img_size).to(self.device) # construct and load 
        self.vit.fc = None
        freeze_model(self.vit)

        self.new_ps = nn.Conv2d(style_embed , style_embed , (1,1))
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

        self.decoder = decoder

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
      assert (input.size() == target.size())
      assert (target.requires_grad is False)
      return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    
    def unpatch(self,hs):
        ### HWxNxC to NxCxHxW to
        N, B, C= hs.shape          
        H = int(np.sqrt(N))
        hs = hs.permute(1, 2, 0)
        hs = hs.view(B, C, -1,H)

        return hs
       

    def forward(self,content_img,style_img,mask=None):
        '''
        @param img: (N, 3, 512, 512)
        @od        
        '''
        if content_img.shape != style_img.shape:
            print('content image shape: ',content_img.shape)
            print('style image shape: ',style_img.shape)
            raise ValueError("Error: content_img and style_img must have the same shape")

        N,_,H,W = content_img.shape
        ### Feature for calculating loss
        content_vgg = self.encode_with_intermediate(content_img)
        style_vgg = self.encode_with_intermediate(style_img)

        content_input = content_img.clone()
        style_input = style_img.clone()
        ########################################################################
        # Model
        # content-aware positional embedding
        content_img = self.vit.patch_embedding(content_img)
        style_img = self.vit.patch_embedding(style_img)
        
        # position embedding
        content_pool = self.averagepooling(content_img)
        #print('conten_pool shape: ',content_pool.shape)       
        pos_c = self.new_ps(content_pool)
        pos_embed_c = F.interpolate(pos_c, mode='bilinear',size= style_img.shape[-2:])

        ###flatten NxCxHxW to HWxNxC     
        style_img = style_img.flatten(2).permute(2, 0, 1)
        content_img = content_img.flatten(2).permute(2, 0, 1) # 512 x N x 768

        #print('shape of content_img patches',content_img.shape)    
        if pos_embed_c is not None:
            pos_embed_c = pos_embed_c.flatten(2).permute(2, 0, 1) # 512 x N x 768
        content_img = content_img + pos_embed_c

        # clone for identity loss
        content_img1 = content_img.clone()
        content_img2 = content_img.clone()
        style_img1 = style_img.clone()
        style_img2 = style_img.clone()
        
        for blk in self.fusion: # Weakness: Have to forward 3 times 
            style_img,content_img = blk(style_feats=style_img,content_feats=content_img)
            
            content_img1,content_img2 = blk(style_feats=content_img1,content_feats=content_img2) # for loss
            style_img1, style_img2 = blk(style_feats=style_img1,content_feats=style_img2) # for loss

        style_img = self.unpatch(style_img)
        content_img1 = self.unpatch(content_img1)
        style_img1 = self.unpatch(style_img1)
        content_img = self.unpatch(content_img)

        #print('style_img output shape:',style_img.shape)
        Ics = self.decoder(content_img) # result image
        #print('ICS shape: ', Ics.shape)

        ######Calculating loss#######################################################
        # stage 1: Perceptual loss##################################################
        Ics_vgg = self.encode_with_intermediate(Ics)
        # Content loss
        loss_c = self.calc_content_loss(normal(Ics_vgg[-1]), normal(content_vgg[-1]))+ \
            self.calc_content_loss(normal(Ics_vgg[-2]), normal(content_vgg[-2]))
        # Style loss
        loss_s = self.calc_style_loss(Ics_vgg[0], style_vgg[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(Ics_vgg[i], style_vgg[i])

        #############################################################################
        ################################################################################
        # stage2: identity loss   
        style_img_i = self.decoder(style_img1)
        content_img_i = self.decoder(content_img1)

        loss_lambda1 = self.calc_content_loss(content_img_i,content_input)+self.calc_content_loss(style_img_i,style_input)

        #Identity losses lambda 2
        Icc_feats=self.encode_with_intermediate(content_img_i)
        Iss_feats=self.encode_with_intermediate(style_img_i)
        loss_lambda2 = self.calc_content_loss(Icc_feats[0], content_vgg[0])+self.calc_content_loss(Iss_feats[0], style_vgg[0])
        for i in range(1, 5):
            loss_lambda2 += self.calc_content_loss(Icc_feats[i], content_vgg[i])+self.calc_content_loss(Iss_feats[i], style_vgg[i])

        if self.is_train:
            return Ics,  loss_c, loss_s, loss_lambda1, loss_lambda2   #train
        
        else: return Ics    #test 



