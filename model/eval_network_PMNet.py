import os
from os.path import join as join
import cv2
import math
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.encoder_modules import *
from modules.decoder import Decoder
from modules.attention_pool import AttentionPool2d
from modules.global_reasoning_unit import GloRe_Unit_3D

class PMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_encoder = KeyEncoder() 
        self.value_encoder = ValueEncoder() 

        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(1024, keydim=64)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.decoder = Decoder()
        
        self.sp_dim = 1024
        self.num_sp = 5
        
        self.top_sp = 5

        if self.num_sp > 0:
            self.ctr = torch.nn.Parameter(torch.rand(self.num_sp * 2, self.sp_dim), requires_grad=True) # ctr即可学习的meta-protot+ype
        else:
            self.ctr = None

        self.num_sp_all = self.num_sp + 1
        self.num_node = 3  

        self.W = 1

        self.folder_name = None

        n_ctx = self.num_sp
        vis_dim = self.sp_dim
        ctx_dim = self.sp_dim
# =============================================================================
#         # random initialization
#         ctx_vectors = torch.empty(n_ctx, ctx_dim)
#         nn.init.normal_(ctx_vectors, std=0.02)
#         self.ctx = nn.Parameter(ctx_vectors)
# =============================================================================

        self.cond_net = AttentionPool2d(spacial_dim=24, embed_dim=self.sp_dim, num_heads=8, output_dim=self.sp_dim)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim * 4))
        ]))
        
        self.GloRe = GloRe_Unit_3D(self.sp_dim, self.sp_dim//2, self.num_node, normalize=False)

    def encode_value(self, frame, kf16, masks): 
        k, _, h, w = masks.shape

        # Extract memory key/value for a frame with multiple masks
        frame = frame.view(1, 3, h, w).repeat(k, 1, 1, 1)
        # Compute the "others" mask
        if k != 1:
            others = torch.cat([
                torch.sum(
                    masks[[j for j in range(k) if i!=j]]
                , dim=0, keepdim=True)
            for i in range(k)], 0)
        else:
            others = torch.zeros_like(masks) 

        f16 = self.value_encoder(frame, kf16.repeat(k,1,1,1), masks, others)

        return f16.unsqueeze(2)

    def encode_key(self, frame):
        f16, f8, f4 = self.key_encoder(frame)
        k16 = self.key_proj(f16)
        f16_thin = self.key_comp(f16)

        return k16, f16_thin, f16, f8, f4

    def read_proto(self, mk, qk):
        #mk:[B, C, T*K]
        #qk:[B, C, h, w]
        B, C, h, w = qk.shape
        #mk = mk.flatten(start_dim=2)
        qk = qk.flatten(start_dim=2)# B, C, HW

        a = mk.pow(2).sum(1).unsqueeze(2)
        b = 2 * (mk.transpose(1, 2) @ qk) # B, TK, HW
        # this term will be cancelled out in the softmax
        # c = qk.pow(2).sum(1).unsqueeze(1)

        affinity = (-a+b) / math.sqrt(C)   # B, TK, HW
        
        # softmax operation; aligned the evaluation style
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]# B, 1, HW
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum  # B, TK, HW

        mem_proto = torch.bmm(mk, affinity) # Weighted-sum B, C, HW
        mem_proto = mem_proto.view(B, C, h, w)

        return mem_proto # B, C, h, w

    def softmax_w_top(self, x, top):
        values, indices = torch.topk(x, k=top, dim=1)
        x_exp = values.exp_()
    
        x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
        x.zero_().scatter_(1, indices, x_exp) # B * THW * HW # 除topk匹配位置外其余位置相似度权重置为0
    
        return x

    def segment_with_query(self, mem_bank, qf8, qf4, qk16, qv16, qf16):
        k = mem_bank.num_objects

        # get prompt
        cond_feas = self.cond_net(qf16) # [bs, ctx_dim]
        prompt = self.meta_net(cond_feas)  # [bs, ctx_dim]

        mem_proto_list = []
        query_emb_k_list = []
        bs, C, _, _ = qf16.shape
        # get protos and embeddings
        for kk in range(k):
            fg_proto, bg_proto = mem_bank.mem_p_fg[kk].view(1, C, -1), mem_bank.mem_p_bg[kk].view(1, C, -1)  # [bs, C, num_sp]

            query_emb_k_ = self.get_proto_emb(qf16, fg_proto[:,:,:-1], bg_proto[:,:,:-1], protos=self.num_sp)  # [bs, C, H, W]
            query_emb_k_list.append(query_emb_k_)

            fg_proto = self.GloRe(fg_proto.view([1, C, self.num_sp+1, 1, 1])) # [bs, C, self.num_sp_all, 1, 1]
            bg_proto = self.GloRe(bg_proto.view([1, C, self.num_sp+1, 1, 1])) # [bs, C, self.num_sp_all, 1, 1]

            fg_proto = fg_proto.view(*fg_proto.shape[:3]) # [bs, C, self.num_sp_all]
            bg_proto = bg_proto.view(*bg_proto.shape[:3]) # [bs, C, self.num_sp_all]

            fg_proto = fg_proto.unsqueeze(2) # [bs, C, 1, self.num_sp_all]
            bg_proto = bg_proto.unsqueeze(2) # [bs, C, 1, self.num_sp_all]

            fg_proto[:,:,:,:-1] = prompt[:,:C].view(bs, C, 1, 1) + fg_proto[:,:,:,:-1] #[bs, C, 1, num_sp] 
            bg_proto[:,:,:,:-1] = prompt[:,C:2*C].view(bs, C, 1, 1) + bg_proto[:,:,:,:-1] #[bs, C, 1, num_sp] 
            
            fg_proto[:,:,:,-1] = prompt[:,2*C:3*C].view(bs, C, 1) + fg_proto[:,:,:,-1] #[bs, C, 1]
            bg_proto[:,:,:,-1] = prompt[:,3*C:4*C].view(bs, C, 1) + bg_proto[:,:,:,-1] #[bs, C, 1]

            fg_proto = fg_proto.view(bs, C, -1) # [bs, C, 1, num_sp+1]
            bg_proto = bg_proto.view(bs, C, -1) # [bs, C, 1, num_sp+1]

            # get proto-aggre feas
            mem_proto_fg = self.read_proto(fg_proto, qf16)  # 1, C, h, w
            mem_proto_bg = self.read_proto(bg_proto, qf16)  # 1, C, h, w
            mem_proto_feat = torch.cat((mem_proto_fg, mem_proto_bg), 1)  # bs, 2C, h, w
            mem_proto_list.append(mem_proto_feat)

        query_emb_k = torch.cat(query_emb_k_list)  # n_objects x c x h x w
        guide_feat = torch.cat(mem_proto_list)  # n_objects x c x h x w

        # get point-wise aggregation
        readout_mem = mem_bank.match_memory_emb(qk16, query_emb_k, weight=self.W, norm_p='channel_num')
        qv16 = qv16.expand(k, -1, -1, -1)
        qv16 = torch.cat([readout_mem, qv16], 1)  # [k,c,h,w]

        pred_mask = torch.sigmoid(self.decoder(qv16, qf8, qf4, guide_feat))  # n_objects x 1 x H x W

        return pred_mask
        
    def get_proto(self, sup_fts, sup_fg, sup_bg, protos):
        B, S, c, h, w = sup_fts.shape # B=bs, S=n_shot
        sup_fts = sup_fts.reshape(-1, c, h * w) # [BS, c, hw]

        sup_fg = sup_fg.view(-1, 1, h * w)  # [BS, 1, hw]
        sup_bg = sup_bg.view(-1, 1, h * w)  # [BS, 1, hw]

        if self.ctr is not None:
            ctr = self.ctr.view(1, c, protos * 2)                                                   # [1, c, 2p]
            mask = torch.stack((sup_fg, sup_bg), dim=1)                                             # [BS, 2, 1, hw]

            # get proto acvivation
            D = -((sup_fts.unsqueeze(dim=2) - ctr.unsqueeze(dim=3)) ** 2).sum(dim=1)                # [BS, 2p, hw]
            D = D.view(-1, 2, protos, h * w)
            #print(torch.softmax(D, dim=2).shape, mask.shape)                                       # [BS, 2, p, hw]
            D = (torch.softmax(D, dim=2) * mask).view(-1, 1, protos * 2, h * w)                     # [BS, 1, 2p, hw]
            D_ = D.view(-1, protos * 2, h * w).permute(0,2,1).contiguous()         # [BS, hw, 2p]

            # get memory fg/bg proto
            masked_fts = sup_fts.view(-1, c, 1, h * w) * D                                          # [BS, c, 2p, hw]
            ctr = (masked_fts.sum(dim=3) / (D.sum(dim=3) + 1e-6)).view(B, S, c, 2, protos)          # [B, S, c, 2, p]
            ctr = ctr.transpose(3, 4).reshape(B, S, c * protos, 2)                                  # [B, S, cp, 2]
            ctr = ctr.mean(dim=1)                                                                   # [B, cp, 2]

            fg_proto, bg_proto = ctr.view(B, c, protos, 2).unbind(dim=3) # [B, c, protos]

            # get proto embedding
            ctr_emb = D_.permute(0,2,1).view([B*S, self.num_sp * 2, h, w]).contiguous() # [BS, 2p, h, w]

        return fg_proto, bg_proto, ctr_emb
        
    def get_proto_g(self, sup_fts, sup_fg, sup_bg, mode=1):
        B, S, c, h, w = sup_fts.shape # B=bs, S=n_shot
        sup_fts = sup_fts.reshape(-1, c, h * w) # [BS, c, hw]
        
        sup_fg = sup_fg.view(-1, 1, h * w)  # [BS, 1, hw]
        sup_bg = sup_bg.view(-1, 1, h * w)  # [BS, 1, hw]

        fg_vecs = torch.sum(sup_fts * sup_fg, dim=-1) / (sup_fg.sum(dim=-1) + 1e-5)     # [BS, c]
        bg_vecs = torch.sum(sup_fts * sup_bg, dim=-1) / (sup_bg.sum(dim=-1) + 1e-5)     # [BS, c]
        fg_proto = fg_vecs.view(B, S, c).mean(dim=1)
        bg_proto = bg_vecs.view(B, S, c).mean(dim=1)
        
        return fg_proto, bg_proto
        
    def get_proto_emb(self, qu_fts, fg_proto, bg_proto, protos):
        # fg_proto/bg_proto: [B, c, p]
        B, c, h, w = qu_fts.shape # B=bs
        qu_fts = qu_fts.view([B, c, h*w]) # [B,c,hw]

        if self.ctr is not None:
            ctr = torch.cat([fg_proto, bg_proto], 2)                                                # [B, c, 2p]

            # get proto response
            D = -((qu_fts.unsqueeze(dim=2) - ctr.unsqueeze(dim=3)) ** 2).sum(dim=1)                 # [B, 2p, hw]
            D = torch.softmax(D, dim=1).permute(0,2,1).contiguous()     # [B, hw, 2p]

            # get proto embedding
            ctr_emb = D.permute(0,2,1).view([B, self.num_sp * 2, h, w]).contiguous() # [B, 2p, h, w]

        return ctr_emb