# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

from inference_memory_bank import MemoryBank
from model.aggregate import aggregate

from util.tensor_util import pad_divide_by

import os
from os.path import join as join
import cv2
import numpy as np

class InferenceCore:
    def __init__(self, prop_net, images, num_objects, top_k=20, mem_every=5, proto_every=1, include_last=False, vis=False):
        self.prop_net = prop_net
        self.mem_every = mem_every
        self.proto_every = proto_every
        self.include_last = include_last

        # True dimensions
        t = images.shape[1]
        h, w = images.shape[-2:]

        # Pad each side to multiple of 16
        images, self.pad = pad_divide_by(images, 16)
        # Padded dimensions
        nh, nw = images.shape[-2:]

        self.images = images
        self.device = 'cuda'

        self.k = num_objects

        # Background included, not always consistent (i.e. sum up to 1)
        self.prob = torch.zeros((self.k+1, t, 1, nh, nw), dtype=torch.float32, device=self.device)
        self.prob[0] = 1e-7

        self.t, self.h, self.w = t, h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh//16
        self.kw = self.nw//16

        self.mem_bank = MemoryBank(k=self.k, top_k=top_k)
        
        self.vis = vis

        self.prob_pixel = torch.zeros((self.k+1, t, 1, nh, nw), dtype=torch.float32, device=self.device)
        self.prob_pixel[0] = 1e-7

        self.prob_proto = torch.zeros((self.k+1, t, 1, nh, nw), dtype=torch.float32, device=self.device)
        self.prob_proto[0] = 1e-7

    def encode_key(self, idx):
        result = self.prop_net.encode_key(self.images[:,idx].cuda())
        return result

    def do_pass_CondProto_v3_5_new2_proto_affinity_indep_sample(self, key_k, key_v, key_k_emb, idx, end_idx, name):
        self.mem_bank.add_memory_allmem(key_k, key_v, key_k_emb)
        closest_ti = end_idx

        # Note that we never reach closest_ti, just the frame before it
        this_range = range(idx+1, closest_ti)
        end = closest_ti - 1

        for ti in this_range:
            k16, qv16, qf16, qf8, qf4 = self.encode_key(ti)
        
            out_mask = self.prop_net.segment_with_query(self.mem_bank, qf8, qf4, k16, qv16, qf16)

            out_mask = aggregate(out_mask, keep_bg=True)
            self.prob[:,ti] = out_mask

            C = qf16.shape[1]

            if ti != end:                    
                is_proto_frame = ((ti % self.proto_every) == 0)
                if self.include_last or is_proto_frame:
                    # add prototype memory
                    sp_center_list_fg = []
                    sp_center_list_bg = []
                    k16_emb_list = []
                    #n_objects = len(self.prob[1:])

                    supp_feat_ = qf16.unsqueeze(1)  # [bs, 1, C, h, w]

                    for n in range(self.k):
                        supp_mask_fg = out_mask[n+1].unsqueeze(0).contiguous()#.view([1, 1, H, W]) # [1, 1, H, W]
                        #max_prob_map = out_mask.max(0)[0].unsqueeze(0).contiguous()
                        #supp_mask_fg = torch.eq(supp_mask_fg, max_prob_map).float() #[1, 1, H, W]
                        supp_mask_fg = F.interpolate(supp_mask_fg, size=(qf16.size(2), qf16.size(3)), mode='bilinear', align_corners=True)
                        #supp_mask_fg = (supp_mask_fg>0.5).float()
                        supp_mask_bg = 1 - supp_mask_fg  # [bs*T, 1, h, w]

                        # （1）get prototype embedding with raw responses 
                        fg_proto, bg_proto, supp_emb = self.prop_net.get_proto(supp_feat_, supp_mask_fg, supp_mask_bg, protos=self.prop_net.num_sp) #[1, C, num_sp] 
                        #fg_proto, bg_proto, _ = self.prop_net.get_proto(supp_feat_, supp_mask_fg, supp_mask_bg, protos=self.prop_net.num_sp)                        
                        
                        fg_proto_g, bg_proto_g = self.prop_net.get_proto_g(supp_feat_, supp_mask_fg, supp_mask_bg) #[1, C]

                        _, C, num_sp = fg_proto.shape
                        protos_all_fg = torch.cat([fg_proto, fg_proto_g.unsqueeze(2)], 2)
                        protos_all_bg = torch.cat([bg_proto, bg_proto_g.unsqueeze(2)], 2)

# =============================================================================
#                         # (2) get prototype embedding after graph reasoning
#                         _, C, num_sp = fg_proto.shape
#                         protos_all_fg = torch.cat([fg_proto, fg_proto_g.unsqueeze(2)], 2).view([1, C, num_sp+1, 1, 1])
#                         protos_all_bg = torch.cat([bg_proto, bg_proto_g.unsqueeze(2)], 2).view([1, C, num_sp+1, 1, 1])
#                         protos_all_fg = self.prop_net.GloRe(protos_all_fg) # [bs, C, self.num_sp_all, 1, 1]
#                         protos_all_bg = self.prop_net.GloRe(protos_all_bg) # [bs, C, self.num_sp_all, 1, 1]
#                         protos_all_fg = protos_all_fg.view(*protos_all_fg.shape[:3]) # [bs, C, self.num_sp_all]
#                         protos_all_bg = protos_all_bg.view(*protos_all_bg.shape[:3]) # [bs, C, self.num_sp_all]
# 
#                         if self.prop_net.use_global: # 计算emb时包含全局proto
#                             supp_emb = self.prop_net.get_proto_query(supp_feat_.squeeze(1), protos_all_fg, protos_all_bg, protos=self.prop_net.num_sp)
#                         else:
#                             supp_emb = self.prop_net.get_proto_query(supp_feat_.squeeze(1), protos_all_fg[:,:,:-1], protos_all_bg[:,:,:-1], protos=self.prop_net.num_sp)                
# =============================================================================

                        sp_center_list_fg.append(protos_all_fg.unsqueeze(2)) # [1,C,K] -> [1, C, T, K]
                        sp_center_list_bg.append(protos_all_bg.unsqueeze(2))
                        k16_emb_list.append(supp_emb)
            
                    self.mem_bank.add_proto_fgbg(sp_center_list_fg, sp_center_list_bg, mode='mean')
                    #self.mem_bank.add_proto_fgbg(sp_center_list_fg, sp_center_list_bg, mode='cat')
                    #self.mem_bank.add_proto_fgbg(sp_center_list_fg, sp_center_list_bg, mode='first_last')  

                    k16_emb = torch.cat(k16_emb_list)  # [n_object, 64, h, w]

                is_mem_frame = ((ti % self.mem_every) == 0)
                if self.include_last or is_mem_frame:
                    # add point-wise memory
                    prev_value = self.prop_net.encode_value(self.images[:,ti].cuda(), qf16, out_mask[1:])
                    prev_key = k16.unsqueeze(2)
                    k16_emb = k16_emb.unsqueeze(2)
                    self.mem_bank.add_memory_allmem(prev_key, prev_value, k16_emb, is_temp=not is_mem_frame)
            
        return closest_ti
    
    def interact_CondProto_v3_5_new2_proto_affinity(self, mask, frame_idx, end_idx, name):
        mask, _ = pad_divide_by(mask.cuda(), 16)

        self.prop_net.folder_name = name

        self.prob[:, frame_idx] = aggregate(mask, keep_bg=True)

        # KV pair for the interacting frame
        key_k, _, qf16, _, _ = self.encode_key(frame_idx)
        key_v = self.prop_net.encode_value(self.images[:,frame_idx].cuda(), qf16, self.prob[1:,frame_idx].cuda())
        key_k = key_k.unsqueeze(2)

        # Initialize proto
        sp_center_list_fg = []
        sp_center_list_bg = []
        key_k_emb_list = []
        #sp_center_list = [None] * self.k
        n_objects = self.k
        #print(self.prob[1:].min(), self.prob[1:].max())
        C = qf16.shape[1]

        # key_k: [1,n_dim,1,h,w]
        # self.prob: [n_objects+1,n_frames,1,h,w]
        for n in range(n_objects):
            supp_feat_ = qf16.unsqueeze(1)  # [bs, 1, C, h, w]

            supp_mask_fg = self.prob[n+1, frame_idx].unsqueeze(0).contiguous()#.view([1, 1, H, W]) # [1, 1, H, W]
            #max_prob_map = self.prob[:,frame_idx].max(0)[0].unsqueeze(0).contiguous()
            #supp_mask_fg = torch.eq(supp_mask_fg, max_prob_map).float() #[1, 1, H, W]
            supp_mask_fg = F.interpolate(supp_mask_fg, size=(qf16.size(2), qf16.size(3)), mode='bilinear', align_corners=True)
            #supp_mask_fg = (supp_mask_fg>0.5).float()
            supp_mask_bg = 1 - supp_mask_fg  # [bs*T, 1, h, w]
            
            # （1）get prototype embedding with raw responses 
            fg_proto, bg_proto, supp_emb = self.prop_net.get_proto(supp_feat_, supp_mask_fg, supp_mask_bg, protos=self.prop_net.num_sp) #[1, C, num_sp] 
            #fg_proto, bg_proto, _ = self.prop_net.get_proto(supp_feat_, supp_mask_fg, supp_mask_bg, protos=self.prop_net.num_sp) #[1, C, num_sp] 

            fg_proto_g, bg_proto_g = self.prop_net.get_proto_g(supp_feat_, supp_mask_fg, supp_mask_bg) #[1, C]    

            _, C, num_sp = fg_proto.shape
            protos_all_fg = torch.cat([fg_proto, fg_proto_g.unsqueeze(2)], 2)
            protos_all_bg = torch.cat([bg_proto, bg_proto_g.unsqueeze(2)], 2)

# =============================================================================
#             # (2) get prototype embedding after graph reasoning
#             _, C, num_sp = fg_proto.shape
#             protos_all_fg = torch.cat([fg_proto, fg_proto_g.unsqueeze(2)], 2).view([1, C, num_sp+1, 1, 1])
#             protos_all_bg = torch.cat([bg_proto, bg_proto_g.unsqueeze(2)], 2).view([1, C, num_sp+1, 1, 1])
#             protos_all_fg = self.prop_net.GloRe(protos_all_fg) # [bs, C, self.num_sp_all, 1, 1]
#             protos_all_bg = self.prop_net.GloRe(protos_all_bg) # [bs, C, self.num_sp_all, 1, 1]
#             protos_all_fg = protos_all_fg.view(*protos_all_fg.shape[:3]) # [bs, C, self.num_sp_all]
#             protos_all_bg = protos_all_bg.view(*protos_all_bg.shape[:3]) # [bs, C, self.num_sp_all]
#             
#             if self.prop_net.use_global: # 计算emb时包含全局proto
#                 supp_emb = self.prop_net.get_proto_query(supp_feat_.squeeze(1), protos_all_fg, protos_all_bg, protos=self.prop_net.num_sp)
#             else:
#                 supp_emb = self.prop_net.get_proto_query(supp_feat_.squeeze(1), protos_all_fg[:,:,:-1], protos_all_bg[:,:,:-1], protos=self.prop_net.num_sp)                
# =============================================================================

            sp_center_list_fg.append(protos_all_fg.unsqueeze(2)) # [1,C,K] -> [1, C, T, K]
            sp_center_list_bg.append(protos_all_bg.unsqueeze(2)) 
            key_k_emb_list.append(supp_emb)

        key_k_emb = torch.cat(key_k_emb_list)  # [n_object, 64, h, w]
        key_k_emb = key_k_emb.unsqueeze(2)
            
        self.mem_bank.add_proto_fgbg(sp_center_list_fg, sp_center_list_bg, mode='mean')
        #self.mem_bank.add_proto_fgbg(sp_center_list_fg, sp_center_list_bg, mode='cat')
        #self.mem_bank.add_proto_fgbg(sp_center_list_fg, sp_center_list_bg, mode='first_last')

        # Propagate
        #self.do_pass_MetaProto(key_k, key_v, frame_idx, end_idx)
        self.do_pass_CondProto_v3_5_new2_proto_affinity_indep_sample(key_k, key_v, key_k_emb, frame_idx, end_idx, name)

    def do_pass_CondProto_v3_5_new3_proto_affinity_indep_sample(self, key_k, key_v, key_k_emb, idx, end_idx, name):
        self.mem_bank.add_memory_allmem(key_k, key_v, key_k_emb)
        closest_ti = end_idx

        # Note that we never reach closest_ti, just the frame before it
        this_range = range(idx+1, closest_ti)
        end = closest_ti - 1

        for ti in this_range:
            k16, qv16, qf16, qf8, qf4 = self.encode_key(ti)
        
            out_mask = self.prop_net.segment_with_query(self.mem_bank, qf8, qf4, k16, qv16, qf16)

            out_mask = aggregate(out_mask, keep_bg=True)
            self.prob[:,ti] = out_mask

            C = qf16.shape[1]

            if ti != end:                    
                is_proto_frame = ((ti % self.proto_every) == 0)
                if self.include_last or is_proto_frame:
                    # add prototype memory
                    sp_center_list_fg = []
                    sp_center_list_bg = []
                    k16_emb_list = []
                    update_index = torch.ones(self.k)
                    #n_objects = len(self.prob[1:])

                    supp_feat_ = qf16.unsqueeze(1)  # [bs, 1, C, h, w]

                    for n in range(self.k):
                        supp_mask_fg = out_mask[n+1].unsqueeze(0).contiguous()#.view([1, 1, H, W]) # [1, 1, H, W]
                        #max_prob_map = out_mask.max(0)[0].unsqueeze(0).contiguous()
                        #supp_mask_fg = torch.eq(supp_mask_fg, max_prob_map).float() #[1, 1, H, W]
                        supp_mask_fg = F.interpolate(supp_mask_fg, size=(qf16.size(2), qf16.size(3)), mode='bilinear', align_corners=True)
                        #supp_mask_fg = (supp_mask_fg>0.5).float()
                        supp_mask_bg = 1 - supp_mask_fg  # [bs*T, 1, h, w]
                        if self.vis:
                            fg_proto, bg_proto = self.mpm_vis(supp_feat_, supp_mask_fg, supp_mask_bg, self.prop_net.num_sp, name, ti, mask_raw=out_mask[n+1].unsqueeze(0))
                        else:
                            # （1）get prototype embedding with raw responses 
                            fg_proto, bg_proto, supp_emb = self.prop_net.get_proto(supp_feat_, supp_mask_fg, supp_mask_bg, protos=self.prop_net.num_sp) #[1, C, num_sp] 
                            #fg_proto, bg_proto, _ = self.prop_net.get_proto(supp_feat_, supp_mask_fg, supp_mask_bg, protos=self.prop_net.num_sp)                        
                        
                        fg_proto_g, bg_proto_g = self.prop_net.get_proto_g(supp_feat_, supp_mask_fg, supp_mask_bg) #[1, C]

                        _, C, num_sp = fg_proto.shape
                        protos_all_fg = torch.cat([fg_proto, fg_proto_g.unsqueeze(2)], 2)
                        protos_all_bg = torch.cat([bg_proto, bg_proto_g.unsqueeze(2)], 2)

# =============================================================================
#                         # (2) get prototype embedding after graph reasoning
#                         _, C, num_sp = fg_proto.shape
#                         protos_all_fg = torch.cat([fg_proto, fg_proto_g.unsqueeze(2)], 2).view([1, C, num_sp+1, 1, 1])
#                         protos_all_bg = torch.cat([bg_proto, bg_proto_g.unsqueeze(2)], 2).view([1, C, num_sp+1, 1, 1])
#                         protos_all_fg = self.prop_net.GloRe(protos_all_fg) # [bs, C, self.num_sp_all, 1, 1]
#                         protos_all_bg = self.prop_net.GloRe(protos_all_bg) # [bs, C, self.num_sp_all, 1, 1]
#                         protos_all_fg = protos_all_fg.view(*protos_all_fg.shape[:3]) # [bs, C, self.num_sp_all]
#                         protos_all_bg = protos_all_bg.view(*protos_all_bg.shape[:3]) # [bs, C, self.num_sp_all]
# 
#                         if self.prop_net.use_global: # 计算emb时包含全局proto
#                             supp_emb = self.prop_net.get_proto_query(supp_feat_.squeeze(1), protos_all_fg, protos_all_bg, protos=self.prop_net.num_sp)
#                         else:
#                             supp_emb = self.prop_net.get_proto_query(supp_feat_.squeeze(1), protos_all_fg[:,:,:-1], protos_all_bg[:,:,:-1], protos=self.prop_net.num_sp)                
# =============================================================================

                        sp_center_list_fg.append(protos_all_fg.unsqueeze(2)) # [1,C,K] -> [1, C, T, K]
                        sp_center_list_bg.append(protos_all_bg.unsqueeze(2))
                        k16_emb_list.append(supp_emb)

                        # 只在当前帧出现目标时更新proto
                        if (supp_mask_fg>0.5).sum() == 0:
                            update_index[n] = 0
            
                    self.mem_bank.add_proto_fgbg_select(sp_center_list_fg, sp_center_list_bg, update_index, mode='mean')
                    #self.mem_bank.add_proto_fgbg(sp_center_list_fg, sp_center_list_bg, mode='cat')
                    #self.mem_bank.add_proto_fgbg(sp_center_list_fg, sp_center_list_bg, mode='first_last')  
                    #self.mem_bank.add_proto_fgbg_select(sp_center_list_fg, sp_center_list_bg, update_index, mode='ema')

                    k16_emb = torch.cat(k16_emb_list)  # [n_object, 64, h, w]

                is_mem_frame = ((ti % self.mem_every) == 0)
                if self.include_last or is_mem_frame:
                    # add point-wise memory
                    prev_value = self.prop_net.encode_value(self.images[:,ti].cuda(), qf16, out_mask[1:])
                    prev_key = k16.unsqueeze(2)
                    k16_emb = k16_emb.unsqueeze(2)
                    self.mem_bank.add_memory_allmem(prev_key, prev_value, k16_emb, is_temp=not is_mem_frame)
            
        return closest_ti
    
    def interact_CondProto_v3_5_new3_proto_affinity(self, mask, frame_idx, end_idx, name):
        mask, _ = pad_divide_by(mask.cuda(), 16)

        self.prop_net.folder_name = name

        self.prob[:, frame_idx] = aggregate(mask, keep_bg=True)

        # KV pair for the interacting frame
        key_k, _, qf16, _, _ = self.encode_key(frame_idx)
        key_v = self.prop_net.encode_value(self.images[:,frame_idx].cuda(), qf16, self.prob[1:,frame_idx].cuda())
        key_k = key_k.unsqueeze(2)

        # Initialize proto
        sp_center_list_fg = []
        sp_center_list_bg = []
        key_k_emb_list = []
        #sp_center_list = [None] * self.k
        n_objects = self.k
        #print(self.prob[1:].min(), self.prob[1:].max())
        C = qf16.shape[1]

        # key_k: [1,n_dim,1,h,w]
        # self.prob: [n_objects+1,n_frames,1,h,w]
        for n in range(n_objects):
            supp_feat_ = qf16.unsqueeze(1)  # [bs, 1, C, h, w]

            supp_mask_fg = self.prob[n+1, frame_idx].unsqueeze(0).contiguous()#.view([1, 1, H, W]) # [1, 1, H, W]
            #max_prob_map = self.prob[:,frame_idx].max(0)[0].unsqueeze(0).contiguous()
            #supp_mask_fg = torch.eq(supp_mask_fg, max_prob_map).float() #[1, 1, H, W]
            supp_mask_fg = F.interpolate(supp_mask_fg, size=(qf16.size(2), qf16.size(3)), mode='bilinear', align_corners=True)
            #supp_mask_fg = (supp_mask_fg>0.5).float()
            supp_mask_bg = 1 - supp_mask_fg  # [bs*T, 1, h, w]
            
            if self.vis:
                fg_proto, bg_proto = self.mpm_vis(supp_feat_, supp_mask_fg, supp_mask_bg, self.prop_net.num_sp, name, ti=0, mask_raw=self.prob[n+1, frame_idx].unsqueeze(0))
            else:
                # （1）get prototype embedding with raw responses 
                fg_proto, bg_proto, supp_emb = self.prop_net.get_proto(supp_feat_, supp_mask_fg, supp_mask_bg, protos=self.prop_net.num_sp) #[1, C, num_sp] 
                #fg_proto, bg_proto, _ = self.prop_net.get_proto(supp_feat_, supp_mask_fg, supp_mask_bg, protos=self.prop_net.num_sp) #[1, C, num_sp] 

            fg_proto_g, bg_proto_g = self.prop_net.get_proto_g(supp_feat_, supp_mask_fg, supp_mask_bg) #[1, C]    

            _, C, num_sp = fg_proto.shape
            protos_all_fg = torch.cat([fg_proto, fg_proto_g.unsqueeze(2)], 2)
            protos_all_bg = torch.cat([bg_proto, bg_proto_g.unsqueeze(2)], 2)

# =============================================================================
#             # (2) get prototype embedding after graph reasoning
#             _, C, num_sp = fg_proto.shape
#             protos_all_fg = torch.cat([fg_proto, fg_proto_g.unsqueeze(2)], 2).view([1, C, num_sp+1, 1, 1])
#             protos_all_bg = torch.cat([bg_proto, bg_proto_g.unsqueeze(2)], 2).view([1, C, num_sp+1, 1, 1])
#             protos_all_fg = self.prop_net.GloRe(protos_all_fg) # [bs, C, self.num_sp_all, 1, 1]
#             protos_all_bg = self.prop_net.GloRe(protos_all_bg) # [bs, C, self.num_sp_all, 1, 1]
#             protos_all_fg = protos_all_fg.view(*protos_all_fg.shape[:3]) # [bs, C, self.num_sp_all]
#             protos_all_bg = protos_all_bg.view(*protos_all_bg.shape[:3]) # [bs, C, self.num_sp_all]
#             
#             if self.prop_net.use_global: # 计算emb时包含全局proto
#                 supp_emb = self.prop_net.get_proto_query(supp_feat_.squeeze(1), protos_all_fg, protos_all_bg, protos=self.prop_net.num_sp)
#             else:
#                 supp_emb = self.prop_net.get_proto_query(supp_feat_.squeeze(1), protos_all_fg[:,:,:-1], protos_all_bg[:,:,:-1], protos=self.prop_net.num_sp)                
# =============================================================================

            sp_center_list_fg.append(protos_all_fg.unsqueeze(2)) # [1,C,K] -> [1, C, T, K]
            sp_center_list_bg.append(protos_all_bg.unsqueeze(2)) 
            key_k_emb_list.append(supp_emb)

        key_k_emb = torch.cat(key_k_emb_list)  # [n_object, 64, h, w]
        key_k_emb = key_k_emb.unsqueeze(2)

        update_index = torch.ones(self.k) # 第一帧proto全部保留
        self.mem_bank.add_proto_fgbg(sp_center_list_fg, sp_center_list_bg, mode='mean')
        #self.mem_bank.add_proto_fgbg(sp_center_list_fg, sp_center_list_bg, mode='cat')
        #self.mem_bank.add_proto_fgbg(sp_center_list_fg, sp_center_list_bg, mode='first_last')
        #self.mem_bank.add_proto_fgbg_select(sp_center_list_fg, sp_center_list_bg, update_index, mode='ema')

        # Propagate
        #self.do_pass_MetaProto(key_k, key_v, frame_idx, end_idx)
        self.do_pass_CondProto_v3_5_new3_proto_affinity_indep_sample(key_k, key_v, key_k_emb, frame_idx, end_idx, name)
