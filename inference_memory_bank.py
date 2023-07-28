import math
import torch
import torch.nn.functional as F

# =============================================================================
# no_top = False
# 
# if no_top:
#     def softmax_w_top(x, top):
#         x_exp = x.exp_() # B, THW, HW
#         x_exp /= torch.sum(x_exp, dim=1, keepdim=True) # B, THW, HW
#     
#         return x
#     
#     def softmax_w_top_ASPP(x, top):
#         # x: B, 4, THW, HW
#         x_exp = x.exp_() # B, 4, THW, H
#         x_exp /= torch.sum(x_exp, dim=2, keepdim=True) # B, 4, THW, HW
#     
#         return x    
#     
# else:    
# =============================================================================
def softmax_w_top(x, top):
    values, indices = torch.topk(x, k=top, dim=1)
    x_exp = values.exp_()

    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    x.zero_().scatter_(1, indices, x_exp)  # B * THW * HW # 除topk匹配位置外其余位置相似度权重置为0

    return x

def softmax_w_top_ASPP(x, top):
    # x: B, 4, THW, HW
    values, indices = torch.topk(x, k=top, dim=2)  # B, 4, top, HW
    x_exp = values.exp_()  # B, 4, top, HW

    x_exp /= torch.sum(x_exp, dim=2, keepdim=True)  # B, 4, top, HW
    x.zero_().scatter_(2, indices, x_exp)  # B, 4, THW, HW # 除topk匹配位置外其余位置相似度权重置为0

    return x

def make_gaussian(y_idx, x_idx, height, width, sigma=7):
    yv, xv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])

    yv = yv.reshape(height*width).unsqueeze(0).float().cuda()
    xv = xv.reshape(height*width).unsqueeze(0).float().cuda()

    y_idx = y_idx.transpose(0, 1)
    x_idx = x_idx.transpose(0, 1)

    g = torch.exp(- ((yv-y_idx)**2 + (xv-x_idx)**2) / (2*sigma**2) )

    return g

def softmax_w_g_top(x, top=None, gauss=None):
    if top is not None:
        if gauss is not None:
            maxes = torch.max(x, dim=1, keepdim=True)[0]
            x_exp = torch.exp(x - maxes)*gauss
            x_exp, indices = torch.topk(x_exp, k=top, dim=1)
        else:
            values, indices = torch.topk(x, k=top, dim=1)
            x_exp = torch.exp(values - values[:,0])

        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        x_exp /= x_exp_sum
        x.zero_().scatter_(1, indices, x_exp) # B * THW * HW

        output = x
    else:
        maxes = torch.max(x, dim=1, keepdim=True)[0]
        if gauss is not None:
            x_exp = torch.exp(x-maxes)*gauss

        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        x_exp /= x_exp_sum
        output = x_exp

    return output

class MemoryBank:
    def __init__(self, k, top_k=20):
        self.top_k = top_k

        self.CK = None
        self.CV = None

        self.mem_k = None
        self.mem_v = None

        # extra mem
        self.mem_m = None
        self.mem_f = None

        self.mem_p = None

        # 1st group
        self.mem_p_fg_all = None
        self.mem_p_bg_all = None

        self.mem_p_fg = None
        self.mem_p_bg = None

        self.mem_p_fg_ema = None
        self.mem_p_bg_ema = None

        self.num_proto = torch.zeros(k)

        self.num_proto_fg = torch.zeros(k)
        self.num_proto_bg = torch.zeros(k)

        # 2nd group(possible)
        self.mem_p_fg_all2 = None
        self.mem_p_bg_all2 = None

        self.mem_p_fg2 = None
        self.mem_p_bg2 = None

        self.num_proto_fg2 = torch.zeros(k)
        self.num_proto_bg2 = torch.zeros(k)
        
        # 记忆多尺度特征
        self.mem_ms = None

        # 记录前景/背景的全局池化向量
        self.mem_vec_fg = None
        self.mem_vec_bg = None
        self.num_pix_fg_all = torch.zeros(k)
        self.num_pix_bg_all = torch.zeros(k)

        # 记录proto的置信度
        self.conf_list_fg_all = None
        self.conf_list_bg_all = None        

        self.temp_k_emb = None

        # hyper parameters
        self.num_objects = k

        self.num_scales = None

        self.mem_ind = []
        
        self.w = 0.9

    def _global_matching(self, mk, qk):
        # mk: [k,c,thw] qk: [1,c,hw]
        # NE means number of elements -- typically T*H*W
        B, CK, NE = mk.shape

        a = mk.pow(2).sum(1).unsqueeze(2)  # k,thw,1
        b = 2 * (mk.transpose(1, 2) @ qk)  #
        # We don't actually need this, will update paper later
        # c = qk.pow(2).expand(B, -1, -1).sum(1).unsqueeze(1)

        affinity = (-a + b) / math.sqrt(CK)  # k, thw, hw
        affinity = softmax_w_top(affinity, top=self.top_k)  # k, thw, hw

        return affinity
    
    def _global_matching_dropout(self, mk, qk, ):
        # mk: [k,c,thw] qk: [1,c,hw]
        # NE means number of elements -- typically T*H*W
        B, CK, NE = mk.shape

        a = mk.pow(2).sum(1).unsqueeze(2)  # k,thw,1
        b = 2 * (mk.transpose(1, 2) @ qk)  #
        # We don't actually need this, will update paper later
        # c = qk.pow(2).expand(B, -1, -1).sum(1).unsqueeze(1)

        affinity = (-a + b) / math.sqrt(CK)  # k, thw, hw
        affinity = softmax_w_top(affinity, top=self.top_k)  # k, thw, hw

        return affinity

    def _global_matching_emb(self, mk, qk, mk_emb, qk_emb, weight, norm_p, gauss):
        # mk: [k,c,thw] qk: [1,c,hw]
        # NE means number of elements -- typically T*H*W
        B, CK, NE = mk.shape

        # get point-wise affinity
        a = mk.pow(2).sum(1).unsqueeze(2)  # k,thw,1
        b = 2 * (mk.transpose(1, 2) @ qk)  #
        affinity_point = (-a + b) / math.sqrt(CK) # k, thw, hw

        # get proto-embedding affinity
        if norm_p == 'channel_num':
            C_emb = mk_emb.shape[1] # 后期使用通道数归一化
        else:
            C_emb = norm_p  # 前期代码使用key通道数64作归一化，为代码兼容默认norm_p=64

        a_p = mk_emb.pow(2).sum(1).unsqueeze(2)  # B, Thw, 1
        b_p = 2 * (mk_emb.transpose(1, 2) @ qk_emb)  # B, Thw, hw
        affinity_proto = (-a_p + b_p) / math.sqrt(C_emb)  # B, THW, HW

        # 整合两种affinity
        #print('affinity_point, min:{}, max:{}, mean:{}'.format(affinity_point.min(), affinity_point.max(), affinity_point.mean()))
        #print('affinity_proto, min:{}, max:{}, mean:{}'.format(affinity_proto.min(), affinity_proto.max(), affinity_proto.mean()))

        affinity = affinity_point + affinity_proto * weight # [k, thw, hw]

        # softmax operation; aligned the evaluation style
        if gauss:
            _, _, T, H, W = self.mem_f.shape
            k = affinity.shape[0]
            affinity_list = []
            # 每个目标单独施加高斯核约束
            for kk in range(k):
                # Make a bunch of Gaussian distributions
                affinity_k = affinity[kk].unsqueeze(0) # [1, thw, hw]
                argmax_idx = affinity_k.max(2)[1]
                y_idx, x_idx = argmax_idx // W, argmax_idx % W
                g = make_gaussian(y_idx, x_idx, H, W, sigma=7) # 高斯kernel默认大小为7
                g = g.view(B, T * H * W, H * W)
                affinity_k = softmax_w_g_top(affinity_k, top=self.top_k, gauss=g)  # 1, thw, hw
                affinity_list.append(affinity_k)
            affinity = torch.cat(affinity_list) # [k, thw, hw]
        else:
            affinity = softmax_w_top(affinity, top=self.top_k)  # k, thw, hw

        return affinity

    # def vis_affinity(self, affinity, T, H, W, index=[3,20,20]):
    #     # affinity: [K, thw, hw]
    #     object_index = 0
    #     interval = 10
    #     affinity
    #     affinity = affinity[]
    #     affinity = affinity.permute(0,2,1) # [bs, hw, thw]
    #     for h in range(0, H, interval): #
    #         for w in range(0, W, interval):
    #             affinity = affinity[]

    def _global_matching_emb_vis(self, mk, qk, mk_emb, qk_emb, weight, norm_p, gauss):
        # mk: [k,c,thw] qk: [1,c,hw]
        # NE means number of elements -- typically T*H*W
        B, CK, NE = mk.shape

        # get point-wise affinity
        a = mk.pow(2).sum(1).unsqueeze(2)  # k,thw,1
        b = 2 * (mk.transpose(1, 2) @ qk)  #
        affinity_point = (-a + b) / math.sqrt(CK) # k, thw, hw

        # get proto-embedding affinity
        if norm_p == 'channel_num':
            C_emb = mk_emb.shape[1] # 后期使用通道数归一化
        else:
            C_emb = norm_p  # 前期代码使用key通道数64作归一化，为代码兼容默认norm_p=64

        a_p = mk_emb.pow(2).sum(1).unsqueeze(2)  # B, Thw, 1
        b_p = 2 * (mk_emb.transpose(1, 2) @ qk_emb)  # B, Thw, hw
        affinity_proto = (-a_p + b_p) / math.sqrt(C_emb)  # B, THW, HW

        # 整合两种affinity
        #print('affinity_point, min:{}, max:{}, mean:{}'.format(affinity_point.min(), affinity_point.max(), affinity_point.mean()))
        #print('affinity_proto, min:{}, max:{}, mean:{}'.format(affinity_proto.min(), affinity_proto.max(), affinity_proto.mean()))

        affinity = affinity_point + affinity_proto * weight

        self.vis_affinity(affinity_point, affinity_proto, affinity)

        # softmax operation; aligned the evaluation style
        if gauss:
            _, _, T, H, W = self.mem_f.shape
            # Make a bunch of Gaussian distributions
            argmax_idx = affinity.max(2)[1]
            y_idx, x_idx = argmax_idx // W, argmax_idx % W
            g = make_gaussian(y_idx, x_idx, H, W, sigma=7) # 高斯kernel默认大小为7
            g = g.view(B, T * H * W, H * W)
            affinity = softmax_w_g_top(affinity, top=self.top_k, gauss=g)  # k, thw, hw
        else:
            affinity = softmax_w_top(affinity, top=self.top_k)  # k, thw, hw

        return affinity

    def _global_matching_emb_v2(self, mk, qk, mk_emb, qk_emb, weight_point, weight_proto, norm_p):
        # mk: [k,c,thw] qk: [1,c,hw]
        # NE means number of elements -- typically T*H*W
        B, CK, NE = mk.shape

        # get point-wise affinity
        a = mk.pow(2).sum(1).unsqueeze(2)  # k,thw,1
        b = 2 * (mk.transpose(1, 2) @ qk)  #
        affinity_point = (-a + b) / math.sqrt(CK) # k, thw, hw

        # get proto-embedding affinity
        if norm_p == 'channel_num':
            C_emb = mk_emb.shape[1] # 后期使用通道数归一化
        else:
            C_emb = norm_p  # 前期代码使用key通道数64作归一化，为代码兼容默认norm_p=64

        a_p = mk_emb.pow(2).sum(1).unsqueeze(2)  # B, Thw, 1
        b_p = 2 * (mk_emb.transpose(1, 2) @ qk_emb)  # B, Thw, hw
        affinity_proto = (-a_p + b_p) / math.sqrt(C_emb)  # B, THW, HW

        # 整合两种affinity
        #print('affinity_point, min:{}, max:{}, mean:{}'.format(affinity_point.min(), affinity_point.max(), affinity_point.mean()))
        #print('affinity_proto, min:{}, max:{}, mean:{}'.format(affinity_proto.min(), affinity_proto.max(), affinity_proto.mean()))

        affinity = affinity_point * weight_point + affinity_proto * weight_proto

        # softmax operation; aligned the evaluation style
        affinity = softmax_w_top(affinity, top=self.top_k)  # k, thw, hw

        return affinity

    def _global_matching_2affinity(self, mk, qk, mk_emb1, qk_emb1, mk_emb2, qk_emb2, weight1, weight2, norm1, norm2):
        # mk: [k,c,thw] qk: [1,c,hw]
        # NE means number of elements -- typically T*H*W
        B, CK, NE = mk.shape

        # get point-wise affinity
        a = mk.pow(2).sum(1).unsqueeze(2)  # k,thw,1
        b = 2 * (mk.transpose(1, 2) @ qk)  #
        affinity_point = (-a + b) / math.sqrt(CK) # k, thw, hw

        # get proto-embedding affinity
        if norm1 == 'channel_num':
            C_emb1 = mk_emb1.shape[1] # 后期使用通道数归一化
        else:
            C_emb1 = norm1  # 前期代码使用key通道数64作归一化，为代码兼容默认norm_p=64

        a_p = mk_emb1.pow(2).sum(1).unsqueeze(2)  # B, Thw, 1
        b_p = 2 * (mk_emb1.transpose(1, 2) @ qk_emb1)  # B, Thw, hw
        affinity_proto = (-a_p + b_p) / math.sqrt(C_emb1)  # B, THW, HW

        # get position-embedding affinity
        if norm2 == 'channel_num':
            C_emb2 = mk_emb2.shape[1] # 后期使用通道数归一化
        else:
            C_emb2 = norm2  # 前期代码使用key通道数64作归一化，为代码兼容默认norm_p=64

        a_p = mk_emb2.pow(2).sum(1).unsqueeze(2)  # B, Thw, 1
        b_p = 2 * (mk_emb2.transpose(1, 2) @ qk_emb2)  # B, Thw, hw
        affinity_pos = (-a_p + b_p) / math.sqrt(C_emb2)  # B, THW, HW

        # 整合两种affinity
        #print('affinity_point, min:{}, max:{}, mean:{}'.format(affinity_point.min(), affinity_point.max(), affinity_point.mean()))
        #print('affinity_proto, min:{}, max:{}, mean:{}'.format(affinity_proto.min(), affinity_proto.max(), affinity_proto.mean()))

        affinity = affinity_point + affinity_proto * weight1 + affinity_pos * weight2

        # softmax operation; aligned the evaluation style
        affinity = softmax_w_top(affinity, top=self.top_k)  # k, thw, hw

        return affinity

    def _global_matching_ASPP(self, mk, qk):
        # NE means number of elements -- typically T*H*W
        # qk: B, 4, 64, HW, mk: B, 4, 64, THW
        B, _, CK, NE = mk.shape

        a = mk.pow(2).sum(2).unsqueeze(3)  # B, 4, THW, 1
        b = 2 * (mk.transpose(2, 3) @ qk)  # B, 4, THW, HW

        affinity = (-a + b) / math.sqrt(CK)  # B, 4, THW, HW
        affinity = softmax_w_top_ASPP(affinity, top=self.top_k)  # B, 4, THW, HW

        return affinity

    def _readout(self, affinity, mv):
        # affinity: k, THW, HW, mv: k, 512, THW
        # print(mv.shape, affinity.shape)
        return torch.bmm(mv, affinity)  # k, 512, HW

    def _readout_ASPP(self, affinity, mv, sum_flag):
        # affinity: k, 4, THW, HW, mv: k, 512, THW
        C = mv.shape[1]
        k, n_scales = affinity.shape[:2]

        mo = mv.view(k, C, -1).unsqueeze(1).expand(k, n_scales, C, -1)  # k, 4, C, THW
        mem = torch.matmul(mo, affinity)  # Weighted-sum k, 4, C, HW
        if sum_flag:
            mem_out = mem.sum(1)  # k, C, HW
        else:
            mem_out = mem.view(k, n_scales * C, -1)  # k, 4*C, HW
        return mem_out

    def match_memory(self, qk):
        k = self.num_objects
        _, _, h, w = qk.shape  # 1,c,h,w

        qk = qk.flatten(start_dim=2)  # 1,c,hw

        if self.temp_k is not None:
            mk = torch.cat([self.mem_k, self.temp_k], 2)  # 1,c,t,h,w
            mv = torch.cat([self.mem_v, self.temp_v], 2)
        else:
            mk = self.mem_k
            mv = self.mem_v
        affinity = self._global_matching(mk, qk)
        # print(affinity.shape, mk.shape, qk.shape)
        # One affinity for all
        # print(affinity.shape, mv.shape)
        readout_mem = self._readout(affinity.expand(k, -1, -1), mv)

        return readout_mem.view(k, self.CV, h, w)

    def match_memory_update(self, qk16_emb, mk16_emb):
        k = self.num_objects
        _, _, h, w = qk16_emb.shape  # k,c,h,w

        qk16_emb = qk16_emb.flatten(start_dim=2)  # k,c,hw
        # mk16_emb = mk16_emb.flatten(start_dim=2)  # k,c,thw

        if self.temp_k is not None:
            mv = torch.cat([self.mem_v, self.temp_v], 2)
        else:
            mv = self.mem_v

        affinity = self._global_matching(mk16_emb, qk16_emb)
        # One affinity for all
        # print(affinity.shape, mv.shape)
        readout_mem = self._readout(affinity.expand(k, -1, -1), mv)

        return readout_mem.view(k, self.CV, h, w)

    def match_memory_emb(self, qk, qk_emb, weight=1, norm_p=64, gauss=False):
        k = self.num_objects
        _, _, h, w = qk.shape  # 1,c,h,w

        qk = qk.flatten(start_dim=2)  # 1,c,hw
        qk_emb = qk_emb.flatten(start_dim=2)  # 1,c,hw

        if self.temp_k is not None:
            mk = torch.cat([self.mem_k, self.temp_k], 2)  # 1,c,t,h,w
            mv = torch.cat([self.mem_v, self.temp_v], 2)
            mk_emb = torch.cat([self.mem_f, self.temp_f], 2)
        else:
            mk = self.mem_k
            mv = self.mem_v
            mk_emb = self.mem_f

        mk_emb = mk_emb.flatten(start_dim=2)  # 1,c,hw
        # print(mk.shape, qk.shape, mk_emb.shape, qk_emb.shape)
        affinity = self._global_matching_emb(mk, qk, mk_emb, qk_emb, weight, norm_p, gauss)
        # print(affinity.shape, mk.shape, qk.shape)
        # One affinity for all
        # print(affinity.shape, mv.shape)
        readout_mem = self._readout(affinity.expand(k, -1, -1), mv)

        return readout_mem.view(k, self.CV, h, w)

    def match_memory_allmem_emb(self, qk, qk_emb, weight=1, norm_p=64, gauss=False):
        k = self.num_objects
        _, _, h, w = qk.shape  # 1,c,h,w

        qk = qk.flatten(start_dim=2)  # 1,c,hw
        qk_emb = qk_emb.flatten(start_dim=2)  # 1,c,hw

        if self.temp_k is not None:
            mk = torch.cat([self.mem_k, self.temp_k], 2)  # 1,c,t,h,w
            mv = torch.cat([self.mem_v, self.temp_v], 2)
            mk_emb = torch.cat([self.mem_e, self.temp_e], 2)
        else:
            mk = self.mem_k
            mv = self.mem_v
            mk_emb = self.mem_e

        mk_emb = mk_emb.flatten(start_dim=2)  # 1,c,hw
        # print(mk.shape, qk.shape, mk_emb.shape, qk_emb.shape)
        affinity = self._global_matching_emb(mk, qk, mk_emb, qk_emb, weight, norm_p, gauss)
        # print(affinity.shape, mk.shape, qk.shape)
        # One affinity for all
        # print(affinity.shape, mv.shape)
        readout_mem = self._readout(affinity.expand(k, -1, -1), mv)

        return readout_mem.view(k, self.CV, h, w)

    def match_memory_emb_v2(self, qk, qk_emb, weight_point, weight_proto, norm_p=64):
        k = self.num_objects
        _, _, h, w = qk.shape  # 1,c,h,w

        qk = qk.flatten(start_dim=2)  # 1,c,hw
        qk_emb = qk_emb.flatten(start_dim=2)  # 1,c,hw

        if self.temp_k is not None:
            mk = torch.cat([self.mem_k, self.temp_k], 2)  # 1,c,t,h,w
            mv = torch.cat([self.mem_v, self.temp_v], 2)
            mk_emb = torch.cat([self.mem_f, self.temp_f], 2)
        else:
            mk = self.mem_k
            mv = self.mem_v
            mk_emb = self.mem_f

        mk_emb = mk_emb.flatten(start_dim=2)  # 1,c,hw
        # print(mk.shape, qk.shape, mk_emb.shape, qk_emb.shape)
        affinity = self._global_matching_emb_v2(mk, qk, mk_emb, qk_emb, weight_point, weight_proto, norm_p)
        # print(affinity.shape, mk.shape, qk.shape)
        # One affinity for all
        # print(affinity.shape, mv.shape)
        readout_mem = self._readout(affinity.expand(k, -1, -1), mv)

        return readout_mem.view(k, self.CV, h, w)


    def match_memory_pos_affinity(self, qk, qk_pos, pos_generator, weight=1, norm_p=64):
        k = self.num_objects
        _, _, h, w = qk.shape  # 1,c,h,w

        qk = qk.flatten(start_dim=2)  # 1,c,hw
        qk_pos = qk_pos.flatten(start_dim=2)  # 1,c,hw

        if self.temp_k is not None:
            mk = torch.cat([self.mem_k, self.temp_k], 2)  # 1,c,t,h,w
            mv = torch.cat([self.mem_v, self.temp_v], 2)  # 1,c,
        else:
            mk = self.mem_k
            mv = self.mem_v

        mf = self.mem_f
        b, c, t = mf.shape[:3]
        mk_pos = pos_generator(mf[:,:,0]) # [1, 2, h, w]
        mk_pos = mk_pos.unsqueeze(2).expand(-1,-1,t,-1,-1) # [1,2,t,h,w]

        mk_pos = mk_pos.flatten(start_dim=2)  # 1,c,hw
        # print(mk.shape, qk.shape, mk_emb.shape, qk_emb.shape)
        affinity = self._global_matching_emb(mk, qk, mk_pos, qk_pos, weight, norm_p)
        # print(affinity.shape, mk.shape, qk.shape)
        # One affinity for all
        # print(affinity.shape, mv.shape)
        readout_mem = self._readout(affinity.expand(k, -1, -1), mv)

        return readout_mem.view(k, self.CV, h, w)

    def match_memory_2affinity(self, qk, qk_emb, pos_generator, weight1=0.1, weight2=0.1, norm1=64, norm2=2):
        k = self.num_objects
        _, _, h, w = qk.shape  # 1,c,h,w

        qk = qk.flatten(start_dim=2)  # 1,c,hw
        qk_emb = qk_emb.flatten(start_dim=2)  # 1,c,hw

        if self.temp_k is not None:
            mk = torch.cat([self.mem_k, self.temp_k], 2)  # 1,c,thw
            mv = torch.cat([self.mem_v, self.temp_v], 2)
            mk_emb = torch.cat([self.mem_f, self.temp_f], 2)
        else:
            mk = self.mem_k
            mv = self.mem_v
            mk_emb = self.mem_f

        mk_emb = mk_emb.flatten(start_dim=2)  # 1,c,hw

        # get position embedding
        mf = self.mem_f
        b, c, t = mf.shape[:3]
        qk_pos = pos_generator(mf[:,:,0]) # [1, 2, h, w], pos_emb的生成与内容无关，只与大小位置有关
        mk_pos = pos_generator(mf[:,:,0]) # [1, 2, h, w]
        mk_pos = mk_pos.unsqueeze(2).expand(-1,-1,t,-1,-1) # [1,2,t,h,w]

        qk_pos = qk_pos.flatten(start_dim=2)  # 1,2,hw
        mk_pos = mk_pos.flatten(start_dim=2)  # 1,2,thw
        #print(mk_pos.shape, qk_pos.shape)
        affinity = self._global_matching_2affinity(mk, qk, mk_emb, qk_emb, mk_pos, qk_pos, weight1, weight2, norm1, norm2)
        # print(affinity.shape, mk.shape, qk.shape)
        # One affinity for all
        # print(affinity.shape, mv.shape)
        readout_mem = self._readout(affinity.expand(k, -1, -1), mv)

        return readout_mem.view(k, self.CV, h, w)

    def match_memory_indep(self, qk):
        # qk: [k,64,h,w]
        k = self.num_objects
        _, _, h, w = qk.shape  # k,c,h,w
        # print('qk.shape:', qk.shape)

        qk = qk.flatten(start_dim=2)  # k,64,hw

        if self.temp_k is not None:
            mk = torch.cat([self.mem_k, self.temp_k], 2)  # k,64,t,h,w
            mv = torch.cat([self.mem_v, self.temp_v], 2)  # k,512,t,h,w
        else:
            mk = self.mem_k
            mv = self.mem_v

        readout_mem_list = []
        # print(qk.shape, mk.shape, mv.shape)
        for kk in range(k):
            affinity_k = self._global_matching(mk[kk:kk + 1], qk[kk:kk + 1])

            # One affinity for one object
            # print(affinity.shape, mv.shape)
            readout_mem_k = self._readout(affinity_k, mv[kk:kk + 1])  # 1,512,hw
            readout_mem_list.append(readout_mem_k)
        readout_mem = torch.cat(readout_mem_list)  # k,512,hw
        # print('readout_mem.shape:', readout_mem.shape)
        return readout_mem.view(k, self.CV, h, w)

    def match_memory_ASPP(self, qk, sum_flag=False):
        # qk: [1, 4, 64, 1, H, W], mk: [1, 4, 64, THW], mv: [k, 512, THW]
        k = self.num_objects
        _, _, _, _, h, w = qk.shape

        qk = qk.flatten(start_dim=3)  # B, 4, 64, HW

        if self.temp_k is not None:
            mk = torch.cat([self.mem_k, self.temp_k], 3)
            mv = torch.cat([self.mem_v, self.temp_v], 3)
        else:
            mk = self.mem_k
            mv = self.mem_v

        affinity = self._global_matching_ASPP(mk, qk)  # 1, 4, THW, HW

        # One affinity for all
        readout_mem = self._readout_ASPP(affinity.expand(k, -1, -1, -1), mv, sum_flag)  # k, 4*512, HW/ k, 512, HW

        if sum_flag:
            return readout_mem.view(k, self.CV, h, w)  # k, 512, H, W
        else:
            return readout_mem.view(k, self.CV * self.num_scales, h, w)  # k, 4*512, H, W

    def add_memory(self, key, value, is_temp=False):
        # Temp is for "last frame"
        # Not always used
        # But can always be flushed
        self.temp_k = None
        self.temp_v = None
        key = key.flatten(start_dim=2)  # 1,c,hw
        value = value.flatten(start_dim=2)  # k,c,hw

        if self.mem_k is None:
            # First frame, just shove it in
            self.mem_k = key
            self.mem_v = value
            self.CK = key.shape[1]
            self.CV = value.shape[1]
        else:
            if is_temp:
                self.temp_k = key  # B*64*T*H*W
                self.temp_v = value
            else:
                self.mem_k = torch.cat([self.mem_k, key], 2)  # 1,c,thw
                self.mem_v = torch.cat([self.mem_v, value], 2)  # k,c,thw

    def add_memory_ms(self, key, value, fts_ms, is_temp=False):
        # Temp is for "last frame"
        # Not always used
        # But can always be flushed
        self.temp_k = None
        self.temp_v = None
        self.fts_ms = None
        key = key.flatten(start_dim=2)  # 1,c,hw
        value = value.flatten(start_dim=2)  # k,c,hw
        fts_ms = fts_ms.flatten(start_dim=2) # k,1792,hw

        if self.mem_k is None:
            # First frame, just shove it in
            self.mem_k = key
            self.mem_v = value
            self.mem_ms = fts_ms
            self.CK = key.shape[1]
            self.CV = value.shape[1]
        else:
            if is_temp:
                self.temp_k = key  # B*64*T*H*W
                self.temp_v = value
                self.temp_ms = fts_ms
            else:
                self.mem_k = torch.cat([self.mem_k, key], 2)  # 1,c,thw
                self.mem_v = torch.cat([self.mem_v, value], 2)  # k,c,thw
                self.fts_ms = torch.cat([self.mem_ms, fts_ms], 2) # 1,1792,thw
 
    def add_memory_allmem(self, key, value, feat, mask=None, is_temp=False):
        # Temp is for "last frame"
        # Not always used
        # But can always be flushed
        self.temp_k = None
        self.temp_v = None
        self.temp_f = None
        self.temp_m = None
        key = key.flatten(start_dim=2)
        value = value.flatten(start_dim=2)

        if self.mem_k is None:
            # First frame, just shove it in
            self.mem_k = key
            self.mem_v = value
            self.mem_f = feat
            self.mem_m = mask
            self.CK = key.shape[1]
            self.CV = value.shape[1]
        else:
            if is_temp:
                self.temp_k = key  # B*64*THW
                self.temp_v = value  # B*512*THW
                self.temp_f = feat  # B*1024*T*H*W
                self.temp_m = mask  # K*1*T*H*W
            else:
                self.mem_k = torch.cat([self.mem_k, key], 2)  # B*64*THW
                self.mem_v = torch.cat([self.mem_v, value], 2)  # B*512*THW
                self.mem_f = torch.cat([self.mem_f, feat], 2)  # B*1024*T*H*W
                # print(self.mem_k.shape, mask.shape)
                if mask is not None:
                    self.mem_m = torch.cat([self.mem_m, mask], 2) # K*1*T*H*W

    def add_memory_allmem_emb(self, key, value, feat, mask, emb, is_temp=False):
        # Temp is for "last frame"
        # Not always used
        # But can always be flushed
        self.temp_k = None
        self.temp_v = None
        self.temp_f = None
        self.temp_m = None
        self.temp_e = None
        key = key.flatten(start_dim=2)
        value = value.flatten(start_dim=2)

        if self.mem_k is None:
            # First frame, just shove it in
            self.mem_k = key
            self.mem_v = value
            self.mem_f = feat
            self.mem_m = mask
            self.mem_e = emb
            self.CK = key.shape[1]
            self.CV = value.shape[1]
        else:
            if is_temp:
                self.temp_k = key  # B*64*THW
                self.temp_v = value  # B*512*THW
                self.temp_f = feat  # B*1024*T*H*W
                self.temp_m = mask  # K*1*T*H*W
                self.temp_e = emb
            else:
                self.mem_k = torch.cat([self.mem_k, key], 2)  # B*64*THW
                self.mem_v = torch.cat([self.mem_v, value], 2)  # B*512*THW
                self.mem_f = torch.cat([self.mem_f, feat], 2)  # B*1024*T*H*W
                self.mem_m = torch.cat([self.mem_m, mask], 2) # K*1*T*H*W
                self.mem_e = torch.cat([self.mem_e, emb], 2) # K*num_sp*T*H*W

    def add_mem_idx(self, mem_idx):
        self.mem_ind.append(mem_idx)

    def add_memory_ASPP(self, key, value, is_temp=False):
        # Temp is for "last frame"
        # Not always used
        # But can always be flushed
        # key: [B, 4, 64, 1, H, W], value: [B, 512, H, W]
        self.temp_k = None
        self.temp_v = None
        key = key.flatten(start_dim=3)  # 1, 4, 64, HW
        value = value.flatten(start_dim=2)  # K, 512, HW

        if self.mem_k is None:
            # First frame, just shove it in
            self.mem_k = key
            self.mem_v = value
            self.CK = key.shape[2]
            self.CV = value.shape[1]
            self.num_scales = key.shape[1]
        else:
            if is_temp:
                self.temp_k = key  # 1, 4, 64, HW
                self.temp_v = value  # K, 512, HW
            else:
                self.mem_k = torch.cat([self.mem_k, key], 3)  # 1, 4, 64, THW
                self.mem_v = torch.cat([self.mem_v, value], 2)  # K, 512, THW

    def get_guide_prob(self, sp_center, query_feat):
        # sp_center = self.mem_p
        sp_center_rep = sp_center[..., None, None].repeat(1, 1, query_feat.size(1), query_feat.size(2))
        cos_sim_map = F.cosine_similarity(sp_center_rep, query_feat.unsqueeze(1), dim=0, eps=1e-7)  # num_sp x h x w
        prob_map = cos_sim_map.sum(0, keepdim=True)  # 1 x h x w

        guide_map = cos_sim_map.max(0)[1]  # h x w
        sp_guide_feat = sp_center[:, guide_map]  # c x h x w
        guide_feat = torch.cat([query_feat, sp_guide_feat], dim=0)  # 2c x h x w

        return guide_feat.unsqueeze(0), prob_map.unsqueeze(0)

    def get_guide_prob_new(self, sp_center, query_feat, num_proto):
        # sp_center = self.mem_p
        sp_center_rep = sp_center[..., None, None].repeat(1, 1, query_feat.size(1), query_feat.size(2))
        cos_sim_map = F.cosine_similarity(sp_center_rep, query_feat.unsqueeze(1), dim=0, eps=1e-7)  # num_sp x h x w
        prob_map = cos_sim_map.sum(0, keepdim=True)  # 1 x h x w
        prob_map = prob_map / num_proto

        guide_map = cos_sim_map.max(0)[1]  # h x w
        sp_guide_feat = sp_center[:, guide_map]  # c x h x w

        return sp_guide_feat.unsqueeze(0), prob_map.unsqueeze(0)

    def add_proto(self, sp_center_list):
        # mem_p: [n_objects, n_dim, n_protos]
        if self.mem_p is None:
            self.mem_p = sp_center_list
            self.num_proto += 1
        else:
            n_objects = len(sp_center_list)
            for n in range(n_objects):
                if sp_center_list[n] is not None:
                    self.mem_p[n] = torch.cat([self.mem_p[n], sp_center_list[n]], 1)  # [n_dim, n_protos]
                    self.num_proto[n] += 1

    def add_proto_yv(self, sp_center):
        # mem_p: [n_objects, n_dim, n_protos]
        if self.mem_p is None:
            self.mem_p = sp_center
            self.num_proto = 1
        else:
            self.mem_p = torch.cat([self.mem_p, sp_center], 1)  # [n_dim, n_protos]
            self.num_proto += 1

    def add_proto_fgbg(self, sp_center_list_fg, sp_center_list_bg, mode='mean'):
        # sp_center_list_fg[n] : [1, n_dim, 1, n_protos]
        if self.mem_p_fg_all is None:
            self.mem_p_fg_all = sp_center_list_fg #.copy()
            self.mem_p_fg = sp_center_list_fg #.copy()
            self.mem_p_fg_ema = sp_center_list_fg #.copy()
            self.num_proto_fg += 1
        else:
            n_objects = len(sp_center_list_fg)
            for n in range(n_objects):
                # print(self.mem_p_fg_all[n].shape, sp_center_list_fg[n].shape)
                self.mem_p_fg_all[n] = torch.cat([self.mem_p_fg_all[n], sp_center_list_fg[n]],
                                                 2)  # [1, n_dim, T, n_protos]
                # print('n:', n)
                if mode == 'cat':
                    self.mem_p_fg[n] = self.mem_p_fg_all[n]
                if mode == 'mean':
                    self.mem_p_fg[n] = self.mem_p_fg_all[n].mean(dim=2, keepdim=True)  # [1, n_dim, 1, n_protos]
                if mode == 'first':
                    self.mem_p_fg[n] = self.mem_p_fg_all[n][:,:,0].unsqueeze(2)
                if mode == 'last':
                    self.mem_p_fg[n] = self.mem_p_fg_all[n][:,:,-1].unsqueeze(2)
                    # self.mem_p_fg[n] = torch.mean(self.mem_p_fg[n], 2, keepdim=True)
                if mode == 'first_last':
                    self.mem_p_fg[n] = torch.stack([self.mem_p_fg_all[n][:, :, 0], self.mem_p_fg_all[n][:, :, -1]],
                                                   2)  # [1, n_dim, 2, n_protos]
                if mode == 'first_last_mean':
                    if self.mem_p_fg_all[n].shape[2] <= 2:
                        mean_p_fg = self.mem_p_fg_all[n][:, :, 0]
                    else:
                        mean_p_fg = self.mem_p_fg_all[n][:, :, 1:-1].mean(dim=2)
                    self.mem_p_fg[n] = torch.stack([self.mem_p_fg_all[n][:, :, 0], mean_p_fg, self.mem_p_fg_all[n][:, :, -1]],
                                                       2)  # [1, n_dim, 3, n_protos]                           
                if mode == 'first_mean':
                    mean_p_fg = self.mem_p_fg_all[n][:, :, 1:].mean(dim=2)
                    self.mem_p_fg[n] = torch.stack([self.mem_p_fg_all[n][:, :, 0], mean_p_fg], 
                                                   2)  # [1, n_dim, 2, n_protos]
                if mode == 'double_first_mean':
                    if self.mem_p_bg_all[n].shape[2] == 1:
                        mean_p_fg = self.mem_p_fg_all[n][:, :, 0]
                    else:
                        mean_p_fg = self.mem_p_fg_all[n][:, :, 1:].mean(dim=2)
                    self.mem_p_fg[n] = torch.stack([self.mem_p_fg_all[n][:, :, 0], self.mem_p_fg_all[n][:, :, 0], mean_p_fg], 
                                                   2)  # [1, n_dim, 3, n_protos]

                if mode == 'ema': # ema: exponential moving average
                    self.mem_p_fg_ema[n] = self.w * self.mem_p_fg_ema[n] + (1-self.w) * sp_center_list_fg[n]
                    self.mem_p_fg[n] = self.mem_p_fg_ema[n]

                if mode == 'first_ema':
                    #print(n, self.w, self.mem_p_fg_ema[n].shape, 1-self.w, sp_center_list_fg.shape)
                    self.mem_p_fg_ema[n] = self.w * self.mem_p_fg_ema[n] + (1-self.w) * sp_center_list_fg[n]
                    self.mem_p_fg[n] = torch.cat([self.mem_p_fg_all[n][:,:,0].unsqueeze(2), self.mem_p_fg_ema[n]],
                                              2)   # [1, n_dim, 2, n_protos]
                self.num_proto_fg[n] += 1
            # print('3', self.mem_p_fg_all[n].shape)

        if self.mem_p_bg_all is None:
            self.mem_p_bg_all = sp_center_list_bg #.copy()
            self.mem_p_bg = sp_center_list_bg #.copy()
            self.mem_p_bg_ema = sp_center_list_bg #.copy()
            self.num_proto_bg += 1
        else:
            n_objects = len(sp_center_list_bg)
            for n in range(n_objects):
                self.mem_p_bg_all[n] = torch.cat([self.mem_p_bg_all[n], sp_center_list_bg[n]],
                                                 2)  # [1, n_dim, T, n_protos]
                if mode == 'cat':
                    self.mem_p_bg[n] = self.mem_p_bg_all[n]
                if mode == 'mean':
                    self.mem_p_bg[n] = self.mem_p_bg_all[n].mean(dim=2, keepdim=True)  # [1, n_dim, 1, n_protos]
                if mode == 'first':
                    self.mem_p_bg[n] = self.mem_p_bg_all[n][:, :, 0].unsqueeze(2)
                if mode == 'last':
                    self.mem_p_bg[n] = self.mem_p_bg_all[n][:, :, -1].unsqueeze(2)
                    # self.mem_p_bg[n] = torch.mean(self.mem_p_bg[n], 2, keepdim=True)
                if mode == 'first_last':
                    self.mem_p_bg[n] = torch.stack([self.mem_p_bg_all[n][:, :, 0], self.mem_p_bg_all[n][:, :, -1]],
                                                   2)  # [1, n_dim, 2, n_protos]
                if mode == 'first_last_mean':
                    if self.mem_p_bg_all[n].shape[2] <= 2:
                        mean_p_bg = self.mem_p_bg_all[n][:, :, 0]
                    else:
                        mean_p_bg = self.mem_p_bg_all[n][:, :, 1:-1].mean(dim=2)
                    self.mem_p_bg[n] = torch.stack([self.mem_p_bg_all[n][:, :, 0], mean_p_bg, self.mem_p_bg_all[n][:, :, -1]],
                                                   2)  # [1, n_dim, 3, n_protos]

                if mode == 'first_mean':
                    mean_p_bg[n] = self.mem_p_bg_all[n][:, :, 1:].mean(dim=2)
                    self.mem_p_bg[n] = torch.stack([self.mem_p_bg_all[n][:, :, 0], mean_p_bg],
                                                   2)  # [1, n_dim, 2, n_protos]

                if mode == 'double_first_mean':
                    if self.mem_p_bg_all[n].shape[2] == 1:
                        mean_p_bg[n] = self.mem_p_bg_all[n][:, :, 0].mean(dim=2)
                    else:
                        mean_p_bg = self.mem_p_bg_all[n][:, :, 1:].mean(dim=2)
                    self.mem_p_bg[n] = torch.stack([self.mem_p_bg_all[n][:, :, 0], self.mem_p_bg_all[n][:, :, 0], mean_p_bg],
                                                   2)  # [1, n_dim, 3, n_protos]
                if mode == 'ema': # ema: exponential moving average
                    self.mem_p_bg_ema[n] = self.w * self.mem_p_bg_ema[n] + (1-self.w) * sp_center_list_bg[n]
                    self.mem_p_bg[n] = self.mem_p_bg_ema[n]

                if mode == 'first_ema':
                    self.mem_p_bg_ema[n] = self.w * self.mem_p_bg_ema[n] + (1-self.w) * sp_center_list_bg[n]
                    self.mem_p_bg[n] = torch.cat([self.mem_p_bg_all[n][:,:,0].unsqueeze(2), self.mem_p_bg_ema[n]],
                                              2)   # [1, n_dim, 2, n_protos]
                self.num_proto_bg[n] += 1

    def add_proto_fgbg_select(self, sp_center_list_fg, sp_center_list_bg, update_index, mode='mean'):
        # sp_center_list_fg[n] : [1, n_dim, 1, n_protos]
        if self.mem_p_fg_all is None:
            self.mem_p_fg_all = sp_center_list_fg.copy()
            self.mem_p_fg = sp_center_list_fg.copy()
            self.mem_p_fg_ema = sp_center_list_fg.copy()
            self.num_proto_fg += 1
        else:
            n_objects = len(sp_center_list_fg)
            for n in range(n_objects):
                if update_index[n] == 1:
                    # print(self.mem_p_fg_all[n].shape, sp_center_list_fg[n].shape)
                    self.mem_p_fg_all[n] = torch.cat([self.mem_p_fg_all[n], sp_center_list_fg[n]],
                                                     2)  # [1, n_dim, T, n_protos]
                    # print('n:', n)
                    if mode == 'cat':
                        self.mem_p_fg[n] = self.mem_p_fg_all[n]
                    if mode == 'mean':
                        self.mem_p_fg[n] = self.mem_p_fg_all[n].mean(dim=2, keepdim=True)  # [1, n_dim, 1, n_protos]
                    if mode == 'first':
                        self.mem_p_fg[n] = self.mem_p_fg_all[n][:,:,0].unsqueeze(2)
                    if mode == 'last':
                        self.mem_p_fg[n] = self.mem_p_fg_all[n][:,:,-1].unsqueeze(2)
                        # self.mem_p_fg[n] = torch.mean(self.mem_p_fg[n], 2, keepdim=True)
                    if mode == 'first_last':
                        self.mem_p_fg[n] = torch.stack([self.mem_p_fg_all[n][:, :, 0], self.mem_p_fg_all[n][:, :, -1]],
                                                       2)  # [1, n_dim, 2, n_protos]
                    if mode == 'first_last_mean':
                        if self.mem_p_fg_all[n].shape[2] <= 2:
                            mean_p_fg = self.mem_p_fg_all[n][:, :, 0]
                        else:
                            mean_p_fg = self.mem_p_fg_all[n][:, :, 1:-1].mean(dim=2)
                        self.mem_p_fg[n] = torch.stack([self.mem_p_fg_all[n][:, :, 0], mean_p_fg, self.mem_p_fg_all[n][:, :, -1]],
                                                           2)  # [1, n_dim, 3, n_protos]                           
                    if mode == 'first_mean':
                        mean_p_fg = self.mem_p_fg_all[n][:, :, 1:].mean(dim=2)
                        self.mem_p_fg[n] = torch.stack([self.mem_p_fg_all[n][:, :, 0], mean_p_fg], 
                                                       2)  # [1, n_dim, 2, n_protos]
                    if mode == 'double_first_mean':
                        if self.mem_p_bg_all[n].shape[2] == 1:
                            mean_p_fg = self.mem_p_fg_all[n][:, :, 0]
                        else:
                            mean_p_fg = self.mem_p_fg_all[n][:, :, 1:].mean(dim=2)
                        self.mem_p_fg[n] = torch.stack([self.mem_p_fg_all[n][:, :, 0], self.mem_p_fg_all[n][:, :, 0], mean_p_fg], 
                                                       2)  # [1, n_dim, 3, n_protos]

                    if mode == 'ema': # ema: exponential moving average
                        self.mem_p_fg_ema[n] = self.w * self.mem_p_fg_ema[n] + (1-self.w) * sp_center_list_fg[n]
                        self.mem_p_fg[n] = self.mem_p_fg_ema[n]

                    if mode == 'first_ema':
                        #print(n, self.w, self.mem_p_fg_ema[n].shape, 1-self.w, sp_center_list_fg.shape)
                        self.mem_p_fg_ema[n] = self.w * self.mem_p_fg_ema[n] + (1-self.w) * sp_center_list_fg[n]
                        self.mem_p_fg[n] = torch.cat([self.mem_p_fg_all[n][:,:,0].unsqueeze(2), self.mem_p_fg_ema[n]],
                                                  2)   # [1, n_dim, 2, n_protos]

                    self.num_proto_fg[n] += 1
                # print('3', self.mem_p_fg_all[n].shape)

        if self.mem_p_bg_all is None:
            self.mem_p_bg_all = sp_center_list_bg.copy()
            self.mem_p_bg = sp_center_list_bg.copy()
            self.mem_p_bg_ema = sp_center_list_bg.copy()
            self.num_proto_bg += 1
        else:
            n_objects = len(sp_center_list_bg)
            for n in range(n_objects):
                if update_index[n] == 1:
                    self.mem_p_bg_all[n] = torch.cat([self.mem_p_bg_all[n], sp_center_list_bg[n]],
                                                     2)  # [1, n_dim, T, n_protos]
                    if mode == 'cat':
                        self.mem_p_bg[n] = self.mem_p_bg_all[n]
                    if mode == 'mean':
                        self.mem_p_bg[n] = self.mem_p_bg_all[n].mean(dim=2, keepdim=True)  # [1, n_dim, 1, n_protos]
                    if mode == 'first':
                        self.mem_p_bg[n] = self.mem_p_bg_all[n][:, :, 0].unsqueeze(2)
                    if mode == 'last':
                        self.mem_p_bg[n] = self.mem_p_bg_all[n][:, :, -1].unsqueeze(2)
                        # self.mem_p_bg[n] = torch.mean(self.mem_p_bg[n], 2, keepdim=True)
                    if mode == 'first_last':
                        self.mem_p_bg[n] = torch.stack([self.mem_p_bg_all[n][:, :, 0], self.mem_p_bg_all[n][:, :, -1]],
                                                       2)  # [1, n_dim, 2, n_protos]
                    if mode == 'first_last_mean':
                        if self.mem_p_bg_all[n].shape[2] <= 2:
                            mean_p_bg = self.mem_p_bg_all[n][:, :, 0]
                        else:
                            mean_p_bg = self.mem_p_bg_all[n][:, :, 1:-1].mean(dim=2)
                        self.mem_p_bg[n] = torch.stack([self.mem_p_bg_all[n][:, :, 0], mean_p_bg, self.mem_p_bg_all[n][:, :, -1]],
                                                       2)  # [1, n_dim, 3, n_protos]
    
                    if mode == 'first_mean':
                        mean_p_bg[n] = self.mem_p_bg_all[n][:, :, 1:].mean(dim=2)
                        self.mem_p_bg[n] = torch.stack([self.mem_p_bg_all[n][:, :, 0], mean_p_bg],
                                                       2)  # [1, n_dim, 2, n_protos]
    
                    if mode == 'double_first_mean':
                        if self.mem_p_bg_all[n].shape[2] == 1:
                            mean_p_bg[n] = self.mem_p_bg_all[n][:, :, 0].mean(dim=2)
                        else:
                            mean_p_bg = self.mem_p_bg_all[n][:, :, 1:].mean(dim=2)
                        self.mem_p_bg[n] = torch.stack([self.mem_p_bg_all[n][:, :, 0], self.mem_p_bg_all[n][:, :, 0], mean_p_bg],
                                                       2)  # [1, n_dim, 3, n_protos]
                    if mode == 'ema': # ema: exponential moving average
                        self.mem_p_bg_ema[n] = self.w * self.mem_p_bg_ema[n] + (1-self.w) * sp_center_list_bg[n]
                        self.mem_p_bg[n] = self.mem_p_bg_ema[n]

                    if mode == 'first_ema':
                        self.mem_p_bg_ema[n] = self.w * self.mem_p_bg_ema[n] + (1-self.w) * sp_center_list_bg[n]
                        self.mem_p_bg[n] = torch.cat([self.mem_p_bg_all[n][:,:,0].unsqueeze(2), self.mem_p_bg_ema[n]],
                                                  2)   # [1, n_dim, 2, n_protos]
                    self.num_proto_bg[n] += 1

    def add_proto_yv_fgbg(self, sp_center_fg, sp_center_bg, mode='mean'):
        # mem_p: [n_objects, n_dim, n_protos]
        if self.mem_p_fg_all is None:
            self.mem_p_fg_all = sp_center_fg  # .copy()
            self.mem_p_fg = sp_center_fg  # .copy()
            self.num_proto_fg = 1
        else:
            self.mem_p_fg_all = torch.cat([self.mem_p_fg_all, sp_center_fg], 2)  # [1, n_dim, T, n_protos]
            if mode == 'cat':
                self.mem_p_fg = self.mem_p_fg_all
            if mode == 'mean':
                self.mem_p_fg = self.mem_p_fg_all.mean(dim=2, keepdim=True)  # [1, n_dim, 1, n_protos]

            self.num_proto_fg += 1

        if self.mem_p_bg_all is None:
            self.mem_p_bg_all = sp_center_bg  # .copy()
            self.mem_p_bg = sp_center_bg  # .copy()
            self.num_proto_bg = 1
        else:
            self.mem_p_bg_all = torch.cat([self.mem_p_bg_all, sp_center_bg], 2)  # [1, n_dim, T, n_protos]
            if mode == 'cat':
                self.mem_p_bg = self.mem_p_bg_all
            if mode == 'mean':
                self.mem_p_bg = self.mem_p_bg_all.mean(dim=2, keepdim=True)  # [1, n_dim, 1, n_protos]

            self.num_proto_bg += 1

    def add_proto_yv_fgbg_new(self, sp_center_fg, sp_center_bg, mode='ema'):
        # mem_p: [n_objects, n_dim, n_protos]
        if self.mem_p_fg_all is None:
            self.mem_p_fg_all = sp_center_fg.clone()
            self.mem_p_fg = sp_center_fg.clone()
            self.mem_p_fg_ema = sp_center_fg.clone()
            self.num_proto_fg = 1
        else:
            self.mem_p_fg_all = torch.cat([self.mem_p_fg_all, sp_center_fg], 2)  # [1, n_dim, T, n_protos]
            if mode == 'cat':
                self.mem_p_fg = self.mem_p_fg_all
            if mode == 'mean':
                self.mem_p_fg = self.mem_p_fg_all.mean(dim=2, keepdim=True)  # [1, n_dim, 1, n_protos]
            if mode == 'ema': # ema: exponential moving average
                self.mem_p_fg_ema = self.w * self.mem_p_fg_ema + (1-self.w) * sp_center_fg
                self.mem_p_fg = self.mem_p_fg_ema

            self.num_proto_fg += 1

        if self.mem_p_bg_all is None:
            self.mem_p_bg_all = sp_center_bg.clone()
            self.mem_p_bg = sp_center_bg.clone()
            self.mem_p_bg_ema = sp_center_bg.clone()
            self.num_proto_bg = 1
        else:
            self.mem_p_bg_all = torch.cat([self.mem_p_bg_all, sp_center_bg], 2)  # [1, n_dim, T, n_protos]
            if mode == 'cat':
                self.mem_p_bg = self.mem_p_bg_all
            if mode == 'mean':
                self.mem_p_bg = self.mem_p_bg_all.mean(dim=2, keepdim=True)  # [1, n_dim, 1, n_protos]
            if mode == 'ema': # ema: exponential moving average
                self.mem_p_bg_ema = self.w * self.mem_p_bg_ema + (1-self.w) * sp_center_bg
                self.mem_p_bg = self.mem_p_bg_ema

            self.num_proto_bg += 1

    # 添加两组不同方式生成的proto
    def add_proto_fgbg_2group(self, sp_center_list_fg, sp_center_list_bg, sp_center_list_fg2, sp_center_list_bg2,
                              mode='mean'):
        # sp_center_list_fg[n] : [1, n_dim, 1, n_protos]

        # 1st group-bg
        if self.mem_p_fg_all is None:
            self.mem_p_fg_all = sp_center_list_fg  # .copy()
            self.mem_p_fg = sp_center_list_fg  # .copy()
            self.num_proto_fg += 1
        else:
            n_objects = len(sp_center_list_fg)
            for n in range(n_objects):
                self.mem_p_fg_all[n] = torch.cat([self.mem_p_fg_all[n], sp_center_list_fg[n]],
                                                 2)  # [1, n_dim, T, n_protos]
                if mode == 'cat':
                    self.mem_p_fg[n] = self.mem_p_fg_all[n]
                if mode == 'mean':
                    self.mem_p_fg[n] = self.mem_p_fg_all[n].mean(dim=2, keepdim=True)  # [1, n_dim, 1, n_protos]
                self.num_proto_fg[n] += 1
        # 1st group-bg
        if self.mem_p_bg_all is None:
            self.mem_p_bg_all = sp_center_list_bg  # .copy()
            self.mem_p_bg = sp_center_list_bg  # .copy()
            self.num_proto_bg += 1
        else:
            n_objects = len(sp_center_list_bg)
            for n in range(n_objects):
                self.mem_p_bg_all[n] = torch.cat([self.mem_p_bg_all[n], sp_center_list_bg[n]],
                                                 2)  # [1, n_dim, T, n_protos]
                if mode == 'cat':
                    self.mem_p_bg[n] = self.mem_p_bg_all[n]
                if mode == 'mean':
                    self.mem_p_bg[n] = self.mem_p_bg_all[n].mean(dim=2, keepdim=True)  # [1, n_dim, 1, n_protos]
                self.num_proto_bg[n] += 1
        # 2nd group-fg
        if self.mem_p_fg_all2 is None:
            self.mem_p_fg_all2 = sp_center_list_fg2  # .copy()
            self.mem_p_fg2 = sp_center_list_fg2  # .copy()
            self.num_proto_fg2 += 1
        else:
            n_objects = len(sp_center_list_fg2)
            for n in range(n_objects):
                self.mem_p_fg_all2[n] = torch.cat([self.mem_p_fg_all2[n], sp_center_list_fg2[n]],
                                                  2)  # [1, n_dim, T, n_protos]
                if mode == 'cat':
                    self.mem_p_fg2[n] = self.mem_p_fg_all2[n]
                if mode == 'mean':
                    self.mem_p_fg2[n] = self.mem_p_fg_all2[n].mean(dim=2, keepdim=True)  # [1, n_dim, 1, n_protos]
                self.num_proto_fg2[n] += 1
        # 2nd group-bg
        if self.mem_p_bg_all2 is None:
            self.mem_p_bg_all2 = sp_center_list_bg2  # .copy()
            self.mem_p_bg2 = sp_center_list_bg2  # .copy()
            self.num_proto_bg2 += 1
        else:
            n_objects = len(sp_center_list_bg2)
            for n in range(n_objects):
                self.mem_p_bg_all2[n] = torch.cat([self.mem_p_bg_all2[n], sp_center_list_bg2[n]],
                                                  2)  # [1, n_dim, T, n_protos]
                if mode == 'cat':
                    self.mem_p_bg2[n] = self.mem_p_bg_all2[n]
                if mode == 'mean':
                    self.mem_p_bg2[n] = self.mem_p_bg_all2[n].mean(dim=2, keepdim=True)  # [1, n_dim, 1, n_protos]
                self.num_proto_bg2[n] += 1

    def add_proto_yv_fgbg_2group(self, sp_center_fg, sp_center_bg, sp_center_fg2, sp_center_bg2, mode='mean'):
        # mem_p: [n_objects, n_dim, n_protos]     
        # 1st group-fg
        if self.mem_p_fg_all is None:
            self.mem_p_fg_all = sp_center_fg  # .copy()
            self.mem_p_fg = sp_center_fg  # .copy()
            self.num_proto_fg = 1
        else:
            self.mem_p_fg_all = torch.cat([self.mem_p_fg_all, sp_center_fg], 2)  # [1, n_dim, T, n_protos]
            if mode == 'cat':
                self.mem_p_fg = self.mem_p_fg_all
            if mode == 'mean':
                self.mem_p_fg = self.mem_p_fg_all.mean(dim=2, keepdim=True)  # [1, n_dim, 1, n_protos]

            self.num_proto_fg += 1
            # 1st group-bg
        if self.mem_p_bg_all is None:
            self.mem_p_bg_all = sp_center_bg  # .copy()
            self.mem_p_bg = sp_center_bg  # .copy()
            self.num_proto_bg = 1
        else:
            self.mem_p_bg_all = torch.cat([self.mem_p_bg_all, sp_center_bg], 2)  # [1, n_dim, T, n_protos]
            if mode == 'cat':
                self.mem_p_bg = self.mem_p_bg_all
            if mode == 'mean':
                self.mem_p_bg = self.mem_p_bg_all.mean(dim=2, keepdim=True)  # [1, n_dim, 1, n_protos]

            self.num_proto_bg += 1
        # 2nd group-fg
        if self.mem_p_fg_all2 is None:
            self.mem_p_fg_all2 = sp_center_fg2  # .copy()
            self.mem_p_fg2 = sp_center_fg2  # .copy()
            self.num_proto_fg2 = 1
        else:
            self.mem_p_fg_all2 = torch.cat([self.mem_p_fg_all2, sp_center_fg2], 2)  # [1, n_dim, T, n_protos]
            if mode == 'cat':
                self.mem_p_fg2 = self.mem_p_fg_all2
            if mode == 'mean':
                self.mem_p_fg2 = self.mem_p_fg_all2.mean(dim=2, keepdim=True)  # [1, n_dim, 1, n_protos]

            self.num_proto_fg2 += 1
        # 2nd group-bg                
        if self.mem_p_bg_all2 is None:
            self.mem_p_bg_all2 = sp_center_bg2  # .copy()
            self.mem_p_bg2 = sp_center_bg2  # .copy()
            self.num_proto_bg2 = 1
        else:
            self.mem_p_bg_all2 = torch.cat([self.mem_p_bg_all2, sp_center_bg2], 2)  # [1, n_dim, T, n_protos]
            if mode == 'cat':
                self.mem_p_bg2 = self.mem_p_bg_all2
            if mode == 'mean':
                self.mem_p_bg2 = self.mem_p_bg_all2.mean(dim=2, keepdim=True)  # [1, n_dim, 1, n_protos]

            self.num_proto_bg2 += 1

    def add_global_vec(self, vec_fg, vec_bg, num_pix_fg, num_pix_bg):
        # vec_fg/bg: [n_objects, C], num_pix_fg/bg: [n_objects, 1]
        # print(vec_fg.shape, vec_bg.shape, num_pix_fg.shape, num_pix_bg.shape)
        if self.mem_vec_fg is None:
            self.mem_vec_fg = vec_fg
        else:
            n_object = len(vec_fg)
            for n in range(n_object):
                v1, n1, v2, n2 = self.mem_vec_fg[n], self.num_pix_fg_all[n], vec_fg[n], num_pix_fg[n]
                self.mem_vec_fg[n] = (v1 * n1 + v2 * n2) / (n1 + n2)
                self.num_pix_fg_all[n] = self.num_pix_fg_all[n] + num_pix_fg[n]

        if self.mem_vec_bg is None:
            self.mem_vec_bg = vec_bg
        else:
            n_objects = len(vec_bg)
            for n in range(n_objects):
                v1_b, n1_b, v2_b, n2_b = self.mem_vec_bg[n], self.num_pix_bg_all[n], vec_bg[n], num_pix_bg[n]
                self.mem_vec_bg[n] = (v1_b * n1_b + v2_b * n2_b) / (n1_b + n2_b)
                self.num_pix_bg_all[n] = self.num_pix_bg_all[n] + num_pix_bg[n]

        # print('self.num_pix_fg_all:{}, self.num_pix_bg_all:{}'.format(self.num_pix_fg_all, self.num_pix_bg_all))
        
    def add_proto_fgbg_conf(self, sp_center_list_fg, sp_center_list_bg, conf_list_fg, conf_list_bg, mode='mean'):
        # sp_center_list_fg[n]: [1, n_dim, 1, n_protos]
        # conf_list_fg[n]: [n_proto]
        
        thre = 0.6
        
        n_objects = len(sp_center_list_fg)
        n_protos = sp_center_list_fg[0].shape[3]

        num_proto_fg = torch.zeros(n_objects, n_protos)
        num_proto_bg = torch.zeros(n_objects, n_protos)

        if self.mem_p_fg_all is None:
            for n in range(n_objects): # 第n个目标
                for p in range(n_protos): # 第p个proto序号
                    # 两层级list，mem_p_fg_all[n][p]为第n个目标的第p个序号对应的全部proto，不同序号Proto数量可不相同
                    # mem_p_fg_all[n][p].shape: [1, n_dim, 1]
                    self.mem_p_fg_all = [list(f.unbind(dim=3)) for f in sp_center_list_fg]
            self.mem_p_fg = self.mem_p_fg_all.copy()
            num_proto_fg[n] += 1
            
            self.conf_list_fg_all = [f.unsqueeze(0) for f in conf_list_fg]
        else:
            for n in range(n_objects):
                for p in range(n_protos):
                    #print(conf_list_fg[n][p])
                    if conf_list_fg[n][p] > thre: # 仅保置信度留大于阈值的proto
                        #print(self.mem_p_fg_all[n][p].shape, sp_center_list_fg[n][:,:,:,p].shape)
                        self.mem_p_fg_all[n][p] = torch.cat([self.mem_p_fg_all[n][p], sp_center_list_fg[n][:,:,:,p]],
                                                     2)  # [1, n_dim, T]
                        #print('n:{}, p:{}, mem_p_fg_all.shape:{}'.format(n, p, self.mem_p_fg_all[n][p].shape))
       
                        if mode == 'cat':
                            self.mem_p_fg[n][p] = self.mem_p_fg_all[n][p]
                        elif mode == 'mean':
                            self.mem_p_fg[n][p] = self.mem_p_fg_all[n][p].mean(dim=2, keepdim=True)  # [1, n_dim, 1]
                        else:
                            raise ValueError('mode error!')
    
                        num_proto_fg[n][p] += 1

                self.conf_list_fg_all[n] = torch.cat([self.conf_list_fg_all[n], conf_list_fg[n].unsqueeze(0)], 0) # [T, n_protos]

        n_objects = len(sp_center_list_bg)
        n_protos = sp_center_list_bg[0].shape[3]
        if self.mem_p_bg_all is None:
            for n in range(n_objects): # 第n个目标
                for p in range(n_protos): # 第p个proto序号
                    # 两层级list，mem_p_bg_all[n][p]为第n个目标的第p个序号对应的全部proto，不同序号Proto数量可不相同
                    # mem_p_bg_all[n][p].shape: [1, n_dim, 1]
                    self.mem_p_bg_all = [list(f.unbind(dim=3)) for f in sp_center_list_bg]
            self.mem_p_bg = self.mem_p_bg_all.copy()
            num_proto_bg[n] += 1

            self.conf_list_bg_all = [f.unsqueeze(0) for f in conf_list_bg]
        else:
            for n in range(n_objects):
                for p in range(n_protos):
                    if conf_list_bg[n][p] > thre: # 仅保置信度留大于阈值的proto
                        self.mem_p_bg_all[n][p] = torch.cat([self.mem_p_bg_all[n][p], sp_center_list_bg[n][:,:,:,p]],
                                                     2)  # [1, n_dim, T]
                        #print('n:{}, p:{}, mem_p_bg_all.shape:{}'.format(n, p, self.mem_p_bg_all[n][p].shape))
     
                        if mode == 'cat':
                            self.mem_p_bg[n][p] = self.mem_p_bg_all[n][p]
                        elif mode == 'mean':
                            self.mem_p_bg[n][p] = self.mem_p_bg_all[n][p].mean(dim=2, keepdim=True)  # [1, n_dim, 1]
                        else:
                            raise ValueError('mode error!')
    
                        num_proto_bg[n][p] += 1

                self.conf_list_bg_all[n] = torch.cat([self.conf_list_bg_all[n], conf_list_bg[n].unsqueeze(0)], 0) # [T, n_protos]
