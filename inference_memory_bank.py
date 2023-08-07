import math
import torch
import torch.nn.functional as F

def softmax_w_top(x, top):
    values, indices = torch.topk(x, k=top, dim=1)
    x_exp = values.exp_()

    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    x.zero_().scatter_(1, indices, x_exp)  # B * THW * HW # 除topk匹配位置外其余位置相似度权重置为0

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

        self.temp_k_emb = None

        # hyper parameters
        self.num_objects = k

        self.num_scales = None

        self.mem_ind = []
        
        self.w = 0.9

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
            C_emb = mk_emb.shape[1]
        else:
            C_emb = norm_p  

        a_p = mk_emb.pow(2).sum(1).unsqueeze(2)  # B, Thw, 1
        b_p = 2 * (mk_emb.transpose(1, 2) @ qk_emb)  # B, Thw, hw
        affinity_proto = (-a_p + b_p) / math.sqrt(C_emb)  # B, THW, HW
        
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
    
    def _readout(self, affinity, mv):
        # affinity: k, THW, HW, mv: k, 512, THW
        # print(mv.shape, affinity.shape)
        return torch.bmm(mv, affinity)  # k, 512, HW

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

    def add_mem_idx(self, mem_idx):
        self.mem_ind.append(mem_idx)

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
