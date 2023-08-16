import os
from os import path
import time
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from model.eval_network_PMNet import PMNet 
from dataset.davis_test_dataset import DAVISTestDataset
from util.tensor_util import unpad
from inference_core import InferenceCore

from progressbar import progressbar

"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default="/path-to-model")
parser.add_argument('--davis_path', default='/path-to-DAVIS')
parser.add_argument('--output', default=r'/path-to-save')
parser.add_argument('--split', help='val/testdev', default='val')
parser.add_argument('--top', type=int, default=20) 
parser.add_argument('--amp', action='store_true')
parser.add_argument('--mem_every', default=5, type=int)
parser.add_argument('--proto_every', default=1, type=int) 
parser.add_argument('--include_last', help='include last frame as temporary memory?', action='store_true')
args = parser.parse_args()

davis_path = args.davis_path
out_path = args.output

# Simple setup  
os.makedirs(out_path, exist_ok=True)
palette = Image.open(path.expanduser(davis_path + '/Annotations/480p/blackswan/00000.png')).getpalette()

torch.autograd.set_grad_enabled(False)

# Setup Dataset
if args.split == 'val':
    test_dataset = DAVISTestDataset(davis_path, imset='2017/val.txt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
elif args.split == 'testdev':
    test_dataset = DAVISTestDataset(davis_path, imset='2017/test-dev.txt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
else:
    raise NotImplementedError

# Load our checkpoint
prop_saved = torch.load(args.model)
top_k = args.top
prop_model = PMNet().cuda().eval()
#prop_model = STCN(graph_first=False).cuda().eval()
prop_model.load_state_dict(prop_saved)

# Start eval
for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):

    #with torch.cuda.amp.autocast(enabled=args.amp):
    rgb = data['rgb'].cuda()
    msk = data['gt'][0].cuda()
    info = data['info']
    name = info['name'][0]
    k = len(info['labels'][0])
    size = info['size_480p']

    torch.cuda.synchronize()
    process_begin = time.time()

    processor = InferenceCore(prop_model, rgb, k, top_k=top_k, 
                    mem_every=args.mem_every, proto_every=args.proto_every, include_last=args.include_last)
    processor.interact_PMNet(msk[:,0], 0, rgb.shape[1], name)

    # Do unpad -> upsample to original size 
    out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')
    for ti in range(processor.t):
        prob = unpad(processor.prob[:,ti], processor.pad)
        prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
        out_masks[ti] = torch.argmax(prob, dim=0)
    
    out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

    torch.cuda.synchronize()
    total_process_time += time.time() - process_begin   
    total_frames += out_masks.shape[0]

    # Save the results
    this_out_path = path.join(out_path, name)
    os.makedirs(this_out_path, exist_ok=True)
    for f in range(out_masks.shape[0]):
        img_E = Image.fromarray(out_masks[f])
        img_E.putpalette(palette)
        img_E.save(os.path.join(this_out_path, '{:05d}.png'.format(f)))

    del rgb
    del msk
    del processor
