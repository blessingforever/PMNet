import os
from os import path
import time
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from model.eval_network_PMNet import PMNet 
from dataset.davis_test_dataset import DAVISTestDataset
from inference_core_davis import InferenceCore

from progressbar import progressbar

# from tqdm import tqdm

"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default="/path-to-model")
parser.add_argument('--davis_path', default='/path-to-DAVIS')
parser.add_argument('--output', default=r'/path-to-save')
parser.add_argument('--top', type=int, default=20)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--mem_every', default=5, type=int)
args = parser.parse_args()

davis_path = args.davis_path
out_path = args.output

# Simple setup
os.makedirs(out_path, exist_ok=True)

torch.autograd.set_grad_enabled(False)

# Setup Datase
test_dataset = DAVISTestDataset(davis_path, imset='2016/val.txt', single_object=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

# Load our checkpoint
prop_saved = torch.load(args.model)
top_k = args.top
prop_model = PMNet().cuda().eval()
prop_model.load_state_dict(prop_saved)

total_process_time = 0
total_frames = 0

# Start eval
for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):
    # for data in tqdm(test_loader):

    if args.amp:
        with torch.cuda.amp.autocast(enabled=args.amp):
            rgb = data['rgb'].cuda()
            msk = data['gt'][0].cuda()
            info = data['info']
            name = info['name'][0]
            k = len(info['labels'][0])

            torch.cuda.synchronize()
            process_begin = time.time()

            processor = InferenceCore(prop_model, rgb, k, top_k=top_k, mem_every=args.mem_every)
            processor.interact_CondProto_v3_5_new3_proto_affinity(msk[:,0], 0, rgb.shape[1], name)

            # Do unpad -> upsample to original size
            out_masks = torch.zeros((processor.t, 1, *rgb.shape[-2:]), dtype=torch.float32, device='cuda')
            for ti in range(processor.t):
                prob = processor.prob[:, ti]

                if processor.pad[2] + processor.pad[3] > 0:
                    prob = prob[:, :, processor.pad[2]:-processor.pad[3], :]
                if processor.pad[0] + processor.pad[1] > 0:
                    prob = prob[:, :, :, processor.pad[0]:-processor.pad[1]]

                out_masks[ti] = torch.argmax(prob, dim=0) * 255

            out_masks = (out_masks.detach().cpu().numpy()[:, 0]).astype(np.uint8)

            torch.cuda.synchronize()
            total_process_time += time.time() - process_begin
            total_frames += out_masks.shape[0]

            this_out_path = path.join(out_path, name)
            os.makedirs(this_out_path, exist_ok=True)
            for f in range(out_masks.shape[0]):
                img_E = Image.fromarray(out_masks[f])
                img_E.save(os.path.join(this_out_path, '{:05d}.png'.format(f)))

    else:
        rgb = data['rgb'].cuda()
        msk = data['gt'][0].cuda()
        info = data['info']
        name = info['name'][0]
        k = len(info['labels'][0])

        torch.cuda.synchronize()
        process_begin = time.time()

        processor = InferenceCore(prop_model, rgb, k, top_k=top_k, mem_every=args.mem_every)
        processor.interact_CondProto_v3_5_new2_proto_affinity(msk[:,0], 0, rgb.shape[1], name)

        # Do unpad -> upsample to original size
        out_masks = torch.zeros((processor.t, 1, *rgb.shape[-2:]), dtype=torch.float32, device='cuda')
        for ti in range(processor.t):
            prob = processor.prob[:, ti]

            if processor.pad[2] + processor.pad[3] > 0:
                prob = prob[:, :, processor.pad[2]:-processor.pad[3], :]
            if processor.pad[0] + processor.pad[1] > 0:
                prob = prob[:, :, :, processor.pad[0]:-processor.pad[1]]

            out_masks[ti] = torch.argmax(prob, dim=0) * 255

        out_masks = (out_masks.detach().cpu().numpy()[:, 0]).astype(np.uint8)

        torch.cuda.synchronize()
        total_process_time += time.time() - process_begin
        total_frames += out_masks.shape[0]

        this_out_path = path.join(out_path, name)
        os.makedirs(this_out_path, exist_ok=True)
        for f in range(out_masks.shape[0]):
            img_E = Image.fromarray(out_masks[f])
            img_E.save(os.path.join(this_out_path, '{:05d}.png'.format(f)))

    del rgb
    del msk
    del processor

print('Total processing time: ', total_process_time)
print('Total processed frames: ', total_frames)
print('FPS: ', total_frames / total_process_time)

#################################################### 测试完直接计算精度
import sys

sys.path.append(r'/ghome/linfc/davis2017-evaluation')
import pandas as pd
from davis2017.evaluation import DAVISEvaluation

default_davis_path = '/path/to/the/folder/DAVIS'

time_start = time.time()
parser.add_argument('--set', type=str, help='Subset to evaluate the results', default='val')
parser.add_argument('--task', type=str, help='Task to evaluate the results', default='semi-supervised',
                    choices=['semi-supervised', 'unsupervised'])
parser.add_argument('--results_path', type=str, help='Path to the folder containing the sequences folders',
                    default=out_path)
parser.add_argument("--year", type=str, help="Davis dataset year (default: 2017)", default='2016',
                    choices=['2016', '2017', '2019'])
args, _ = parser.parse_known_args()
csv_name_global = f'global_results-{args.set}.csv'
csv_name_per_sequence = f'per-sequence_results-{args.set}.csv'

# Compute the results
csv_name_global_path = os.path.join(args.results_path, csv_name_global)
csv_name_per_sequence_path = os.path.join(args.results_path, csv_name_per_sequence)
print(f'Evaluating sequences for the {args.task} task...')
# Create dataset and evaluate
dataset_eval = DAVISEvaluation(davis_root=args.davis_path, task=args.task, gt_set=args.set, year=args.year)
metrics_res = dataset_eval.evaluate(args.results_path)
J, F = metrics_res['J'], metrics_res['F']

# Generate dataframe for the general results
g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                  np.mean(F["D"])])
g_res = np.reshape(g_res, [1, len(g_res)])
table_g = pd.DataFrame(data=g_res, columns=g_measures)
with open(csv_name_global_path, 'w') as f:
    table_g.to_csv(f, index=False, float_format="%.3f")
print(f'Global results saved in {csv_name_global_path}')

# Generate a dataframe for the per sequence results
seq_names = list(J['M_per_object'].keys())
seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
J_per_object = [J['M_per_object'][x] for x in seq_names]
F_per_object = [F['M_per_object'][x] for x in seq_names]
table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
with open(csv_name_per_sequence_path, 'w') as f:
    table_seq.to_csv(f, index=False, float_format="%.3f")
print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

# Print the results
sys.stdout.write(f"--------------------------- Global results for {args.set} ---------------------------\n")
print(table_g.to_string(index=False))
sys.stdout.write(f"\n---------- Per sequence results for {args.set} ----------\n")
print(table_seq.to_string(index=False))
total_time = time.time() - time_start
sys.stdout.write('\nTotal time:' + str(total_time))

