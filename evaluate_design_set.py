#!/usr/bin/env python3
import os
import numpy as np
from tqdm import tqdm
from DomainSegmentor import *

# TODO make parameters commandline accessible.
eval_dir = sys.argv[1]
if eval_dir[-1] == '/':
    eval_dir = eval_dir[:-1]
target_class_idx = [27]
bypass_prob = True
include_max = False #  Include results using the max of the selected classes.
include_entropy = True

# GPU parameters.
gpu_id = int(sys.argv[2])

# Generate path list.
path_list = []
for root, dirs, files in os.walk(eval_dir):
    for file in files:
        if file.endswith('.pdb') and not file.startswith('.'):
            path_list.append(os.path.join(root, file))

# Check that classes are correct.
print("Selected Classes:")
for t in target_class_idx:
    print(idx_to_class[t])

# Evaluate and extract desired probs.
segmentor = DomainSegmentor(gpu_id=gpu_id)
prob_dict = {}
max_prob_dict = {}
entropy_dict = {}

print("Starting Evaluation.")
for pdb_path in tqdm(path_list):
    if not bypass_prob:
        class_probs, res_nums = segmentor.predict(pdb_path, log=True)
        prob_subset = class_probs[[i for i in target_class_idx][:]]
        target_probs = np.mean(prob_subset, axis=1)
        prob_dict[os.path.split(pdb_path)[1]] = target_probs
    if include_max:
        max_prob_subset = np.max(prob_subset, axis=0)
        max_target_prob = np.mean(max_prob_subset)
        max_prob_dict[os.path.split(pdb_path)[1]] = max_target_prob
    if include_entropy:
        entropy, _ = segmentor.computeEntropy(pdb_path)
        mean_entropy = np.mean(entropy)
        se_entropy = np.std(entropy)/np.sqrt(len(entropy))
        max_entropy = np.max(entropy)
        min_entropy = np.min(entropy)
        entropy_dict[os.path.split(pdb_path)[1]] = [str(mean_entropy), str(se_entropy), str(max_entropy),str(min_entropy)]

# Write the output.
if include_entropy:
    sorted_list = sorted(entropy_dict.items(), key=lambda x:x[1][0])[::-1]
    out_lines = [x[0] + '\t' + '\t'.join(x[1]) + '\n' for x in sorted_list]
    out_file = open(eval_dir +'.entropy', 'w')
    out_file.writelines(out_lines)
    out_file.close()
    print("Wrote: " + eval_dir + '.entropy')

if include_max:
    sorted_list = sorted(max_prob_dict.items(), key=lambda x:x[1])[::-1]
    out_lines = [x[0] + '\t' + str(x[1]) + '\n' for x in sorted_list]
    out_file = open(eval_dir + '_max_' + '-'.join(str(t) for t in target_class_idx) +'.prob', 'w')
    out_file.writelines(out_lines)
    out_file.close()
    print("Wrote: " + eval_dir + '_max_' + '-'.join(str(t) for t in target_class_idx) +'.prob')

if not bypass_prob:
    for i, class_idx in enumerate(target_class_idx):
        temp_dict = {x : prob_dict[x][i] for x in prob_dict}
        sorted_list = sorted(temp_dict.items(), key=lambda x:x[1])[::-1]
        out_lines = [x[0] + '\t' + str(x[1]) + '\n' for x in sorted_list]
        out_file = open(eval_dir + '_' + idx_to_class[class_idx].replace(' ','').lower() + '.prob', 'w')
        out_file.writelines(out_lines)
        out_file.close()
        print("Wrote: " + eval_dir + '_' + idx_to_class[class_idx].replace(' ','').lower() + '.prob')
