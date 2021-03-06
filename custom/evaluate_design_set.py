import os
import numpy as np
from tqdm import tqdm
sys.path.insert(0, 'classes')
from DomainSegmentor import *

# TODO make parameters commandline accessible.
eval_dir = 'nhlrc3_set2'
target_class_idx = [8, 9, 10, 11]
include_max = True #  Include results using the max of the selected classes.

# Generate path list.
path_list = []
for root, dirs, files in os.walk(eval_dir):
    for file in files:
        if file.endswith('.pdb'):
            path_list.append(os.path.join(root, file))

# Check that classes are correct.
print("Selected Classes:")
for t in target_class_idx:
    print(idx_to_class[t])

# Evaluate and extract desired probs.
segmentor = DomainSegmentor()
prob_dict = {}
max_prob_dict = {}

print("Starting Evaluation.")
for pdb_path in tqdm(path_list):
    class_probs, res_nums = segmentor.predict(pdb_path, log=True)
    prob_subset = class_probs[[i for i in target_class_idx][:]]
    target_probs = np.mean(prob_subset, axis=1)
    prob_dict[os.path.split(pdb_path)[1]] = target_probs
    if include_max:
        max_prob_subset = np.max(prob_subset, axis=0)
        max_target_prob = np.mean(max_prob_subset)
        max_prob_dict[os.path.split(pdb_path)[1]] = max_target_prob

# Write the output.
if include_max:
    sorted_list = sorted(max_prob_dict.items(), key=lambda x:x[1])[::-1]
    out_lines = [x[0] + '\t' + str(x[1]) + '\n' for x in sorted_list]
    out_file = open(eval_dir + '_max_' + '-'.join(str(t) for t in target_class_idx) +'.prob', 'w')
    out_file.writelines(out_lines)
    out_file.close()
    print("Wrote: " + eval_dir + '_max_' + '-'.join(str(t) for t in target_class_idx) +'.prob')

for i, class_idx in enumerate(target_class_idx):
    temp_dict = {x : prob_dict[x][i] for x in prob_dict}
    sorted_list = sorted(temp_dict.items(), key=lambda x:x[1])[::-1]
    out_lines = [x[0] + '\t' + str(x[1]) + '\n' for x in sorted_list]
    out_file = open(eval_dir + '_' + idx_to_class[class_idx].replace(' ','').lower() + '.prob', 'w')
    out_file.writelines(out_lines)
    out_file.close()
    print("Wrote: " + eval_dir + '_' + idx_to_class[class_idx].replace(' ','').lower() + '.prob')
