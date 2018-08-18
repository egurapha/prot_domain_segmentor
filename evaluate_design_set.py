import os
import numpy as np
from tqdm import tqdm
from DomainSegmentor import *

eval_dir = 'nhlrc3'
target_class_idx = 9 # 9, 10, 11 correspond to propeller classes

path_list = []
for root, dirs, files in os.walk(eval_dir):
    for file in files:
        path_list.append(os.path.join(root, file))

segmentor = DomainSegmentor()
prob_dict = {}

for pdb_path in tqdm(path_list):
    class_probs, res_nums = segmentor.predict(pdb_path, log=True)
    target_prob = np.mean(class_probs[target_class_idx,:])
    prob_dict[os.path.split(pdb_path)[1]] = target_prob

sorted_list = sorted(prob_dict.items(), key=lambda x:x[1])[::-1]
out_lines = [x[0] + '\t' + str(x[1]) + '\n' for x in sorted_list]

out_file = open(eval_dir + '_class' + str(target_class_idx), 'w')
out_file.writelines(out_lines)
out_file.close()




