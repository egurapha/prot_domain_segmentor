from collections import defaultdict
import os

line_dict = defaultdict(list)
counter = 0
with open('file_list', 'r') as f:
    while True:
        line = f.readline()
        if not line: break
        line = line.strip()
        gpu = counter%3
        line_dict[gpu].append('python evaluate_design_set.py ' + line + ' ' + str(gpu) + '\n')
        counter += 1

for key, entry in line_dict.items():
    out_file = open('gpu_' + str(key) + '.sh', 'w')
    out_file.write('#!/bin/bash\n')
    out_file.writelines(entry)
    out_file.close()
    os.system('chmod +x ' + 'gpu_' + str(key) + '.sh')
