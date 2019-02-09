# Numerical
import torch
from torch.autograd import Variable
import numpy as np
# Pymol
import pymol
from pymol import cmd, stored
import __main__
__main__.pymol_argv = [ 'pymol', '-qei' ]
# Custom
import os, sys
sys.path.insert(0, 'model')
sys.path.insert(0, 'classes')
from segmentor_model_v2 import *
from segmentor_utils import *
from DomainSegmentor import *

model_path = 'model/epoch95_model_v2'
num_classes = 38
class_dict = idx_to_class # stored in DomainSegmentor.py

def segmentFold(input_file, model_path):
    segmentor = DomainSegmentor()
    prediction, numbering = segmentor.predictClass(input_file)
    prediction_dict = {}
    for i in set(prediction):
        prediction_dict[class_dict[i]] = [str(numbering[z]) for z in list(np.where(prediction==i)[0])]
    return prediction_dict

if __name__ == "pymol":
    def splitList(l, n): # Annoying but necessary due to pymol buffering limit.
        if n == 0: 
            yield l
        for i in range(0, len(l), n):
            yield l[i:i + n]
    try:
        input_file = sys.argv[2]
    except:
        raise Exception("No input PDB specified.")
    color_list = ['green', 'violet', 'cyan', 'yellow', 'orange']
    pymol.finish_launching()
    cmd.load(input_file)
    cmd.color('white', "all")
    print("Running Domain Segmentor.")
    prediction_dict = segmentFold(input_file, model_path=model_path)
    for i, class_name in enumerate(prediction_dict):
        color = color_list[i%len(color_list)]
        idx_batches = splitList(prediction_dict[class_name], int(len(prediction_dict[class_name])/100 + 1))
        for idx_list in idx_batches: # Fix for pymol buffering.
            cmd.color(color, "resi %s" %"+".join(idx_list))
        print(color + " : " + class_name)

if __name__ == '__main__':
    try:
        input_file = sys.argv[1]
    except:
        raise Exception("No input PDB specified.")
    prediction_dict = segmentFold(input_file, model_path=model_path)
    for i, class_name in enumerate(prediction_dict):
        print(class_name)
        print(prediction_dict[class_name])


