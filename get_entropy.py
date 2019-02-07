# Numerical
import torch
from torch.autograd import Variable
import numpy as np
import scipy.stats
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

if __name__ == "pymol":
    try:
        input_file = sys.argv[2]
    except:
        raise Exception("No input PDB specified.")
    pymol.finish_launching()
    cmd.load(input_file)
    cmd.color('white', 'all')
    print("Running entropy Calculation.")
    segmentor = DomainSegmentor()
    entropy, numbering = segmentor.computeEntropy(input_file)
    entropy = np.exp(entropy)
    # Color by z-score.
    ent_mean = np.mean(entropy) 
    ent_std = np.std(entropy)
    entropy  = [(x - ent_mean)/ent_std for x in entropy]
    cmd.alter('all', "b=0.0")
    for i, num in enumerate(numbering):
        cmd.alter("all and resi %d and n. CA"%num, "b=%f" %entropy[i])
    maxval = max(entropy)
    minval = min(entropy)
    print("Blue indicates the lowest entropy point, while Red indicates the highest entropy point.")
    print("Highest Exp(Entropy):" + str(maxval))
    print("Lowest Exp(Entropy):" + str(minval))
    cmd.spectrum("b", "blue_white_red", "all and n. CA", minimum=minval, maximum=maxval)

if __name__ == '__main__':
    try:
        input_file = sys.argv[1]
    except:
        raise Exception("No input PDB specified.")
    segmentor = DomainSegmentor()
    entropy, numbering = segmentor.computeEntropy(input_file)
    print(numbering)
    print(entropy)
