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
from segmentor_model_v2 import *
from segmentor_utils import *
from DomainSegmentor import *
import scipy.stats

model_path = 'model/epoch95_model_v2'
num_classes = 38
class_dict = idx_to_class # stored in DomainSegmentor.py

def getEntropy(input_file, model_path):
    segmentor = DomainSegmentor()
    confidence, numbering = segmentor.getEntropy(input_file)
    assert len(confidence) == len(numbering)
    return confidence, numbering
   
if __name__ == "pymol":
    '''
    def splitList(l, n): # Annoying but necessary due to pymol buffering limit.
        if n == 0: 
            yield l
        for i in range(0, len(l), n):
            yield l[i:i + n]
    '''
    try:
        input_file = sys.argv[2]
    except:
        raise Exception("No input PDB specified.")
    pymol.finish_launching()
    cmd.load(input_file)
    cmd.color('white', 'all')
    print("Running Confidence Calculation.")
    confidence, numbering = getEntropy(input_file, model_path=model_path)
    confidence = np.log(confidence)
    cmd.alter('all', "b=0.0")
    for i, num in enumerate(numbering):
        cmd.alter("all and resi %d and n. CA"%num, "b=%f" %confidence[i])
    minval = -3.01177#np.mean(confidence) - np.std(confidence) 
    maxval = -0.52835#max(confidence)#np.log(scipy.stats.entropy(np.repeat(1.00/num_classes, num_classes))/3)
    print(confidence)
    print(maxval)
    print(minval)
    print(min(confidence))
    cmd.spectrum("b", "blue_white_red", "all and n. CA", minimum=minval, maximum=maxval)

if __name__ == '__main__':
    try:
        input_file = sys.argv[1]
    except:
        raise Exception("No input PDB specified.")
    confidence, numbering = getConfidence(input_file, model_path=model_path)
    print(numbering)
    print(confidence)
