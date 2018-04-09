import torch
import pickle
from torch.autograd import Variable
from pymol import cmd, stored
import numpy as np
import os, sys
from segmentor_model_v2 import *
from segmentor_utils import *
import __main__
__main__.pymol_argv = [ 'pymol', '-qei' ]
import pymol


model_path = 'epoch95_model'
num_classes = 38
class_dict = {0: 'Unassigned (Loop)', 1: 'Orthogonal Bundle', 2: 'Up-down Bundle', 3: 'Alpha Horseshoe', 4: 'Alpha/alpha barrel', 5: 'Ribbon', 6: 'Aligned Prism', 7: '3-layer Sandwich', 8: '4 Propeller', 9: '5 Propeller', 
                10: '6 Propeller', 11: '7 Propeller', 12: '2 Solenoid', 13: '3 Solenoid', 14: 'Beta Complex', 15: 'Single Sheet', 16: 'Roll', 17: 'Beta Barrel', 18: 'Clam', 19: 'Sandwich', 20: 'Distorted Sandwich', 21: 'Trefoil', 
                22: 'Orthogonal Prism', 23: 'Roll', 24: 'Ribosomal Protein L15; Chain: K; domain 2', 25: 'Super Roll', 26: 'Alpha-Beta Barrel', 27: '2-Layer Sandwich', 28: '3-Layer(aba) Sandwich', 29: '3-Layer(bba) Sandwich', 
                30: '3-Layer(bab) Sandwich', 31: '4-Layer Sandwich', 32: 'Alpha-beta prism', 33: 'Box', 34: '5-stranded Propeller', 35: 'Alpha-Beta Horseshoe', 36: 'Alpha-Beta Complex', 37: 'Irregular', -1: 'NULL'}

def predictPDB(pdb_file, model, ignore_index=-9999):
    # Initialize Model
    model.eval()
    if torch.cuda.is_available(): 
        model.cuda()
        print("Running on GPU.")
    else:
        print("Running on CPU.")
    # Generate contact map and get sequence numbering.
    seq_len, numbering = get_pdb_info(pdb_file)
    cm, numbering = makeContactMapTensor(pdb_file, seq_len, numbering, target_size=512, upper_tol=512)
    cm_var = Variable(cm)
    #cm_var = Variable(cm.repeat(64,1,1,1)) # funny but necessary...
    if torch.cuda.is_available(): 
        cm_var = cm_var.cuda()
    outputs = model(cm_var)
    _, predicted = torch.max(outputs.data, 1)
    # Format for output.
    predicted = predicted[0,:,:].cpu().numpy().flatten()
    out_pred = []
    out_num = [] 
    for i in range(len(numbering)):
        if numbering[i] != ignore_index:
            out_pred.append(predicted[i])
            out_num.append(numbering[i])
    assert len(out_pred) == len(out_num)
    return out_pred, out_num


def segmentFold(input_file, model_path):
    net = SegmentorModel(1,8,16, num_classes)
    if torch.cuda.is_available(): 
        net.load_state_dict(torch.load(model_path))
    else:
        net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    prediction, numbering = predictPDB(input_file, net)
    prediction_dict = {}
    for i in set(prediction):
        prediction_dict[class_dict[i]] = [str(numbering[z]) for z in list(np.where(prediction==i)[0])]
    return prediction_dict


if __name__ == "pymol":
    try:
        input_file = sys.argv[2] # pdb name # TODO change to commandline argument.
    except:
        raise Exception("No input PDB specified.")
    color_list = ['green', 'violet', 'cyan', 'yellow', 'orange']
    pymol.finish_launching()
    cmd.load(input_file)
    cmd.color('white', "all")
    print("Running Domain Segmentor.")
    prediction_dict = segmentFold(input_file, model_path=model_path)
    for i, class_name in enumerate(prediction_dict): # TODO: Color 100 residues at a time otherwise pymol can't buffer it.
        color = color_list[i%len(color_list)]
        cmd.color(color, "resi %s" %"+".join(prediction_dict[class_name]))
        print(color + " : " + class_name)

if __name__ == '__main__':
    try:
        input_file = sys.argv[1]
    except:
        raise Exception("No input PDB specified.") # pdb name # TODO change to commandline argument.
    prediction_dict = segmentFold(input_file, model_path=model_path)
    for i, class_name in enumerate(prediction_dict):
        print(class_name)
        print(prediction_dict[class_name])


