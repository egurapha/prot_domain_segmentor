import torch
import pickle
from torch.autograd import Variable
from pymol import cmd, stored
import numpy as np
import os, sys
from segmentor_model_v2 import *
from segmentor_utils import *
import torch.nn.functional as F
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.cm as cm
matplotlib.pyplot.switch_backend('agg')


model_path = 'epoch95_model'
num_classes = 38
class_dict = {0: 'Unassigned (Loop)', 1: 'Orthogonal Bundle', 2: 'Up-down Bundle', 3: 'Alpha Horseshoe', 4: 'Alpha/alpha barrel', 5: 'Ribbon', 6: 'Aligned Prism', 7: '3-layer Sandwich', 8: '4 Propeller', 9: '5 Propeller', 
                10: '6 Propeller', 11: '7 Propeller', 12: '2 Solenoid', 13: '3 Solenoid', 14: 'Beta Complex', 15: 'Single Sheet', 16: 'Roll', 17: 'Beta Barrel', 18: 'Clam', 19: 'Sandwich', 20: 'Distorted Sandwich', 21: 'Trefoil', 
                22: 'Orthogonal Prism', 23: 'Roll', 24: 'Ribosomal Protein L15; Chain: K; domain 2', 25: 'Super Roll', 26: 'Alpha-Beta Barrel', 27: '2-Layer Sandwich', 28: '3-Layer(aba) Sandwich', 29: '3-Layer(bba) Sandwich', 
                30: '3-Layer(bab) Sandwich', 31: '4-Layer Sandwich', 32: 'Alpha-beta prism', 33: 'Box', 34: '5-stranded Propeller', 35: 'Alpha-Beta Horseshoe', 36: 'Alpha-Beta Complex', 37: 'Irregular', -1: 'NULL'}

def predictPDB(pdb_file, model, ignore_index=-9999, expected_class=None):
    # Initialize Model
    model.eval()
    if torch.cuda.is_available(): 
        model.cuda()
        #print("Running on GPU.")
    else:
        pass
        #print("Running on CPU.")
    # Generate contact map and get sequence numbering.
    seq_len, numbering = get_pdb_info(pdb_file)
    cm, numbering = makeContactMapTensor(pdb_file, seq_len, numbering, target_size=512, upper_tol=512)
    cm_var = Variable(cm)
    if torch.cuda.is_available(): 
        cm_var = cm_var.cuda()
    outputs = model(cm_var)
    outputs = F.softmax(outputs, dim=1)
    outputs = outputs.data[0,:,:,:]
    # Format for output.
    class_probs = outputs.cpu().numpy().squeeze() # 38x512 matrix.
    out_num = []
    trunc_class_probs = np.array([None])
    for i in range(len(numbering)):
        if numbering[i] != ignore_index:
            out_num.append(numbering[i])
            if not trunc_class_probs.any():
                trunc_class_probs = np.expand_dims(class_probs[:,i], axis=1)
            else:
                trunc_class_probs = np.column_stack([trunc_class_probs, class_probs[:,i]])
    return trunc_class_probs, out_num


def segmentFold(input_file, model_path, path_header='', expected=None):
    net = SegmentorModel(1,8,16, num_classes)
    if torch.cuda.is_available(): 
        net.load_state_dict(torch.load(model_path))
    else:
        net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    class_probs, numbering = predictPDB(path_header + input_file, net)
    if not expected:
        raise Exception("No Expected Class Specified.")
    # normalize class probs
    overall_class_prob = class_probs.sum(axis=1)[expected,]/ class_probs.shape[1]
    print(overall_class_prob)
    return overall_class_prob
 
def getEnergy(file_name, path_header=''):
    for line in reversed(open(path_header + file_name, 'r').readlines()):
        if "silent_score" in line:
            break
    energy = line.split()[1] 
    return energy

def getRms(file_name, path_header=''):
    for line in reversed(open(path_header + file_name, 'r').readlines()):
        if "rms" in line:
            break
    rms = line.split()[1] 
    return rms
           
if __name__ == '__main__':
    decoy_path = "/home/raphael/Desktop/unseen_structures/2w0i/"
    expected_class = 28
    prob_list = []
    rms_list = []
    energy_list = []
    counter = 0
   
    for f in os.listdir(decoy_path):
        if f.endswith(".pdb") and not f.startswith("."):
            energy = getEnergy(f, path_header=decoy_path)
            rms = getRms(f, path_header=decoy_path)
            average_class_prob = segmentFold(f, model_path=model_path, path_header=decoy_path, expected=expected_class)
            print(energy, average_class_prob, rms)
            prob_list.append(float(average_class_prob))
            rms_list.append(float(rms))
            energy_list.append(float(energy))
            counter += 1
            if counter == 1000:
                break
    
    prob_list = np.log(np.array(prob_list))
    # Class Prob vs Energy
    plt.plot(energy_list, prob_list, 'bo')
    plt.xlabel('Energy')
    plt.ylabel('Class Prob')
    plt.tight_layout()
    plt.savefig('energy_prob', ppi=300)
    plt.close()
    
    # Class Prob vs RMSD
    plt.plot(rms_list, prob_list, 'bo')
    plt.xlabel('RMSD')
    plt.ylabel('Class Prob')
    plt.tight_layout()
    plt.savefig('rms_prob', ppi=300)
    plt.close()
    
    # Energy vs RMSD with Class Prob Heat Map
    plt.scatter(rms_list, energy_list, c=prob_list, label='o', s=3, cmap='coolwarm')
    plt.colorbar().set_label('Class Prob')
    plt.xlabel('RMSD')
    plt.ylabel('Energy')
    plt.tight_layout()
    plt.savefig('rms_energy', ppi=300)
    plt.close()
    
