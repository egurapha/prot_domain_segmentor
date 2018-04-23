#from itertools import izip
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

def predictPDB(pdb_file, model, ignore_index=-9999):
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


def segmentFold(input_file, model_path, path_header=''):
    net = SegmentorModel(1,8,16, num_classes)
    if torch.cuda.is_available(): 
        net.load_state_dict(torch.load(model_path))
    else:
        net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    class_probs, numbering = predictPDB(path_header + input_file, net)
    print(class_probs.shape)
    # normalize class probs
    overall_class_prob = class_probs.sum(axis=1)/ class_probs.shape[1]
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
    decoy_path = "/home/raphael/Desktop/unseen_structures/"
    for d in os.listdir(decoy_path):
        if os.path.isdir(decoy_path + d):
            set_path = decoy_path + d + '/'
            prob_list = np.array([None])
            rms_list = []
            energy_list = []
            counter = 0
           
            for f in os.listdir(set_path):
                if f.endswith(".pdb") and not f.startswith("."):
                    energy = getEnergy(f, path_header=set_path)
                    rms = getRms(f, path_header=set_path)
                    average_class_probs = segmentFold(f, model_path=model_path, path_header=set_path)
                    if not prob_list.any():
                        prob_list = np.expand_dims(average_class_probs, axis=1)
                    else:
                        prob_list = np.column_stack([prob_list, average_class_probs])
                    print(energy, average_class_probs, rms)
                    rms_list.append(float(rms))
                    energy_list.append(float(energy))
                    counter += 1
                    if counter == 1000:
                        break
 
            prob_list = np.log(np.array(prob_list))
            for i in range(prob_list.shape[0]): # loop over all classes. 
                x_temp, y_temp = zip(*sorted(zip(rms_list, prob_list[i,:])))
                plt.plot(x_temp, y_temp,'-o', linewidth=0.5, markersize=1)
            plt.ylim(-5.5,0) 
            #print(np.repeat(np.expand_dims(np.array(rms_list), axis=1), 38, axis=1).T.shape)
            #print(prob_list.shape)
            #plt.plot(np.repeat(np.expand_dims(np.array(rms_list), axis=1), 38, axis=1), prob_list.T, 'bo')
            plt.title(d + ' Probabilities')
            plt.savefig(d + '_prob_plot.png', ppi=300)
            plt.close()


            plt.scatter(rms_list, energy_list, label='o', s=3)
            plt.xlabel('RMSD')
            plt.ylabel('Energy')
            plt.tight_layout()
            plt.title(d + ' Energy vs RMSD')
            plt.savefig(d + '_rms_energy', ppi=300)
            plt.close()

            '''
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
            '''