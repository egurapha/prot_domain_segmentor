sys.path.insert(0, 'model')
from segmentor_model_v2 import *
from segmentor_utils import *
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import os, sys

class ProteinSegmentor:
    def __init__(self, model_path='model/epoch95_model_v2', try_gpu=True):
        self.class_dict = {0: 'Unassigned (Loop)', 
                            1: 'Orthogonal Bundle', 
                            2: 'Up-down Bundle', 
                            3: 'Alpha Horseshoe', 
                            4: 'Alpha/alpha barrel', 
                            5: 'Ribbon', 
                            6: 'Aligned Prism', 
                            7: '3-layer Sandwich', 
                            8: '4 Propeller', 
                            9: '5 Propeller', 
                            10: '6 Propeller', 
                            11: '7 Propeller', 
                            12: '2 Solenoid', 
                            13: '3 Solenoid', 
                            14: 'Beta Complex', 
                            15: 'Single Sheet', 
                            16: 'Roll', 
                            17: 'Beta Barrel', 
                            18: 'Clam', 
                            19: 'Sandwich', 
                            20: 'Distorted Sandwich', 
                            21: 'Trefoil', 
                            22: 'Orthogonal Prism', 
                            23: 'Roll',
                            24: 'Ribosomal Protein L15; Chain: K; domain 2',
                            25: 'Super Roll',
                            26: 'Alpha-Beta Barrel',
                            27: '2-Layer Sandwich',
                            28: '3-Layer(aba) Sandwich',
                            29: '3-Layer(bba) Sandwich', 
                            30: '3-Layer(bab) Sandwich',
                            31: '4-Layer Sandwich',
                            32: 'Alpha-beta prism',
                            33: 'Box',
                            34: '5-stranded Propeller',
                            35: 'Alpha-Beta Horseshoe',
                            36: 'Alpha-Beta Complex',
                            37: 'Irregular',
                            -1: 'NULL'}
        self.cuda_avail = torch.cuda.is_available()
        self.num_classes = len(self.class_dict)-1 
        self.model = SegmentorModel(1,8,16, self.num_classes)
        self._init_model(self.model, model_path)
        
    def _init_model(self, net, model_path):
        if self.cuda_avail:
            net.load_state_dict(torch.load(model_path))
            net.cuda()
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count())) 
            print("Model Initialized on GPU.")
        else:
            net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
            print("Model Initialized on CPU.")
        net.eval() # Set model to evaluation mode to remove stochastic regularization.

    def predict(self, pdb_name, ignore_index=-9999):
        '''
        Input: pdb file name.
        Output: 
            trunc_class_probs -- 38 x 512 matrix. Entry (i,j) is the probility of residue j being in class i.
            out_num -- the list of pdb_residue numberings corresponding to the columns of trunc_class_probs. For example, out_num[i] = 50, then the i-th column of trunc_class_probs corresponds to residue 50 in the actual PDB.
        '''
        seq_len, numbering = get_pdb_info(pdb_name)
        cm, numbering = makeContactMapTensor(pdb_name, seq_len, numbering, target_size=512, upper_tol=512)
        if self.cuda_avail:
            cm_var = Variable(cm).cuda()
        else:
            cm_var = Variable(cm)
        outputs = self.model(cm_var)
        outputs = F.softmax(outputs, dim=1)
        outputs = outputs.data[0,:,:,:]
        # Format for output.
        class_probs = outputs.cpu().numpy().squeeze() # 38x512 matrix. # Columns define a probability distribution over the 38 classes for each residue.
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

# Usage Example.
'''
if __name__ == '__main__':
    segmentor = ProteinSegmentor() # Initialize model, no params needed.
    probs, nums = segmentor.predict('test_cases/3gqyA.pdb') # Run prediction. Pass in a path to a pdb file.
'''
