import os, sys
sys.path.insert(0, 'model')
sys.path.insert(0, 'classes')
from DomainSegmentor import idx_to_class
from segmentor_ext_model import *
from segmentor_model_v2 import *
from segmentor_utils import *
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import scipy.stats

class CaspGDTPredictor:
    def __init__(self, base_model_path='model/segmentor_epoch95_model_v2', ext_model_path='model/gdt_epoch500_model', class_dict=idx_to_class, try_gpu=True):
        self.class_dict = class_dict
        self.cuda_avail = torch.cuda.is_available() and try_gpu
        self.num_classes = len(self.class_dict)-1 
        self.base_model = SegmentorModel(1,8,16, self.num_classes)
        self.ext_model = ModelExtension()
        self._init_model(self.base_model, base_model_path, name='Base')
        self._init_model(self.ext_model, ext_model_path, name='Extension')
        
    def _init_model(self, net, model_path, name=''):
        if self.cuda_avail:
            net.load_state_dict(torch.load(model_path))
            net.cuda()
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count())) 
            print(name + " Model Initialized on GPU.")
        else:
            net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
            print(name + " Model Initialized on CPU.")
        net.eval() # Set model to evaluation mode to remove stochastic regularization.

    def _get_input(self, pdb_name):
        seq_len, numbering = get_pdb_info(pdb_name)
        cm, numbering = makeContactMapTensor(pdb_name, seq_len, numbering, target_size=512, upper_tol=512)
        if self.cuda_avail:
            model_input = Variable(cm).cuda()
        else:
            model_input = Variable(cm)
        return numbering, model_input

    def predict(self, pdb_name, ignore_index=-9999, log=False):
        '''
        Input: pdb name as string.
        Output: 
            trunc_class_probs -- 38 x 512 matrix. Entry (i,j) is the probility of residue j being in class i.
            res_num -- the list of pdb_residue numberings corresponding to the columns of trunc_class_probs. For example, out_num[i] = 50, then the i-th column of trunc_class_probs corresponds to residue 50 in the actual PDB.
        '''
        numbering, model_input = self._get_input(pdb_name)
        model_output = self.ext_model(self.base_model(model_input*-100))
        gdt = round(float(model_output.data), 3)
        return gdt

if __name__ == '__main__':
    predictor = CaspGDTPredictor()
    pdb_file = sys.argv[1]
    if not pdb_file:
        raise Exception("No Input PDB Specified!")
    print("Predicted GDT-TS: " + str(predictor.predict(pdb_file)))


