from DomainSegmentor import *
import numpy as np

def getEntropy(input_file):
    segmentor = DomainSegmentor()
    confidence, numbering = segmentor.getEntropy(input_file)
    assert len(confidence) == len(numbering)
    return confidence, numbering

working_dir = '2chf_figure_decoys/' 

accum_entropy = []
max_entropy = float('-inf')
min_entropy = float('+inf')

if __name__ == '__main__':
    for file in os.listdir(working_dir):
        if file.endswith('.pdb') and not file.startswith('.'):
            entropy, numbering = getEntropy(working_dir + file)
            entropy = np.exp(entropy)
            ent_mean = np.mean(entropy)
            ent_std = np.std(entropy)
            entropy = [(x - ent_mean) / ent_std for x in entropy]
            if max(entropy) > max_entropy: max_entropy = max(entropy)
            if min(entropy) < min_entropy: min_entropy = min(entropy)

            accum_entropy.append(entropy)

print(max_entropy)
print(min_entropy)

'''
accum_entropy = np.array(accum_entropy).flatten()
#print(len(accum_entropy))
accum_entropy = np.exp(accum_entropy)
print(np.mean(accum_entropy))
print(np.std(accum_entropy))
new_entropy = [(x - np.mean(accum_entropy)) / np.std(accum_entropy)**2 for x in accum_entropy]
print(np.max(new_entropy))
print(np.min(new_entropy))
'''