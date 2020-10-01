# Deep Semantic Segmentation for Protein Structures
### Author: Raphael R. Eguchi

A CNN for performing semantic segmantation of multi-domain protein structures written in PyTorch. Requires Biopython, Pytorch, Numpy, Scipy.
A step-by-step tutorial can be found at: http://www.proteindesign.org/segmentor-tutorial.html

The model treats each residue in a protein analogously to a pixel in an image:
<p align="center">
<img src="img/concept.png" width="90%">
</p>
  
  
An example segmentation output for a discontinuous protein domain, visualized in PyMOL:
<p align="center">
<img src="img/example.png" width="60%">
</p>


Usage as a Class:
```python
import sys
sys.path.insert(0, '/path/to/classes/DomainSegmentor.py') # add to import path to your script.
from DomainSegmentor import *

segmentor = DomainSegmentor() # Initialize model.
classes, res_nums = segmentor.predictClass('/path/to/pdb')  # Get Class Predictions.
probs, res_nums = segmentor.predict('/path/to/pdb')  # Get Probability Matrix.
```
The predict function returns a matrix of probabilities and a vector of residue numbers indicating the residue number of each column. The predictClass function returns a list class predictions and corresponding vector of residue numbers. See DomainSegmentor.py for more details.
 
Usage as a PyMOL/Python Script:
```bash
pymol run_segmentor.py /path/to/pdb  # Visualize the structure in pymol.
python run_segmentor.py /path/to/pdb  # Prints a list of class predictions and corresponding residue numbers.
```




### Domain Parsing
The domain parsing variant of the model can be run using the DomainParser class or using the run_parser.py script using the same commands described above. For the DomainParser class, the predict() and predictClass() functions return domain probabilities and domain assignments respectively.

```python
import sys
sys.path.insert(0, '/path/to/classes/DomainParser.py') # add to import path to run anywhere.
from DomainParser import *

parser = DomainParser() # Initialize model.
classes, res_nums = parser.predictClass('/path/to/pdb')  # Get Parser Predictions.
probs, res_nums = parser.predict('/path/to/pdb')  # Get Probability Matrix.
```

### CASP GDT Prediction by Transfer Learning
The transfer-learning model for GDT prediction was trained on submissions to the free modelling categories for CASP 10~12. See our published manuscript for more details. 
```python
python predict_cast_gdt.py /path/to/pdb 
```
### Citing
This work was published in Oxford Bioinformatics in 2019 [(Link)](https://academic.oup.com/bioinformatics/article/36/6/1740/5551337).
Please reference using the following citation.

```
@article{10.1093/bioinformatics/btz650,
    author = {Eguchi, Raphael R and Huang, Po-Ssu},
    title = "{Multi-scale structural analysis of proteins by deep semantic segmentation}",
    journal = {Bioinformatics},
    volume = {36},
    number = {6},
    pages = {1740-1749},
    year = {2019},
    month = {08},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btz650},
    url = {https://doi.org/10.1093/bioinformatics/btz650},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/36/6/1740/32915157/btz650.pdf},
}
```


