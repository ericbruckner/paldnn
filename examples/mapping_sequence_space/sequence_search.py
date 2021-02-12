# Search Sequence Space with PALDNN

## Import necessary libraries
from paldnn import main
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools

## Load a Keras DNN for PA nanostructure predition
DNN_model = main.loadPALDNN()

## Load Selected Features
selected_features = main.loadSelectedFeatures()

## Load the Scaler
DNNscaler = main.loadScaler()

## Load Parameter Files for Molecular Descriptors
descriptor_dataframe = main.LoadMordredPickle()
aaindex_dict = main.LoadAAindexDict()

## Set Parameters
seq_len = 2 # Enter maximum number of residues per petide

Nterm_SMILES = 'CCCCCCCCCCCCCCCC(O)=O' # Palmitic Acid tail at the N-terminus
Cterm_SMILES = 'N'                     # Amide C-terminus
pH = 7.4                               # pH for self-assembly conditions

AA_list = ['W','F','L','I','M','Y','V','P','C','A','E','T','D','Q','S','N','G','R','H','K'] # Select all or a subset of amino acids

## Generate list of all peptide sequences
peptide_list = []
for i in range(seq_len):
    combinations = list(itertools.product(AA_list, repeat = i+1))
    ith_peptide_list = [''.join(c) for c in combinations]
    peptide_list = peptide_list + ith_peptide_list

path = str(seq_len) + 'aa_sequence_search.csv'

## Search sequence space and save data file
file = open(path,'w')
file.write('Peptide Sequence,')
file.writelines(s + ',' for s in selected_features)
file.write('micelle score,fiber score,prediction\n')

for i in range(len(peptide_list)):
    peptide_seq = peptide_list[i]
    
    # Calculate Molecular Fingerprint
    fingerprint_raw = main.DNNFeatureCalculator(peptide_seq, Nterm_SMILES, Cterm_SMILES, pH, selected_features, aaindex_dict, descriptor_dataframe)

    # Scale Molecular Fingerprint
    fingerprint = DNNscaler.transform(fingerprint_raw)
    
    # Make Prediction
    pred = DNN_model.predict(fingerprint)
    micelle_score = pred[0][0]
    fiber_score = pred[0][1]
    
    if micelle_score > fiber_score:
        prediction = 'micelle'
    else:
        prediction = 'fiber'
        
    # Write Data
    file.write(peptide_seq + ',')
    file.writelines(str(f) + ',' for f in fingerprint[0])
    file.write(str(micelle_score) + ',' + str(fiber_score) + ',' + prediction + '\n')

file.close()

## Load data file and visualize results as a function of two molecular descriptors
df = pd.read_csv(path, index_col = 0)
selected_features = df.columns[0:-3]
fibrous = df[df['prediction'] == 'fiber']
micellar = df[df['prediction'] == 'micelle']

x_feature = selected_features[0]
y_feature = selected_features[18]

x1 = fibrous[x_feature]
x2 = micellar[x_feature]

y1 = fibrous[y_feature]
y2 = micellar[y_feature]

plt.scatter(x2,y2, alpha=0.5, label = 'Micellar')
plt.scatter(x1,y1, alpha=0.5, label = 'Fibrous')

plt.xlabel(x_feature)
plt.ylabel(y_feature)
plt.legend(loc='upper left')
plt.title('Tripeptide Sequence Space')
plt.savefig("sequence_space_slice.tif", format="tif", dpi = 300)