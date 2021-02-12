# paldnn
Peptide Amphiphile Learning with Deep Neural Networks (PALDNN). Python scripts for creating a deep neural network to predict the nanostructures of peptide amphiphiles. 

    Built using python3.7

## Software prerequisites:
- Python3 (Tested on version 3.7.6)
- numpy (Tested on version 1.17.0)
- pandas (Test on version 1.1.3)
- scikit-learn (Tested on version 0.23.2)
- keras (Tested on version 2.3.1) with Tensorflow backend  (Tested on version 1.14.0)
- mordred  (https://github.com/mordred-descriptor/mordred)
- rdkit (https://github.com/rdkit)

## Description
This package contains Python scripts to build and/or deploy a deep neural network (DNN) to predict whether a peptide amphiphile (PA) will assemble into micellar or fibrous nanostructures. Typical steps in this process include the following:
- Build a SMILES for a PA based on the N-terminus modification, C-terminus modification, and amino acid sequence
- Calculate molecular descriptors for a given PA
- Train and optimize a DNN based on the calcualted molecular descriptors
- Search the chemical space of PAs for a specific nanostructure

main.py is the main script that lets you do all the above as shown by the examples

## Examples
Common usage cases for PALDNN. See example.py for the full scripts.

### Using the PALDNN model for PA nanostructure prediction
#### Load necessary packages
    import pandas as pd
    import numpy as np
    from paldnn import main
#### Load a Keras DNN for PA nanostructure predition
    DNN_model = main.loadPALDNN()

#### Load Selected Features
    selected_features = main.loadSelectedFeatures()

#### Load the Scaler
    DNNscaler = main.loadScaler()

#### Load Parameter Files for Molecular Descriptors
    descriptor_dataframe = main.LoadMordredPickle()
    aaindex_dict = main.LoadAAindexDict()

#### Specify a PA sequence
    peptide_seq = 'VVVAAAEEE'
    Nterm_SMILES = 'CCCCCCCCCCCCCCCC(O)=O' # Palmitic Acid: 'None' for NH2 terminated
    Cterm_SMILES = 'N' # 'N' for Amide terminated: 'None' for COOH terminated
    pH = 7.4

#### Calculate Molecular Fingerprint
    fingerprint_raw = main.DNNFeatureCalculator(peptide_seq, Nterm_SMILES, Cterm_SMILES, pH, selected_features, aaindex_dict, descriptor_dataframe)
    fingerprint = DNNscaler.transform(fingerprint_raw)

### Make a Prediction
    pred = DNN_model.predict(fingerprint)
    print('Micellar Score: %.2f'%pred[0][0])
    print('Fiberous Score: %.2f'%pred[0][1])

    if pred[0][0] > pred[0][1]:
        print('PALDNN predicts a micellar nanostructure')
    else:
        print('PALDNN predicts a fibrous nanostructre')

### Calculate molecular descriptors from the PA_Database.csv dataset
#### Load PA database as dataframe
    pa_database = main.LoadPAdatabase()

#### Sample a subset of the dataframe for quicker calculations
    pa_database_subset = pa_database.sample(10)

#### Extract PA structure from the database
    peptides = [p.strip(' ') for p in pa_database_subset['Pep Seq']]
    Nterm = pa_database_subset['N-Term SMILES']
    Cterm = pa_database_subset['C-Term SMILES']
    pHs = pa_database_subset['pH']

#### Calculated Hydrophobicity Gradient descriptors
    degree = 3 # degree parameter for the HydrophobicityGradients, see documentation for details
    data = []
    for peptide,pH,Nterm_SMILES,Cterm_SMILES in zip(peptides,pHs,Nterm,Cterm):
        [col_labels, col_val] = main.HydrophobicityGradients(peptide, Nterm_SMILES, Cterm_SMILES, pH, degree)
        data.append(col_val)
    df_HG = pd.DataFrame(np.array(data))
    df_HG.columns = col_labels

#### Calculate Charge Descriptors
    order = 3 # order of autocorrelation functions
    data = []
    for peptide,pH in zip(peptides,pHs):
        [col_labels, col_val] = main.ChargeDescriptors(peptide, pH, order)
        data.append(col_val)
    df_charge = pd.DataFrame(np.array(data))
    df_charge.columns = col_labels

#### Calculate Hydrophobicity Descriptors
    order = 3 # order of autocorrelation functions
    data = []
    for peptide,pH in zip(peptides,pHs):
        [col_labels, col_val] = main.HydrophobicityDescriptors(peptide, pH, order)
        data.append(col_val)
    df_HD = pd.DataFrame(np.array(data))
    df_HD.columns = col_labels

#### Calculate Peptide Descriptors from AAindex
    aaindex_dict = main.LoadAAindexDict() # load aaindex parameters from DOI: 10.1093/nar/gkm998
    order = 3 # order of autocorrelation functions
    data = []
    for peptide in peptides:
        [col_labels, col_val] = main.PeptideDescriptors(peptide, order, aaindex_dict)
        data.append(col_val)
    df_PD = pd.DataFrame(np.array(data))
    df_PD.columns = col_labels

    df = pd.concat([df_HG,df_charge,df_HD,df_PD],axis = 1)
    df.to_csv('Peptide_Seq_Custom_Descriptors.csv', index = False)        
