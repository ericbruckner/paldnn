import numpy as np
import pandas as pd
import os
import sys
import glob
import pickle


from IPython.display import SVG
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
from rdkit.Chem.Crippen import MolLogP

from mordred import Calculator, descriptors

from tensorflow.keras.models import load_model


this_path = os.path.dirname(os.path.abspath(__file__))

def DistanceMatrix(peptide,order):
    n = len(peptide)
    lag = order

    if order == 0:
        a = np.zeros((1, n))[0]
        b = np.ones((1, n-order))[0]
        m = np.diag(a, 0) + np.diag(b, -order)
    elif order < n:
        a = np.zeros((1, n))[0]
        b = np.ones((1, n-order))[0]
        m = np.diag(a, 0) + np.diag(b, -order) + np.diag(b, order)
    else:
        m = np.zeros((n, n))
            
    return m
    
def HydrophobicityDescriptor(peptide, pH, order, autocorrelation_type):
    
    #Calculate Distance Matrix
    DistMat = DistanceMatrix(peptide,order)

    #Hydropbicity Index Measured from Kovacs et al. Biopolymers. 2006; 84(3): 283–297 (DOI: 10.1002/bip.20417)
    AA_index = ['W','F','L','I','M','Y','V','P','C','A','E','T','D','Q','S','N','G','R','H','K']
    pH2_hydrophilicity_index = [32.2,29.1,23.4,21.3,16.1,15.4,13.8,9.4,8.1,3.6,3.6,2.8,2.2,0.5,0,0,0,-5,-7,-7]
    pH5_hydrophilicity_index = [33.2,30.1,24.1,22.2,16.4,15.2,14,9.4,7.9,3.3,-.5,2.8,-1,0.6,0,0,0,-3.7,-5.1,-3.7]
    pH7_hydrophilicity_index = [32.9,29.9,24.2,33.4,16.3,15.4,14.4,9.7,8.3,3.9,-0.9,3.9,-0.9,0.5,0.5,0.5,0,3.9,3.4,-1.1]
    
    #Calculate w, the amino acid property vector
    w = []
    for aa in peptide:
        i = AA_index.index(aa)
        if pH < 3:
            w.append(pH2_hydrophilicity_index[i])
        elif pH >= 3 and pH < 6:
            w.append(pH5_hydrophilicity_index[i])
        elif pH >= 6:
            w.append(pH5_hydrophilicity_index[i])     
    w = np.matrix(w)
  
    descriptor_list = ['ATS','AATS','ATSC','AATSC','MATS']
    
    #Calculate Autocorrelation Functions
  
    w_c = w - np.average(w)
    delta = len(np.argwhere(DistMat == 1))

    B = np.matrix(DistMat)
    ATS = float(w*B*w.transpose())
    ATSC = float(w_c*B*w_c.transpose())

    if delta == 0:
            AATS = np.nan
            AATSC = np.nan
    else:
        AATS = ATS / delta
        AATSC = ATSC / delta
    if w_c*w_c.transpose() == 0:
        MATS = np.nan
    else:
        MATS = AATSC / float(w_c*w_c.transpose()) / len(peptide)
    
    autocorrelation_dict = {'ATS': ATS, 'AATS':AATS, 'ATSC':ATSC, 'AATSC':AATSC, 'MATS': MATS}
    return autocorrelation_dict[autocorrelation_type]

def HydrophobicityGradient(peptide, Nterm_SMILES, Cterm_SMILES, pH, degree):
    from rdkit import Chem

    if Cterm_SMILES == 'None':
        Cterm_SMILES = 'O'
    if Nterm_SMILES == 'None':
        Nterm_SMILES = 'N'
        
    Cterm = Chem.MolFromSmiles(Cterm_SMILES)
    Nterm = Chem.MolFromSmiles(Nterm_SMILES)
  
    Cterm_SLogP = MolLogP(Cterm)
    Nterm_SLogP = MolLogP(Nterm)

    #Hydropbicity Index Measured from Kovacs et al. Biopolymers. 2006; 84(3): 283–297 (DOI: 10.1002/bip.20417)
    AA_index = ['W','F','L','I','M','Y','V','P','C','A','E','T','D','Q','S','N','G','R','H','K']
    pH2_hydrophilicity_index = [32.2,29.1,23.4,21.3,16.1,15.4,13.8,9.4,8.1,3.6,3.6,2.8,2.2,0.5,0,0,0,-5,-7,-7]
    pH5_hydrophilicity_index = [33.2,30.1,24.1,22.2,16.4,15.2,14,9.4,7.9,3.3,-.5,2.8,-1,0.6,0,0,0,-3.7,-5.1,-3.7]
    pH7_hydrophilicity_index = [32.9,29.9,24.2,33.4,16.3,15.4,14.4,9.7,8.3,3.9,-0.9,3.9,-0.9,0.5,0.5,0.5,0,3.9,3.4,-1.1]
    
    w = []
    for aa in peptide:
        i = AA_index.index(aa)
        if pH < 3:
            w.append(pH2_hydrophilicity_index[i])
        elif pH >= 3 and pH < 6:
            w.append(pH5_hydrophilicity_index[i])
        elif pH >= 6:
            w.append(pH5_hydrophilicity_index[i])     
    
    hydrograd = 0
    if Nterm_SLogP > Cterm_SLogP:
        for i in range(len(w)):
                hydrograd = hydrograd + w[i]/(i+1)**degree
    else:
        w = w[::-1]
        for i in range(len(w)):
                hydrograd = hydrograd + w[i]/(i+1)**degree
    
    return hydrograd
    
def ChargeDescriptor(peptide, pH, order, autocorrelation_type):
    
    #Calculate Distance Matrix
    DistMat = DistanceMatrix(peptide,order)

    #pKa of Side Chain from D.R. Lide, Handbook of Chemistry and Physics, 72nd Edition, CRC Press, Boca Raton, FL, 1991.
    AA_index = ['A','R','N','D','C','E','Q','G','H','O','I','L','K','M','F','P','U','S','T','W','Y','V']
    pKa = [-1,12.48,-1,3.65,8.18,4.25,-1,-1,6,-1,-1,-1,10.53,-1,-1,-1,-1,-1,-1,-1,10.07,-1]
    charge = [0,1,0,-1,0,-1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0]
    
    w = []
    for aa in peptide:
        i = AA_index.index(aa)
        if charge[i] == 1:
            if pH > pKa[i]:
                w.append(0)     
            else:
                w.append(charge[i])                          
        elif charge[i] == -1:
            if pH > pKa[i]:
                w.append(charge[i])     
            else:
                w.append(0)
        elif charge[i] == 0:
            w.append(charge[i])     

        
    w = np.matrix(w)
  
    descriptor_list = ['ATS','AATS','ATSC','AATSC','MATS']
    
    #Calculate Autocorrelation Functions
  
    w_c = w - np.average(w)
    delta = len(np.argwhere(DistMat == 1))

    B = np.matrix(DistMat)
    ATS = float(w*B*w.transpose())
    ATSC = float(w_c*B*w_c.transpose())

    if delta == 0:
            AATS = np.nan
            AATSC = np.nan
    else:
        AATS = ATS / delta
        AATSC = ATSC / delta
    if w_c*w_c.transpose() == 0:
        MATS = np.nan
    else:
        MATS = AATSC / float(w_c*w_c.transpose()) / len(peptide)
    
    autocorrelation_dict = {'ATS': ATS, 'AATS':AATS, 'ATSC':ATSC, 'AATSC':AATSC, 'MATS': MATS}
    return autocorrelation_dict[autocorrelation_type]

def aaindex2dict(path):
    file = open(path,'r')
    raw_lines = file.readlines()
    lines = [line.rstrip() for line in raw_lines]

    AAvalues = []
    AAkeys = []


    for i in range(len(lines)):
        line = lines[i]
        if line[0] == 'H':
            split_line = line.split(' ')
            AAkeys.append(split_line[1])
        if line[0] == 'I':
            index1 = lines[i+1]
            index2 = lines[i+2]
            index = index1 + index2
            if 'NA' not in index:
                num_index = [float(i) for i in index.split(' ') if len(i) > 0]
                AAvalues.append(num_index)
    AAindex = dict(zip(AAkeys, AAvalues))
    return AAindex
    
def PeptideDescriptor(peptide, order, index, aaindex_dict, autocorrelation_type):
    
    #Calculate Distance Matrix
    DistMat = DistanceMatrix(peptide, order)
    
    #Load aaindex into dict
    #file_path = this_path + '\\aaindex1'
    #aaindex_dict = aaindex2dict(file_path)
    
    #Calculate w amino acid propery vector based on aaindex
    aa_labels = np.array(['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V'])

    w = []
    for aa in peptide:
        N = np.argwhere(aa == aa_labels)
        i = N[0][0]    
        w.append(aaindex_dict[index][i])
    w = np.matrix(w)
 
    descriptor_list = ['ATS','AATS','ATSC','AATSC','MATS']
    
    #Calculate Autocorrelation Functions
  
    w_c = w - np.average(w)
    delta = len(np.argwhere(DistMat == 1))

    B = np.matrix(DistMat)
    ATS = float(w*B*w.transpose())
    ATSC = float(w_c*B*w_c.transpose())

    if delta == 0:
            AATS = np.nan
            AATSC = np.nan
    else:
        AATS = ATS / delta
        AATSC = ATSC / delta
    if w_c*w_c.transpose() == 0:
        MATS = np.nan
    else:
        MATS = AATSC / float(w_c*w_c.transpose()) / len(peptide)
    
    autocorrelation_dict = {'ATS': ATS, 'AATS':AATS, 'ATSC':ATSC, 'AATSC':AATSC, 'MATS': MATS}
    return autocorrelation_dict[autocorrelation_type]

def MordredSingleDescriptor(smi, descriptor_name, descriptor_dataframe):    
    df = descriptor_dataframe

    module = df['module'][descriptor_name]
    constructor = df['constructor'][descriptor_name]
    method = getattr(descriptors,module)
    method = getattr(method,constructor)   

    #Calculate Property            
    if pd.isnull(df['argument1'][descriptor_name]):
        calc = Calculator(method)
    elif pd.isnull(df['argument2'][descriptor_name]):
        calc = Calculator(method(df['argument1'][descriptor_name]))
    elif pd.isnull(df['argument3'][descriptor_name]):
        calc = Calculator(method(df['argument1'][descriptor_name],df['argument2'][descriptor_name]))
    elif pd.isnull(df['argument4'][descriptor_name]):
        calc = Calculator(method(df['argument1'][descriptor_name],df['argument2'][descriptor_name],df['argument3'][descriptor_name]))
    elif pd.isnull(df['argument5'][descriptor_name]):
        calc = Calculator(method(df['argument1'][descriptor_name],df['argument2'][descriptor_name],df['argument3'][descriptor_name],df['argument4'][descriptor_name]))
    else:
        calc = Calculator(method(df['argument1'][descriptor_name],df['argument2'][descriptor_name],df['argument3'][descriptor_name],df['argument4'][descriptor_name],df['argument5'][descriptor_name]))

    mol = Chem.MolFromSmiles(smi)
    single_property = calc(mol)
    
    return single_property[0]

def PA_Builder(N_terminus, Peptide, C_terminus):
    #N_terminus: mol file of hydrophobic tail
    #Peptide: mol file of peptide sequence generated from a HELM sequence
    #C_terminus: mol file of C_terminus modificiation
    
    #Generate 2D Coordinates
    AllChem.Compute2DCoords(Peptide)


    #Build molecular fragment by reacting the N_terminus molecule with the amine of the N-terminus of the peptide
    rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])-[OD1].[N!H0:3]>>[C:1](=[O:2])[N:3]')    
    if N_terminus == -1:
        frag = Peptide
    else:
        AllChem.Compute2DCoords(N_terminus)
        products = rxn.RunReactants((N_terminus,Peptide))
        frag = Chem.rdmolops.RemoveHs(products[0][0])
    
   
    #React C_terminus molecule with the C-terminus of the PA molecular fragment
    if C_terminus != -1:
        AllChem.Compute2DCoords(C_terminus)
        amine = Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]')
        alcohol = Chem.MolFromSmarts('[#6][OX2H]')
        if C_terminus.HasSubstructMatch(amine):
            rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])-[OD1].[N!H0:3]>>[C:1](=[O:2])[N:3]')       
        elif C_terminus.HasSubstructMatch(alcohol):
            rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])-[OD1].[O!H0:3]>>[C:1](=[O:2])[O:3]')   

        products = rxn.RunReactants((frag, C_terminus))

        COOH = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
        match = frag.GetSubstructMatches(COOH)
        num = len(match)

        glu = Chem.MolFromSmarts('O=C(CC[C@H](NC)C(O)=O)O')
        asp = Chem.MolFromSmarts('O=C(C[C@H](NC)C(O)=O)O')
        amide = Chem.MolFromSmarts('[OX1]=CN')
        if frag.HasSubstructMatch(glu):
            PA = Chem.rdmolops.RemoveHs(products[-2][0])
        elif frag.HasSubstructMatch(asp):
            PA = Chem.rdmolops.RemoveHs(products[-2][0])
        elif C_terminus.HasSubstructMatch(amide):
            PA = Chem.rdmolops.RemoveHs(products[0][0])
        else:
            PA = Chem.rdmolops.RemoveHs(products[-1][0])

        AllChem.Compute2DCoords(PA)
    else:
        PA = frag
        AllChem.Compute2DCoords(PA)

    
    return PA
    return products


def Mol_Viewer(mol, window_height = 500, window_width = 500):
    molSize=(window_width,window_height)
    mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        Chem.rdDepictor.Compute2DCoords(mc)
    # init the drawer with the size
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    
    #
    #mc.GetAtomWithIdx(2).SetProp('atomNote', 'foo')
    #mc.GetBondWithIdx(0).SetProp('bondNote', 'bar')
    
    drawer.drawOptions().addStereoAnnotation = True
    drawer.drawOptions().addAtomIndices = True

    #draw the molcule
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    # get the SVG string
    svg = drawer.GetDrawingText()
    # fix the svg string and display it
    display(SVG(svg.replace('svg:','')))
    
def SeqToHELM(pep):
    HELM = 'PEPTIDE1{'
    for s in pep:
        HELM = HELM + s +'.'
    HELM = HELM[0:-1] +'}$$$$'
    Pep = Chem.rdmolfiles.MolFromHELM(HELM)
    return Pep

def LoadMordredPickle():
    path = this_path + '\\mordred_descriptor_list_clean.pkl'
    df = pd.read_pickle(path)
    return df
    
def LoadAAindexDict():
    path = this_path + '\\aaindex1'
    aaindex_dict = aaindex2dict(path)
    return aaindex_dict

def LoadPAdatabase():
    path = this_path + '\\PA_Database.csv'
    df = pd.read_csv(path)
    return df

def loadPALDNN():
    path = this_path + '\\paldnn.h5'
    model = load_model(path)
    return model

def loadScaler():
    path = this_path + '\\scaler.pkl'
    scaler = pickle.load(open(path, 'rb'))
    return scaler

def loadSelectedFeatures():
    path = this_path + '\\selected_features.csv'
    selected_features_df = pd.read_csv(path)
    selected_features = list(selected_features_df['features'])    
    return selected_features
    
def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()
    
def DistanceMatrices(peptide,order):
    n = len(peptide)
    lag = order
    DistMat = []

    for i in range(lag):
        if i == 0:
            a = np.zeros((1, n))[0]
            b = np.ones((1, n-i))[0]
            m = np.diag(a, 0) + np.diag(b, -i)
            DistMat.append(m)
        elif i < n:
            a = np.zeros((1, n))[0]
            b = np.ones((1, n-i))[0]
            m = np.diag(a, 0) + np.diag(b, -i) + np.diag(b, i)
            DistMat.append(m)
        else:
            DistMat.append(np.zeros((n, n)))
            
    return DistMat

def PeptideDescriptors(peptide, order, aaindex_dict):
    
    #Calculate Distance Matrix
    DistMat = DistanceMatrices(peptide,order)
    lag = order

    #Construct AA Property Vector
    AA_labels = np.array(['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V'])

        
    col_labels = []
    col_val = []
    descriptor_list = ['ATS','AATS','ATSC','AATSC','MATS']
    
    for AA_property in list(aaindex_dict.keys()):
        w = []
        for aa in peptide:
            N = np.argwhere(aa == AA_labels)
            i = N[0][0]    
            w.append(aaindex_dict[AA_property][i])
        w = np.matrix(w)

        #Calculate Molecular Descriptors
        ATS = []
        AATS = []
        ATSC = []
        AATSC = []
        MATS = []
    
        w_c = w - np.average(w)
        for i in range(lag):
            delta = len(np.argwhere(DistMat[i] == 1))

            B = np.matrix(DistMat[i])
            ATS.append(float(w*B*w.transpose()))
            ATSC.append(float(w_c*B*w_c.transpose()))

        
            if delta == 0:
                AATS.append(np.nan)
                AATSC.append(np.nan)

            else:
                AATS.append(ATS[i] / delta)
                AATSC.append(ATSC[i] / delta)
        
            if w_c*w_c.transpose() == 0:
                MATS.append(np.nan)
            else:
                MATS.append(AATSC[i] / float(w_c*w_c.transpose()) / len(peptide))
        
        descriptor_val = [ATS,AATS,ATSC,AATSC,MATS]
        for i in range(len(descriptor_list)):
            for j in range(lag):
                col_labels.append('Peptide_' + 'AAindex_' + AA_property + '_' + descriptor_list[i] + str(j))
                col_val.append(descriptor_val[i][j])

    return col_labels, col_val

def HydrophobicityDescriptors(peptide, pH, order):
    
    #Calculate Distance Matrix
    DistMat = DistanceMatrices(peptide,order)
    lag = order

    #Construct AA Property Vector
    #pH2_AA =['L','I','F','W','M','C','Y','A','T','E','G','S','Q','D','R','K','N','H','P']
    #pH2_normalized_index = [100, 100, 92, 84, 79, 74, 52, 49, 47,13,8,0,-7,-18,-18,-26,-37,-41,-42,-46] #Normalized from Sereda et al., J. Chrom. 676: 139-153 (1994).
    
    #pH7_AA = ['F','I','W','L','V','M','Y','C','A','T','H','G','S','Q','R','K','N','E','P','D']
    #pH7_normalized_index = [100,99,97,97,76,74,63,49,41,13,8,0,-5,-10,-14,-23,-28,-46,-55] #Normalized from Monera et al., J. Protein Sci. 1: 319-329 (1995).
    
    #Hydropbicity Index Measured from Kovacs et al. Biopolymers. 2006; 84(3): 283–297 (DOI: 10.1002/bip.20417)
    AA_index = ['W','F','L','I','M','Y','V','P','C','A','E','T','D','Q','S','N','G','R','H','K']
    pH2_hydrophilicity_index = [32.2,29.1,23.4,21.3,16.1,15.4,13.8,9.4,8.1,3.6,3.6,2.8,2.2,0.5,0,0,0,-5,-7,-7]
    pH5_hydrophilicity_index = [33.2,30.1,24.1,22.2,16.4,15.2,14,9.4,7.9,3.3,-.5,2.8,-1,0.6,0,0,0,-3.7,-5.1,-3.7]
    pH7_hydrophilicity_index = [32.9,29.9,24.2,33.4,16.3,15.4,14.4,9.7,8.3,3.9,-0.9,3.9,-0.9,0.5,0.5,0.5,0,3.9,3.4,-1.1]
    
    w = []
    for aa in peptide:
        i = AA_index.index(aa)
        if pH < 3:
            w.append(pH2_hydrophilicity_index[i])
        elif pH >= 3 and pH < 6:
            w.append(pH5_hydrophilicity_index[i])
        elif pH >= 6:
            w.append(pH5_hydrophilicity_index[i])     
    w = np.matrix(w)
  

    col_labels = []
    col_val = []
    descriptor_list = ['ATS','AATS','ATSC','AATSC','MATS']
    
    #Calculate Molecular Descriptors
    ATS = []
    AATS = []
    ATSC = []
    AATSC = []
    MATS = []
    
    w_c = w - np.average(w)
    for i in range(lag):
        delta = len(np.argwhere(DistMat[i] == 1))

        B = np.matrix(DistMat[i])
        ATS.append(float(w*B*w.transpose()))
        ATSC.append(float(w_c*B*w_c.transpose()))

        
        if delta == 0:
                AATS.append(np.nan)
                AATSC.append(np.nan)

        else:
            AATS.append(ATS[i] / delta)
            AATSC.append(ATSC[i] / delta)
        
        if w_c*w_c.transpose() == 0:
            MATS.append(np.nan)
        else:
            MATS.append(AATSC[i] / float(w_c*w_c.transpose()) / len(peptide))
        
    descriptor_val = [ATS,AATS,ATSC,AATSC,MATS]
    for i in range(len(descriptor_list)):
        for j in range(lag):
            col_labels.append('Peptide_'+'Hydrophobicity' + '_' + descriptor_list[i] + str(j))
            col_val.append(descriptor_val[i][j])

    return col_labels, col_val

def HydrophobicityGradients(peptide, Nterm_SMILES, Cterm_SMILES, pH, degree):
    from rdkit import Chem

    if Cterm_SMILES == 'None':
        Cterm_SMILES = 'O'
    if Nterm_SMILES == 'None':
        Nterm_SMILES = 'N'
        
    Cterm = Chem.MolFromSmiles(Cterm_SMILES)
    Nterm = Chem.MolFromSmiles(Nterm_SMILES)
  
    Cterm_SLogP = MolLogP(Cterm)
    Nterm_SLogP = MolLogP(Nterm)

    #Hydropbicity Index Measured from Kovacs et al. Biopolymers. 2006; 84(3): 283–297 (DOI: 10.1002/bip.20417)
    AA_index = ['W','F','L','I','M','Y','V','P','C','A','E','T','D','Q','S','N','G','R','H','K']
    pH2_hydrophilicity_index = [32.2,29.1,23.4,21.3,16.1,15.4,13.8,9.4,8.1,3.6,3.6,2.8,2.2,0.5,0,0,0,-5,-7,-7]
    pH5_hydrophilicity_index = [33.2,30.1,24.1,22.2,16.4,15.2,14,9.4,7.9,3.3,-.5,2.8,-1,0.6,0,0,0,-3.7,-5.1,-3.7]
    pH7_hydrophilicity_index = [32.9,29.9,24.2,33.4,16.3,15.4,14.4,9.7,8.3,3.9,-0.9,3.9,-0.9,0.5,0.5,0.5,0,3.9,3.4,-1.1]
    
    w = []
    for aa in peptide:
        i = AA_index.index(aa)
        if pH < 3:
            w.append(pH2_hydrophilicity_index[i])
        elif pH >= 3 and pH < 6:
            w.append(pH5_hydrophilicity_index[i])
        elif pH >= 6:
            w.append(pH5_hydrophilicity_index[i])     
    
    col_vals = []
    col_labels = []
    for d in range(degree):
        hydrograd = 0
        if Nterm_SLogP > Cterm_SLogP:
            for i in range(len(w)):
                    hydrograd = hydrograd + w[i]/(i+1)**d
        else:
            w = w[::-1]
            for i in range(len(w)):
                    hydrograd = hydrograd + w[i]/(i+1)**d
        col_vals.append(hydrograd)
        col_labels.append('Peptide_' + 'HydroGrad_' + 'Degree_' + str(d))

    return col_labels, col_vals


def ChargeDescriptors(peptide, pH, order):
    
    #Calculate Distance Matrix
    DistMat = DistanceMatrices(peptide,order)
    lag = order

    #Construct AA Property Vector
    
    #pKa of Side Chain from D.R. Lide, Handbook of Chemistry and Physics, 72nd Edition, CRC Press, Boca Raton, FL, 1991.
    AA_index = ['A','R','N','D','C','E','Q','G','H','O','I','L','K','M','F','P','U','S','T','W','Y','V']
    pKa = [-1,12.48,-1,3.65,8.18,4.25,-1,-1,6,-1,-1,-1,10.53,-1,-1,-1,-1,-1,-1,-1,10.07,-1]
    charge = [0,1,0,-1,0,-1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0]
    
    w = []
    for aa in peptide:
        i = AA_index.index(aa)
        if charge[i] == 1:
            if pH > pKa[i]:
                w.append(0)     
            else:
                w.append(charge[i])                          
        elif charge[i] == -1:
            if pH > pKa[i]:
                w.append(charge[i])     
            else:
                w.append(0)
        elif charge[i] == 0:
            w.append(charge[i])     

        
    w = np.matrix(w)
  

    col_labels = []
    col_val = []
    descriptor_list = ['ATS','AATS','ATSC','AATSC','MATS']
    
    #Calculate Molecular Descriptors
    ATS = []
    AATS = []
    ATSC = []
    AATSC = []
    MATS = []
    
    w_c = w - np.average(w)
    for i in range(lag):
        delta = len(np.argwhere(DistMat[i] == 1))

        B = np.matrix(DistMat[i])
        ATS.append(float(w*B*w.transpose()))
        ATSC.append(float(w_c*B*w_c.transpose()))

        
        if delta == 0:
                AATS.append(np.nan)
                AATSC.append(np.nan)

        else:
            AATS.append(ATS[i] / delta)
            AATSC.append(ATSC[i] / delta)
        
        if w_c*w_c.transpose() == 0:
            MATS.append(np.nan)
        else:
            MATS.append(AATSC[i] / float(w_c*w_c.transpose()) / len(peptide))
        
    descriptor_val = [ATS,AATS,ATSC,AATSC,MATS]
    for i in range(len(descriptor_list)):
        for j in range(lag):
            col_labels.append('Peptide_'+'Charge' + '_' + descriptor_list[i] + str(j))
            col_val.append(descriptor_val[i][j])

    return col_labels, col_val

def DNNFeatureCalculator(peptide_seq, Nterm_SMILES, Cterm_SMILES, pH, selected_features, aaindex_dict, descriptor_dataframe):
    
    #Build PA
    peptide_mol = SeqToHELM(peptide_seq)
    if Cterm_SMILES == 'None' and Nterm_SMILES != 'None':
        Cterm_SMILES = 'O'
        Cterm_mol = Chem.MolFromSmiles(Cterm_SMILES)
        Nterm_mol = Chem.MolFromSmiles(Nterm_SMILES)    
        PA_mol = PA_Builder(Nterm_mol, peptide_mol, -1)
        PA_SMILES = Chem.MolToSmiles(PA_mol)
    elif Nterm_SMILES == 'None' and Cterm_SMILES != 'None':
        Nterm_SMILES = 'N'
        Cterm_mol = Chem.MolFromSmiles(Cterm_SMILES)
        Nterm_mol = Chem.MolFromSmiles(Nterm_SMILES)    
        PA_mol = PA_Builder(-1, peptide_mol, Cterm_mol)
        PA_SMILES = Chem.MolToSmiles(PA_mol)
    elif Nterm_SMILES == 'None' and Cterm_SMILES == 'None':
        Cterm_SMILES = 'O'
        Nterm_SMILES = 'N'
        Cterm_mol = Chem.MolFromSmiles(Cterm_SMILES)
        Nterm_mol = Chem.MolFromSmiles(Nterm_SMILES)    
        PA_mol = PA_Builder(-1, peptide_mol, -1)
        PA_SMILES = Chem.MolToSmiles(PA_mol)
    else:
        Cterm_mol = Chem.MolFromSmiles(Cterm_SMILES)
        Nterm_mol = Chem.MolFromSmiles(Nterm_SMILES)    
        PA_mol = PA_Builder(Nterm_mol, peptide_mol, Cterm_mol)
        PA_SMILES = Chem.MolToSmiles(PA_mol)

    #Determine Location of Hydrophobic Tail
    Cterm_SLogP = MolLogP(Cterm_mol)
    Nterm_SLogP = MolLogP(Nterm_mol)
    
    if Nterm_SLogP > Cterm_SLogP:
        Tail_SMILES = Nterm_SMILES
    else:
        Tail_SMILES = Cterm_SMILES
    
    #Calculate Features
    col_vals = []
    for f in selected_features:
        f_code = f.split('_')
        if f_code[0] == 'PA':
            if len(f_code) == 2:
                descriptor_name = f_code[1]
                single_property = MordredSingleDescriptor(PA_SMILES, descriptor_name, descriptor_dataframe)
            elif len(f_code) == 3:
                descriptor_name = f_code[1] + '_' + f_code[2]
                single_property = MordredSingleDescriptor(PA_SMILES, descriptor_name, descriptor_dataframe)
        elif f_code[0] == 'Tail':
            if len(f_code) == 2:
                descriptor_name = f_code[1]
                single_property = MordredSingleDescriptor(Tail_SMILES, descriptor_name, descriptor_dataframe)
            elif len(f_code) == 3:
                descriptor_name = f_code[1] + '_' + f_code[2]
                single_property = MordredSingleDescriptor(NTERM_SMILES, descriptor_name, descriptor_dataframe)
        elif f_code[0] == 'Peptide':
            if f_code[1] == 'AAindex':
                peptide = peptide_seq
                order = int(f_code[3][-1])
                index = f_code[2]
                autocorrelation_type = f_code[3][0:-1]
                single_property = PeptideDescriptor(peptide, order, index, aaindex_dict, autocorrelation_type)
            elif f_code[1] == 'HydroGrad':
                peptide = peptide_seq
                degree = int(f_code[3])
                single_property = HydrophobicityGradient(peptide, Nterm_SMILES, Cterm_SMILES, pH, degree)
            elif f_code[1] == 'Charge':
                peptide = peptide_seq
                order = int(f_code[2][-1])
                autocorrelation_type = f_code[2][0:-1]
                single_property = HydrophobicityDescriptor(peptide, pH, order, autocorrelation_type)
            elif f_code[1] == 'Hydrophobicity':
                peptide = peptide_seq
                order = int(f_code[2][-1])
                autocorrelation_type = f_code[2][0:-1]
                single_property = HydrophobicityDescriptor(peptide, pH, order, autocorrelation_type)
        col_vals.append(single_property)
    fingerprint = pd.DataFrame([col_vals], columns = selected_features)
    return fingerprint
