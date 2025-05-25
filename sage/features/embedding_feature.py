"""
Copyright (c) 2024 Hocheol Lim.
"""
import numpy as np
from .embedding_utils import smiles_char_to_int, smiles_int_to_char

def onehot_from_smiles(smiles: str, start_end=True, matrix_output=True):
    
    correct_term = 0
    if not(start_end):
        correct_term = 2
    
    num_classes = 46 - correct_term
    encode_smiles = smiles_char_to_int(smiles, start_end)
    
    if matrix_output:
        one_hot_matrix = np.zeros((len(encode_seq), num_classes))

        for i, val in enumerate(encode_smiles):
            if 0 < val <= 46:
                one_hot_matrix[i, val-1-correct_term] = 1
        
        return one_hot_matrix
    else:
        return encode_smiles

def mfbert_from_smiles(smiles: str):
    import sys
    sys.path.append('/home/MFBERT')
    from Tokenizer.MFBERT_Tokenizer import MFBERTTokenizer
    from Model.model import MFBERT
    
    tokenizer = MFBERTTokenizer.from_pretrained('/home/MFBERT/Tokenizer/Model/sentencepiece.unigram.model', dict_file = '/home/MFBERT/Tokenizer/Model/dict.txt')
    model = MFBERT(weights_dir='/home/MFBERT/Model/pre-trained', return_attention=False, inference_method='mean')
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        fp = None
        
        if mol is not None:
            fp = model(tokenizer(Chem.MolToSmiles(mol), return_tensors='pt')).detach().numpy()[0].tolist()
        
        return fp
        
    except:
        return None
