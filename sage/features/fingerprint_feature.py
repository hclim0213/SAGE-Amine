"""
Copyright (c) 2024 Hocheol Lim.
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints

def fp_extension(fp1: list, fp2: list):
    try:
        fp1.extend(fp2)
        return fp1
    except:
        return None
def maccs_from_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        fp = None
        
        if mol is not None:
            fp = MACCSkeys.GenMACCSKeys(mol).ToList()[:166]
        
        return fp
        
    except:
        return None
def ecfp6_from_smiles(smiles: str, n_bits=1024):
    try:
        mol = Chem.MolFromSmiles(smiles)
        fp = None
        
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=n_bits).ToList()
        
        return fp
        
    except:
        return None
def ecfp4_from_smiles(smiles: str, n_bits=1024):
    try:
        mol = Chem.MolFromSmiles(smiles)
        fp = None
        
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits).ToList()
        
        return fp
        
    except:
        return None
def fcfp6_from_smiles(smiles: str, n_bits=1024):
    try:
        mol = Chem.MolFromSmiles(smiles)
        fp = None
        
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, useFeatures=True, nBits=n_bits).ToList()
        
        return fp
        
    except:
        return None
def fcfp4_from_smiles(smiles: str, n_bits=1024):
    try:
        mol = Chem.MolFromSmiles(smiles)
        fp = None
        
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=n_bits).ToList()
        
        return fp
        
    except:
        return None
def pcfp_from_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        fp = None
        
        if mol is not None:
            fp = get_fingerprint(Chem.MolToSmiles(mol), 'pubchem').to_numpy().tolist()
        
        return fp
        
    except:
        return None
def standard_from_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        fp = None
        
        if mol is not None:
            fp = get_fingerprint(Chem.MolToSmiles(mol), 'standard').to_numpy().tolist()
        
        return fp
        
    except:
        return None
def extended_from_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        fp = None
        
        if mol is not None:
            fp = get_fingerprint(Chem.MolToSmiles(mol), 'extended').to_numpy().tolist()
        
        return fp
        
    except:
        return None
def morgan_from_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        fp = None
        
        if mol is not None:
            fp = get_fingerprint(Chem.MolToSmiles(mol), 'morgan').to_numpy().tolist()
        
        return fp
        
    except:
        return None
def avalon_from_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        fp = None
        
        if mol is not None:
            fp = get_fingerprint(Chem.MolToSmiles(mol), 'avalon').to_numpy().tolist()
        
        return fp
        
    except:
        return None
def rpair_from_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        fp = None
        
        if mol is not None:
            fp = get_fingerprint(Chem.MolToSmiles(mol), 'atom-pair').to_numpy().tolist()
        
        return fp
        
    except:
        return None
def rtorsion_from_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        fp = None
        
        if mol is not None:
            fp = get_fingerprint(Chem.MolToSmiles(mol), 'topological-torsion').to_numpy().tolist()
        
        return fp
        
    except:
        return None
def rdesc_from_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        fp = None
        
        if mol is not None:
            fp = get_fingerprint(Chem.MolToSmiles(mol), 'rdk-descriptor').to_numpy().tolist()
        
        return fp
        
    except:
        return None
def mol2vec_from_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        fp = None
        
        if mol is not None:
            fp = get_fingerprint(Chem.MolToSmiles(mol), 'mol2vec').to_numpy().tolist()
        
        return fp
        
    except:
        return None
