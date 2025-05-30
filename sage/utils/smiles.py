"""
Copyright (c) 2022 Hocheol Lim.
"""
import os
from itertools import islice
from typing import List, Tuple
import numpy as np
import selfies as sf
from sage.scoring.scoring_function import ScoringFunction
from sage.utils.chemistry import canonicalize
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.DataManip.Metric.rdMetricMatrixCalc import GetTanimotoSimMat
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity

from sage.data.char_dict import SmilesCharDictionary
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import logging
logging.getLogger().setLevel(logging.ERROR)

def score_wt_timeout(smile: List[str], scoring_function: ScoringFunction, timeout=900):
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(scoring_function.score, smile)
            try:
                result = future.result(timeout=timeout)
                return result
            except TimeoutError:
                print(f"Timeout occurred for smile: {smile}")
                return -1.0
            except Exception as e:
                print(f"Error occurred for smile: {smile}, error: {str(e)}")
                return -1.0

def smiles_to_actions(char_dict: SmilesCharDictionary, smis: List[str]):
    max_seq_length = char_dict.max_smi_len + 1
    enc_smis = list(map(lambda smi: char_dict.encode(smi) + char_dict.END, smis))
    actions = np.zeros((len(smis), max_seq_length), dtype=np.int32)
    seq_lengths = np.zeros((len(smis),), dtype=np.long)

    for i, enc_smi in list(enumerate(enc_smis)):
        for c in range(len(enc_smi)):
            try:
                actions[i, c] = char_dict.char_idx[enc_smi[c]]
            except:
                print(char_dict.char_idx)
                print(enc_smi)
                print(enc_smi[c])
                assert False

        seq_lengths[i] = len(enc_smi)

    return actions, seq_lengths


def canonicalize_and_score_smiles(
    smiles: List[str],
    scoring_function: ScoringFunction,
    char_dict: SmilesCharDictionary,
    pool: Parallel,
) -> Tuple[List[str], List[float]]:
    
    
    canon_smiles = pool(
        delayed(lambda smile: canonicalize(smile, include_stereocenters=False))(smile)
        for smile in smiles
    )
    
    canon_smiles = list(
        filter(
            lambda smile: (smile is not None) and char_dict.is_allowed(smile),
            canon_smiles,
        )
    )
    
    canon_scores = Parallel(n_jobs=int(pool.n_jobs))(
        delayed(score_wt_timeout)(smile, scoring_function) for smile in canon_smiles
    )
        
    filted_smiles_and_scores = list(
        filter(
            lambda smile_and_score: smile_and_score[1]
            > scoring_function.scoring_function.corrupt_score,  # type: ignore
            zip(canon_smiles, canon_scores),
        )
    )

    canon_smiles, canon_scores = (
        map(list, zip(*filted_smiles_and_scores))  # type: ignore
        if len(filted_smiles_and_scores) > 0
        else ([], [])
    )

    return canon_smiles, canon_scores

def get_fp_scores(smiles_back: List[str], target_smi: str) -> List[float]:
    smiles_back_scores = []
    target = Chem.MolFromSmiles(target_smi)
    fp_target = Chem.AllChem.GetMorganFingerprint(target, 2)
    for item in smiles_back:
        mol = Chem.MolFromSmiles(item)
        if mol is not None:
            fp_mol = Chem.AllChem.GetMorganFingerprint(mol, 2)
            score = TanimotoSimilarity(fp_mol, fp_target)
            smiles_back_scores.append(score)
        else:
            smiles_back_scores.append(0.0)
    return smiles_back_scores


def partial_sanitized_selfie(smiles) -> str:
    ps_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    ps_mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(
        ps_mol,
        Chem.SanitizeFlags.SANITIZE_FINDRADICALS
        | Chem.SanitizeFlags.SANITIZE_KEKULIZE
        | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
        | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
        | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
        | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
        catchErrors=True,
    )
    ps_smiles = Chem.MolToSmiles(ps_mol)
    return sf.encoder(ps_smiles)


def randomize_smiles(smiles_a: str, smiles_b: str) -> Tuple[str, str]:
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    Chem.Kekulize(mol_a)
    Chem.Kekulize(mol_b)

    rand_smiles_a = Chem.MolToSmiles(
        mol_a,
        canonical=False,
        doRandom=True,
        isomericSmiles=False,
        kekuleSmiles=True,
    )
    rand_smiles_b = Chem.MolToSmiles(
        mol_b,
        canonical=False,
        doRandom=True,
        isomericSmiles=False,
        kekuleSmiles=True,
    )

    if rand_smiles_a is None:
        rand_smiles_a = smiles_a
    if rand_smiles_b is None:
        rand_smiles_b = smiles_b

    return rand_smiles_a, rand_smiles_b


def calculate_similarity(smiles: List[str]) -> float:
    all_fps = []

    for smiles_str in smiles:
        mol = Chem.MolFromSmiles(smiles_str)
        fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        all_fps.append(fp)

    sim_mat = GetTanimotoSimMat(all_fps)
    return np.mean(sim_mat)
