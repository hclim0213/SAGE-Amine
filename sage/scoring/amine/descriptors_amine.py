"""
Copyright (c) 2024 Hocheol Lim.
"""
import os
from rdkit import Chem
from rdkit.Chem import Mol
import numpy as np

import sys
sys.path.append('/workspace/_ext')

from sage.scoring.filters import (
    filter_mw,
    filter_rascore,
    filter_solubility,
    score_solubility,
    score_rascore,
)

def classify_amine(mol: Mol):
    from rdkit.Chem import rdqueries

    nitrogen = rdqueries.AtomNumEqualsQueryAtom(7)  # 질소 원자 찾기
    amines = mol.GetAtomsMatchingQuery(nitrogen)
    
    results = {
        "amine": 0,
        "primary": 0,
        "secondary": 0,
        "tertiary": 0,
        "cyclic": 0,
        "multiple": 0
    }
    
    count = 0
    cyclic_count = 0
    
    for amine in amines:
        if amine.IsInRing():
            results["amine"] = 1
            results["cyclic"] += 1
            cyclic_count += 1
        elif len(amine.GetNeighbors())==1 and amine.GetTotalNumHs()==2:
            results["amine"] = 1
            results["primary"] += 1
            count += 1
        elif len(amine.GetNeighbors())==2 and amine.GetTotalNumHs()==1:
            results["amine"] = 1
            results["secondary"] += 1
            count += 1
        elif len(amine.GetNeighbors())==3 and amine.GetTotalNumHs()==0:
            results["amine"] = 1
            results["tertiary"] += 1
            count += 1
            
        if count > 1 and cyclic_count == 0:
            results["multiple"] += 1
    
    return results

def filter_amine(mol: Mol) -> bool:
    flag = False
    
    results = classify_amine(mol)
    
    if results['amine'] == 0:
        return False
    else:
        flag = True
    
    return flag

def filter_ps(mol: Mol) -> bool:
    flag = False
    
    results = classify_amine(mol)
    
    if results['amine'] == 0:
        return False
    elif results['multiple'] > 0:
        return False
    elif results['cyclic'] > 0:
        return False
    elif results['tertiary'] > 0:
        return False
    elif results['primary'] > 0 or results['secondary'] > 0:
        return True
    
    return flag

def filter_pst(mol: Mol) -> bool:
    flag = False
    
    results = classify_amine(mol)
    
    if results['amine'] == 0:
        return False
    elif results['multiple'] > 0:
        return False
    elif results['cyclic'] > 0:
        return False
    elif results['primary'] > 0 or results['secondary'] > 0 or results['tertiary'] > 0:
        return True
    
    return flag

def filter_tcm(mol: Mol) -> bool:
    flag = False
    
    results = classify_amine(mol)
    
    if results['amine'] == 0:
        return False
    elif results['multiple'] > 0:
        return True
    elif results['cyclic'] > 0:
        return True
    elif results['tertiary'] > 0:
        return True
    elif results['primary'] > 0 or results['secondary'] > 0:
        flag = False
    
    return flag

def filter_CHONS(mol: Mol) -> bool:
    CHONS_elements = {"C", "H", "O", "N", "S"}
    
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in CHONS_elements:
            return False
    return True

def name_amine(mol: Mol) -> str:
    result_name = 'amine'
    
    results = classify_amine(mol)
    
    if results['amine'] == 0:
        result_name = 'non-amine'
    elif results['cyclic'] > 0:
        result_name = 'cyclic'
    elif results['multiple'] > 0:
        result_name = 'multiple'
    elif results['tertiary'] > 0:
        result_name = 'tertiary'
    elif results['secondary'] > 0:
        result_name = 'secondary'
    elif results['primary'] > 0:
        result_name = 'primary'
    
    return result_name

def score_price(mol: Mol, scoring_gram=True, upper_limit=10, lower_limit=0) -> float:
    import numpy as np
    from sage.scoring.amine.coprinet.predict_price import predict_price
    
    score_ori = predict_price(Chem.MolToSmiles(mol), scoring_gram=scoring_gram)
    score = (upper_limit - score_ori) / (upper_limit - lower_limit)
    
    return float(score), float(score_ori)

def score_melting_point(mol: Mol, lower_limit=40, upper_limit=80) -> float:
    from sage.predict.gnn_regression import get_model_regression, load_model_regression, get_dataloader, GNN_Handler
    
    temp_model = 'melting_point'

    d_mpnn_model = load_model_regression(gnn_type='DMPNN', use_best=True, save_dir='/home/amine/'+temp_model)
    gnn_handler = GNN_Handler(model_name=temp_model, model=d_mpnn_model, use_cuda=False, only_predict=True)
    score_ori = gnn_handler.predict(
        smiles_data = [Chem.MolToSmiles(mol)],
    )[0]
    
    if score_ori <= lower_limit:
        score = 1
    elif score_ori > upper_limit:
        score = 0
    else:
        score = (upper_limit - score_ori) / (upper_limit-lower_limit)
    
    del d_mpnn_model, gnn_handler
    return float(score), float(score_ori)

def score_boiling_point(mol: Mol, lower_limit=80, upper_limit=250) -> float:
    from sage.predict.gnn_regression import get_model_regression, load_model_regression, get_dataloader, GNN_Handler
    
    temp_model = 'boiling_point'

    d_mpnn_model = load_model_regression(gnn_type='DMPNN', use_best=True, save_dir='/home/amine/'+temp_model)
    gnn_handler = GNN_Handler(model_name=temp_model, model=d_mpnn_model, use_cuda=False, only_predict=True)
    score_ori = gnn_handler.predict(
        smiles_data = [Chem.MolToSmiles(mol)],
    )[0]
    
    if score_ori < lower_limit:
        score = 0
    elif score_ori >= upper_limit:
        score = 1
    else:
        score = (score_ori - lower_limit) / (upper_limit-lower_limit)
    
    del d_mpnn_model, gnn_handler
    return float(score), float(score_ori)

def score_vapor_pressure(mol: Mol, temp: float = 298.15, upper_limit=3, lower_limit=-3) -> float:
    from sage.predict.gnn_regression import get_model_regression, load_model_regression, get_dataloader, GNN_Handler
    
    temp_model = 'vapor_pressure'
    inverse_temp = 1 / temp

    d_mpnn_model = load_model_regression(gnn_type='DMPNN', use_best=True, save_dir='/home/amine/'+temp_model)
    gnn_handler = GNN_Handler(model_name=temp_model, model=d_mpnn_model, use_cuda=False, only_predict=True)
    score_ori = gnn_handler.predict(
        smiles_data = [Chem.MolToSmiles(mol)],
        addon_data = [inverse_temp],
    )[0]
    
    if score_ori < lower_limit:
        score = 0
    elif score_ori >= upper_limit:
        score = 1
    else:
        score = (upper_limit - score_ori) / (upper_limit - lower_limit)
    
    del d_mpnn_model, gnn_handler
    return float(score), float(score_ori)

def score_viscosity(mol: Mol, temp: float = 298.15, upper_limit=2, lower_limit=-1) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    temp_model = 'viscosity_rdesc_LGBM'
    clf = pickle.load(open('/home/amine/viscosity/'+temp_model+'.pkl', 'rb'))
    
    temp_rdesc = get_fingerprint(Chem.MolToSmiles(mol), 'rdk-descriptor').to_numpy().tolist()
    inverse_temp = 1 / temp
    temp_fp = np.expand_dims(np.concatenate(([inverse_temp], temp_rdesc)), axis=0)
    
    score_ori = clf.predict(temp_fp)
    
    if score_ori < lower_limit:
        score = 0
    elif score_ori >= upper_limit:
        score = 1
    else:
        score = (upper_limit - score_ori) / (upper_limit - lower_limit)
    
    return float(score), float(score_ori)

def score_pka(mol: Mol, scoring='average', upper_limit=14, lower_limit=7) -> float:
    import numpy as np
    from rdkit.Chem import MACCSkeys, AllChem
    from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    from sage.scoring.amine.molgpka.predict_pka import predict as predict_pka
    from rdkit.Chem import rdqueries
    
    base_dict = predict_pka(mol, base=True, acid=False)
    amine_dict = {}
    
    nitrogen = rdqueries.AtomNumEqualsQueryAtom(7)
    nitrogen_sites = mol.GetAtomsMatchingQuery(nitrogen)
    
    for nitrogen_site in nitrogen_sites:
        flag = False
        try:
            
            if nitrogen_site.IsInRing():
                flag = True
            elif len(nitrogen_site.GetNeighbors())==1 and nitrogen_site.GetTotalNumHs()==2:
                flag = True
            elif len(nitrogen_site.GetNeighbors())==2 and nitrogen_site.GetTotalNumHs()==1:
                flag = True
            elif len(nitrogen_site.GetNeighbors())==3 and nitrogen_site.GetTotalNumHs()==0:
                flag = True
            
            if flag:
                amine_dict[nitrogen_site.GetIdx()] = base_dict[nitrogen_site.GetIdx()]
            
        except:
            pass
    
    values = list(amine_dict.values())

    if scoring == 'max' or scoring == 'maximum':
        score_ori = max(values)
    elif scoring == 'min' or scoring == 'minimum':
        score_ori = min(values)
    elif scoring == 'average' or scoring == 'mean':
        score_ori = np.mean(values)
    elif scoring == 'median':
        score_ori = np.median(values)
    else:
        score_ori = np.mean(values)
    
    if score_ori == 'nan':
        score = -1000
    else:
        if score_ori < lower_limit:
            score = 0.0
        elif score_ori >= upper_limit:
            score = 1.0
        else:
            score = (score_ori - lower_limit) / (upper_limit - lower_limit)

    return score, score_ori

def score_amine_solubility(mol: Mol, lower_limit=-4, upper_limit=2) -> float:
    solubility_ori = score_solubility(mol)
    
    if solubility_ori < lower_limit:
        score = 0
    elif solubility_ori >= upper_limit:
        score = 1
    else:
        score = (solubility_ori - lower_limit) / (upper_limit-lower_limit)
    
    return float(score), float(solubility_ori)

def high_pka_ps_score(mol: Mol) -> float:
    
    flag = filter_ps(mol)
    penalty = 1
    if(flag == False):
        penalty = 0.1
    
    compound = Chem.MolToSmiles(mol)
    
    score, pka_score_ori = score_pka(mol)
    
    os.system("echo '"+str(compound)+"\t"+str(round(pka_score_ori,3))+"\t"+name_amine(mol)+"' >> High_pKa_PS_raw_score.txt")
    
    return float(score) * penalty

def high_pka_tcm_score(mol: Mol) -> float:
    
    flag = filter_tcm(mol)
    penalty = 1
    if(flag == False):
        penalty = 0.1
    
    compound = Chem.MolToSmiles(mol)
    
    score, pka_score_ori = score_pka(mol)
    
    os.system("echo '"+str(compound)+"\t"+str(round(pka_score_ori,3))+"\t"+name_amine(mol)+"' >> High_pKa_TCM_raw_score.txt")
    
    return float(score) * penalty

def high_pka_score(mol: Mol) -> float:
    
    flag = filter_amine(mol)
    penalty = 1
    if(flag == False):
        penalty = 0.01
    
    compound = Chem.MolToSmiles(mol)
    
    score, pka_score_ori = score_pka(mol)
    
    os.system("echo '"+str(compound)+"\t"+str(round(pka_score_ori,3))+"\t"+name_amine(mol)+"' >> High_pKa_raw_score.txt")
    
    return float(score) * penalty

def low_viscosity_ps_score(mol: Mol, temp: float = 298.15) -> float:
    
    flag = filter_ps(mol)
    penalty = 1
    if(flag == False):
        penalty = 0.1
    
    compound = Chem.MolToSmiles(mol)
    
    score, viscosity_score_ori = score_viscosity(mol, temp)
    
    os.system("echo '"+str(compound)+"\t"+str(round(viscosity_score_ori,3))+"\t"+name_amine(mol)+"' >> Low_Viscosity_PS_raw_score.txt")
    
    return float(score) * penalty

def low_viscosity_tcm_score(mol: Mol, temp: float = 298.15) -> float:
    
    flag = filter_tcm(mol)
    penalty = 1
    if(flag == False):
        penalty = 0.1
    
    compound = Chem.MolToSmiles(mol)
    
    score, viscosity_score_ori = score_viscosity(mol, temp)
    
    os.system("echo '"+str(compound)+"\t"+str(round(viscosity_score_ori,3))+"\t"+name_amine(mol)+"' >> Low_Viscosity_TCM_raw_score.txt")
    
    return float(score) * penalty

def low_viscosity_score(mol: Mol, temp: float = 298.15) -> float:
    
    flag = filter_amine(mol)
    penalty = 1
    if(flag == False):
        penalty = 0.01
    
    compound = Chem.MolToSmiles(mol)
    
    score, viscosity_score_ori = score_viscosity(mol, temp)

    os.system("echo '"+str(compound)+"\t"+str(round(viscosity_score_ori,3))+"\t"+name_amine(mol)+"' >> Low_Viscosity_raw_score.txt")
    
    return float(score) * penalty

def low_vapor_pressure_ps_score(mol: Mol) -> float:
    
    flag = filter_ps(mol)
    penalty = 1
    if(flag == False):
        penalty = 0.1
    
    compound = Chem.MolToSmiles(mol)
    
    score, vapor_pressure_score_ori = score_vapor_pressure(mol)
    
    os.system("echo '"+str(compound)+"\t"+str(round(vapor_pressure_score_ori,3))+"\t"+name_amine(mol)+"' >> Low_VaporPressure_PS_raw_score.txt")
    
    return float(score) * penalty

def low_vapor_pressure_tcm_score(mol: Mol) -> float:

    flag = filter_tcm(mol)
    penalty = 1
    if(flag == False):
        penalty = 0.1
    
    compound = Chem.MolToSmiles(mol)
    
    score, vapor_pressure_score_ori = score_vapor_pressure(mol)
    
    os.system("echo '"+str(compound)+"\t"+str(round(vapor_pressure_score_ori,3))+"\t"+name_amine(mol)+"' >> Low_VaporPressure_TCM_raw_score.txt")
    
    return float(score) * penalty

def low_vapor_pressure_score(mol: Mol) -> float:

    flag = filter_amine(mol)
    penalty = 1
    if(flag == False):
        penalty = 0.01
    
    compound = Chem.MolToSmiles(mol)
    
    score, vapor_pressure_score_ori = score_vapor_pressure(mol)
    
    os.system("echo '"+str(compound)+"\t"+str(round(vapor_pressure_score_ori,3))+"\t"+name_amine(mol)+"' >> Low_VaporPressure_raw_score.txt")
    
    return float(score) * penalty

def high_co2_absorption_ps_score(mol: Mol) -> float:
    
    flag = filter_ps(mol)
    flag_CHONS = filter_CHONS(mol)
    penalty = 1
    if(flag == False or flag_CHONS == False):
        penalty = 0.1
    
    compound = Chem.MolToSmiles(mol)
    
    pka_score, pka_score_ori = score_pka(mol)
    viscosity_score, viscosity_score_ori = score_viscosity(mol)
    vapor_pressure_score, vapor_pressure_score_ori = score_vapor_pressure(mol)
    boiling_point_score, boiling_point_score_ori = score_boiling_point(mol)
    melting_point_score, melting_point_score_ori = score_melting_point(mol)
    solubility_score, solubility_score_ori = score_amine_solubility(mol)
    rascore_score_ori = score_rascore(mol)
    price_score, price_score_ori = score_price(mol)
    
    score = np.mean([pka_score*2, viscosity_score, vapor_pressure_score, np.mean([boiling_point_score, melting_point_score]), solubility_score, np.mean([rascore_score_ori, price_score])])
    
    os.system("echo '"+str(compound)+"\t"+str(round(pka_score_ori,3))+"\t"+str(round(viscosity_score_ori,3))+"\t"+str(round(vapor_pressure_score_ori,3))+"\t"+str(round(boiling_point_score_ori,3))+"\t"+str(round(melting_point_score_ori,3))+"\t"+str(round(rascore_score_ori,3))+"\t"+str(round(price_score_ori,3))+"\t"+str(round(solubility_score_ori,3))+"\t"+name_amine(mol)+"' >> High_CO2_Absorption_PS_raw_score.txt")
    
    return float(score) * penalty

def high_co2_absorption_tcm_score(mol: Mol) -> float:
    
    flag = filter_tcm(mol)
    flag_CHONS = filter_CHONS(mol)
    penalty = 1
    if(flag == False or flag_CHONS == False):
        penalty = 0.1
    
    compound = Chem.MolToSmiles(mol)
    
    pka_score, pka_score_ori = score_pka(mol)
    viscosity_score, viscosity_score_ori = score_viscosity(mol)
    vapor_pressure_score, vapor_pressure_score_ori = score_vapor_pressure(mol)
    boiling_point_score, boiling_point_score_ori = score_boiling_point(mol)
    melting_point_score, melting_point_score_ori = score_melting_point(mol)
    solubility_score, solubility_score_ori = score_amine_solubility(mol)
    rascore_score_ori = score_rascore(mol)
    price_score, price_score_ori = score_price(mol)
    
    score = np.mean([pka_score*2, viscosity_score, vapor_pressure_score, np.mean([boiling_point_score, melting_point_score]), solubility_score, np.mean([rascore_score_ori, price_score])])
    
    os.system("echo '"+str(compound)+"\t"+str(round(pka_score_ori,3))+"\t"+str(round(viscosity_score_ori,3))+"\t"+str(round(vapor_pressure_score_ori,3))+"\t"+str(round(boiling_point_score_ori,3))+"\t"+str(round(melting_point_score_ori,3))+"\t"+str(round(rascore_score_ori,3))+"\t"+str(round(price_score_ori,3))+"\t"+str(round(solubility_score_ori,3))+"\t"+name_amine(mol)+"' >> High_CO2_Absorption_TCM_raw_score.txt")
    
    return float(score) * penalty

def high_co2_absorption_score(mol: Mol) -> float:
    
    flag = filter_amine(mol)
    flag_CHONS = filter_CHONS(mol)
    penalty = 1
    if(flag == False or flag_CHONS == False):
        penalty = 0.01
    
    compound = Chem.MolToSmiles(mol)
    
    pka_score, pka_score_ori = score_pka(mol)
    viscosity_score, viscosity_score_ori = score_viscosity(mol)
    vapor_pressure_score, vapor_pressure_score_ori = score_vapor_pressure(mol)
    boiling_point_score, boiling_point_score_ori = score_boiling_point(mol)
    melting_point_score, melting_point_score_ori = score_melting_point(mol)
    solubility_score, solubility_score_ori = score_amine_solubility(mol)
    rascore_score_ori = score_rascore(mol)
    price_score, price_score_ori = score_price(mol)
    
    score = np.mean([pka_score*2, viscosity_score, vapor_pressure_score, np.mean([boiling_point_score, melting_point_score]), solubility_score, np.mean([rascore_score_ori, price_score])])
    
    os.system("echo '"+str(compound)+"\t"+str(round(pka_score_ori,3))+"\t"+str(round(viscosity_score_ori,3))+"\t"+str(round(vapor_pressure_score_ori,3))+"\t"+str(round(boiling_point_score_ori,3))+"\t"+str(round(melting_point_score_ori,3))+"\t"+str(round(rascore_score_ori,3))+"\t"+str(round(price_score_ori,3))+"\t"+str(round(solubility_score_ori,3))+"\t"+name_amine(mol)+"' >> High_CO2_Absorption_raw_score.txt")
    
    return float(score) * penalty

def high_co2_absorption_pst_score(mol: Mol) -> float:
    
    flag = filter_pst(mol)
    flag_CHONS = filter_CHONS(mol)
    flag_mw = filter_mw(mol)
    
    penalty = 1
    if(flag == False or flag_CHONS == False or flag_mw == False):
        penalty = 0.1
    
    compound = Chem.MolToSmiles(mol)
    
    pka_score, pka_score_ori = score_pka(mol)
    viscosity_score, viscosity_score_ori = score_viscosity(mol)
    vapor_pressure_score, vapor_pressure_score_ori = score_vapor_pressure(mol)
    boiling_point_score, boiling_point_score_ori = score_boiling_point(mol=mol, lower_limit=80, upper_limit=120)
    melting_point_score, melting_point_score_ori = score_melting_point(mol)
    solubility_score, solubility_score_ori = score_amine_solubility(mol)
    rascore_score_ori = score_rascore(mol)
    price_score, price_score_ori = score_price(mol)
    
    score = np.mean([pka_score*2, viscosity_score, vapor_pressure_score, np.mean([boiling_point_score, melting_point_score]), solubility_score, np.mean([rascore_score_ori, price_score])])
    
    os.system("echo '"+str(compound)+"\t"+str(round(pka_score_ori,3))+"\t"+str(round(viscosity_score_ori,3))+"\t"+str(round(vapor_pressure_score_ori,3))+"\t"+str(round(boiling_point_score_ori,3))+"\t"+str(round(melting_point_score_ori,3))+"\t"+str(round(rascore_score_ori,3))+"\t"+str(round(price_score_ori,3))+"\t"+str(round(solubility_score_ori,3))+"\t"+name_amine(mol)+"' >> High_CO2_Absorption_PS_raw_score.txt")
    
    return float(score) * penalty

def high_co2_absorption_all_score(mol: Mol) -> float:
    
    flag = filter_amine(mol)
    flag_CHONS = filter_CHONS(mol)
    flag_mw = filter_mw(mol)
    
    penalty = 1
    if(flag == False or flag_CHONS == False or flag_mw == False):
        penalty = 0.1
    
    compound = Chem.MolToSmiles(mol)
    
    pka_score, pka_score_ori = score_pka(mol)
    viscosity_score, viscosity_score_ori = score_viscosity(mol)
    vapor_pressure_score, vapor_pressure_score_ori = score_vapor_pressure(mol)
    boiling_point_score, boiling_point_score_ori = score_boiling_point(mol=mol, lower_limit=80, upper_limit=120)
    melting_point_score, melting_point_score_ori = score_melting_point(mol)
    solubility_score, solubility_score_ori = score_amine_solubility(mol)
    rascore_score_ori = score_rascore(mol)
    price_score, price_score_ori = score_price(mol)
    
    score = np.mean([pka_score*2, viscosity_score, vapor_pressure_score, np.mean([boiling_point_score, melting_point_score]), solubility_score, np.mean([rascore_score_ori, price_score])])
    
    os.system("echo '"+str(compound)+"\t"+str(round(pka_score_ori,3))+"\t"+str(round(viscosity_score_ori,3))+"\t"+str(round(vapor_pressure_score_ori,3))+"\t"+str(round(boiling_point_score_ori,3))+"\t"+str(round(melting_point_score_ori,3))+"\t"+str(round(rascore_score_ori,3))+"\t"+str(round(price_score_ori,3))+"\t"+str(round(solubility_score_ori,3))+"\t"+name_amine(mol)+"' >> High_CO2_Absorption_raw_score.txt")
    
    return float(score) * penalty