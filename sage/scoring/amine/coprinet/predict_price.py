import torch
import numpy as np
import pandas as pd
from typing import Optional
from rdkit import Chem
from sage.scoring.amine.coprinet.nets.netsGraph import PricePredictorModule
from sage.scoring.amine.coprinet.preprocessData.smilesToGraph import smiles_to_graph
from torch_geometric.loader import DataLoader

def predict_price(input_smiles: Optional[str], scoring_gram: bool = False) -> float:
    
    smiles_list = pd.Series(input_smiles)
    smiles_list.apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    
    graphs_list = []
    for temp_smiles in smiles_list:
        graphs_list.append(smiles_to_graph(temp_smiles))
    
    dataloader = DataLoader(dataset=graphs_list, batch_size=1)
    
    model = PricePredictorModule.load_from_checkpoint("/home/coprinet/epoch=238-step=1295857.ckpt", batch_size=1)
    
    model.eval()
    results = []
    for batch_idx, batch in enumerate(dataloader):
        results.append(model(batch).detach().to("cpu").numpy().tolist())

    price_list = []
    for temp_smiles, temp_pred in zip(smiles_list, results):
        if scoring_gram:
            temp_mol = Chem.MolFromSmiles(temp_smiles)
            mw = Chem.Descriptors.ExactMolWt(temp_mol)
            price = np.log(np.exp(temp_pred)*1000/mw)
        else:
            price = temp_pred
        
        price_list.append(price)
    
    if len(price_list) == 1:
        return float(price_list[0])
    else:
        return price_list