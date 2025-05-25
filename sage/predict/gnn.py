"""
Copyright (c) 2024 Hocheol Lim.
"""
import numpy as np
from typing import Optional
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sage.models.attribution import (
    AbstractExplainer,
    DirectedMessagePassingNetwork,
    GraphConvNetwork,
    GraphConvNetworkV2,
    GraphAttnTransformer,
    GraphAttnTransformerV2,
    EGConvNetwork,
    TransformerConvNetwork,
)

import random
import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader

# import pandas as pd
# from sage.predict.gnn_regression import get_model_regression, load_model_regression, get_dataloader, GNN_Handler

# data = pd.read_csv('data/vapor_pressure_240624.tsv', sep='\t')

# Example for Training

# d_mpnn_model = get_model_regression(
#     gnn_type='DMPNN', num_feature_add=1,
#     input_size=84, output_size=1,
#     edge_size=12, edge_hidden_size=512,
#     hidden_size=1024, steps=1, dropout=0.1,
# )

# train_dataloader_cv0 = get_dataloader(
#     smiles_data=data[data['cv_fold'] != 0]['RDKit_SMILES'],
#     score_data=data[data['cv_fold'] != 0]['y'],
#     addon_data=data[data['cv_fold'] != 0]['inverse_temp'],
#     batch_size=128, shuffle=True, drop_last=True,
# )
# valid_dataloader_cv0 = get_dataloader(
#     smiles_data=data[data['cv_fold'] == 0]['RDKit_SMILES'],
#     score_data=data[data['cv_fold'] == 0]['y'],
#     addon_data=data[data['cv_fold'] == 0]['inverse_temp'],
#     batch_size=128, shuffle=True, drop_last=True,
# )

# gnn_handler = GNN_Handler(
#    model_name='DMPNN_vapor_pressure', model=d_mpnn_model,
#    num_epochs=10, use_cuda=True, num_gpu=0,)

# results = gnn_handler.train(
#     train_loader=train_dataloader_cv0,
#     valid_loader=valid_dataloader_cv0,
# )

# Example for Prediction

# d_mpnn_model = load_model_regression(
#     gnn_type='DMPNN', use_best=True,
#     save_dir='DMPNN_vapor_pressure/240713_063617',)

# gnn_handler = GNN_Handler(
#    model_name='DMPNN_vapor_pressure', model=d_mpnn_model,
#    use_cuda=True, num_gpu=0,)

# output = gnn_handler.predict(
#     smiles_data = ['CCC(CC)O[N+](=O)[O-]'],
#     addon_data = [0.003354],
# )

def get_model_regression(
    gnn_type: str == 'DMPNN',
    input_size: int = 84,
    num_feature_add: int = 0,
    output_size: int = 1,
    num_layers: int = 3,
    hidden_size: int = 1024,
    edge_size: int = 12,
    edge_hidden_size: int = 512,
    steps: int = 1,
    num_heads: int = 8,
    dropout: float = 0.1,
):
    model = None
    if gnn_type == "GCN":
        model = GraphConvNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                num_layers=num_layers,
                dropout=dropout,
                num_feature_add=num_feature_add,
            )
    elif gnn_type == "GAT": 
        model = GraphAttnTransformer(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                num_feature_add=num_feature_add,
            )
    elif gnn_type == "GATV2":
        model = GraphAttnTransformerV2(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                num_feature_add=num_feature_add,
            )
    elif gnn_type == "GraphConv":
        model = GraphConvNetworkV2(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                num_layers=num_layers,
                dropout=dropout,
                num_feature_add=num_feature_add,
            )
    elif gnn_type == "TransformerConv":
        model = TransformerConvNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                num_feature_add=num_feature_add,
            )
    elif gnn_type == "EGConv":
        model = EGConvNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                num_feature_add=num_feature_add,
            )
    else:
        model = DirectedMessagePassingNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                edge_size=edge_size,
                steps=steps,
                dropout=dropout,
                num_feature_add=num_feature_add,
                )
        
    return model

def load_model_regression(
    gnn_type: str == 'DMPNN', 
    save_dir: str, 
    use_best: bool = True
    ):
    model = None
    if gnn_type == 'GCN':
        model = GraphConvNetwork.load(load_dir=save_dir, best=use_best)
    elif gnn_type == 'GAT':
        model = GraphAttnTransformer.load(load_dir=save_dir, best=use_best)
    elif gnn_type == 'GATV2':
        model = GraphAttnTransformerV2.load(load_dir=save_dir, best=use_best)
    elif gnn_type == 'EGConv':
        model = EGConvNetwork.load(load_dir=save_dir, best=use_best)
    elif gnn_type == 'TransformerConv':
        model = TransformerConvNetwork.load(load_dir=save_dir, best=use_best)
    else:
        model = DirectedMessagePassingNetwork.load(load_dir=save_dir, best=use_best)
    
    return model

def get_dataloader(
    smiles_data: [str],
    score_data: [float],
    addon_data: Optional[any] = None,
    batch_size: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    ):
    from rdkit import Chem
    from sage.utils.featurizer import CanonicalFeaturizer
    from torch_geometric.loader import DataLoader
    
    datas = []
    
    if addon_data is None:
        addon_data = [None] * len(smiles_data)
    
    for temp_smiles, temp_score, temp_addon in zip(smiles_data, score_data, addon_data):
        temp_mol = Chem.MolFromSmiles(temp_smiles)
        temp_featurizer = CanonicalFeaturizer()
        
        if temp_addon is None:
            data = temp_featurizer.process(temp_mol, temp_score)
        elif isinstance(temp_addon, float) or isinstance(temp_addon, int):
            data = temp_featurizer.process(temp_mol, temp_score, [temp_addon])
        else:
            data = temp_featurizer.process(temp_mol, temp_score, temp_addon)
        
        datas.append(data)
    
    temp_dataloader = DataLoader(datas, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    
    return temp_dataloader

class GNN_Handler:
    def __init__(
        self,
        model_name: str,
        model: AbstractExplainer,
        learning_rate: float = 1e-3,
        num_epochs: int = 100,
        save_root: str = './',
        seed: int = 404,
        use_cuda: bool = True,
        num_gpu: int = 0,
        only_predict: bool = False,
    ):  
        if use_cuda:
            device = torch.device(num_gpu)
        else:
            device = torch.device("cpu")
            
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        self.num_epochs = num_epochs
        
        now = datetime.datetime.now()
        experiment_id = now.strftime("%y%m%d_%H%M%S")
        
        self.save_dir = (
            save_root 
            + "{}/".format(model_name)
            + "{}/".format(experiment_id)
            )
        
        if only_predict is False:
            save_path = Path(self.save_dir)
            if not save_path.exists():
                save_path.mkdir(parents=True)
            
        self.seed = seed
    
    def predict(
    self,
    smiles_data: [str],
    addon_data: Optional[any] = None,
    batch_size: int = 128,
    ):
        from rdkit import Chem
        from sage.utils.featurizer import CanonicalFeaturizer
        from torch_geometric.loader import DataLoader
        
        datas = []
        
        if addon_data is None:
            addon_data = [None] * len(smiles_data)
        
        for temp_smiles, temp_addon in zip(smiles_data, addon_data):
            temp_mol = Chem.MolFromSmiles(temp_smiles)
            temp_featurizer = CanonicalFeaturizer()
            temp_score = 0
            
            if temp_addon is None:
                data = temp_featurizer.process(temp_mol, temp_score)
            elif isinstance(temp_addon, float) or isinstance(temp_addon, int):
                data = temp_featurizer.process(temp_mol, temp_score, [temp_addon])
            else:
                data = temp_featurizer.process(temp_mol, temp_score, temp_addon)
            
            datas.append(data)
        
        temp_dataloader = DataLoader(datas, batch_size=batch_size, shuffle=False, drop_last=False)
        
        random.seed(self.seed)
        
        self.model.eval()
        preds = []
        with torch.no_grad():
            for data in temp_dataloader:
                data = data.to(self.device)
                node_feats = data.x
                edge_feats = data.edge_attr
                edge_index = data.edge_index
                batch = data.batch
                
                if hasattr(data, 'feature_add'):
                    feature_add = data.feature_add
                    
                    if isinstance(self.model, DirectedMessagePassingNetwork):
                        output = self.model(node_feats=node_feats, edge_feats=edge_feats, edge_index=edge_index, batch=batch, feature_add=feature_add)
                    else:
                        output = self.model(node_feats=node_feats, edge_index=edge_index, batch=batch, feature_add=feature_add)
                else:
                    if isinstance(self.model, DirectedMessagePassingNetwork):
                        output = self.model(node_feats=node_feats, edge_feats=edge_feats, edge_index=edge_index, batch=batch)
                    else:
                            output = self.model(node_feats=node_feats, edge_index=edge_index, batch=batch)
                
                pred = output.cpu().numpy()[0][0]
                preds.append(pred)
        
        return preds
        
    def train(
    self,
    train_loader: DataLoader,
    valid_loader: Optional[DataLoader] = None,
    best_metric: str = 'r2',
    ):
        
        random.seed(self.seed)
        
        if valid_loader is None:
            metric_scores = {
            'epochs': [],
            'train_loss' : [], 
            'train_R2': [], 
            'train_RMSE' : [], 
            'train_MSE' : [], 
            'train_MAE' : [],
            } 
            
            print('epoch', '\t',
                'train_loss', '\t', 'train_r2', '\t', 'train_mse', '\t', 'train_mae')
            for epoch in range(self.num_epochs):
                metric_scores['epochs'].append(epoch)
                self.model.train()
                train_loss_all = 0
                train_i = 0
                
                for data in train_loader:
                    data = data.to(self.device)
                    node_feats = data.x
                    edge_feats = data.edge_attr
                    edge_index = data.edge_index
                    batch = data.batch
                    
                    self.optimizer.zero_grad()
                    
                    if hasattr(data, 'feature_add'):
                        feature_add = data.feature_add
                        
                        if isinstance(self.model, DirectedMessagePassingNetwork):
                            output = self.model(node_feats=node_feats, edge_feats=edge_feats, edge_index=edge_index, batch=batch, feature_add=feature_add)
                        else:
                            output = self.model(node_feats=node_feats, edge_index=edge_index, batch=batch, feature_add=feature_add)
                    else:
                        if isinstance(self.model, DirectedMessagePassingNetwork):
                            output = self.model(node_feats=node_feats, edge_feats=edge_feats, edge_index=edge_index, batch=batch)
                        else:
                            output = self.model(node_feats=node_feats, edge_index=edge_index, batch=batch)

                    train_temp_loss = F.mse_loss(output.flatten(), data.y)
                    train_temp_loss.backward()
            
                    train_loss_all += train_temp_loss.item()
                    self.optimizer.step()
                    train_i += 1
                    train_loss = train_loss_all / train_i
                    metric_scores['train_loss'].append(train_loss)
                
                self.model.eval()
                
                train_MSE, train_MAE, train_RMSE = np.inf, np.inf, np.inf
                train_trues, train_preds = [], []
                
                with torch.no_grad():
                    for data in train_loader:
                        data = data.to(self.device)
                        node_feats = data.x
                        edge_feats = data.edge_attr
                        edge_index = data.edge_index
                        batch = data.batch
                        
                        if hasattr(data, 'feature_add'):
                            feature_add = data.feature_add
                            
                            if isinstance(self.model, DirectedMessagePassingNetwork):
                                output = self.model(node_feats=node_feats, edge_feats=edge_feats, edge_index=edge_index, batch=batch, feature_add=feature_add)
                            else:
                                output = self.model(node_feats=node_feats, edge_index=edge_index, batch=batch, feature_add=feature_add)
                        else:
                            if isinstance(self.model, DirectedMessagePassingNetwork):
                                output = self.model(node_feats=node_feats, edge_feats=edge_feats, edge_index=edge_index, batch=batch)
                            else:
                                output = self.model(node_feats=node_feats, edge_index=edge_index, batch=batch)
                        
                        train_pred = output.cpu().numpy()[0][0]
                        train_true = data.y.cpu().numpy()[0]
            
                        train_trues.append(train_true)
                        train_preds.append(train_pred)
                            
                    train_MAE = mean_absolute_error(train_trues, train_preds)
                    train_MSE = mean_squared_error(train_trues, train_preds)
                    train_RMSE = np.square(train_MSE)
                    train_R2 = r2_score(train_trues, train_preds)
                    
                    metric_scores['train_R2'].append(train_R2)
                    metric_scores['train_RMSE'].append(train_RMSE)
                    metric_scores['train_MSE'].append(train_MSE)
                    metric_scores['train_MAE'].append(train_MAE)
                        
                self.model.save(self.save_dir)
                
                print(epoch, '\t', 
                    train_loss, '\t', train_R2, '\t', train_MSE, '\t', train_MAE)
            
            return metric_scores
            
        else:
            metric_scores = {
            'epochs': [],
            'train_loss' : [], 
            'train_R2': [], 
            'train_RMSE' : [], 
            'train_MSE' : [], 
            'train_MAE' : [],
            'valid_loss' : [], 
            'valid_R2': [], 
            'valid_RMSE' : [], 
            'valid_MSE' : [], 
            'valid_MAE' : [],
            } 
            
            if best_metric == 'r2':
                best_score = 0
            else:
                best_score = np.inf
            
            
            print('epoch', '\t',
                'train_loss', '\t', 'train_r2', '\t', 'train_mse', '\t', 'train_mae', '\t',
                'valid_loss', '\t', 'valid_r2', '\t', 'valid_mse', '\t', 'valid_mae', '\t', 'best')
            for epoch in range(self.num_epochs):
                metric_scores['epochs'].append(epoch)
                self.model.train()
                train_loss_all = 0
                valid_loss_all = 0
                train_i = 0
                valid_i = 0
                
                for data in train_loader:
                    data = data.to(self.device)
                    node_feats = data.x
                    edge_feats = data.edge_attr
                    edge_index = data.edge_index
                    batch = data.batch
                    
                    self.optimizer.zero_grad()
                    
                    if hasattr(data, 'feature_add'):
                        feature_add = data.feature_add
                        
                        if isinstance(self.model, DirectedMessagePassingNetwork):
                            output = self.model(node_feats=node_feats, edge_feats=edge_feats, edge_index=edge_index, batch=batch, feature_add=feature_add)
                        else:
                            output = self.model(node_feats=node_feats, edge_index=edge_index, batch=batch, feature_add=feature_add)
                    else:
                        if isinstance(self.model, DirectedMessagePassingNetwork):
                            output = self.model(node_feats=node_feats, edge_feats=edge_feats, edge_index=edge_index, batch=batch)
                        else:
                            output = self.model(node_feats=node_feats, edge_index=edge_index, batch=batch)

                    train_temp_loss = F.mse_loss(output.flatten(), data.y)
                    train_temp_loss.backward()
            
                    train_loss_all += train_temp_loss.item()
                    self.optimizer.step()
                    train_i += 1
                    train_loss = train_loss_all / train_i
                    metric_scores['train_loss'].append(train_loss)
                
                self.model.eval()
                
                train_MSE, train_MAE, train_RMSE = np.inf, np.inf, np.inf
                train_trues, train_preds = [], []
                
                with torch.no_grad():
                    for data in train_loader:
                        data = data.to(self.device)
                        node_feats = data.x
                        edge_feats = data.edge_attr
                        edge_index = data.edge_index
                        batch = data.batch
                        
                        if hasattr(data, 'feature_add'):
                            feature_add = data.feature_add
                            
                            if isinstance(self.model, DirectedMessagePassingNetwork):
                                output = self.model(node_feats=node_feats, edge_feats=edge_feats, edge_index=edge_index, batch=batch, feature_add=feature_add)
                            else:
                                output = self.model(node_feats=node_feats, edge_index=edge_index, batch=batch, feature_add=feature_add)
                        else:
                            if isinstance(self.model, DirectedMessagePassingNetwork):
                                output = self.model(node_feats=node_feats, edge_feats=edge_feats, edge_index=edge_index, batch=batch)
                            else:
                                output = self.model(node_feats=node_feats, edge_index=edge_index, batch=batch)
                        
                        train_pred = output.cpu().numpy()[0][0]
                        train_true = data.y.cpu().numpy()[0]
            
                        train_trues.append(train_true)
                        train_preds.append(train_pred)
                            
                    train_MAE = mean_absolute_error(train_trues, train_preds)
                    train_MSE = mean_squared_error(train_trues, train_preds)
                    train_RMSE = np.square(train_MSE)
                    train_R2 = r2_score(train_trues, train_preds)
                    
                    metric_scores['train_R2'].append(train_R2)
                    metric_scores['train_RMSE'].append(train_RMSE)
                    metric_scores['train_MSE'].append(train_MSE)
                    metric_scores['train_MAE'].append(train_MAE)
                        
                self.model.eval()
                
                valid_MSE, valid_MAE, valid_RMSE = np.inf, np.inf, np.inf
                valid_trues, valid_preds = [], []
                
                with torch.no_grad():
                    for data in valid_loader:
                        data = data.to(self.device)
                        node_feats = data.x
                        edge_feats = data.edge_attr
                        edge_index = data.edge_index
                        batch = data.batch
                        
                        if hasattr(data, 'feature_add'):
                            feature_add = data.feature_add
                            
                            if isinstance(self.model, DirectedMessagePassingNetwork):
                                output = self.model(node_feats=node_feats, edge_feats=edge_feats, edge_index=edge_index, batch=batch, feature_add=feature_add)
                            else:
                                output = self.model(node_feats=node_feats, edge_index=edge_index, batch=batch, feature_add=feature_add)
                        else:
                            if isinstance(self.model, DirectedMessagePassingNetwork):
                                output = self.model(node_feats=node_feats, edge_feats=edge_feats, edge_index=edge_index, batch=batch)
                            else:
                                output = self.model(node_feats=node_feats, edge_index=edge_index, batch=batch)
                        
                        valid_temp_loss = F.mse_loss(output.flatten(), data.y)
            
                        valid_loss_all += valid_temp_loss.item()
                        valid_i += 1
                        valid_loss = valid_loss_all / valid_i
                        metric_scores['valid_loss'].append(valid_loss)
                        
                        valid_pred = output.cpu().numpy()[0][0]
                        valid_true = data.y.cpu().numpy()[0]
            
                        valid_trues.append(valid_true)
                        valid_preds.append(valid_pred)
                        
                    valid_MAE = mean_absolute_error(valid_trues, valid_preds)
                    valid_MSE = mean_squared_error(valid_trues, valid_preds)
                    valid_RMSE = np.square(valid_MSE)
                    valid_R2 = r2_score(valid_trues, valid_preds)
            
                    metric_scores['valid_R2'].append(valid_R2)
                    metric_scores['valid_RMSE'].append(valid_RMSE)
                    metric_scores['valid_MSE'].append(valid_MSE)
                    metric_scores['valid_MAE'].append(valid_MAE)
                
                best_valid = False
                if best_metric == 'r2' or best_metric == 'R2':
                    if train_R2 >= 0 and valid_R2 >= 0:
                        valid_score = train_R2 * valid_R2
                        if best_score <= valid_score:
                            best_score = valid_score
                            self.model.save(self.save_dir, best=True)
                            best_valid = True
                else:
                    if best_metric == 'mae' or best_metric == 'MAE':
                        valid_score = train_MAE * valid_MAE
                    elif best_metric == 'mse' or best_metric == 'MSE':
                        valid_score = train_MSE * valid_MSE
                    elif best_metric == 'rmse' or best_metric == 'RMSE':
                        valid_score = train_RMSE * valid_RMSE
                    else:
                        valid_score = valid_loss
                    
                    if valid_score <= best_score:
                        best_score = valid_score
                        self.model.save(self.save_dir, best=True)
                        best_valid = True
                        
                self.model.save(self.save_dir)
                
                print(epoch, '\t', 
                    train_loss, '\t', train_R2, '\t', train_MSE, '\t', train_MAE, '\t',
                    valid_loss, '\t', valid_R2, '\t', valid_MSE, '\t', valid_MAE, '\t', best_valid)
            
            return metric_scores