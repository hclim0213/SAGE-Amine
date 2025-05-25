"""
Copyright (c) 2024 Hocheol Lim.
"""
import sys
sys.path.append('/workspace/_ext')

import numpy as np
from rdkit import Chem

from typing import Optional, Any, Dict, Tuple, Type, Callable, List
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from sage.utils.featurizer import CanonicalFeaturizer
from sage.models.attribution_cla import (
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

import torch.utils.data
import torchvision

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

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(
        self,
        dataset,
        labels: list = None,
        indices: list = None,
        num_samples: int = None,
        callback_get_label: Callable = None,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset) if labels is None else labels
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torch.utils.data.TensorDataset):
            return dataset.tensors[1]
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

def get_model_classification(
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

def load_model_classification(
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
    from torch_geometric.loader import DataLoader
    
    datas = []
    datas_y = []
    
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
        datas_y.append(temp_score)
    
    temp_dataloader = DataLoader(datas, sampler=ImbalancedDatasetSampler(dataset=datas, labels=datas_y), shuffle=shuffle, batch_size=batch_size, drop_last=drop_last)
    
    return temp_dataloader

class GNN_Handler:
    def __init__(
        self,
        model_name: str,
        model: AbstractExplainer,
        learning_rate: float = 1e-3,
        num_epochs: int = 300,
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
    best_metric: str = 'f1',
    ):
        
        random.seed(self.seed)
        
        if valid_loader is None:
            metric_scores = {
            'epochs': [],
            'train_loss' : [], 
            'train_accuracy': [], 
            'train_precision' : [], 
            'train_recall' : [], 
            'train_f1score' : [],
            'train_mcc': [],
            'train_auc': [],
            } 
            
            print('epoch', '\t',
                'train_loss', '\t', 'train_accuracy', '\t', 'train_precision', '\t', 'train_recall', '\t', 'train_f1score', '\t', 'train_mcc', '\t', 'train_auc')
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

                    train_temp_loss = F.binary_cross_entropy(output.flatten(), data.y)
                    train_temp_loss.backward()
            
                    train_loss_all += train_temp_loss.item()
                    self.optimizer.step()
                    train_i += 1
                    train_loss = train_loss_all / train_i
                    metric_scores['train_loss'].append(train_loss)
                
                self.model.eval()
                
                train_ACC, train_PRE, train_REC, train_F1, train_MCC, train_AUC = 0., 0., 0., 0., 0., 0.
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
                        
                        train_pred = output.cpu().numpy()[:, 0].reshape(-1).tolist()
                        train_true = data.y.cpu().numpy().reshape(-1).tolist()
            
                        train_trues.extend(train_true)
                        train_preds.extend(train_pred)
                    
                    train_preds_binary = list((np.array(train_preds) >= 0.5).astype(int))
                    train_ACC = accuracy_score(train_trues, train_preds_binary)
                    train_PRE = precision_score(train_trues, train_preds_binary, zero_division=0)
                    train_REC = recall_score(train_trues, train_preds_binary, zero_division=0)
                    train_F1 = f1_score(train_trues, train_preds_binary, zero_division=0)
                    train_MCC = matthews_corrcoef(train_trues, train_preds_binary)
                    train_AUC = roc_auc_score(train_trues, train_preds)
                    
                    metric_scores['train_accuracy'].append(train_ACC)
                    metric_scores['train_precision'].append(train_PRE)
                    metric_scores['train_recall'].append(train_REC)
                    metric_scores['train_f1score'].append(train_F1)
                    metric_scores['train_mcc'].append(train_MCC)
                    metric_scores['train_auc'].append(train_AUC)
                        
                self.model.save(self.save_dir)
                
                print(epoch, '\t', 
                    train_loss, '\t', train_ACC, '\t', train_PRE, '\t', train_REC, '\t',
                    train_F1, '\t', train_MCC, '\t', train_AUC)
            
            return metric_scores
            
        else:
            metric_scores = {
            'epochs': [],
            'train_loss' : [], 
            'train_accuracy': [], 
            'train_precision' : [], 
            'train_recall' : [], 
            'train_f1score' : [],
            'train_mcc': [],
            'train_auc': [],
            'valid_loss' : [], 
            'valid_accuracy': [], 
            'valid_precision' : [], 
            'valid_recall' : [], 
            'valid_f1score' : [],
            'valid_mcc': [],
            'valid_auc': [],
            } 
            
            best_score = 0
            
            print('epoch', '\t',
                'train_loss', '\t', 'train_accuracy', '\t', 'train_precision', '\t', 'train_recall', '\t', 'train_f1score', '\t', 'train_mcc', '\t', 'train_auc', '\t',
                'valid_loss', '\t', 'valid_accuracy', '\t', 'valid_precision', '\t', 'valid_recall', '\t', 'valid_f1score', '\t', 'valid_mcc', '\t', 'valid_auc', '\t',
                'best')
                
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
                    
                    train_temp_loss = F.binary_cross_entropy(output.flatten(), data.y)
                    train_temp_loss.backward()
            
                    train_loss_all += train_temp_loss.item()
                    self.optimizer.step()
                    train_i += 1
                    train_loss = train_loss_all / train_i
                    metric_scores['train_loss'].append(train_loss)
                
                self.model.eval()
                
                train_ACC, train_PRE, train_REC, train_F1, train_MCC, train_AUC = 0., 0., 0., 0., 0., 0.
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
                        
                        train_pred = output.cpu().numpy()[:, 0].reshape(-1).tolist()
                        train_true = data.y.cpu().numpy().reshape(-1).tolist()
                        
                        train_trues.extend(train_true)
                        train_preds.extend(train_pred)
                    
                    train_preds_binary = list((np.array(train_preds) >= 0.5).astype(int))
                    train_ACC = accuracy_score(train_trues, train_preds_binary)
                    train_PRE = precision_score(train_trues, train_preds_binary, zero_division=0)
                    train_REC = recall_score(train_trues, train_preds_binary, zero_division=0)
                    train_F1 = f1_score(train_trues, train_preds_binary, zero_division=0)
                    train_MCC = matthews_corrcoef(train_trues, train_preds_binary)
                    train_AUC = roc_auc_score(train_trues, train_preds)
                    
                    metric_scores['train_accuracy'].append(train_ACC)
                    metric_scores['train_precision'].append(train_PRE)
                    metric_scores['train_recall'].append(train_REC)
                    metric_scores['train_f1score'].append(train_F1)
                    metric_scores['train_mcc'].append(train_MCC)
                    metric_scores['train_auc'].append(train_AUC)
                        
                self.model.eval()
                
                valid_ACC, valid_PRE, valid_REC, valid_F1, valid_MCC, valid_AUC = 0., 0., 0., 0., 0., 0.
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
                        
                        valid_temp_loss = F.binary_cross_entropy(output.flatten(), data.y)
            
                        valid_loss_all += valid_temp_loss.item()
                        valid_i += 1
                        valid_loss = valid_loss_all / valid_i
                        metric_scores['valid_loss'].append(valid_loss)
                        
                        valid_pred = output.cpu().numpy()[:, 0].reshape(-1).tolist()
                        valid_true = data.y.cpu().numpy().reshape(-1).tolist()
            
                        valid_trues.extend(valid_true)
                        valid_preds.extend(valid_pred)
                        
                    valid_preds_binary = list((np.array(valid_preds) >= 0.5).astype(int))
                    valid_ACC = accuracy_score(valid_trues, valid_preds_binary)
                    valid_PRE = precision_score(valid_trues, valid_preds_binary, zero_division=0)
                    valid_REC = recall_score(valid_trues, valid_preds_binary, zero_division=0)
                    valid_F1 = f1_score(valid_trues, valid_preds_binary, zero_division=0)
                    valid_MCC = matthews_corrcoef(valid_trues, valid_preds_binary)
                    valid_AUC = roc_auc_score(valid_trues, valid_preds)
                    
                    metric_scores['valid_accuracy'].append(valid_ACC)
                    metric_scores['valid_precision'].append(valid_PRE)
                    metric_scores['valid_recall'].append(valid_REC)
                    metric_scores['valid_f1score'].append(valid_F1)
                    metric_scores['valid_mcc'].append(valid_MCC)
                    metric_scores['valid_auc'].append(valid_AUC)
                
                best_valid = False
                if best_metric != 'loss':
                    if best_metric == 'f1' or best_metric == 'F1':
                        valid_score = train_F1 * valid_F1
                    elif best_metric == 'mcc' or best_metric == 'MCC':
                        valid_score = train_MCC * valid_MCC
                    elif best_metric == 'auc' or best_metric == 'AUC':
                        valid_score = train_AUC * valid_AUC
                    
                    if best_score <= valid_score:
                        best_score = valid_score
                        self.model.save(self.save_dir, best=True)
                        best_valid = True
                else:
                    valid_score = valid_loss
                    
                    if valid_score <= best_score:
                        best_score = valid_score
                        self.model.save(self.save_dir, best=True)
                        best_valid = True
                        
                self.model.save(self.save_dir)
                
                print(epoch, '\t', 
                    train_loss, '\t', train_ACC, '\t', train_PRE, '\t', train_REC, '\t', train_F1, '\t', train_MCC, '\t', train_AUC, '\t',
                    valid_loss, '\t', valid_ACC, '\t', valid_PRE, '\t', valid_REC, '\t', valid_F1, '\t', valid_MCC, '\t', valid_AUC, '\t',
                    best_valid)
                
            return metric_scores