"""
Copyright (c) 2022 Hocheol Lim.
"""

from typing import List, Optional

import torch
from rdkit import Chem
from torch_geometric.data import DataLoader
from tqdm import tqdm

from sage.data import GraphDataset
from sage.logger.abstract_logger import AbstractLogger
from sage.models.handlers import ExplainerHandler
from sage.utils.featurizer import CanonicalFeaturizer
from sage.predict.gbm import get_scoring, calculate_metrics

class ExplainerPreTrainer:
    def __init__(
        self,
        train_smiles: List[str],
        train_scores: List[float],
        explainer_handler: ExplainerHandler,
        num_epochs: int,
        batch_size: int,
        save_dir: str,
        num_workers: int,
        device: torch.device,
        logger: AbstractLogger,
        valid_smiles: Optional[List[str]] = None,
        valid_scores: Optional[List[float]] = None,
    ):
        self.explainer_handler = explainer_handler
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.device = device
        self.logger = logger

        featurizer = CanonicalFeaturizer()
        train_graph_dataset = GraphDataset(
            canon_smiles=train_smiles,
            canon_scores=train_scores,
            featurizer=featurizer,
        )

        self.train_dataset_loader = DataLoader(
            train_graph_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        
        self.valid_smiles = valid_smiles
        self.valid_scores = valid_scores
        
        if valid_smiles is not None and valid_scores is not None:
            self.best_valid_loss = float('inf')
            valid_graph_dataset = GraphDataset(
                canon_smiles=valid_smiles,
                canon_scores=valid_scores,
                featurizer=featurizer,
            )
            
            self.valid_dataset_loader = DataLoader(
                valid_graph_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

    def pretrain(self) -> None:
        
        if self.valid_smiles is not None and self.valid_scores is not None:
            print('epoch', '\t',
            'train_loss', '\t', 'train_r2', '\t', 'train_mse', '\t', 'train_mae', '\t',
            'valid_loss', '\t', 'valid_r2', '\t', 'valid_mse', '\t', 'valie_mae', '\t', 'best')
        else:
            print('epoch', '\t',
            'train_loss', '\t', 'train_r2', '\t', 'train_mse', '\t', 'train_mae')
            
        for epoch in tqdm(range(self.num_epochs)):
            train_loss = []
            train_preds = []
            train_targets = []
            
            for batch in tqdm(self.train_dataset_loader):
                loss_train, preds_train, targets_train = self.explainer_handler.train_on_graph_batch(
                    batch=batch, device=self.device
                )
                self.logger.log_metric("explainer_train_loss", loss_train)
                train_loss.append(loss_train)
                train_preds.append(preds_train)
                train_targets.append(targets_train)
            
            train_loss_ = sum([float(i) for i in train_loss]) / len(train_loss)
            train_preds = torch.cat(train_preds)
            train_targets = torch.cat(train_targets)
            train_metrics = calculate_metrics(train_preds, train_targets, get_scoring(keyword='regression'))
            
            if self.valid_smiles is not None and self.valid_scores is not None:
                valid_loss = []
                valid_preds = []
                valid_targets = []
                best_valid = False
                
                for batch in tqdm(self.valid_dataset_loader):
                    loss_valid, preds_valid, targets_valid = self.explainer_handler.valid_on_graph_batch(
                        batch=batch, device=self.device
                    )
                    self.logger.log_metric("explainer_valid_loss", loss_valid)
                    valid_loss.append(loss_valid)
                    valid_preds.append(preds_valid)
                    valid_targets.append(targets_valid)
                
                valid_loss_ = sum([float(i) for i in valid_loss]) / len(valid_loss)
                if valid_loss_ <= self.best_valid_loss:
                    self.best_valid_loss = valid_loss_
                    best_valid = True
                
                valid_preds = torch.cat(valid_preds)
                valid_targets = torch.cat(valid_targets)
                valid_metrics = calculate_metrics(valid_preds, valid_targets, get_scoring(keyword='regression'))
                
                print(epoch, '\t', 
                train_loss_, '\t', train_metrics['r2'], '\t', train_metrics['mse'], '\t', train_metrics['mae'], '\t',
                valid_loss_, '\t', valid_metrics['r2'], '\t', valid_metrics['mse'], '\t', valid_metrics['mae'], '\t', best_valid)
                
                if best_valid:
                    self.explainer_handler.save(self.save_dir, best=True)
                
            else:
                print(epoch, '\t', 
                train_loss_, '\t', train_metrics['r2'], '\t', train_metrics['mse'], '\t', train_metrics['mae'])
            
            self.explainer_handler.save(self.save_dir)
