"""
Copyright (c) 2022 Hocheol Lim.
"""

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch_geometric.data import Batch

from sage.models.attribution import (
    AbstractExplainer,
    DirectedMessagePassingNetwork,
    GraphConvNetwork,
    GraphAttnTransformer,
)


class ExplainerHandler:
    def __init__(
        self,
        model: AbstractExplainer,
        optimizer,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.MSELoss()

    def train_on_graph_batch(self, batch: Batch, device=torch.device) -> float:
        pred = self.generate_preds(batch.to(device))
        loss = self.criterion(pred.squeeze(), batch.y.to(device))

        self.model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item(), pred.squeeze().detach().cpu(), batch.y.detach().cpu()
    
    def valid_on_graph_batch(self, batch: Batch, device=torch.device) -> float:
        pred = self.generate_preds(batch.to(device))
        loss = self.criterion(pred.squeeze(), batch.y.to(device))

        #self.model.zero_grad()
        #loss.backward()
        #nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        #self.optimizer.step()

        return loss.item(), pred.squeeze().detach().cpu(), batch.y.detach().cpu()
    
    def save(self, save_dir: str, best: bool = False) -> None:
        self.model.save(save_dir, best=best)

    def generate_preds(
        self, batch: Batch, return_activations: bool = False
    ) -> torch.Tensor:
        """
        Interface to generate predictions depending on the model instance from Batch data
        """
        if isinstance(self.model, DirectedMessagePassingNetwork):
            preds = self.model(
                batch.x,
                batch.edge_attr,
                batch.edge_index,
                batch.batch,
                return_activations,
            )
        elif isinstance(self.model, GraphConvNetwork) or isinstance(self.model, GraphAttnTransformer):
            preds = self.model(
                batch.x, batch.edge_index, batch.batch, return_activations
            )
        else:
            preds = self.model(
                batch.x, batch.edge_index, batch.batch, return_activations
            )
        return preds

    def generate_attributions(self, batch: Batch) -> torch.Tensor:
        with torch.no_grad():
            activations = self.generate_preds(batch, return_activations=True)
            cam = torch.matmul(
                activations, torch.transpose(self.model.final_dense.weight.data, 0, 1)  # type: ignore
            )
            return cam
