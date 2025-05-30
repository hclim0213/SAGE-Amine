"""
Copyright (c) 2022 Hocheol Lim.
"""

from typing import Dict, Optional

import torch
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_add

from .abstract_explainer import AbstractExplainer


class DirectedMessagePassingNetwork(AbstractExplainer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        edge_size: int,
        steps: int,
        dropout: float,
        num_feature_add: int = 0,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.edge_size = edge_size
        self.steps = steps
        self.dropout = dropout
        self.num_feature_add = num_feature_add

        activation = torch.nn.ReLU

        self.gnn_layers = torch.nn.ModuleList()
        self.gnn_layers.append(
            DirectedMessagePassingLayer(
                input_size, hidden_size, edge_size, steps, dropout
            )
        )

        self.final_dense = torch.nn.Linear(hidden_size + num_feature_add, output_size, bias=False)

    def forward(  # type: ignore
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        return_activations: bool = False,
        feature_add: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        x = node_feats.float()
        for i, layer in enumerate(self.gnn_layers):
            x = layer(node_feats, edge_feats, edge_index)

        if return_activations:
            return x

        x = global_mean_pool(x, batch)
        
        if feature_add is None:
            x = x
        else:
            if feature_add.dim() == 0:
                x = torch.cat((x, feature_add.unsqueeze(0)), dim=1)
            else:
                x = torch.cat((x, feature_add.view(-1, 1)), dim=1)
        
        x = self.final_dense(x)
        return x

    def config(self) -> Dict:
        return dict(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            edge_size=self.edge_size,
            steps=self.steps,
            dropout=self.dropout,
            num_feature_add=self.num_feature_add,
        )

class DirectedMessagePassingLayer(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        edge_in: int,
        steps: int,
        dropout: float,
    ):
        super().__init__()
        self.norm_layer = torch.nn.LayerNorm(output_size)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.activation = torch.nn.ReLU()

        self.dense_init = torch.nn.Linear(input_size + edge_in, output_size)
        self.dense_hidden = torch.nn.Sequential(
            torch.nn.Linear(output_size, output_size),
            torch.nn.ReLU(),
            torch.nn.Linear(output_size, output_size),
        )
        self.dense_final = torch.nn.Linear(input_size + output_size, output_size)
        self.steps = steps

    def forward(
        self,
        x: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = self.apply_conv(x, edge_feats, edge_index)
        x = self.norm_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

    def apply_conv(self, x: torch.Tensor, edge_feats, edge_index):
        edge_index = edge_index
        edge_attr = edge_feats.float()

        h0 = torch.cat((x[edge_index[0, :]], edge_attr), dim=1)
        h0 = self.activation(self.dense_init(h0))

        h = h0
        for step in range(self.steps):
            h_ = scatter_add(h, edge_index[1, :], dim=0)
            m = h_[edge_index[1, :]] - h
            h = self.activation(h0 + self.dense_hidden(m))

        m = scatter_add(h, edge_index[0, :], dim=0, dim_size=x.size(0))
        h = self.activation(self.dense_final(torch.cat((x, m), dim=1)))
        return h
