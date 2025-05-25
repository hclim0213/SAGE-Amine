"""
Copyright (c) 2022 Hocheol Lim.
"""

from typing import Any, Dict, Optional, Tuple, Type

import torch
from torch_geometric.nn import GCNConv, global_mean_pool

from .abstract_explainer import AbstractExplainer


class GraphConvNetwork(AbstractExplainer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        dropout: float,
        num_feature_add: int = 0,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_feature_add = num_feature_add

        activation = torch.nn.ReLU

        self.gnn_layers = torch.nn.ModuleList()
        self.gnn_layers.append(GraphConvLayer(input_size, hidden_size, dropout=dropout))
        for i in range(num_layers - 1):
            self.gnn_layers.append(
                GraphConvLayer(hidden_size, hidden_size, dropout=dropout)
            )
        
        # We don't want to add a bias term in the CAM attribution method
        self.final_dense = torch.nn.Linear(hidden_size + num_feature_add, output_size, bias=False)

    def forward(  # type: ignore
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        return_activations: bool = False,
        feature_add: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        x = node_feats.float()
        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index)

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
            num_layers=self.num_layers,
            dropout=self.dropout,
            num_feature_add=self.num_feature_add,
        )


class GraphConvLayer(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = GCNConv(in_features, out_features)

        self.res_connection = torch.nn.Linear(in_features, out_features)
        self.norm_layer = torch.nn.LayerNorm(out_features)

        self.dropout = torch.nn.Dropout(p=dropout)
        self.activation = torch.nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv(x, edge_index)

        # residual connection is applied before normalization and activation
        # according to https://arxiv.org/pdf/2006.07739.pdf
        x += self.res_connection(identity)
        x = self.norm_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
