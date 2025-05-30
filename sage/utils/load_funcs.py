"""
Copyright (c) 2022 Hocheol Lim.
"""

import logging
from typing import List, Optional, Union

import torch

from sage.logger import CommandLineLogger, NeptuneLogger
from sage.memory.fragment_lib import FragmentLibrary
from sage.models.apprentice import LSTMGenerator, TransformerGenerator, TransformerDecoderGenerator
from sage.models.apprentice import XTransformerGenerator, Decoder, Encoder, CrossAttender
from sage.models.apprentice import xLSTMGenerator
from sage.models.attribution import DirectedMessagePassingNetwork, GraphConvNetwork, GraphConvNetworkV2
from sage.models.attribution import GraphAttnTransformer, GraphAttnTransformerV2
from sage.models.attribution import EGConvNetwork, TransformerConvNetwork

from sage.models.handlers import (
    ExplainerHandler,
    GeneticOperatorHandler,
    LSTMGeneratorHandler,
    TransformerGeneratorHandler,
    TransformerDecoderGeneratorHandler,
    XTransformerGeneratorHandler,
    xLSTMGeneratorHandler,
)

def load_logger(args, tags=None):
    if args.logger_type == "Neptune":
        logger = NeptuneLogger(args, tags)
    elif args.logger_type == "CommandLine":
        logger = CommandLineLogger(args)
    else:
        raise NotImplementedError

    return logger


def load_neural_apprentice(args):
    if args.model_type == "LSTM":
        neural_apprentice = LSTMGenerator.load(load_dir=args.apprentice_load_dir, best=args.use_best)
    elif args.model_type == "Transformer":
        neural_apprentice = TransformerGenerator.load(load_dir=args.apprentice_load_dir, best=args.use_best)
    elif args.model_type == "TransformerDecoder":
        neural_apprentice = TransformerDecoderGenerator.load(load_dir=args.apprentice_load_dir, best=args.use_best)
    elif args.model_type == "XTransformer":
        neural_apprentice = XTransformerGenerator.load(load_dir=args.apprentice_load_dir, best=args.use_best)
    elif args.model_type == "xLSTM":
        neural_apprentice = xLSTMGenerator.load(load_dir=args.apprentice_load_dir, best=args.use_best)
    else:
        raise ValueError(f"{args.model_type} is not a valid model-type")

    return neural_apprentice


def load_apprentice_handler(model, optimizer, char_dict, max_sampling_batch_size, args):
    if args.model_type == "LSTM":
        apprentice_handler = LSTMGeneratorHandler(
            model=model,
            optimizer=optimizer,
            char_dict=char_dict,
            max_sampling_batch_size=max_sampling_batch_size,
        )
    elif args.model_type == "Transformer":
        apprentice_handler = TransformerGeneratorHandler(
            model=model,
            optimizer=optimizer,
            char_dict=char_dict,
            max_sampling_batch_size=max_sampling_batch_size,
        )
    elif args.model_type == "TransformerDecoder":
        apprentice_handler = TransformerDecoderGeneratorHandler(
            model=model,
            optimizer=optimizer,
            char_dict=char_dict,
            max_sampling_batch_size=max_sampling_batch_size,
        )
    elif args.model_type == "XTransformer":
        apprentice_handler = XTransformerGeneratorHandler(
            model=model,
            optimizer=optimizer,
            char_dict=char_dict,
            max_sampling_batch_size=max_sampling_batch_size,
        )
    elif args.model_type == "xLSTM":
        apprentice_handler = xLSTMGeneratorHandler(
            model=model,
            optimizer=optimizer,
            char_dict=char_dict,
            max_sampling_batch_size=max_sampling_batch_size,
        )
    return apprentice_handler


def load_genetic_experts(
    expert_types: List[str],
    args,
) -> List[GeneticOperatorHandler]:
    experts = []
    for ge_type in expert_types:
        expert_handler = GeneticOperatorHandler(
            crossover_type=ge_type,
            mutation_type=ge_type,
            mutation_initial_rate=args.mutation_initial_rate,
        )
        experts.append(expert_handler)
    return experts


def load_generator(input_size: int, args):
    if args.model_type == "LSTM":
        generator = LSTMGenerator(
            input_size=input_size,
            hidden_size=args.hidden_size,
            output_size=input_size,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )
    elif args.model_type == "Transformer":
        generator = TransformerGenerator(  # type: ignore
            n_token=input_size,
            n_embed=args.embed_size,
            n_head=args.n_head,
            n_hidden=args.hidden_size,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )
    elif args.model_type == "TransformerDecoder":
        generator = TransformerDecoderGenerator(  # type: ignore
            n_token=input_size,
            n_embed=args.embed_size,
            n_head=args.n_head,
            n_hidden=args.hidden_size,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )
    elif args.model_type == "XTransformer":
        generator = XTransformerGenerator(  # type: ignore
            num_tokens=input_size,
            max_seq_len=args.max_smiles_length,
            emb_dropout=args.dropout,
            num_memory_tokens = 20,
            l2norm_embed = True,
            post_emb_norm = True,
            attn_layers = Decoder(
                dim = args.embed_size,
                depth = args.n_layers,
                heads = args.n_head,
                layer_dropout = args.dropout,
                attn_dropout = args.dropout,
                ff_dropout = args.dropout,

                # select one of causal, rotary_xpos, rotary_pos_emb
                rotary_xpos=True,
                #causal = True,
                #rotary_pos_emb = True,
                
                # select one of rel_pos_bias, alibi_pos_bias, dynamic_pos_bias, flash_attn
                rel_pos_bias = True,
                #alibi_pos_bias = True,
                #alibi_num_heads = 4
                #dynamic_pos_bias = True,
                #dynamic_pos_bias_log_distance = False,
                #attn_flash = True, # only works in torch >= 2.0
                
                # select one of pre_norm, sandwich_norm, resi_dual
                pre_norm = True,
                #sandwich_norm = True,
                #resi_dual = True,
                #resi_dual_scale = 0.1,
                
                # select one of use_simple_rmsnorm, use_scalenorm, use_rmsnorm
                use_simple_rmsnorm = True,
                #use_scalenorm = True,
                #use_rmsnorm = True,
                
                # select one of shift_tokens
                shift_tokens = 1,
                #shift_tokens = (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)
                
                # Others
                gate_residual = True,
                scale_residual = True,
                #cross_attend = True,
                
                # select one of attn_kv_heads, attn_one_kv_head
                #attn_kv_heads = 2,
                attn_one_kv_head = True,
                
                attn_qk_norm = True,
                attn_qk_norm_dim_scale = True,
                #attn_qk_norm_groups = 10
                
                attn_sparse_topk = 8,
                attn_talking_heads = True,
                
                attn_head_scale = True,
                
                # select one of ff_swish, relu_squared
                ff_glu = True,
                #ff_swish = True,
                #ff_relu_squared = True,
                
                ff_no_bias = True,
                ff_post_act_ln = True,
            )
        )
    elif args.model_type == "xLSTM":
        generator = xLSTMGenerator(
            n_token=input_size,
            n_embed=args.embed_size,
            n_block=args.max_smiles_length+1,
            n_head=args.n_head,
            n_hidden=args.hidden_size,
            dropout=args.dropout,
            config_xlstm=args.config_xlstm,
            n_depth=args.n_depth,
            n_factor=args.n_factor,
            use_cuda = args.use_cuda,
            use_rotary_embedding = args.use_rotary_embedding,
        )
    return generator

def load_explainer(args, save_dir: str = None):
    explainer: Union[GraphConvNetwork, DirectedMessagePassingNetwork, 
                    GraphConvNetworkV2, GraphAttnTransformer, GraphAttnTransformerV2, 
                    EGConvNetwork, TransformerConvNetwork]
    if save_dir is None:
        logging.info("Loading the explainer without weights")
        if args.explainer_type == "MPNN":
            raise NotImplementedError
        elif args.explainer_type == "DMPNN":
            explainer = DirectedMessagePassingNetwork(
                input_size=args.input_size,
                hidden_size=args.hidden_size,
                output_size=args.output_size,
                edge_size=args.edge_size,
                steps=args.steps,
                dropout=args.dropout,
            )
        elif args.explainer_type == "GCN":
            explainer = GraphConvNetwork(
                input_size=args.input_size,
                hidden_size=args.hidden_size,
                output_size=args.output_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
        elif args.explainer_type == "GraphConv":
            explainer = GraphConvNetworkV2(
                input_size=args.input_size,
                hidden_size=args.hidden_size,
                output_size=args.output_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
        elif args.explainer_type == "GAT":
            explainer = GraphAttnTransformer(
                input_size=args.input_size,
                hidden_size=args.hidden_size,
                output_size=args.output_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
            )
        elif args.explainer_type == "GATV2":
            explainer = GraphAttnTransformerV2(
                input_size=args.input_size,
                hidden_size=args.hidden_size,
                output_size=args.output_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
            )
        elif args.explainer_type == "EGConv":
            explainer = EGConvNetwork(
                input_size=args.input_size,
                hidden_size=args.hidden_size,
                output_size=args.output_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
            )
        elif args.explainer_type == "TransformerConv":
            explainer = TransformerConvNetwork(
                input_size=args.input_size,
                hidden_size=args.hidden_size,
                output_size=args.output_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
            )
        else:
            raise ValueError(f"The explainer_type {args.explainer_type} is invalid")

    elif isinstance(save_dir, str):
        logging.info("Loading the explainer from pretrained weights!")
        if args.explainer_type == "DMPNN":
            explainer = DirectedMessagePassingNetwork.load(load_dir=save_dir, best=args.use_best)  # type: ignore
        elif args.explainer_type == "GCN":
            explainer = GraphConvNetwork.load(load_dir=save_dir, best=args.use_best)  # type: ignore
        elif args.explainer_type == "GAT":
            explainer = GraphAttnTransformer.load(load_dir=save_dir, best=args.use_best)
        elif args.explainer_type == "GATV2":
            explainer = GraphAttnTransformerV2.load(load_dir=save_dir, best=args.use_best)
        elif args.explainer_type == "EGConv":
            explainer = EGConvNetwork.load(load_dir=save_dir, best=args.use_best)
        elif args.explainer_type == "TransformerConv":
            explainer = TransformerConvNetwork.load(load_dir=save_dir, best=args.use_best)
        else:
            raise ValueError(f"The explainer_type {args.explainer_type} is invalid")

    return explainer


def load_explainer_handler(
    model,
    optimizer,
):
    return ExplainerHandler(
        model=model,
        optimizer=optimizer,
    )
