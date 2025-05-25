"""
Copyright (c) 2024 Hocheol Lim.
"""

import math
import json
from typing import Dict, Optional, List
from collections import OrderedDict
from pathlib import Path
from random import random

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from functools import partial

from einops import rearrange, repeat, reduce, pack, unpack

from x_transformers.x_transformers import (
    AbsolutePositionalEmbedding,
    AlibiPositionalBias,
    always,
    Attention,
    cast_tuple,
    default,
    DynamicPositionBias,
    equals,
    exists,
    FeedForward,
    groupby_prefix_and_trim,
    GRUGating,
    LayerIntermediates,
    RelativePositionBias,
    Residual,
    RMSNorm,
    RotaryEmbedding,
    ScaledSinusoidalEmbedding,
    ScaleNorm,
    ShiftTokens,
    SimpleRMSNorm,
    TokenEmbedding,
)

from sage.models.apprentice import AbstractGenerator

DEFAULT_DIM_HEAD = 64

class AttentionLayers(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        causal = False,
        cross_attend = False,
        only_cross = False,
        use_scalenorm = False,
        use_rmsnorm = False,
        use_simple_rmsnorm = False,
        alibi_pos_bias = False,
        alibi_num_heads = None,
        rel_pos_bias = False,
        rel_pos_num_buckets = 32,
        rel_pos_max_distance = 128,
        dynamic_pos_bias = False,
        dynamic_pos_bias_log_distance = False,
        dynamic_pos_bias_mlp_depth = 2,
        dynamic_pos_bias_norm = False,
        rotary_pos_emb = False,
        rotary_emb_dim = None,
        rotary_xpos = False,
        rotary_interpolation_factor = 1.,
        rotary_xpos_scale_base = 512,
        rotary_base_rescale_factor = 1.,
        custom_layers = None,
        sandwich_coef = None,
        par_ratio = None,
        weight_tie_layers = False,   # Albert - https://arxiv.org/abs/1909.11942
        layers_execute_order = None, # generalizes weight tying, can do arbitrary layer execution orders
        residual_attn = False,
        cross_residual_attn = False,
        macaron = False,
        pre_norm = True,
        pre_norm_has_final_norm = True,
        gate_residual = False,
        scale_residual = False,
        scale_residual_constant = 1.,
        shift_tokens = 0,
        sandwich_norm = False,
        resi_dual = False,
        resi_dual_scale = 1.,
        zero_init_branch_output = False,
        layer_dropout = 0.,
        cross_attn_tokens_dropout = 0.,
        **kwargs
    ):
        super().__init__()
        rotary_pos_emb = rotary_pos_emb or rotary_xpos

        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, kwargs = groupby_prefix_and_trim('attn_', kwargs)

        dim_head = attn_kwargs.get('dim_head', DEFAULT_DIM_HEAD)

        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.layer_drop = layer_dropout
        self.attn_drop = attn_kwargs.get('dropout') 
        self.ff_drop = ff_kwargs.get('dropout')
        
        self.causal = causal
        self.layers = nn.ModuleList([])
        
        

        self.has_pos_emb = rel_pos_bias or rotary_pos_emb

        rotary_emb_dim = max(default(rotary_emb_dim, dim_head // 2), 32)

        assert not (rotary_xpos and not causal), 'rotary xpos is not compatible with bidirectional attention'
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim, use_xpos = rotary_xpos, scale_base = rotary_xpos_scale_base, interpolation_factor = rotary_interpolation_factor, base_rescale_factor = rotary_base_rescale_factor) if rotary_pos_emb else None

        assert not (alibi_pos_bias and rel_pos_bias), 'you can only choose Alibi positional bias or T5 relative positional bias, not both'
        assert rel_pos_num_buckets <= rel_pos_max_distance, 'number of relative position buckets must be less than the relative position max distance'

        # relative positional bias

        flash_attn = attn_kwargs.get('flash', False)
        assert (int(rel_pos_bias) + int(dynamic_pos_bias) + int(alibi_pos_bias)) <= 1, 'you can only choose up to one of t5, alibi, or dynamic positional bias'

        self.rel_pos = None
        if rel_pos_bias:
            assert not flash_attn, 'flash attention not compatible with t5 relative positional bias'
            self.rel_pos = RelativePositionBias(scale = dim_head ** 0.5, causal = causal, heads = heads, num_buckets = rel_pos_num_buckets, max_distance = rel_pos_max_distance)
        elif dynamic_pos_bias:
            assert not flash_attn, 'flash attention not compatible with dynamic positional bias'
            self.rel_pos = DynamicPositionBias(dim = dim // 4, heads = heads, log_distance = dynamic_pos_bias_log_distance, depth = dynamic_pos_bias_mlp_depth, norm = dynamic_pos_bias_norm)
        elif alibi_pos_bias:
            alibi_num_heads = default(alibi_num_heads, heads)
            assert alibi_num_heads <= heads, 'number of ALiBi heads must be less than the total number of heads'
            self.rel_pos = AlibiPositionalBias(heads = alibi_num_heads, total_heads = heads)

        assert (int(sandwich_norm) + int(resi_dual)) <= 1, 'either sandwich norm or resiDual is selected, but not both'
        assert not (not pre_norm and sandwich_norm), 'sandwich norm cannot be used when not using prenorm'

        if resi_dual:
            pre_norm = False

        self.pre_norm = pre_norm
        self.sandwich_norm = sandwich_norm

        self.resi_dual = resi_dual
        assert 0 < resi_dual_scale <= 1., 'resiDual prenorm residual must be scaled by a factor greater than 0 and less than or equal to 1.'
        self.resi_dual_scale = resi_dual_scale

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn
        assert not (flash_attn and (residual_attn or cross_residual_attn)), 'flash attention is not compatible with residual attention'

        self.cross_attend = cross_attend

        assert (int(use_scalenorm) + int(use_rmsnorm) + int(use_simple_rmsnorm)) <= 1, 'you can only use either scalenorm, rmsnorm, or simple rmsnorm'

        if use_scalenorm:
            norm_class = ScaleNorm
        elif use_rmsnorm:
            norm_class = RMSNorm
        elif use_simple_rmsnorm:
            norm_class = SimpleRMSNorm
        else:
            norm_class = nn.LayerNorm

        norm_fn = partial(norm_class, dim)

        if cross_attend and not only_cross:
            default_block = ('a', 'c', 'f')
        elif cross_attend and only_cross:
            default_block = ('c', 'f')
        else:
            default_block = ('a', 'f')

        if macaron:
            default_block = ('f',) + default_block

        # zero init

        if zero_init_branch_output:
            attn_kwargs = {**attn_kwargs, 'zero_init_output':  True}
            ff_kwargs = {**ff_kwargs, 'zero_init_output':  True}

        # setup weight tying, which is a special case of `layer_execute_order`

        assert not (weight_tie_layers and any([*map(exists, (custom_layers, par_ratio, sandwich_coef))]))

        if weight_tie_layers:
            assert not exists(layers_execute_order)
            layers_execute_order = tuple(range(len(default_block))) * depth
            depth = 1

        # calculate layer block order

        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, 'par ratio out of range'
            default_block = tuple(filter(not_equals('f'), default_block))
            par_attn  = par_depth // par_ratio
            depth_cut = par_depth * 2 // 3  # 2 / 3 attention layer cutoff suggested by PAR paper
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert len(default_block) <= par_width, 'default block is too large for par_ratio'
            par_block = default_block + ('f',) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ('f',) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert sandwich_coef > 0 and sandwich_coef <= depth, 'sandwich coefficient should be less than the depth'
            layer_types = ('a',) * sandwich_coef + default_block * (depth - sandwich_coef) + ('f',) * sandwich_coef
        else:
            layer_types = default_block * depth

        self.layer_types = layer_types
        self.layers_execute_order = default(layers_execute_order, tuple(range(len(layer_types))))

        assert all([i < len(self.layer_types) for i in self.layers_execute_order])

        self.num_attn_layers = len(list(filter(equals('a'), layer_types)))

        # stochastic depth

        self.layer_dropouts = cast_tuple(layer_dropout, len(layer_types))

        # structured dropout for cross attending

        self.cross_attn_tokens_dropout = cross_attn_tokens_dropout

        # calculate token shifting

        shift_tokens = cast_tuple(shift_tokens, len(layer_types))

        # whether it has post norm

        self.final_norm = norm_fn() if pre_norm or resi_dual else nn.Identity()

        # iterate and construct layers

        for ind, (layer_type, layer_shift_tokens) in enumerate(zip(self.layer_types, shift_tokens)):
            is_last_layer = ind == (len(self.layer_types) - 1)

            if layer_type == 'a':
                layer = Attention(dim, heads = heads, causal = causal, **attn_kwargs)
            elif layer_type == 'c':
                layer = Attention(dim, heads = heads, **attn_kwargs)
            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f'invalid layer type {layer_type}')

            if layer_shift_tokens > 0:
                shift_range_upper = layer_shift_tokens + 1
                shift_range_lower = -layer_shift_tokens if not causal else 0
                layer = ShiftTokens(range(shift_range_lower, shift_range_upper), layer)

            residual_fn = GRUGating if gate_residual else Residual
            residual = residual_fn(dim, scale_residual = scale_residual, scale_residual_constant = scale_residual_constant)

            pre_branch_norm = norm_fn() if pre_norm else None
            post_branch_norm = norm_fn() if sandwich_norm else None
            post_main_norm = norm_fn() if not pre_norm else None

            norms = nn.ModuleList([
                pre_branch_norm,
                post_branch_norm,
                post_main_norm
            ])

            self.layers.append(nn.ModuleList([
                norms,
                layer,
                residual
            ]))

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        attn_mask = None,
        self_attn_kv_mask = None,
        mems = None,
        seq_start_pos: Optional[Tensor] = None,
        cache: Optional[LayerIntermediates] = None,
        cache_age = 1,
        return_hiddens = False
    ):
        assert not (self.cross_attend ^ exists(context)), 'context must be passed in if cross_attend is set to True'

        # initialize accums

        hiddens = []
        layer_hiddens = []
        intermediates = []

        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers

        # handle left padded sequences

        if exists(seq_start_pos):
            seq_arange = torch.arange(x.shape[-2], device = x.device, dtype = torch.long)
            left_pad_mask = seq_arange >= seq_start_pos[..., None]

            if exists(self_attn_kv_mask):
                self_attn_kv_mask = self_attn_kv_mask & left_pad_mask
            else:
                self_attn_kv_mask = left_pad_mask

        # rotary positions

        rotary_pos_emb = None

        if exists(self.rotary_pos_emb):
            max_rotary_emb_length = max(list(map(lambda m: (m.shape[1] if exists(m) else 0) + x.shape[1], mems)))
            rotary_pos_emb = self.rotary_pos_emb(max_rotary_emb_length)

        # assume cached key / values

        attn_cache = []

        if exists(cache):
            assert not self.training and self.causal and not any([*map(exists, (mask, attn_mask))])

            if cache_age > 0:
                x = x[:, -cache_age:] # for spec decoding, may be greater than 1

            attn_cache = cache.attn_intermediates

        iter_attn_cache = iter(attn_cache)

        # outer residual - for resiDual paper

        outer_residual = x * self.resi_dual_scale

        # get layers to be executed

        layer_variables = (
            self.layer_types,
            self.layers,
            self.layer_dropouts
        )

        layer_variables = tuple(tuple(layer_variable[i] for i in self.layers_execute_order) for layer_variable in layer_variables)

        # go through the attention and feedforward layers

        for ind, (layer_type, (norm, block, residual_fn), layer_dropout) in enumerate(zip(*layer_variables)):
            is_last = ind == (len(self.layers) - 1)

            if self.training and layer_dropout > 0. and random() < layer_dropout:
                continue

            if layer_type == 'a':
                if return_hiddens:
                    hiddens.append(x)
                layer_mem = mems.pop(0) if mems else None

            if layer_type == 'c':
                if self.training and self.cross_attn_tokens_dropout > 0.:
                    context, context_mask = dropout_seq(context, context_mask, self.cross_attn_tokens_dropout)

            inner_residual = x

            if return_hiddens:
                layer_hiddens.append(x)

            pre_norm, post_branch_norm, post_main_norm = norm

            if exists(pre_norm):
                x = pre_norm(x)

            if layer_type == 'a':
                out, inter = block(x, mask = mask, context_mask = self_attn_kv_mask, attn_mask = attn_mask, rel_pos = self.rel_pos, rotary_pos_emb = rotary_pos_emb, prev_attn = prev_attn, cache = next(iter_attn_cache, None), mem = layer_mem, return_intermediates = True)
            elif layer_type == 'c':
                out, inter = block(x, context = context, mask = mask, context_mask = context_mask, prev_attn = prev_cross_attn, cache = next(iter_attn_cache, None), return_intermediates = True)
            elif layer_type == 'f':
                out = block(x)

            if self.resi_dual:
                outer_residual = outer_residual + out * self.resi_dual_scale

            if exists(post_branch_norm):
                out = post_branch_norm(out)

            x = residual_fn(out, inner_residual)

            if layer_type in ('a', 'c') and return_hiddens:
                intermediates.append(inter)

            if layer_type == 'a' and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == 'c' and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if exists(post_main_norm):
                x = post_main_norm(x)

        if return_hiddens:
            layer_hiddens.append(x)

        if self.resi_dual:
            x = x + self.final_norm(outer_residual)
        else:
            x = self.final_norm(x)

        if not return_hiddens:
            return x

        intermediates = LayerIntermediates(
            hiddens = hiddens,
            attn_intermediates = intermediates,
            layer_hiddens = layer_hiddens
        )

        return x, intermediates

class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on encoder'
        super().__init__(causal = False, **kwargs)

class Decoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on decoder'
        super().__init__(causal = True, **kwargs)

class CrossAttender(AttentionLayers):
    def __init__(self, **kwargs):
        super().__init__(cross_attend = True, only_cross = True, **kwargs)

class XTransformerGenerator(AbstractGenerator):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        attn_layers,
        emb_dim = None,
        max_mem_len = 0,
        shift_mem_down = 0,
        emb_dropout = 0.,
        post_emb_norm = False,
        num_memory_tokens = None,
        tie_embedding = False,
        logits_dim = None,
        use_abs_pos_emb = True,
        scaled_sinu_pos_emb = False,
        l2norm_embed = False,
        emb_frac_gradient = 1., # GLM-130B and Cogview successfully used this, set at 0.1
        attn_z_loss_weight = 1e-4
    ):
        super().__init__()

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)
        self.emb_dim = emb_dim
        self.num_tokens = num_tokens
        self.emb_drop = emb_dropout
        
        max_seq_len = max_seq_len + 1
        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down

        self.l2norm_embed = l2norm_embed
        self.token_emb = TokenEmbedding(emb_dim, num_tokens, l2norm_embed = l2norm_embed)

        if not (use_abs_pos_emb and not attn_layers.has_pos_emb):
            self.pos_emb = always(0)
        elif scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(emb_dim)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len, l2norm_embed = l2norm_embed)

        self.emb_frac_gradient = emb_frac_gradient # fraction of the gradient that should go to the embedding, https://arxiv.org/abs/2105.13290

        self.post_emb_norm = nn.LayerNorm(emb_dim) if post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers

        self.init_()

        logits_dim = default(logits_dim, num_tokens)
        self.to_logits = nn.Linear(dim, logits_dim) if not tie_embedding else lambda t: t @ self.token_emb.emb.weight.t()

        # memory tokens (like [cls]) from Memory Transformers paper

        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        # whether can do cached kv decoding

        self.can_cache_kv = self.num_memory_tokens == 0
    
    def init_(self):
        if self.l2norm_embed:
            nn.init.normal_(self.token_emb.emb.weight, std = 1e-5)
            if not isinstance(self.pos_emb, always):
                nn.init.normal_(self.pos_emb.emb.weight, std = 1e-5)
            return

        nn.init.kaiming_normal_(self.token_emb.emb.weight)
    
    def forward(
        self,
        x,
        return_embeddings = False,
        return_logits_and_embeddings = False,
        return_intermediates = False,
        mask = None,
        return_mems = False,
        return_attn = False,
        mems = None,
        pos = None,
        prepend_embeds = None,
        sum_embeds = None,
        return_attn_z_loss = False,
        attn_z_loss_weight = 1e-4,
        seq_start_pos = None,
        cache: Optional[LayerIntermediates] = None,
        **kwargs
    ):
        b, n, device, num_mem, emb_frac_gradient = *x.shape, x.device, self.num_memory_tokens, self.emb_frac_gradient
        return_hiddens = return_mems | return_attn | return_intermediates | return_attn_z_loss

        # absolute positional embedding

        external_pos_emb = exists(pos) and pos.dtype != torch.long
        pos_emb = self.pos_emb(x, pos = pos, seq_start_pos = seq_start_pos) if not external_pos_emb else pos
        x = self.token_emb(x) + pos_emb

        # for summing embeddings passed externally - needs this for self-conditioning in non-autoregressive training

        if exists(sum_embeds):
            x = x + sum_embeds

        # post embedding norm, purportedly leads to greater stabilization

        x = self.post_emb_norm(x)

        # whether to append embeds, as in PaLI, for image embeddings

        if exists(prepend_embeds):
            prepend_seq, prepend_dim = prepend_embeds.shape[1:]
            assert prepend_dim == x.shape[-1], 'prepended embeddings need to have same dimensions as text model dimensions'

            x = torch.cat((prepend_embeds, x), dim = -2)

        # whether to reduce the gradient going to the embedding, from cogview paper, corroborated by GLM-130B model

        if emb_frac_gradient < 1:
            assert emb_frac_gradient > 0
            x = x * emb_frac_gradient + x.detach() * (1 - emb_frac_gradient)

        # embedding dropout

        x = self.emb_dropout(x)

        x = self.project_emb(x)

        if num_mem > 0:
            mem = repeat(self.memory_tokens, 'n d -> b n d', b = b)
            x = torch.cat((mem, x), dim = 1)

            # auto-handle masking after appending memory tokens
            if exists(mask):
                mask = pad_at_dim(mask, (num_mem, 0), dim = -1, value = True)

        if self.shift_mem_down and exists(mems):
            mems_l, mems_r = mems[:self.shift_mem_down], mems[self.shift_mem_down:]
            mems = [*mems_r, *mems_l]

        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, cache = cache, return_hiddens = True, seq_start_pos = seq_start_pos, **kwargs)

        mem, x = x[:, :num_mem], x[:, num_mem:]

        if return_logits_and_embeddings:
            out = (self.to_logits(x), x)
        elif return_embeddings:
            out = x
        else:
            out = self.to_logits(x)

        if return_attn_z_loss:
            pre_softmax_attns = list(map(lambda t: t.pre_softmax_attn, intermediates.attn_intermediates))
            intermediates.attn_z_loss = calc_z_loss(pre_softmax_attns, weight = attn_z_loss_weight)
            return_intermediates = True

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = list(map(lambda pair: torch.cat(pair, dim = -2), zip(mems, hiddens))) if exists(mems) else hiddens
            new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), new_mems))

            if not return_intermediates:
                return out, new_mems

            intermediates.mems = new_mems

        if return_intermediates:
            return out, intermediates

        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        return out

    def config(self) -> Dict:
        return dict(
            num_tokens=self.num_tokens,
            max_seq_len=self.max_seq_len-1,
            emb_dropout = self.emb_drop,
            atten_type=self.attn_layers.__class__.__name__,
            atten_dim=self.attn_layers.dim,
            atten_depth=self.attn_layers.depth,
            atten_heads=self.attn_layers.heads,
            layer_dropout=self.attn_layers.layer_drop,
            attn_dropout=self.attn_layers.attn_drop,
            ff_dropout=self.attn_layers.ff_drop,
        )

    @classmethod
    def load(cls, load_dir: str, best: bool = False) -> "AbstractGenerator":
        model_config_path = Path(load_dir) / "generator_config.json"
        if best:
            model_config_path = Path(load_dir) / "generator_config_best.json"
        
        with open(str(model_config_path), "r") as file:
            config = json.load(file)

        #model = cls(**config)  # type: ignore
        if config['atten_type'] == 'Decoder':
            
            model = cls(
                num_tokens = config['num_tokens'],
                max_seq_len = config['max_seq_len'],
                emb_dropout = config['emb_dropout'],
                num_memory_tokens = 20,
                l2norm_embed = True,
                post_emb_norm = True,
                attn_layers = Decoder(
                    dim = config['atten_dim'],
                    depth = config['atten_depth'],
                    heads = config['atten_heads'],
                    layer_dropout = config['layer_dropout'],
                    attn_dropout = config['attn_dropout'],    # dropout post-attention
                    ff_dropout = config['ff_dropout'],

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
            ))
            
        elif config['atten_type'] == 'Encoder':
            model = cls(
                num_tokens = config['num_tokens'],
                max_seq_len = config['max_seq_len'],
                emb_dropout = config['emb_dropout'],
                attn_layers = Encoder(
                    dim = config['atten_dim'],
                    depth = config['atten_depth'],
                    heads = config['atten_heads'],
                    layer_dropout = config['layer_dropout'],
                    attn_dropout = config['attn_dropout'],    # dropout post-attention
                    ff_dropout = config['ff_dropout'],
            ))
        
        model_weight_path = Path(load_dir) / "generator_weight.pt"
        if best:
            model_weight_path = Path(load_dir) / "generator_weight_best.pt"
        
        try:
            model_state_dict = torch.load(model_weight_path, map_location="cpu")

            # Change the keyword if external state-dicts are used because of naming missmatch
            new_model_state_dict = OrderedDict()
            for name in model_state_dict.keys():
                new_model_state_dict[name] = model_state_dict[name]

            model_state_dict = new_model_state_dict
            model.load_state_dict(model_state_dict)
        except:
            print("No pretrained weight for SmilesGenerator")

        return model