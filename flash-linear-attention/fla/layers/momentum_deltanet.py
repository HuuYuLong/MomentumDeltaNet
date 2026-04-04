# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Dict, Optional, Tuple, List
from copy import copy, deepcopy

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution, L2Norm 
from fla.ops.momentum_delta_rule import chunk_mode_rule, fused_recurrent_mode_rule

import csv
import os
from datetime import datetime


if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class MomentumDeltaNet(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2,
        head_dim: int = 256,
        num_heads: int = 6,
        num_v_heads: int = None,
        mode: str = 'chunk',
        use_gate: bool = True,
        use_lora_gate: bool = False,
        lora_gate_rank: int = None,
        use_p_times_alpha: bool = True, # defaultly as True, to make p = alpha * Norm(k)
        use_short_conv: bool = True,
        conv_size: int = 4,
        min_log_mu: float = -2.,
        tau_factor: int = 1,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        use_output_correction: bool= True,
        **kwargs
    ) -> MomentumDeltaNet:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.min_log_mu = min_log_mu
        self.expand_v = expand_v

        self.use_gate = use_gate
        self.use_p_times_alpha = use_p_times_alpha
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.use_output_correction = use_output_correction
        self.layer_idx = layer_idx
    
        # Consistency check: Ensure expand_v produces integer values
        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.num_v_heads * self.head_dim * expand_v}, which is invalid for nn.Linear."
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}."
            )

        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}. "
                f"Resulting head_v_dim would be {head_dim * expand_v}, which is invalid for FusedRMSNormGated."
            )
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # for module keep the same init
        self.a_min_init = 0
        self.a_max_init = 16
        self.m_min_init = 0
        self.m_max_init = 16 
        self.factor_scale = 4 
        self.tau = math.sqrt(self.hidden_size/tau_factor)

        self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(self.a_min_init, self.a_max_init)
        self.A_log = nn.Parameter(torch.log(A.clone()))
        # # hard coded for now
        self.dt_min = 0.001
        self.dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_v_heads) * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        #sig_inv_dt = torch.logit(dt)
        # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt.clone())
        self.mu_bias = nn.Parameter(inv_dt.clone())

        self.m_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.e_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        Mu = torch.empty(self.num_heads, dtype=torch.float32).uniform_(self.m_min_init, self.m_max_init)
        self.Mu_log = nn.Parameter(torch.log((Mu.clone())))

        self.log_factor = nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, self.factor_scale)))    
        self.log_factor._no_weight_decay = True

        self.A_log._no_weight_decay = True
        self.Mu_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True
        self.mu_bias._no_weight_decay = True

        if use_output_correction:   # credits to comba 
            self.D_log = nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 1)))
            self.D_log._no_weight_decay = True
 
        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu'
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu'
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu'
            )
        else:
            warnings.warn(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing."
            )
        if use_gate:
            if use_lora_gate:
                rank = self.head_v_dim if lora_gate_rank is None else lora_gate_rank
                self.g_proj = nn.Sequential(
                    nn.Linear(hidden_size, rank, bias=False),
                    nn.Linear(rank, self.value_dim, bias=True),
                )
            else:
                self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, activation='sigmoid', eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

 
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        # change to inference mode.
        mode = 'fused_recurrent' if q_len <= 64 else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        if self.num_v_heads > self.num_heads:
            q, k = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_v_heads // self.num_heads), (q, k))

        a = self.a_proj(hidden_states).float()
        b = self.b_proj(hidden_states).float() 
        m = self.m_proj(hidden_states).float()
        e = self.e_proj(hidden_states).float() / self.tau

        log_mu = - self.Mu_log.float().exp() * F.softplus(m + self.mu_bias)    # ~1  
        log_alpha = - self.A_log.float().exp() * F.softplus(a + self.dt_bias)  # ~1
        eta = F.tanh(e) + 1                      # range (0, 2)  
        beta =  F.sigmoid(b)

        #### below is constraint #### 
        theta = torch.arctan(eta * self.log_factor.float().exp())
        beta_upper  = torch.sin(theta) ** 2
        alpha_upper = torch.cos(theta) ** 2  
        beta = beta_upper * beta         
        log_alpha = alpha_upper.log() + log_alpha 

        if self.min_log_mu is not None:
            log_mu.clamp_min_(self.min_log_mu)  # -2 can achive better result but may occur upper overflow for larger model 

        if self.use_output_correction:   # credits to comba
            q = (q - self.D_log.float().exp()[None, None, :, None] * k).to(k.dtype) 

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None

        if mode == 'chunk': 
            o, recurrent_state = chunk_mode_rule(
                q=q,
                k=k,
                v=v, 
                log_alpha=log_alpha,
                beta=beta,
                log_mu=log_mu,
                eta=eta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
                use_p_times_alpha=self.use_p_times_alpha
            )
        elif mode == 'fused_recurrent': 
            o, recurrent_state = fused_recurrent_mode_rule(
                q=q,
                k=k,
                v=v, 
                log_alpha=log_alpha,
                beta=beta,
                log_mu=log_mu,
                eta=eta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
                use_p_times_alpha=self.use_p_times_alpha
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len
            )

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
 
