# -*- coding: utf-8 -*-
# v5

import os
from typing import List

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def recurrent_momentum_delta_rule_ref(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        log_alpha: torch.Tensor,
        log_mu: torch.Tensor,
        p: torch.Tensor, # defaultly make p = alpha * Norm(k) outside the kernel
        beta: torch.Tensor = None,
        eta: torch.Tensor = None,
        scale: float = None,
        initial_S: torch.Tensor = None,
        initial_M: torch.Tensor = None,
        output_final_state: bool = False,
):
    q, k, v, p, log_alpha, log_mu, beta, eta = map(lambda x: x.to(torch.float32),
                                                   [q, k, v, p, log_alpha, log_mu, beta, eta]
                                                   )
    B, T, H, DK, DV = *k.shape, v.shape[-1]

    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)

    q = q * scale
    S_prev = torch.zeros(B, H, DK, DV).to(v)  # [b*h, dk, dv]
    M_prev = torch.zeros(B, H, DK, DV).to(v)

    if initial_M is not None:
        M_prev = initial_M  # for save momentum state
    if initial_S is not None:
        S_prev = initial_S

    out = torch.zeros_like(v)
    for i in range(T):
        k_t = k[:, i]   # B, H, DK
        q_t = q[:, i]   # B, H, DK
        v_t = v[:, i]   # B, H, DV
        p_t = p[:, i]   # For GatedDeltaNet: p = alpha * Norm(k), mu = 0

        mu_i = log_mu[:, i].exp().view(B, H, 1, 1)        # B, H, 1, 1
        beta_i = beta[:, i].view(B, H, 1, 1)              # B, H, 1, 1
        alpha_i = log_alpha[:, i].exp().view(B, H, 1, 1)  # B, H, 1, 1

        eta_i = eta[:, i].unsqueeze(-1)  # B, H, 1

        # delta grad
        w_t = -(v_t.unsqueeze(-2) - p_t.unsqueeze(-2) @ S_prev)     # B, H, 1, V
        # momentum update
        Mt = mu_i * M_prev + (eta_i * k_t).unsqueeze(-1) @ w_t   # (B, H, k, 1)  @  (B, H, 1, V) = (B, H, K, V)
        St = alpha_i * S_prev - beta_i * Mt

        out[:, i] = (q_t.unsqueeze(-1) * St).sum(-2)  # (B, H, 1, k)  @  (B, H, k, V) = ((B, H, k, 1) *  (B, H, k, V)).sum(-2)

        M_prev = Mt
        S_prev = St

    o = out
    if output_final_state:
        final_state = torch.stack([S_prev, M_prev], dim=0)
    else:
        final_state = None

    return o, final_state



def chunk_momentum_delta_rule_ref(    
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,   # defaultly make p = alpha * Norm(k) outside the kernel
        log_alpha: torch.Tensor,
        log_mu: torch.Tensor,
        beta: torch.Tensor,
        eta: torch.Tensor,
        scale: float = None,
        initial_S: torch.Tensor = None,
        initial_M: torch.Tensor = None,
        output_final_state: bool = False,
        chunk_size: int = 64,
):
    # assert not torch.any(torch.eq(beta, 0))
    BT = chunk_size
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    # Calculate padding needed to make T a multiple of BT
    q, k, v, p, log_alpha, log_mu, beta, eta = map(lambda x: x.to(torch.float32),
                                                   [q, k, v, p, log_alpha, log_mu, beta, eta]
                                                   )
    T = q.shape[1]
    pad_len = (BT - (T % BT)) % BT

    if pad_len > 0:
        # Pad all tensors
        q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
        p = F.pad(p, (0, 0, 0, 0, 0, pad_len))
        log_alpha = F.pad(log_alpha, (0, 0, 0, pad_len))
        log_mu = F.pad(log_mu, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, 0, 0, pad_len))
        eta = F.pad(eta, (0, 0, 0, pad_len))

    B, l, H, DK = q.shape  # l is the sequence lenght after padding
    DV = v.shape[-1]
    q = q * scale
    assert l % chunk_size == 0
    assert q.shape == (B, pad_len+T, H, DK)
    assert log_alpha.shape == (B, pad_len+T, H)

    k_eta = eta[..., None] * k
    q, k, v, p, log_alpha, log_mu, beta = map(
        lambda x: rearrange(x, 'b (n c) h d -> b h n c d', c=chunk_size),
        [q, k_eta, v, p, log_alpha.unsqueeze(-1), log_mu.unsqueeze(-1), beta.unsqueeze(-1)]
    )

    log_a_cum = log_alpha.squeeze(-1).cumsum(-1)
    log_m_cum = log_mu.squeeze(-1).cumsum(-1)
    log_beta  = (beta + 1e-6).squeeze(-1).log()

    # Computing w parallel part
    log_c_before = log_beta + log_m_cum - log_a_cum
    log_ct       = torch.logcumsumexp(log_c_before, dim=-1)                   # \sum _{j=1}^{t} c_j
    log_ct_tm1   = torch.cat([torch.full_like(log_ct[:, :, :, :1], float('-inf')), log_ct[:, :, :, :-1]], dim=-1)

    a = log_ct.unsqueeze(-1)
    b = log_ct_tm1.unsqueeze(-2)
    x = (b - a).tril()                           # x <= 0
    temp = 1 - torch.exp(x)

    log_bar_a_tm1 = torch.cat([torch.zeros_like(log_a_cum[:, :, :, :1]), log_a_cum[:, :, :, :-1]], dim=3)
    bar_a_tm1 = log_bar_a_tm1.exp()                                                    # \bar{a}_{t-1}

    b_t   = (log_a_cum + log_ct).exp()   # b_t
    b_tm1 = torch.cat([torch.zeros_like(b_t[:, :, :, :1]), b_t[:, :, :, :-1]], dim=3)  # b_{t-1}

    gamma_mask_q = (log_a_cum.unsqueeze(-1) - log_m_cum.unsqueeze(-2) + a
                    ).exp().float().tril() * temp
    gamma_mask = torch.cat([torch.zeros_like(gamma_mask_q[:, :, :, :1]), gamma_mask_q[:, :, :, :-1]], dim=3)

    attn = (p @ k.transpose(-1, -2)) * gamma_mask

    attn_inv = -attn
    for i in range(1, chunk_size):
        attn_inv[..., i, :i] += (attn_inv[..., i, :, None].clone() * attn_inv[..., :, :i].clone()).sum(-2)
    attn_inv = attn_inv + torch.eye(chunk_size, dtype=attn_inv.dtype, device=q.device)  # diagonal with I

    alpha_tm1_p = bar_a_tm1[..., None] * p
    b_tm1_p     = b_tm1[..., None]     * p

    u_c = attn_inv @ v
    y_c = attn_inv @ alpha_tm1_p
    z_c = attn_inv @ b_tm1_p

    S_pre = k.new_zeros(B, H, DK, DV)
    M_pre = k.new_zeros(B, H, DK, DV)

    if initial_S is not None:
        S_pre = initial_S

    if initial_M is not None:
        M_pre = initial_M

    log_m_cum_test = log_m_cum + torch.zeros_like(log_m_cum)
    # log_m_cum_test.register_hook(lambda grad: print("grad for log_m_cum_test", grad))

    o = torch.zeros_like(v)
    num_chunks = q.shape[2]
    for i in range(num_chunks):
        q_i, k_i, = q[:, :, i], k[:, :, i]                                # the difference: v_i is from w_i
        v_i = u_c[:, :, i] - y_c[:, :, i] @ S_pre + z_c[:, :, i] @ M_pre  # pseudo Value v, objective loss ||Sk - V||^2

        # qS read out
        attn_inner = (q_i @ k_i.transpose(-1, -2)) * gamma_mask_q[:, :, i]
        bar_alpha_t_q = q_i * log_a_cum[:, :, i, :].exp().unsqueeze(-1)
        b_t_q = q_i * b_t[:, :, i, :].unsqueeze(-1)
        qS_inter = bar_alpha_t_q @ S_pre - b_t_q @ M_pre
        o[:, :, i] = qS_inter + attn_inner @ v_i

        # update S, M
        decay_s = gamma_mask_q[:, :, i, -1].unsqueeze(-1)
        S = log_a_cum[:, :, i, -1, None, None].exp() * S_pre \
            - b_t[:, :, i, -1, None, None] * M_pre \
            + (k_i * decay_s).transpose(-1, -2) @ v_i

        decay_m = (log_m_cum_test[:, :, i, -1, None] - log_m_cum_test[:, :, i]).exp()[..., None]
        M = log_m_cum_test[:, :, i, -1, None, None].exp() * M_pre \
            - (k_i * decay_m).transpose(-1, -2) @ v_i

        S_pre, M_pre = S, M

    if output_final_state:
        final_state = torch.stack([S_pre, M_pre], dim=0)
    else:
        final_state = None

    # unpad
    o = rearrange(o, 'b h n c d -> b (n c) h d')
    o = o[:, :T]
    return o, final_state

