# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, 


from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp
from fla.utils import autotune_cache_kwargs, check_shared_mem, is_tf32_supported

NUM_WARPS = [2, 4, 8] 
NUM_WARPS = [2, 4]

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None, 
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [16, 32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'BT', 'IS_VARLEN' ],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_mode_rule_pkt_fwd_kernel(
    k,
    p, 
    log_a_cum,
    log_mu_cum,
    log_ct,
    A,
    gamma_mask_q,
    bt,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # notation:
    # b_ means data
    # p_ means pointer
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # prepare mask
    o_t = i_t * BT + tl.arange(0, BT)   # The index of inside the chunk BT
    m_t = o_t < T   # mask
    # mask_A    = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)   # tril mask
    mask_tril = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)   # tril mask
 
 
    i, j = tl.arange(0, BT)[:, None], tl.arange(0, BT)[None, :]
    S_m = tl.where(i == (j + 1), 1.0, 0.0)                   # [BT, BT]

    log_a_cum  += bos*H + i_h
    log_mu_cum += bos*H + i_h
    log_ct     += bos*H + i_h
    
    # loading data
    p_log_acum = tl.make_block_ptr(log_a_cum , (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_log_mcum = tl.make_block_ptr(log_mu_cum  , (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_log_ct   = tl.make_block_ptr(log_ct  , (T,), (H,), (i_t * BT,), (BT,), (0,))

    b_log_a_cum   = tl.load(p_log_acum, boundary_check=(0,))                  # [BT]
    b_log_m_cum   = tl.load(p_log_mcum, boundary_check=(0,))                   # [BT]
    b_log_ct = tl.load(p_log_ct, boundary_check=(0,))                 # [BT]

    # c_{t}
    # b_ct = tl.exp(b_log_ct)
    r = tl.arange(0, BT)
    # \bar{a}_{t-1}

    # c_{t-1}
    neg_inf = tl.zeros([1], dtype=tl.float32) - float("inf")
    b_log_c_tm1 = tl.where(r==0, neg_inf, tl.sum(S_m * b_log_ct[None, :], axis=1)) 
    b_bt = tl.exp(b_log_a_cum + b_log_ct)    # b_t
    
    a = b_log_ct[:, None]
    b = b_log_c_tm1[None, :]
    x = b - a                    # <=0 du
    x = 1 - tl.exp(x)
    
    b_log_gamma = tl.where(mask_tril, (b_log_a_cum + b_log_ct)[:, None] - b_log_m_cum[None, :], 0)  
    b_gamma_mask_q = tl.where(mask_tril, tl.exp(b_log_gamma) * x, 0)

    b_gamma_mask = tl.dot(S_m, b_gamma_mask_q)               # first row is zero
    b_gamma_mask = tl.where(mask_tril, b_gamma_mask, 0.0)    # strict tril

    # p * k
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1)) # trans access
        p_p = tl.make_block_ptr(p + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

        b_k = tl.load(p_k, boundary_check=(0, 1)) 
        b_p = tl.load(p_p, boundary_check=(0, 1)) 
        b_A += tl.dot(b_p, b_k)

    b_A = b_A * b_gamma_mask

    # store results
    p_A       = tl.make_block_ptr(A            + (bos*H + i_h) * BT, (T, BT), (BT*H, 1), (i_t * BT, 0), (BT, BT), (1, 0)) # pointer
    p_gamma_q = tl.make_block_ptr(gamma_mask_q + (bos*H + i_h) * BT, (T, BT), (BT*H, 1), (i_t * BT, 0), (BT, BT), (1, 0)) # pointer
    p_bt        = tl.make_block_ptr(bt        + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))  # pointer
    
    tl.store(p_A,         b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_gamma_q,   b_gamma_mask_q.to(p_gamma_q.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_bt,        b_bt.to(p_bt.dtype.element_ty),               boundary_check=(0,))
    

def chunk_scaled_dot_mode_rule_pkt_fwd(
    k: torch.Tensor,
    p: torch.Tensor,
    log_a_cum: Optional[torch.Tensor] = None,
    log_mu_cum: Optional[torch.Tensor] = None,
    log_ct: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Compute beta \mathcal{A}(i-1/j) * P * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        p (torch.Tensor):
            The auxiliary key tensor of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        g0 (torch.Tensor):
            The cumulative sum minus the original one of the gate tensor of shape `[B, T, H]`.
            Default: None
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`.
            Default: None
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """
    B, T, H, K = k.shape
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # need solve reverse
    A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
    # for computing readout
    gamma_mask_q = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
 
    bt        = torch.empty(B, T, H, device=k.device, dtype=output_dtype)
    
    chunk_scaled_dot_mode_rule_pkt_fwd_kernel[(NT, B * H)](
        k=k,
        p=p,
        log_a_cum=log_a_cum,
        log_mu_cum=log_mu_cum,
        log_ct=log_ct,
        A=A,
        gamma_mask_q=gamma_mask_q,
        bt=bt,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
    )
    return A,  bt,  gamma_mask_q


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        # for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4, 5]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_uyz_repr_bwd_kernel(
    q,
    k,
    v,
    p,
    beta,
    log_a_cum,
    log_m_cum,
    gamma_mask_q,
    d_Attn_do_v,
    d_decay_s,
    bt,
    log_ct,
    A,
    du,
    dy,
    dz,
    dk,
    dv,
    dp,
    dbt,
    dlog_a,
    dlog_mu,
    d_log_mu_cum,
    d_log_a_cum,
    dbeta,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q  += (bos * H + i_h) * K
    k  += (bos * H + i_h) * K
    p  += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dp += (bos * H + i_h) * K
    dy += (bos * H + i_h) * K
    dz += (bos * H + i_h) * K

    v  += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V
    du += (bos * H + i_h) * V

    bt += bos * H + i_h
    log_a_cum += bos * H + i_h
    log_m_cum += bos * H + i_h
    beta      += bos * H + i_h
    log_ct       += bos * H + i_h
    dlog_a  += bos * H + i_h
    dlog_mu += bos * H + i_h
    dbeta   += bos * H + i_h
    dbt     += bos * H + i_h
    d_log_mu_cum += bos * H + i_h
    d_log_a_cum  += bos * H + i_h

    d_decay_s += bos * H + i_h

    A              += (bos*H + i_h) * BT
    gamma_mask_q   += (bos*H + i_h) * BT
    d_Attn_do_v += (bos*H + i_h) * BT
    
    p_bt         = tl.make_block_ptr(bt  ,     (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A             = tl.make_block_ptr(A  ,   (BT, T), (1, H*BT), (0, i_t*BT), (BT, BT), (0, 1)) # trans
    p_gamma_q       = tl.make_block_ptr(gamma_mask_q  , (T, BT), (H*BT, 1), (i_t*BT, 0), (BT, BT), (1, 0))
    p_log_acum      = tl.make_block_ptr(log_a_cum   ,  (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_log_mcum      = tl.make_block_ptr(log_m_cum   ,  (T,), (H,), (i_t * BT,), (BT,), (0,))

    b_log_a_cum     = tl.load(p_log_acum,     boundary_check=(0,))                        # [BT]
    b_log_m_cum     = tl.load(p_log_mcum,     boundary_check=(0,))                        # [BT]
    b_bt            = tl.load(p_bt, boundary_check=(0,))                                  # [BT]
    b_A             = tl.load(p_A, boundary_check=(0, 1))                               # [BT, BT]
    b_gamma_mask_q  = tl.load(p_gamma_q, boundary_check=(0, 1))
    
    o_t = i_t*BT + tl.arange(0, BT)   # The index of inside the chunk BT
    m_t = o_t < T   # mask
    mask_A    = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)   # tril mask
    mask_tril = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)   # tril mask

    i, j = tl.arange(0, BT)[:, None], tl.arange(0, BT)[None, :]
    S_m1 = tl.where(i == (j + 1), 1.0, 0.0)                                                        # [BT, BT]
    S_p1 = tl.where((i + 1) == j, 1.0, 0.0)                                                        # [BT, BT]
    b_gamma_mask = tl.dot(S_m1, b_gamma_mask_q)
    b_gamma_mask = tl.where(mask_A, b_gamma_mask, 0.0)                          # strict tril
 
    b_bar_a_tm1 = tl.exp(tl.sum(S_m1 * b_log_a_cum[None, :], axis=1))        # [BT]
    b_b_tm1 = tl.sum(S_m1 * b_bt[None, :], axis=1)                           # [BT]

    b_d_bar_a_tm1 = tl.zeros([BT], dtype=tl.float32)
    b_d_btm1      = tl.zeros([BT], dtype=tl.float32)
    b_d_log_ct    = tl.zeros([BT], dtype=tl.float32)
    b_dA          = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_p  = tl.make_block_ptr(p,   (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dp = tl.make_block_ptr(dp , (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dy = tl.make_block_ptr(dy , (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dz = tl.make_block_ptr(dz , (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        """
        # The 'd' means partial when 'd' in dL/dx 
        # for BK part:
        d_bar_a_tm1 = dL/dy * dy/d_bar_a_tm1_p * d_bar_a_tm1_p/d_bar_a_tm1
                       = dy * A * k                                                          [BT, BK] -> [BT, 1]
        d_b_tm1 = dL/dz * dz/d_b_tm1_p * d_b_tm1_p/d_b_tm1
                       = dz * A * k                                                          [BT, BK] -> [BT, 1]
        dp = dL/dy * dy/d_bar_a_tm1_p * d_bar_a_tm1_p/d_p + dL/dz * dz/d_b_tm1_p * d_b_tm1_p/d_p
                       = d_bar_a_tm1 * bar_a_tm1  + d_b_tm1 * b_tm1                                     [BT, BK]
        dL/dAttn_inv = dL/du * du/dAttn_inv + dL/dy * dy/dAttn_inv + dL/dz * dz/dAttn_inv
                   = du * v + (dy * alpha_tm1_p + dz * d_b_tm1_p)                                       [BT, BT]
        """
        b_p  = tl.load(p_p, boundary_check=(0, 1))
        b_dy = tl.load(p_dy, boundary_check=(0, 1))
        b_dz = tl.load(p_dz, boundary_check=(0, 1))
        b_dalpha_tm1_p = tl.dot(b_A.to(b_dy.dtype), b_dy)
        b_db_tm1_p     = tl.dot(b_A.to(b_dz.dtype), b_dz)                                                                # [BT]
        b_dp           = b_dalpha_tm1_p * b_bar_a_tm1[:, None] + b_db_tm1_p * b_b_tm1[:, None]

        b_p_bar_a_tm1 = b_p * b_bar_a_tm1[:, None]
        b_p_b_tm1     = b_p * b_b_tm1[:, None]
        b_dA += tl.dot(b_dy, tl.trans(b_p_bar_a_tm1).to(b_dy.dtype)) + tl.dot(b_dz, tl.trans(b_p_b_tm1).to(b_dz.dtype))
        b_d_bar_a_tm1 += tl.sum(b_dalpha_tm1_p * b_p, axis=1)
        b_d_btm1      += tl.sum(b_db_tm1_p * b_p, axis=1)

        tl.store(p_dp, b_dp.to(p_dp.dtype.element_ty), boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v  = tl.make_block_ptr(v,  (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_du = tl.make_block_ptr(du, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        """
        # for BV part:
        dv = dL/du * du/dv = du * A                                                                        [BT, BV]
        dL/dAttn_inv = dL/du * du/dAttn_inv + dL/dy * dy/dAttn_inv + dL/dz * dz/dAttn_inv
                       = (du * v) + dy * alpha_tm1_p + dz * d_b_tm1_p                                      [BT, BT]
        """
        b_v  = tl.load(p_v, boundary_check=(0, 1))
        b_du = tl.load(p_du, boundary_check=(0, 1))
        b_dv = tl.dot(b_A.to(b_du.dtype), b_du)
        b_dA += tl.dot(b_du, tl.trans(b_v))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    """
    from  Attn_inv * A = I                                                                                     
    get   dA = - A * d(Attn_inv) * A                                                                     [BT, BT]
    """
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))      # A * dA_inv
    b_dA = -tl.dot(b_dA.to(b_A.dtype), b_A)      # A * dA_inv * A
    
    b_dG = b_dA * b_gamma_mask  # b_gamma_mask already tril
    b_G = tl.zeros([BT, BT], dtype=tl.float32)
    b_Gqk = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q  = tl.make_block_ptr(q , (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

        p_k  = tl.make_block_ptr(k , (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_p  = tl.make_block_ptr(p , (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dp = tl.make_block_ptr(dp, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        """
        # for next BK part
        pkt = (p @ k.transpose(-1, -2))                                                                 [BT, BT]
        d_gamma_mask = dL/dA * dA/dgamma_mask
                       = dA * kkt                                                                       [BT, BT]
        d_pkt = dA * gamma_mask                                                                         [BT, BT]
        d_p = dL/d_kkt * d_kkt/dp = d_kkt^T * k                                                         [BT, BK]
        d_k = dL/d_kkt * d_kkt/dk = d_kkt^T * p                                                         [BT, BK]
        """
        b_q  = tl.load(p_q,  boundary_check=(0, 1))
        b_k  = tl.load(p_k,  boundary_check=(0, 1))
        b_p  = tl.load(p_p,  boundary_check=(0, 1))
        b_dp = tl.load(p_dp, boundary_check=(0, 1))

        b_G   += tl.dot(b_p, tl.trans(b_k))
        b_Gqk += tl.dot(b_q, tl.trans(b_k))
        b_dp  += tl.dot(b_dG.to(b_k.dtype), b_k)
        b_dk  =  tl.dot(tl.trans(b_dG).to(b_k.dtype), b_p)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dp, b_dp.to(p_dp.dtype.element_ty), boundary_check=(0, 1))
 
    p_d_bt         = tl.make_block_ptr(dbt,           (T,), (H,), (i_t * BT,), (BT,), (0,))             # [BT]
    p_log_ct       = tl.make_block_ptr(log_ct ,       (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_d_log_mu_cum = tl.make_block_ptr(d_log_mu_cum,  (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_d_log_a_cum  = tl.make_block_ptr(d_log_a_cum,   (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_beta         = tl.make_block_ptr(beta,          (T,), (H,), (i_t * BT,), (BT,), (0,))

    b_d_bt         = tl.load(p_d_bt,         boundary_check=(0,))                         # [BT]
    b_log_ct       = tl.load(p_log_ct,       boundary_check=(0,))                         # [BT]
    b_beta         = tl.load(p_beta,         boundary_check=(0,))                         # [BT]
    b_d_log_mu_cum = tl.load(p_d_log_mu_cum, boundary_check=(0,))                         # [BT]
    b_d_log_a_cum  = tl.load(p_d_log_a_cum,  boundary_check=(0,))                         # [BT]

    """
    # dL/dgamma_mask_q = dL/dAttn * dAttn/dgamma_mask_q + dL/d_decay_s
    #                         = dL/dAttn * q * k                                            [BT, BT]
    """
    p_dAttn_do_v = tl.make_block_ptr(d_Attn_do_v, (T, BT), (H*BT, 1), (i_t*BT, 0), (BT, BT), (1, 0))
    p_d_decay_s   = tl.make_block_ptr(d_decay_s , (T,), (H,), (i_t * BT,), (BT,), (0,))

    b_dAttn_do_v  = tl.load(p_dAttn_do_v, boundary_check=(0, 1)) 
    b_d_decay_s  = tl.load(p_d_decay_s, boundary_check=(0,)) 
    
    b_dgamma_mask_q = b_dAttn_do_v * b_Gqk * scale                       # [BT, BT]

    rel_last = (min((i_t + 1) * BT, T) - 1) - i_t * BT  # scalar in [0, BT-1]
    rows = tl.arange(0, BT)[:, None]  # [BT, 1]
    # last_mask = rows == rel_last
    b_last_row = tl.where(rows < rel_last, 0.0, b_d_decay_s[None, :])  # [BT, BT]
    b_dgamma_mask_q += b_last_row   # b_d_gamma_mask_q_last = b_ddecay_s
    
    b_dgamma_mask = b_dA * b_G                          # dL/dgamma = dA * G
    b_dgamma_mask_q += tl.dot(S_p1, b_dgamma_mask)                                                   # [BT, BT]

    NEG_INF = tl.zeros([1], dtype=tl.float32) - float("inf")
    b_log_c_tm1 = tl.sum(S_m1 * b_log_ct[None, :], axis=1)
    r = tl.arange(0, BT)
    b_log_c_tm1 = tl.where(r == 0, NEG_INF, b_log_c_tm1)
    b_bt = tl.exp(b_log_a_cum + b_log_ct)    # b_t

    # according to the:
    #     gamma_mask_q = (log_a_cum.unsqueeze(-1) - log_m_cum.unsqueeze(-2) + log_c_jt).exp().float().tril()
    #     gamma_mask = torch.cat([torch.zeros_like(gamma_mask_q[:, :, :, :1]), gamma_mask_q[:, :, :, :-1]], dim=3)
    b_d_log_gamma = tl.where(mask_tril, b_dgamma_mask_q * b_gamma_mask_q, 0.0)                       # [BT, BT]

    b_d_log_a_cum += tl.sum(b_d_log_gamma,  axis=1)      # row sum                                     # [BT]
    b_d_log_mu_cum -= tl.sum(b_d_log_gamma, axis=0)      # col sum                                      # [BT]
    b_d_log_c_jt = b_d_log_gamma

    # according to the:
    #     b_t   = (log_a_cum + log_ct).exp()   # b_t
    #     b_tm1 = torch.cat([torch.zeros_like(b_t[:, :, :, :1]), b_t[:, :, :, :-1]], dim=3)  # b_{t-1}
    b_d_bt += tl.sum(S_p1 * b_d_btm1[None, :], axis=1)

    temp = b_d_bt * b_bt
    b_d_log_a_cum += temp
    b_d_log_ct += temp

    # according to:
    #     log_bar_a_tm1 = torch.cat([torch.zeros_like(log_a_cum[:, :, :, :1]), log_a_cum[:, :, :, :-1]], dim=3)
    #     bar_a_tm1 = log_bar_a_tm1.exp()                                                    # \bar{a}_{t-1}
    b_dlog_bar_a_tm1 = b_d_bar_a_tm1 * b_bar_a_tm1
    b_d_log_a_cum += tl.sum(S_p1 * b_dlog_bar_a_tm1[None, :], axis=1)

    # according to:
    #     log_ct_tm1 = torch.cat([torch.full_like(log_ct[:, :, :, :1], float('-inf')), log_ct[:, :, :, :-1]], dim=-1)
    #     a = log_ct.unsqueeze(-1)
    #     b = log_ct_tm1.unsqueeze(-2)
    #     x = (b - a).tril()                          # x <= 0
    #     log_c_jt = a + torch.log(1 - torch.exp(x))   # a + log(1 - exp(x))
    # b_x = tl.where(mask_tril, b_log_c_tm1[None, :] - b_log_ct[:, None], 0.0)
    eps = tl.zeros([1], dtype=tl.float32) + 1e-6

    b_x = b_log_c_tm1[None, :] - b_log_ct[:, None]  #<= 0
    b_x = 1 - tl.exp(b_x)
    b_d_x = tl.where(mask_tril, - b_d_log_c_jt * (1 - b_x) / (b_x + eps), 0.0)

    b_d_log_ct += tl.sum(b_d_log_c_jt - b_d_x, axis=1)
    b_d_b = tl.sum(b_d_x, axis=0)
    b_d_log_ct += tl.sum(S_p1 * b_d_b[None, :], axis=1)

    # according to:
    #     log_c_before = log_beta + log_m_cum - log_a_cum
    #     log_ct = torch.logcumsumexp(log_c_before, dim=-1)                   # \sum _{j=1}^{t} c_j
    # b_d_ct = b_d_log_ct * tl.exp(-b_log_ct)
    b_log_c_before = tl.log(b_beta + eps) + b_log_m_cum - b_log_a_cum  # todo: bug here

    # # 2) w[t, j] = exp(log_c_j - log_ct_t)
    # #    = exp_shift[t, j] / ct[t]
    # weight = b_exp_shift / b_ct[:, None]                                      # [BT, BT]

    # # 3)d_log_c[j] = sum_t g[t] * w[t, j]
    i = tl.arange(0, BT)[:, None]
    j = tl.arange(0, BT)[None, :]
    mask = i >= j
    
    log_expo = tl.where(mask, b_log_c_before[None, :] - b_log_ct[:, None] , NEG_INF) # [BT, BT]
    col_max = tl.maximum(tl.max(log_expo, axis=1), 0.)
    b_d_log_c_before = tl.sum(tl.exp(log_expo - col_max[None, :]) * b_d_log_ct[:, None], axis=0) * tl.exp(col_max)
    """
    # b_d_log_c_before = b_d_cumsum * tl.exp(b_log_c_before)
    # b_d_log_mu_cum += b_d_log_c_before  
    # b_d_log_a_cum  += -b_d_log_c_before
    # b_dlog_alpha = tl.cumsum(b_d_log_a_cum,  axis=0, reverse=True)
    # b_dlog_mu    = tl.cumsum(b_d_log_mu_cum, axis=0, reverse=True)
    # b_d_beta = tl.where(m_t, b_d_log_c_before / b_beta, 0.0)
    """
    b_d_beta = b_d_log_c_before / (b_beta + eps) 
    b_d_log_mu_cum += b_d_log_c_before  
    b_d_log_a_cum  += -b_d_log_c_before
    b_dlog_alpha = tl.cumsum(b_d_log_a_cum,  axis=0, reverse=True)
    b_dlog_mu    = tl.cumsum(b_d_log_mu_cum, axis=0, reverse=True)
    
    p_dlog_alpha = tl.make_block_ptr(dlog_a , (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dlog_mu    = tl.make_block_ptr(dlog_mu, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dbeta      = tl.make_block_ptr(dbeta,   (T,), (H,), (i_t * BT,), (BT,), (0,))

    tl.store(p_dbeta,      b_d_beta.to(p_dbeta.dtype.element_ty),          boundary_check=(0,))
    tl.store(p_dlog_alpha, b_dlog_alpha.to(p_dlog_alpha.dtype.element_ty), boundary_check=(0,))
    tl.store(p_dlog_mu,    b_dlog_mu.to(p_dlog_mu.dtype.element_ty),       boundary_check=(0,))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        # for num_warps in [2, 4, 8]
        for num_warps in NUM_WARPS
        for num_stages in [ 2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def recompute_u_y_z_fwd_kernel(
    p,
    v,
    A,
    log_a_cum,
    bt,
    u,
    y,
    z,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_log_a_cum = tl.make_block_ptr(log_a_cum + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_bt     = tl.make_block_ptr(bt + (bos*H + i_h),    (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A      = tl.make_block_ptr(A + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))

    b_log_a_cum = tl.load(p_log_a_cum, boundary_check=(0,))  # [BT]
    b_bt    = tl.load(p_bt, boundary_check=(0,))             # [BT]
    b_A     = tl.load(p_A, boundary_check=(0, 1))        

    i, j = tl.arange(0, BT)[:, None], tl.arange(0, BT)[None, :]
    S_m1 = tl.where(i == (j + 1), 1.0, 0.0)        
    
    
    """
    # b_log_bar_a_tm1 = tl.sum(S_m * b_log_a_cum[None, :], axis=1)
    # is_first_chunk = (i_t == 0)
    # b_bar_a_tm1 = tl.exp(tl.where(r == 0, 0.0, b_log_bar_a_tm1))


    # NEG_INF = tl.zeros([1], dtype=tl.float32) - float("inf")
    # b_log_c_tm1 = tl.sum(S_m1 * b_log_ct[None, :], axis=1)
    # r = tl.arange(0, BT)
    # b_log_c_tm1 = tl.where(r == 0, NEG_INF, b_log_c_tm1)

    # c_{t-1}
    neg_inf = tl.zeros([1], dtype=tl.float32) - float("inf")
    b_log_c_tm1 = tl.where(r==0, neg_inf, tl.sum(S_m * b_log_ct[None, :], axis=1)) 
    # b_log_c_tm1 = tl.where(r==0, neg_inf, tl.load(p_log_c_tm1, boundary_check=(0,)).to(tl.float32))
    # b_ctm1 = tl.exp(b_log_c_tm1)
    """
    b_bar_a_tm1 = tl.exp(tl.sum(S_m1 * b_log_a_cum[None, :], axis=1))        # [BT]
    b_b_tm1 = tl.sum(S_m1 * b_bt[None, :], axis=1)                           # [BT]

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        b_v = tl.load(p_v, boundary_check=(0, 1))     # [BT, BV]
        b_u = tl.dot(b_A, b_v)                # [BT, BT] @ [BT, BV] -> [BT, BV]
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(p + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_y = tl.make_block_ptr(y + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_z = tl.make_block_ptr(z + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

        b_k     = tl.load(p_k, boundary_check=(0, 1))
        b_kbara = b_bar_a_tm1[:, None] * b_k
        b_kbtm1 = b_b_tm1[:, None] * b_k

        b_y = tl.dot(b_A, b_kbara.to(b_A.dtype))
        b_z = tl.dot(b_A, b_kbtm1.to(b_A.dtype))

        tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_z, b_z.to(p_z.dtype.element_ty), boundary_check=(0, 1))


def recompute_u_y_z_fwd(
    p: torch.Tensor,
    v: torch.Tensor,
    A: torch.Tensor,
    log_a_cum: torch.Tensor,
    bt: torch.Tensor, 
    cu_seqlens: Optional[torch.LongTensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *p.shape, v.shape[-1]
    BT = A.shape[-1]

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    CONST_TILING = 64
        
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    u = torch.empty_like(v)
    y = torch.empty_like(p)  # pseudo k
    z = torch.empty_like(p)  # pseudo k
    recompute_u_y_z_fwd_kernel[(NT, B*H)](
        p=p,
        v=v,
        A=A,
        log_a_cum=log_a_cum,
        bt=bt,
        u=u,
        y=y,
        z=z,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return u, y, z


def prepare_uyz_repr_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    beta: torch.Tensor,
    log_a_cum: torch.Tensor,
    log_mu_cum: torch.Tensor,
    log_ct: torch.Tensor,
    gamma_mask_q: torch.Tensor,
    d_Attn_do_v: torch.Tensor,
    d_decay_s: torch.Tensor,
    A: torch.Tensor,
    bt: torch.Tensor,
    dbt: torch.Tensor,
    d_log_mu_cum: torch.Tensor,
    d_log_a_cum: torch.Tensor,
    du: torch.Tensor,
    dy: torch.Tensor,
    dz: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor ]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = gamma_mask_q.shape[-1]
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # H100 can have larger block size
    # if check_shared_mem('hopper', k.device.index):
    #     CONST_TILING = 128
    if check_shared_mem:
        CONST_TILING = 64
    else:
        CONST_TILING = 32
    # CONST_TILING = 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    dv = torch.empty_like(v)

    dk = torch.empty_like(k, dtype=torch.float)
    dp = torch.empty_like(p, dtype=torch.float)

    dlog_a = torch.empty_like(log_a_cum, dtype=torch.float)
    dlog_mu = torch.empty_like(log_mu_cum, dtype=torch.float)
    dbeta = torch.empty_like(log_mu_cum, dtype=torch.float)

    prepare_uyz_repr_bwd_kernel[(NT, B * H)](
        q=q,
        k=k,
        v=v,
        p=p,
        beta=beta,
        log_a_cum=log_a_cum,
        log_m_cum=log_mu_cum,
        gamma_mask_q=gamma_mask_q,
        d_Attn_do_v=d_Attn_do_v,
        d_decay_s=d_decay_s,
        bt=bt,
        log_ct=log_ct,
        A=A,
        du=du,
        dy=dy,
        dz=dz,
        dk=dk,
        dv=dv,
        dp=dp,
        dlog_a=dlog_a,
        dlog_mu=dlog_mu,
        dbeta=dbeta,
        dbt=dbt,
        d_log_mu_cum=d_log_mu_cum,
        d_log_a_cum=d_log_a_cum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    
    return dk, dv, dp, dlog_a, dlog_mu, dbeta
