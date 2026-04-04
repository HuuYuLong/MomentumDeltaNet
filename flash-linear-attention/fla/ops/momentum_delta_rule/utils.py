# v5
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils.index import prepare_chunk_indices

NUM_WARPS = [1, 2, 4, 8, 16]
# NUM_WARPS = [ 4]

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        # for num_warps in [1, 2, 4, 8, 16]
        for num_warps in NUM_WARPS
    ],
    key=['B', 'H', 'BT', 'IS_VARLEN']
)
@triton.jit(do_not_specialize=['T'])
def chunk_mode_rule_cumsum_scalar_fwd_kernel(
        log_alpha,
        log_mu,
        beta,
        log_a_cum,
        log_mu_cum,
        log_ct,
        # log_c_before,
        cu_seqlens,
        chunk_indices,
        T,
        B: tl.constexpr,
        H: tl.constexpr,
        BT: tl.constexpr,
        IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_log_a       = tl.make_block_ptr(log_alpha   + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_log_m       = tl.make_block_ptr(log_mu      + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_beta        = tl.make_block_ptr(beta        + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_log_a_cum   = tl.make_block_ptr(log_a_cum   + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_log_mu_cum  = tl.make_block_ptr(log_mu_cum  + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_log_ct      = tl.make_block_ptr(log_ct      + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    
    # [BT]
    b_log_a = tl.load(p_log_a, boundary_check=(0,)).to(tl.float32)
    b_log_m = tl.load(p_log_m, boundary_check=(0,)).to(tl.float32)
    b_beta  = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
    
    eps = tl.zeros([1], dtype=tl.float32) + 1e-6

    b_log_a_cum = tl.cumsum(b_log_a, axis=0)
    b_log_m_cum = tl.cumsum(b_log_m, axis=0)
    b_log_beta  = tl.log(b_beta + eps)                               # the assert should be added at outer of kernel

    b_log_c = b_log_beta + tl.cumsum(b_log_m - b_log_a, axis=0)   

    # Directly use global max value also will reulting '-inf' value  
    
    # Safe logcumsumexp with O(Chunk_length^2)
    #   safe_log_c_matrix = torch.where(i[:, None] >= i[None, :], log_c_before[None, :] , float('-inf')) 
    #   row_max = safe_log_c_matrix.max(1).values 
    #   row_sum = torch.sum(torch.exp(safe_log_c_matrix - row_max[:, None]), dim=1)
    #   b_log_ct = torch.log(row_sum) + row_max
    neg_inf = tl.zeros([1], dtype=tl.float32) - float("inf")
    o_t = i_t * BT + tl.arange(0, BT)
    m_A = (o_t[:, None] >= o_t[None, :])                                              # tril mask
    b_log_c_matrix = tl.where(m_A, b_log_c[None, :], neg_inf)
    b_row_max = tl.max(b_log_c_matrix, axis=1)
    b_ct = tl.sum(tl.exp(b_log_c_matrix - b_row_max[:, None]), axis=1)
    b_log_ct = tl.log(b_ct) + b_row_max

    tl.store(p_log_a_cum, b_log_a_cum.to(p_log_a_cum.dtype.element_ty), boundary_check=(0,))
    tl.store(p_log_mu_cum, b_log_m_cum.to(p_log_mu_cum.dtype.element_ty), boundary_check=(0,))
    tl.store(p_log_ct, b_log_ct.to(p_log_ct.dtype.element_ty), boundary_check=(0,))
    

def chunk_mode_rule_cumsum_scalar_fwd(
        log_alpha: torch.Tensor,
        log_mu: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int,
        cu_seqlens: Optional[torch.Tensor] = None,
        output_dtype: Optional[torch.dtype] = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    B, T, H = log_alpha.shape
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"

    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    log_a_cum = torch.empty_like(log_alpha, dtype=output_dtype or log_alpha.dtype)
    log_mu_cum = torch.empty_like(log_mu, dtype=output_dtype or log_mu.dtype)
    log_ct     = torch.empty_like(log_alpha, dtype=output_dtype or log_alpha.dtype)
    
    grid = (NT, B * H)
    chunk_mode_rule_cumsum_scalar_fwd_kernel[grid](
        log_alpha=log_alpha,
        log_mu=log_mu,
        beta=beta,
        log_a_cum=log_a_cum,
        log_mu_cum=log_mu_cum,
        log_ct=log_ct,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
    )
    return log_a_cum, log_mu_cum, log_ct


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        # for num_warps in [1, 2, 4, 8]
        for num_warps in NUM_WARPS
    ],
    key=['B', 'H', 'BT', 'IS_VARLEN']
)
@triton.jit(do_not_specialize=['T'])
def chunk_mode_rule_cumsum_scalar_bwd_kernel(
        d_log_a_cum,
        d_log_mu_cum,
        d_log_bar_a_tm1,
        d_log_cum_1_t,
        log_c_cum1t,
        log_c_before,
        dbeta,
        dlog_alpha,
        dlog_mu,
        cu_seqlens,
        chunk_indices,
        T,
        B: tl.constexpr,
        H: tl.constexpr,
        BT: tl.constexpr,
        IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_dlog_a_cum = tl.make_block_ptr(d_log_a_cum + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dlog_mu_cum = tl.make_block_ptr(d_log_mu_cum + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dlog_bar_a_tm1 = tl.make_block_ptr(d_log_bar_a_tm1 + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dlog_cum_1_t = tl.make_block_ptr(d_log_cum_1_t + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_log_c_cum_1_t = tl.make_block_ptr(log_c_cum1t + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_log_c_before = tl.make_block_ptr(log_c_before + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))

    #todo process d_log_bar_a_tm1
    b_dlog_a_cum = tl.load(p_dlog_a_cum, boundary_check=(0,)).to(tl.float32)
    b_dlog_mu_cum = tl.load(p_dlog_mu_cum, boundary_check=(0,)).to(tl.float32)
    b_dlog_bar_a_tm1 = tl.load(p_dlog_bar_a_tm1, boundary_check=(0,)).to(tl.float32)
    b_dlog_cum_1_t_shift = tl.load(p_dlog_cum_1_t, boundary_check=(0,)).to(tl.float32)
    b_log_cum_1_t = tl.load(p_log_c_cum_1_t, boundary_check=(0,)).to(tl.float32)
    b_log_c_before = tl.load(p_log_c_before, boundary_check=(0,)).to(tl.float32)

    w = b_dlog_bar_a_tm1 * tl.exp(-b_log_cum_1_t)
    r = tl.cumsum(w, axis=0, reverse=True)

    offs = tl.arange(0, BT)
    mask = (offs > 0) & (offs < BT - 1)
    b_dlog_cum_1_t_shift = tl.where(mask, b_dlog_cum_1_t_shift, 0.0)

    d_log_c_before = tl.exp(b_log_c_before) * r

    b_dlog_a_cum += b_dlog_cum_1_t_shift - d_log_c_before
    b_dlog_mu_cum += d_log_c_before
    b_dlog_beta = d_log_c_before

    b_dlog_alpha = tl.cumsum(b_dlog_beta, axis=0, reverse=True)
    b_dlog_mu = tl.cumsum(b_dlog_mu_cum, axis=0, reverse=True)

    # [BT]
    """
    b_dg:   1,2,3,4
    b_dg0:  0,1,2,3
    b_temp: 0,1,3,6
    b_dz:   6
    b_dgr:  6,5,3,0
    """
    # b_temp = tl.cumsum(b_dg0, axis=0)
    # b_dz = tl.sum(b_dg0, axis=0)
    # b_dgr = -b_temp + b_dz[None]
    # dbeta, dlog_alpha, dlog_mu

    p_dlog_alpha = tl.make_block_ptr(dlog_alpha + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dlog_mu = tl.make_block_ptr(dlog_mu + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dbeta = tl.make_block_ptr(dbeta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))

    tl.store(p_dlog_alpha, b_dlog_alpha.to(p_dlog_alpha.dtype.element_ty), boundary_check=(0,))
    tl.store(p_dlog_mu, b_dlog_mu.to(p_dlog_mu.dtype.element_ty), boundary_check=(0,))
    tl.store(p_dbeta, b_dlog_beta.to(p_dbeta.dtype.element_ty), boundary_check=(0,))


def chunk_mode_rule_cumsum_scalar_bwd(
        d_log_a_cum: torch.Tensor,
        d_log_mu_cum: torch.Tensor,
        d_log_bar_a_tm1: torch.Tensor,
        d_log_cum_1_t: torch.Tensor,
        log_c_cum1t: torch.Tensor,
        log_c_before: torch.Tensor,
        chunk_size: int,
        cu_seqlens: Optional[torch.Tensor] = None,
        output_dtype: Optional[torch.dtype] = torch.float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    B, T, H = d_log_a_cum.shape
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dbeta       = torch.empty_like(d_log_a_cum, dtype=output_dtype or d_log_a_cum.dtype)
    dlog_alpha  = torch.empty_like(d_log_a_cum, dtype=output_dtype or d_log_a_cum.dtype)
    dlog_mu     = torch.empty_like(d_log_a_cum, dtype=output_dtype or d_log_a_cum.dtype)

    grid = (NT, B * H)
    chunk_mode_rule_cumsum_scalar_bwd_kernel[grid](
        d_log_a_cum=d_log_a_cum,
        d_log_mu_cum=d_log_mu_cum,
        d_log_bar_a_tm1=d_log_bar_a_tm1,
        d_log_cum_1_t=d_log_cum_1_t,
        dbeta=dbeta,
        dlog_alpha=dlog_alpha,
        dlog_mu=dlog_mu,
        log_c_cum1t=log_c_cum1t,
        log_c_before=log_c_before,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
    )
    return dbeta, dlog_alpha, dlog_mu



