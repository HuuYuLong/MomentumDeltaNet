# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, v5

from typing import Optional

import pandas as pd
import torch
from einops import rearrange

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.momentum_delta_rule.utils import chunk_mode_rule_cumsum_scalar_bwd, chunk_mode_rule_cumsum_scalar_fwd
from fla.ops.momentum_delta_rule.wy_fast import chunk_scaled_dot_mode_rule_pkt_fwd, prepare_uyz_repr_bwd, recompute_u_y_z_fwd
from fla.ops.momentum_delta_rule.chunk_delta_h import chunk_mode_rule_bwd_dhu, chunk_mode_rule_fwd_h_recompute_by_vnew, chunk_mode_rule_fwd_inter_qS_qM
from fla.ops.momentum_delta_rule.chunk_o import chunk_mode_rule_bwd_dqkyz, chunk_mode_rule_fwd_o, chunk_mode_bwd_dv_local
 

from fla.ops.utils import chunk_local_cumsum, solve_tril
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def chunk_mode_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    log_alpha: torch.Tensor,
    log_mu: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_S: torch.Tensor,
    initial_M: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size:int = 64,
):
    assert chunk_size in [16, 32, 64]
    log_a_cum, log_mu_cum, log_ct = chunk_mode_rule_cumsum_scalar_fwd(
        log_alpha=log_alpha,
        log_mu=log_mu,
        beta=beta,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32
        )

    A, bt, gamma_mask_q = chunk_scaled_dot_mode_rule_pkt_fwd(
        k=k,
        p=p,
        log_a_cum=log_a_cum,
        log_mu_cum=log_mu_cum,
        log_ct=log_ct,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
        chunk_size=chunk_size
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        output_dtype=k.dtype
    )

    u, y, z = recompute_u_y_z_fwd(
        p=p,  # pseudo k
        v=v,
        A=A,
        log_a_cum=log_a_cum,
        bt=bt,
        cu_seqlens=cu_seqlens,
    )

    o_inter, v_new, final_S, final_M = chunk_mode_rule_fwd_inter_qS_qM(
        q=q,
        k=k,
        v=v,
        u=u,
        y=y,
        z=z,
        log_a_cum=log_a_cum,
        log_mu_cum=log_mu_cum,
        bt=bt,
        gamma_mask_q=gamma_mask_q,
        initial_S=initial_S,
        initial_M=initial_M,
        output_final_state=output_final_state,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

    o = chunk_mode_rule_fwd_o(
        q=q,
        k=k,
        v=v_new,
        o_inter=o_inter,
        gamma_mask_q=gamma_mask_q,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    ) 
    o.add_(o_inter)
    return o, A, final_S, final_M, bt, log_a_cum, log_mu_cum, log_ct, gamma_mask_q, v_new 


def chunk_mode_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    beta: torch.Tensor,
    log_ct: torch.Tensor,
    log_a_cum: torch.Tensor,
    log_mu_cum: torch.Tensor,
    gamma_mask_q: torch.Tensor,
    bt: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_S: torch.Tensor,
    initial_M: torch.Tensor,
    do: torch.Tensor,
    dst: torch.Tensor,
    dmt: torch.Tensor,
    hS: torch.Tensor = None, 
    hM: torch.Tensor = None, 
    v_new: torch.Tensor = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size = 64,
):
    assert chunk_size in [16, 32, 64] 
    
    u, y, z = recompute_u_y_z_fwd(
        p=p,  # pseudo k
        v=v,
        A=A,
        log_a_cum=log_a_cum,
        bt=bt,
        cu_seqlens=cu_seqlens,
    )
    

    hS, hM = chunk_mode_rule_fwd_h_recompute_by_vnew(
        k=k,
        v_new=v_new,
        log_a_cum=log_a_cum,
        log_mu_cum=log_mu_cum,
        bt=bt,
        gamma_mask_q=gamma_mask_q,
        initial_S=initial_S,
        initial_M=initial_M,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

    dv = chunk_mode_bwd_dv_local(
        q=q,
        k=k,
        gamma_mask_q=gamma_mask_q,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    
    ds, dm, ds0, dm0, dv = chunk_mode_rule_bwd_dhu(
        q=q,
        k=k,
        u=u,
        y=y,
        z=z,
        log_mu_cum=log_mu_cum,
        log_a_cum=log_a_cum,
        bt=bt,
        gamma_mask_q=gamma_mask_q,
        s0=initial_S,
        m0=initial_M,
        dst=dst,
        dmt=dmt,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

    dq, dk, dy, dz, d_log_mu_cum, d_log_a_cum, d_bt, d_Attn_do_v, d_decay_s  = chunk_mode_rule_bwd_dqkyz(
        q=q,
        k=k,
        v=v_new,
        do=do,
        s=hS,
        m=hM,
        ds=ds,
        dm=dm,  # 
        log_mu_cum=log_mu_cum,
        log_a_cum=log_a_cum,
        bt=bt,
        gamma_mask_q=gamma_mask_q,
        dv=dv,
        y=y,
        z=z,
        cu_seqlens=cu_seqlens,
        scale=scale,
    )
    del hS, hM, ds, dm
    
    dk2, dv, dp, dlog_alpha, dlog_mu, dbeta = prepare_uyz_repr_bwd(
        q=q,
        k=k,
        v=v,
        p=p,
        beta=beta,
        log_a_cum=log_a_cum,
        log_mu_cum=log_mu_cum,
        log_ct=log_ct,
        gamma_mask_q=gamma_mask_q,
        d_Attn_do_v=d_Attn_do_v,
        d_decay_s=d_decay_s,
        A=A,
        bt=bt,
        dbt=d_bt,
        d_log_mu_cum=d_log_mu_cum,
        d_log_a_cum=d_log_a_cum,
        du=dv,
        dy=dy,
        dz=dz,
        cu_seqlens=cu_seqlens,
        scale=scale,
    )
    
    dk.add_(dk2)
    return dq, dk, dv, dp, dlog_alpha, dlog_mu, dbeta, ds0, dm0


class Chunkmode_ruleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        log_alpha: torch.Tensor,
        log_mu: torch.Tensor,
        beta: torch.Tensor,
        eta: torch.Tensor,
        scale: float,
        initial_S: torch.Tensor,
        initial_M: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = True,
        use_p_times_alpha: bool = True,  # defaultly as True, to make p = alpha * Norm(k)
    ):
 
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
            p, p_rstd = l2norm_fwd(p)
        else:
            q_rstd, k_rstd, p_rstd = None, None, None

        o, A, final_S, final_M, bt,   \
            log_a_cum, log_mu_cum, log_ct, gamma_mask_q, v_new  = chunk_mode_rule_fwd(
            q=q,
            k=k if eta is None else (k * eta.unsqueeze(-1)).to(q.dtype),
            v=v,
            p=p if not use_p_times_alpha else (p * log_alpha.exp().unsqueeze(-1)).to(q.dtype),
            log_alpha=log_alpha,
            log_mu=log_mu,
            beta=beta,
            scale=scale,
            initial_S=initial_S,
            initial_M=initial_M,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        
        ctx.save_for_backward(q, k, v, p, eta, beta, A,
                            log_a_cum, log_mu_cum, log_ct, bt, gamma_mask_q,
                            initial_S, initial_M, cu_seqlens,
                            q_rstd, k_rstd, p_rstd, v_new, log_alpha
                            )
                            
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_p_times_alpha = use_p_times_alpha
        
        return o.to(q.dtype), final_S, final_M

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dst: torch.Tensor,
        dmt: torch.Tensor,
    ):
    
        q, k, v, p, eta, beta, A, \
            log_a_cum, log_mu_cum, log_ct, bt,  gamma_mask_q, \
            initial_S, initial_M, cu_seqlens, \
            q_rstd, k_rstd, p_rstd, v_new, log_alpha  = ctx.saved_tensors

        dq, dk, dv, dp, dlog_alpha, dlog_mu, dbeta, ds0, dm0 = chunk_mode_rule_bwd(
            q=q,
            k=k if eta is None else (k * eta.unsqueeze(-1)).to(q.dtype),
            v=v,
            p=p if not ctx.use_p_times_alpha else (p * log_alpha.exp().unsqueeze(-1)).to(q.dtype),
            beta=beta,
            log_ct=log_ct,
            log_a_cum=log_a_cum,
            log_mu_cum=log_mu_cum,
            gamma_mask_q=gamma_mask_q,
            bt=bt,
            A=A,
            scale=ctx.scale,
            initial_S=initial_S,
            initial_M=initial_M,
            do=do,
            dst=dst,
            dmt=dmt,
            v_new=v_new,
            cu_seqlens=cu_seqlens,
        )

        if eta is not None:
            deta = (dk * k).sum(-1).to(beta)
            dk = dk * eta.unsqueeze(-1)
        else:
            deta = None
        
        if ctx.use_p_times_alpha:
            alpha = log_alpha.exp()
            dlog_alpha.add_((dp * p).sum(-1).to(alpha) * alpha)
            dp = dp * alpha.unsqueeze(-1)

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
            dp = l2norm_bwd(p, p_rstd, dp)

        return dq.to(q), dk.to(k), dv.to(v), dp.to(p), \
            dlog_alpha.to(beta), dlog_mu.to(beta), dbeta.to(beta), deta, \
            None, ds0, dm0, None, None, None, None


@torch.compiler.disable
def chunk_mode_rule(     # MODE: MOmentum DElta
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_alpha: torch.Tensor,
    log_mu: torch.Tensor,
    p: torch.Tensor = None,
    beta: torch.Tensor = None,
    eta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_p_times_alpha: bool = True, 
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        p (torch.Tensor):
            auxiliary keys of shape `[B, T, H, K]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        # >>> from fla.ops.mode_rule import chunk_mode_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> b = torch.rand(H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> p = k * b[:, None]
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_mode_rule(
            q, k, v, p, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_mode_rule(
            q, k, v, p, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    # if initial_M is not None and initial_S is not None:
    #     assert initial_S.shape == initial_M.shape

    if p is None:
        p = k

    if scale is None:
        scale = k.shape[-1] ** -0.5
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    if eta is None:
        eta = torch.ones_like(q[..., 0])
    if p is None:
        p = k.clone()
    
    if initial_state is not None:
        initial_S, initial_M = initial_state[0], initial_state[1]
    else:
        initial_S, initial_M = None, None
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_S is not None and initial_S.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_S.shape[0]}."
            )

    o, final_S, final_M = Chunkmode_ruleFunction.apply(
        q,
        k,
        v,
        p,
        log_alpha,
        log_mu,
        beta,
        eta,
        scale,
        initial_S,
        initial_M,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        use_p_times_alpha,
    )
    if output_final_state:
        final_state = torch.stack([final_S, final_M], dim=0)
    else:
        final_state = None

    return o, final_state
