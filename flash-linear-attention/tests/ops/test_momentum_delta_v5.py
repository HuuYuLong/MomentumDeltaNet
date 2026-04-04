# -*- coding: utf-8 -*-

import os
from typing import List

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.utils import  assert_close, device, is_intel_alchemist 
from fla.ops.momentum_delta_rule import fused_recurrent_mode_rule, chunk_mode_rule
from fla.ops.momentum_delta_rule.naive import recurrent_momentum_delta_rule_ref, chunk_momentum_delta_rule_ref


test_b_list = [2]
test_t_list = [1, 15, 63, 300,  512, 2048, 4096] 
test_d_list = [ 64, 32, 100, 256,]
test_gate_list = [0.8, 1.0, 2.0, 4.0, 8.0]


test_h_list = [2]
test_hv_list = [4]

 

@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('HV', test_hv_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('gate_logit_normalizer', test_gate_list)
@pytest.mark.parametrize('scale', [1, 0.1])
@pytest.mark.parametrize('use_qk_l2norm_in_kernel', [False, True])
@pytest.mark.parametrize('use_p_times_alpha', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '0',
    reason='Skipping test because TEST_CHUNK_VARLEN is enabled'
)
def test_torch_recurrent_torch_chunk_forward(
        B: int,
        T: int,
        H: int,
        HV: int,
        D: int,
        scale: float,
        dtype: torch.dtype,
        use_qk_l2norm_in_kernel: bool,
        use_p_times_alpha: bool,
        gate_logit_normalizer: float,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=torch.float32)
    k = torch.randn(B, T, H, D, dtype=torch.float32)
    p = torch.randn(B, T, H, D, dtype=torch.float32)
    v = torch.randn(B, T, HV, D, dtype=dtype)
    beta = (torch.rand(B, T, HV, dtype=dtype).sigmoid() *0.5)
    eta = torch.randn(B, T, HV, dtype=dtype)
    log_a = F.logsigmoid(torch.rand(B, T, HV, dtype=torch.float32)) / gate_logit_normalizer
    log_mu = F.logsigmoid(torch.rand(B, T, HV, dtype=torch.float32)) / gate_logit_normalizer
    s0 = torch.randn(B, HV, D, D, dtype=torch.float32)
    m0 = torch.randn(B, HV, D, D, dtype=torch.float32)
    q, k, v, p, beta, eta, log_a, log_mu, s0, m0 = map(lambda x: x.to(device).requires_grad_(),
                                                       (q, k, v, p, beta, eta, log_a, log_mu, s0, m0))
    p = F.normalize(repeat(k.clone(), 'b t h d -> b t (h g) d', g=HV // H), p=2, dim=-1).to(dtype)
    ref, ref_ht = recurrent_momentum_delta_rule_ref(
        q=F.normalize(repeat(q.clone(), 'b t h d -> b t (h g) d', g=HV // H), p=2, dim=-1).to(dtype),
        k=F.normalize(repeat(k.clone(), 'b t h d -> b t (h g) d', g=HV // H), p=2, dim=-1).to(dtype),
        p=p if not use_p_times_alpha else p*log_a.exp().unsqueeze(-1),
        v=v.clone(),
        beta=beta.clone(),
        eta=eta.clone(),
        log_alpha=log_a.clone(),
        log_mu=log_mu.clone(),
        scale=scale,
        initial_S=s0.clone(),
        initial_M=m0.clone(),
        output_final_state=True,
    )

    tri, tri_ht = chunk_momentum_delta_rule_ref(
        q=F.normalize(repeat(q.clone(), 'b t h d -> b t (h g) d', g=HV // H), p=2, dim=-1).to(dtype),
        k=F.normalize(repeat(k.clone(), 'b t h d -> b t (h g) d', g=HV // H), p=2, dim=-1).to(dtype),
        p=p.clone() if not use_p_times_alpha else p.clone()*log_a.exp().unsqueeze(-1),
        v=v.clone(),
        beta=beta.clone(),
        eta=eta.clone(),
        log_alpha=log_a.clone(),
        log_mu=log_mu.clone(),
        scale=scale,
        initial_S=s0.clone(),
        initial_M=m0.clone(),
        output_final_state=True,
    )

    assert_close('  o', ref, tri, 0.004)
    assert_close(' ht', ref_ht, tri_ht, 0.004)


@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('HV', test_hv_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('gate_logit_normalizer', test_gate_list)
@pytest.mark.parametrize('scale', [1, 0.1])
@pytest.mark.parametrize('use_qk_l2norm_in_kernel', [False, True])
@pytest.mark.parametrize('use_p_times_alpha', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '0',
    reason='Skipping test because TEST_CHUNK_VARLEN is enabled'
)
def test_recurrent_forward(
        B: int,
        T: int,
        H: int,
        HV: int,
        D: int,
        scale: float,
        dtype: torch.dtype,
        use_qk_l2norm_in_kernel: bool,
        use_p_times_alpha: bool,
        gate_logit_normalizer: float,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=torch.float32)
    k = torch.randn(B, T, H, D, dtype=torch.float32)
    p = torch.randn(B, T, H, D, dtype=torch.float32)
    v = torch.randn(B, T, HV, D, dtype=dtype)
    beta = (torch.rand(B, T, HV, dtype=dtype).sigmoid() * 0.25)
    eta = torch.rand(B, T, HV, dtype=dtype).sigmoid()
    log_a = F.logsigmoid(torch.rand(B, T, HV, dtype=torch.float32)) / gate_logit_normalizer
    log_mu = F.logsigmoid(torch.rand(B, T, HV, dtype=torch.float32)) / gate_logit_normalizer
    s0 = torch.randn(B, HV, D, D, dtype=torch.float32)
    m0 = torch.randn(B, HV, D, D, dtype=torch.float32)
    q, k, v, p, beta, eta, log_a, log_mu, s0, m0 = map(lambda x: x.to(device).requires_grad_(),
                                                       (q, k, v, p, beta, eta, log_a, log_mu, s0, m0))
        
    p=F.normalize(repeat(k.clone(), 'b t h d -> b t (h g) d', g=HV // H), p=2, dim=-1).to(dtype)
    ref, ref_ht = recurrent_momentum_delta_rule_ref(
        q=F.normalize(repeat(q.clone(), 'b t h d -> b t (h g) d', g=HV // H), p=2, dim=-1).to(dtype),
        k=F.normalize(repeat(k.clone(), 'b t h d -> b t (h g) d', g=HV // H), p=2, dim=-1).to(dtype),
        p=p.clone() if not use_p_times_alpha else p.clone()*log_a.clone().exp().unsqueeze(-1),
        v=v.clone(),
        beta=beta.clone(),
        eta=eta.clone(),
        log_alpha=log_a.clone(),
        log_mu=log_mu.clone(),
        scale=scale,
        initial_S=s0.clone(),
        initial_M=m0.clone(),
        output_final_state=True,
    )
    tri, tri_ht = fused_recurrent_mode_rule(
        q=F.normalize(q.clone(), p=2, dim=-1).to(dtype) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1).to(dtype) if not use_qk_l2norm_in_kernel else k.clone(), 
        p=F.normalize(k.clone(), p=2, dim=-1).to(dtype) if not use_qk_l2norm_in_kernel else k.clone(), 
        v=v.clone(),
        beta=beta.clone(),
        eta=eta.clone(),
        log_alpha=log_a.clone(),
        log_mu=log_mu.clone(),
        scale=scale,
        initial_state=torch.stack([s0.clone(),m0.clone()], dim=0),
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_p_times_alpha=use_p_times_alpha,
        output_final_state=True,
    )
    assert_close('  o', ref, tri, 0.002)
    assert_close(' ht', ref_ht, tri_ht, 0.002)

@pytest.mark.parametrize('B', test_b_list) 
@pytest.mark.parametrize('T', [64, 128, 1024, 2048, 4096]) 
# @pytest.mark.parametrize('T', [64]) 
@pytest.mark.parametrize('H', test_h_list + [4, 8]) 
@pytest.mark.parametrize('D', test_d_list + [128]) 
@pytest.mark.parametrize('gate_logit_normalizer', test_gate_list)
@pytest.mark.parametrize('scale', [1, 0.1]) 
@pytest.mark.parametrize('use_p_times_alpha', [False, True])
# @pytest.mark.parametrize('scale', [ 0.1]) 
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '0',
    reason='Skipping test because TEST_CHUNK_VARLEN is enabled'
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    scale: float,
    gate_logit_normalizer: float,
    use_p_times_alpha: bool
    # mask_p: float,
):
    if is_intel_alchemist and D > 128:
        pytest.skip(reason='mdn is not supported on alchemist for D>128')

    # q = torch.randn(B, T, H, D, dtype=dtype)
    # k = F.normalize(torch.randn(B, T, H, D, dtype=torch.floatc32), p=2, dim=-1).to(dtype)
    # v = torch.randn(B, T, H, D, dtype=dtype)
    # beta = torch.rand(B, T, H, dtype=dtype).sigmoid().fill_(1)
    # g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32))
    # h0 = torch.zeros(B, H, D, D, dtype=torch.float32)
    # g = g / gate_logit_normalizer
    # g = g * (torch.rand_like(g) > mask_p)
    # q, k, v, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, g, h0))
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=torch.float32)
    k = torch.randn(B, T, H, D, dtype=torch.float32)
    v = torch.randn(B, T, H, D, dtype=torch.float32)
    p = k.clone()
    beta = (torch.randn(B, T, H, dtype=dtype).sigmoid() * 0.5) 
    eta = torch.randn(B, T, H, dtype=dtype).sigmoid()

    log_a = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32)) / gate_logit_normalizer
    # log_a = log_a * (torch.rand_like(log_a) > mask_p)
    # log_a += torch.log(torch.Tensor([0.9]))

    log_mu = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32)) / gate_logit_normalizer
    # log_mu = log_mu * (torch.rand_like(log_mu) > mask_p)

    # s0 = torch.randn(B, H, D, D, dtype=torch.float32)
    # m0 = torch.randn(B, H, D, D, dtype=torch.float32)
    s0 = torch.randn(B, H, D, D, dtype=torch.float32)
    m0 = torch.randn(B, H, D, D, dtype=torch.float32)
    # h0 = torch.stack([s0, m0], dim=0)
    q, k, v, p, beta, eta, log_a, log_mu, s0, m0 = map(lambda x: x.to(device).requires_grad_(),
                                                       (q, k, v, p, beta, eta, log_a, log_mu, s0, m0))
    tri, tri_ht = chunk_mode_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        p=p.clone(),
        beta=beta.clone(),
        eta=eta.clone(),
        log_alpha=log_a.clone(),
        log_mu=log_mu.clone(),
        scale=scale,
        initial_state=torch.stack([s0.clone(), m0.clone()], dim=0),
        use_qk_l2norm_in_kernel=True,
        use_p_times_alpha=use_p_times_alpha,
        output_final_state=True,
    )
    do = torch.randn_like(v)
    dst = torch.randn_like(s0)
    dmt = torch.randn_like(m0)
    dht = torch.stack([dst, dmt], dim=0)

    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dp, tri_dbeta, tri_deta, tri_dlog_a, tri_dlog_mu, tri_ds0, tri_dm0 = \
        q.grad, k.grad, v.grad, p.grad, beta.grad, eta.grad, log_a.grad, log_mu.grad, s0.grad, m0.grad

    q.grad = k.grad = v.grad = p.grad = beta.grad = eta.grad = log_a.grad = log_mu.grad = s0.grad = m0.grad = None

    # ref, ref_ht = recurrent_momentum_delta_rule_ref(
    ref, ref_ht = chunk_momentum_delta_rule_ref(
        q=F.normalize(q.clone(), p=2, dim=-1).to(dtype),
        k=F.normalize(k.clone(), p=2, dim=-1).to(dtype),
        v=v.clone(),
        p=F.normalize(p.clone(), p=2, dim=-1).to(dtype) if not use_p_times_alpha else F.normalize(p.clone(), p=2, dim=-1).to(dtype) * log_a.exp().unsqueeze(-1),
        beta=beta.clone(),
        eta=eta.clone(),
        log_alpha=log_a.clone(),
        log_mu=log_mu.clone(),
        scale=scale,
        initial_S=s0.clone(),
        initial_M=m0.clone(),
        output_final_state=True,
    )

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dp, ref_dbeta, ref_deta, ref_dlog_a, ref_dlog_mu, ref_ds0, ref_dm0 = \
        q.grad, k.grad, v.grad, p.grad, beta.grad, eta.grad, log_a.grad, log_mu.grad, s0.grad, m0.grad

    assert_close('  o', ref, tri, 0.005)
    assert_close(' ht', ref_ht, tri_ht, 0.005)
    assert_close(' dq', ref_dq, tri_dq, 0.005)
    assert_close(' dk', ref_dk, tri_dk, 0.005)
    assert_close(' dv', ref_dv, tri_dv, 0.005)
    assert_close(' dp', ref_dp, tri_dp, 0.0065)
    assert_close(' ds0', ref_ds0, tri_ds0, 0.006)
    assert_close(' dm0', ref_dm0, tri_dm0, 0.006)
    assert_close(' deta', ref_deta, tri_deta, 0.007)
    assert_close(' dbeta', ref_dbeta, tri_dbeta, 0.007)
    assert_close(' dlog_a', ref_dlog_a, tri_dlog_a, 0.0166)
    assert_close(' dlog_m', ref_dlog_mu, tri_dlog_mu, 0.015)
 

# @pytest.mark.parametrize('H', [2])
# @pytest.mark.parametrize('D', [128])
# @pytest.mark.parametrize('cu_seqlens', [[0, 122, 229, 400, 1000]])
# @pytest.mark.parametrize('scale', [1])
# @pytest.mark.parametrize('mask_p', [0.5])
# @pytest.mark.parametrize('dtype', [torch.float16])
# @pytest.mark.skipif(
#     os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
#     reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set'
# )
# def test_chunk_varlen(
#     cu_seqlens: List[int],
#     H: int,
#     D: int,
#     scale: float,
#     mask_p: float,
#     dtype: torch.dtype,
# ):
#     if is_intel_alchemist and D > 128:
#         pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')
#     torch.manual_seed(42)
#     os.environ['TRITON_F32_DEFAULT'] = 'ieee'
#     # randomly split the sequence into N segments
#     cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
#     T = cu_seqlens[-1]
#     N = len(cu_seqlens) - 1
#
#     # seq-first required for inputs with variable lengths
#     q = torch.randn((1, T, H, D), dtype=dtype)
#     k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
#     v = torch.randn((1, T, H, D), dtype=dtype)
#     g = F.logsigmoid(torch.rand(1, T, H, dtype=dtype))
#     g = g * (torch.rand_like(g) > mask_p)
#     beta = torch.rand(1, T, H, dtype=dtype).sigmoid()
#     h0 = torch.randn((N, H, D, D), dtype=dtype)
#
#     q, k, v, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, beta, g, h0))
#     do = torch.randn_like(v)
#     dht = torch.rand_like(h0)
#
#     tri, tri_ht = chunk_gated_delta_rule(
#         q=q.clone(),
#         k=k.clone(),
#         v=v.clone(),
#         beta=beta.clone(),
#         g=g.clone(),
#         scale=scale,
#         output_final_state=True,
#         initial_state=h0.clone(),
#         cu_seqlens=cu_seqlens,
#     )
#     ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
#     tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
#     q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None
#
#     ref = []
#     ref_ht = []
#     for i in range(N):
#         ref_i, ref_ht_i = recurrent_gated_delta_rule_ref(
#             q=q[:, cu_seqlens[i]:cu_seqlens[i+1]],
#             k=k[:, cu_seqlens[i]:cu_seqlens[i+1]],
#             v=v[:, cu_seqlens[i]:cu_seqlens[i+1]],
#             beta=beta[:, cu_seqlens[i]:cu_seqlens[i+1]],
#             g=g[:, cu_seqlens[i]:cu_seqlens[i+1]],
#             scale=scale,
#             initial_state=h0[i],
#             output_final_state=True,
#         )
#         ref.append(ref_i)
#         ref_ht.append(ref_ht_i)
#     ref = torch.cat(ref, 1)
#     ref_ht = torch.cat(ref_ht, 0)
#
#     ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
#     ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
#
#     assert_close('  o', ref, tri, 0.005)
#     assert_close(' ht', ref_ht, tri_ht, 0.005)
#     assert_close(' dq', ref_dq, tri_dq, 0.007)
#     assert_close(' dk', ref_dk, tri_dk, 0.008)
#     assert_close(' dv', ref_dv, tri_dv, 0.007)
#     assert_close(' db', ref_dbeta, tri_dbeta, 0.015)
#     assert_close('dh0', ref_dh0, tri_dh0, 0.007)
#     assert_close(' dg', ref_dg, tri_dg, 0.015)
