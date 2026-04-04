# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.momentum_deltanet.configuration_momentum_deltanet import MomentumDeltaNetConfig
from fla.models.momentum_deltanet.modeling_momentum_deltanet import MomentumDeltaNetForCausalLM, MomentumDeltaNetModel

AutoConfig.register(MomentumDeltaNetConfig.model_type, MomentumDeltaNetConfig, exist_ok=True)
AutoModel.register(MomentumDeltaNetConfig, MomentumDeltaNetModel, exist_ok=True)
AutoModelForCausalLM.register(MomentumDeltaNetConfig, MomentumDeltaNetForCausalLM, exist_ok=True)

__all__ = ['MomentumDeltaNetConfig', 'MomentumDeltaNetForCausalLM', 'MomentumDeltaNetModel']
