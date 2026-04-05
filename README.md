# Anonymized Code Submission

This repository contains the anonymized code package for the submitted paper. All identifying author and organization information has been removed from the package metadata and documentation.

## Contents

The layer implementation, triton kernel implementation and training framework are mainly based on the great open-source projects `flash-linear-attention` and `flame`. This codebase mainly includes:

- `flash-linear-attention/`: Main implementation of linear-attention layers, fused modules, and model definitions. Includes Triton kernels for high-performance computation.
- `flame/`: Training and experiment management framework. Contains all model configurations and training scripts for running experiments.
- `environment.yml`: Conda environment definition for reproducing experiments, including all necessary dependencies.

## Core Implementations

### Stepwise Momentum Delta Rule

The repository includes the key implementations of the stepwise momentum delta rule:

1. **PyTorch Reference Implementation** ([fla/ops/momentum_delta_rule/naive.py](flash-linear-attention/fla/ops/momentum_delta_rule/naive.py))
   - Functions: `recurrent_momentum_delta_rule_ref()`, `chunk_momentum_delta_rule_ref()`
   - Pure PyTorch implementation for understanding and validation
   - Suitable for testing and debugging

2. **Triton Kernel Implementation** ([fla/ops/momentum_delta_rule/chunk.py](flash-linear-attention/fla/ops/momentum_delta_rule/chunk.py))
   - Function: `chunk_mode_rule()`: Supports chunkwise forward and backward passes
   - Function: `fused_recurrent_mode_rule()`: Supports recurrent forward pass (in `fused_recurrent.py`)
   - High-performance CUDA kernel implementation using Triton

3. **Momentum DeltaNet Layer Implementation** ([fla/layers/momentum_deltanet.py](flash-linear-attention/fla/layers/momentum_deltanet.py))
   - Complete model layer combining Triton kernel implementations with gate constraints
   - Integrates with HuggingFace Transformers API
   - Supports both chunk and recurrent inference modes

### Configuration and Training Scripts

1. **Model Configurations** ([flame/configs](flame/configs)): JSON files defining model architectures and hyperparameters
2. **Training Scripts** ([flame/training_scripts](flame/training_scripts)): Bash scripts for launching training jobs with specific configurations


## Setup

The following requirements should be satisfied:

- PyTorch >= 2.5
- Triton >= 3.0 (or nightly version, see flash-linear-attention/FAQs)
- einops
- transformers >= 4.45.0
- datasets >= 3.3.0
- causal-conv1d >= 1.4.0

### Option 1: Create Conda Environment

```bash
conda env create -f environment.yml
conda activate llm
```

### Option 2: Install Minimal Dependencies

```bash
pip install torch>=2.5 triton>=3.0 einops transformers>=4.45 datasets>=3.3.0
```

## Install the Package

```bash
cd flame
pip install .
cd ..
cd flash-linear-attention
pip install -e .
```

## Minimal Usage Example

```python
import torch
from fla.layers import MomentumDeltaNet

model = MomentumDeltaNet(hidden_size=1024, num_heads=4)
x = torch.randn(2, 1024, 1024, device='cuda')
y, *_ = model(x)
print(y.shape)
```

## Reproduce Main Experiments

The `flame/configs/` directory contains all model configurations for different architectures and sizes.

The `flame/training_scripts/` directory contains training scripts for various models.

A typical reproducible command is:

```bash
cd flame
bash training_scripts/training_mdn_400M.sh
```

Before training, please check and replace `--model.tokenizer_path` and `--training.dataset` with the actual tokenizer and dataset paths used for reproduction. Ensure the dataset is preprocessed and tokenized appropriately.

## Notes

- This repository contains anonymized code used for paper submission.
- The codebase has been sanitized to remove author names, emails, and repository URLs.
