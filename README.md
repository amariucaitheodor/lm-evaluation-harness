# `lm-evaluation-harness` + `promptsource`

![](https://github.com/EleutherAI/lm-evaluation-harness/workflows/Build/badge.svg)
[![codecov](https://codecov.io/gh/EleutherAI/lm-evaluation-harness/branch/master/graph/badge.svg?token=JSG3O2427J)](https://codecov.io/gh/EleutherAI/lm-evaluation-harness)

## Overview

This project provides the backend for the BabyLM Challenge's evaluation pipeline. This is based on the `bigscience-workshop` branch of the `lm-evaluation-harness` repository, originally from EleutherAI.

The BLiMP prompts are provided via `promptsource`, while the COMPS and GLUE prompts are implemented by the BabyLM organizing committee.

If you have questions about or suggestions for this code, please contact Aaron Mueller. We also welcome pull requests!

We also provide a [Colab version](https://colab.research.google.com/drive/1HX2D3wztO81tKcqCeV_ecRcEUseBVuTc?usp=sharing) of the evaluation pipeline as a demonstration.

## Installation

Assuming you're using a GPU enabled for CUDA 11.3:

```bash
git clone https://github.com/bigscience-workshop/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e ".[dev]"
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 promptsource==0.2.3 --extra-index-url https://download.pytorch.org/whl/cu113
```

## CLI Usage 🖥️
### Zero-shot
To evaluate a model on zero-shot tasks like BLiMP:

```bash
python babylm_eval.py \
    'path/to/model_and_tokenizer' \
    'model_type' \
    --tasks 'blimp' \
```

Where `model_type` is one of "encoder", "decoder" or "encoder-decoder".

### Fine-tuning
To evaluate a model on tasks that require fine-tuning, like the (Super)GLUE tasks:

```bash
./finetune_all_tasks.sh 'path/to/model_and_tokenizer'
```

This script contains strong defaults that should work for a variety of model sizes. You may adjust these hyperparameters as you wish, though we ask that you submit the best hyperparmeter settings in a README file if you don't use the defaults.