# BabyLM Evaluation Pipeline
![BabyLM Challenge](assets/babylm.png)

## Overview

This code provides the backend for the BabyLM Challenge's evaluation pipeline. 

We provide support for zero-shot evaluations on BLiMP, as well as scripts for fine-tuning HuggingFace-based models on GLUE tasks.

We also provide a [Colab demo](https://colab.research.google.com/drive/1HX2D3wztO81tKcqCeV_ecRcEUseBVuTc?usp=sharing) of the evaluation pipeline as a demonstration of how to use the code.

If you have questions about or suggestions for this code, please contact Aaron Mueller. We also welcome pull requests!

## Installation

This code assumes you have access using a GPU enabled for CUDA 11.3. Other CUDA versions should work as long as the `torch` and `transformers` versions are sufficiently up-to-date.

```bash
git clone https://github.com/bigscience-workshop/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e ".[dev]"
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 promptsource==0.2.3 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Usage
### Zero-shot Evaluation
To evaluate a model on zero-shot tasks like BLiMP:

```bash
python babylm_eval.py 'path/to/model_and_tokenizer' 'model_type'
```

Where `model_type` is one of "encoder", "decoder" or "encoder-decoder".

### Fine-tuning
To fine-tune and evaluate a model on tasks that require fine-tuning, like the (Super)GLUE tasks:

```bash
./finetune_all_tasks.sh 'path/to/model_and_tokenizer'
```

This script contains strong hyperparameter defaults that should work for a variety of model sizes. You may adjust these hyperparameters as you wish, though we ask that you submit the best hyperparmeter settings in a README file if you don't use the defaults.

## Uploading Results
We provide a shell script that will collect your results into a single file:

```bash
./collect_results.sh path/to/model_and_tokenizer
```

We will ask you to share your results, model, and tokenizer. We will evaluate on held-out tasks (TBA) as part of the final evaluation.