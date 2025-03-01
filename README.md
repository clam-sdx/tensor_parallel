# tensor_parallel

## Requirements

BF16 (bfloat16) operations are only available on NVIDIA GPUs with compute capability 8.0 (Ampere architecture) or higher.

## Setup

```
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv python install 3.10
uv venv .venv --python 3.10
source .venv/bin/activate
pip config set global.index-url https://pypi.org/simple
pip install -e .
```


## Flash Attention

FlashAttention is only supported on CUDA 11.7 and above. 

If you run into trouble retrying with `pip3 install flash-attn --no-build-isolation`, it might be that your system is not configured in a way that the compiler can find the necessary CUDA components through default paths. Try these:

1. Make the version of cuda you see when doing `nvidia-smi -l` matche the version you see when doing `nvcc -V`

2. Make the folder printed when you do `which nvcc` within `echo $CUDA_HOME`

- For example, supposed your CUDA is version 12.1, then `which nvcc` is `/usr/local/cuda-12.1/bin/nvcc` and  `echo $CUDA_HOME` needs to be set to `export CUDA_HOME=/usr/local/cuda-12.1`


## Quickstart

```
torchrun --nproc_per_node 4 train.py --tp_size 4 --micro_batch_size 4 --gradient_accumulation_steps 8 --seq_len 1024 --max_tokens 4096000 --num_proc 16 --model_name TinyLlama/TinyLlama_v1.1 --num_hidden_layers 22 --num_attention_heads 32 --num_key_value_heads 4 --run_name tp_1B
```