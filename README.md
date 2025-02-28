# tensor_parallel

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

FlashAttention is only supported on CUDA 11.7 and above, so `pip3 install flash-attn --no-build-isolation` will only succeed if: 

1. the version of cuda you see when doing `nvidia-smi -l` matches the version you see when doing `nvcc -V`

2. the folder printed when you do `which nvcc` is within `echo $CUDA_HOME`

- For example, supposed your CUDA is version 12.1, then `which nvcc` is `/usr/local/cuda-12.1/bin/nvcc` and  `echo $CUDA_HOME` needs to be set to `export CUDA_HOME=/usr/local/cuda-12.1`