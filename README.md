# Fast Block Sparse Matrices for Sparse Quantized Representation

This PyTorch extension provides a **drop-in replacement** for torch.nn.Linear using **block sparse matrices** instead of dense ones for SpQR [Dettmers et. al.].

It enables very easy experimentation with sparse matrices since you can directly replace Linear layers in your model with sparse ones.

## Motivation
The goal of this library is to show that **sparse matrices can be used in neural networks**, instead of dense ones, without significantly altering the precision.  

Combined with dense and sparse quantization using block structures based on SpQR, this method produces models which are 50% smaller than the original with lesser overheads due to clock sparse representation of the weights.

## Original code
This work is based on the [pytorch block sparse]([https://github.com/huggingface/pytorch_block_sparse]) proof of concept by [Yulhwa Kim](https://github.com/YulhwaKim).

It is using C++ CUDA templates for block-sparse matrix multiplication based on [CUTLASS](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/).

## Basic usage
You can use the BlockSparseLinear drop in replacement for torch.nn.Linear in your own model:

```python
# from torch.nn import Linear
from pytorch_block_sparse import BlockSparseLinear

...

# self.fc = nn.Linear(1024, 256)
self.fc = BlockSparseLinear(1024, 256, density=0.1)
```

## Advanced usage: converting whole models

Or you can use a utility called BlockSparseModelPatcher to modify easily an existing model before training it. (you will need to train it from scratch rather than sparsifying a pre-trained model).

## Installation

python setup.py install


