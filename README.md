# Per-Layer Based Precision in PyTorch

One can assign different precision for different layers as the idea from [this paper](http://proceedings.mlr.press/v70/sakr17a/sakr17a.pdf)


## Requiremnts

- CUDA

pip packages:

- torch
- torchvision
- numpy
- pandas

## Usage

Use uniform precision as the same as original repo: (runs only one test)
```
python quantize.py --type cifar10 --quant_method linear --param_bits 8 --fwd_bits 8 --bn_bits 8 --gpu 0
```

Define precision of each layer in quantize_runner.py `param_bits`, `batch_norm_bits`, `layer_output_bits` and other settings and then run:
(currently only support uniform precision for `layer_output_bits`)
```
python quantize_runner.py
```

Results will be written to both result.csv and result.pkl.


## Attribution

This code is modified from [this](https://github.com/aaron-xichen/pytorch-playground) repository, which adopts uniform precision.