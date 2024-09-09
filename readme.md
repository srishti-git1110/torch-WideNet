# WideNet: Go Wider Instead of Deeper
PyTorch implementation of [WideNet: Go Wider Instead of Deeper](https://arxiv.org/abs/2105.04703). 

![WideNet](arch.png)

## Usage
1. Clone:

```
git clone https://github.com/srishti-git1110/torch-WideNet.git
cd widenet
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Usage

```python
import torch

from widenet import WideNet

inp_dim = 512
num_experts = 8
num_heads = 8
vocab_size = 50000
num_layers = 8

wide_net = WideNet(
    inp_dim, 
    num_experts, 
    num_heads, 
    vocab_size, 
    num_layers
).cuda()

x = torch.randn(2, 1024, inp_dim).cuda()
output, total_aux_loss = switch_transformer(x)
```
