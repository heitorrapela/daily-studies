from __future__ import print_function
import torch
import numpy as np

# a = [1,1]
# b = [2,2]
a = torch.Tensor([1,1])
b = torch.Tensor([2,2])

print(a+b)

if (torch.cuda.is_available):
	a = a.cuda()
	b = b.cuda()
	print(a+b)

# Point mult
d = (a+a)*(b+b)

print(d)