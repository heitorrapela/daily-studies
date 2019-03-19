import torch
import numpy as np
# Tensor of 5x7
# a = torch.FloatTensor(5,7)
# print a

# Initialize tensor with randomized normal distribution with mean = 0 and var = 1
a = torch.randn(5,7)
# print a
# print a.size()

# Difference from inplace and out-of-place
# Operations inplace have _
# Inplace : add_
# Outplace: add

# Fill tensor with value 3.5
a.fill_(3.5)
print a

b = a.add(4.0)

print(a,b)

# a.add_(4.0)
# print a

# Zero Indexing
b = a[0,3] # First row, and 4th column of a
# print b

# Tensor can be also indexed with Python Slicing
b = a[:, 3:5]
# print b

x = torch.ones(5,5)
print x

z = torch.Tensor(5,3)
z[:,0] = 10
z[:,1] = 100
z[:,2] = 13
print z

x.index_add_(1, torch.LongTensor([0,1,2]), z)
print x

# Convert torch Tensro to numpy array
a = torch.ones(5)
print a

b = a.numpy()
print b

a.add_(1)
# print(a)
# print b

# Convert numpy array to torch tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a,1,out=a)
print a
print b


if torch.cuda.is_available():
	a = torch.LongTensor(10).fill_(3).cuda()
	print(type(a))
	b = a.cpu