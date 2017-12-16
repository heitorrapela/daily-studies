from __future__ import print_function
import torch
import numpy
# construct a 5x3 matrix, uninitialized:

x = torch.Tensor(5,3)
print (x)

# construct a randomly initialized matrix
y = torch.rand(3,5)
print (y)

# Get x or y size
print(x.size())
print(y.size())

# Operations - Addition
z = torch.rand(5,3)
print(x+z)
print(torch.add(x,z))

# Output tensor
result = torch.Tensor(5,3)
torch.add(x,z,out=result)
print(result)

# Addition in-place
# adds x to z
z.add_(x)
print(z)

# Numpy-like indexing
print(x)
print(x[:,1])

# Numpy bridge
a = torch.ones(5)
print (a)

b = a.numpy()
print (b)

######################

a.add_(1)
print(a)
# b is a in numpy form 
print(b)

a = numpy.ones(5)
b = torch.from_numpy(a)
numpy.add(a,1,out=a)
print (a)
print (b)

# Tensors can be moved onto GPU using the .cuda function
x = torch.ones(5)
z = torch.ones(5)
if torch.cuda.is_available():
	x = x.cuda()
	z = z.cuda()
	print(x + z)

