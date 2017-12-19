# The autograd package provides automatic differentiation for all operations on Tensors. It is a define-by-run framework, which means that your backprop is defined by how your code is run, and that every single iteration can be different.
import torch
from torch.autograd import Variable

x = Variable(torch.ones(2,2), requires_grad=True)
print (x)

y = x+2
print (y)

print(y.grad_fn)

z = y*y*3
out = z.mean()
print z,out

out.backward()

print(x.grad)

x = torch.randn(3)
x = Variable(x, requires_grad=True)
print (x)
y = x*2

while y.data.norm() < 1000:
	y = y*2

print (y)

gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)
print(x)