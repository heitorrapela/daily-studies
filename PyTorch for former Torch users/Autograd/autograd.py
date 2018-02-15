import torch
from torch.autograd import Variable

x = Variable(torch.ones(2,2), requires_grad=True)
print x

print x.data

print x.grad

print x.grad_fn

y = x+2

print y

print y.grad_fn

z = y * y * 3
out = z.mean()
print out
print z, out

out.backward()
print(x.grad)

print z.grad_fn
print out.grad_fn
